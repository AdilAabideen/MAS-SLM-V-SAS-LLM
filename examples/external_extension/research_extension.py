"""A self-contained researcher-owned word-count extension, outside toolkit source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict

from mas_slm_research.agents.definition import AgentDefinition
from mas_slm_research.configuration import WorkflowFileSpec
from mas_slm_research.grading import BaseGrader, GradeDecision, GradeResult, GradeStatus
from mas_slm_research.handoff import HandoffDefinition, create_handoff_tools, define_handoff
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.mas_contract import MASState
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.workflows.definition import WorkflowDefinition


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class WordCountOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int
    ok: bool = True


class CountHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int
    method: str


@tool
def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


WORKER_HANDOFF = define_handoff(
    source_agent="counter_agent", target_agent="reviewer_agent", payload_model=CountHandoff,
    description="Send the counted words to the reviewer.", tool_name="send_count_to_reviewer",
)


@dataclass(frozen=True)
class WordCountAgent(AgentDefinition):
    name: str
    system_prompt: str
    output_schema: type[BaseModel] = WordCountOutput

    def build_kernel(
        self, *, model: Any, runtime_config: RuntimeConfig,
        workflow: WorkflowDefinition | None, handoff_schemas: Mapping[str, str], registry: Any,
    ) -> AgentKernel:
        tools = [registry.resolve("tools", "research.count_words_v1")]
        handoff_names: tuple[str, ...] = ()
        if runtime_config.multi_agent and self.name == "counter_agent":
            if workflow is None or tuple(workflow.allowed_targets_for(self.name)) != ("reviewer_agent",):
                raise ValueError("counter requires the reviewer route")
            if handoff_schemas != {"reviewer_agent": "research.count_handoff_v1"}:
                raise ValueError("counter requires the registered handoff schema")
            handoff = HandoffDefinition(
                source_agent=self.name, target_agent="reviewer_agent",
                payload_model=registry.resolve("schemas", handoff_schemas["reviewer_agent"]),
                description=WORKER_HANDOFF.description, tool_name=WORKER_HANDOFF.tool_name,
            )
            tools.extend(create_handoff_tools(self.name, [handoff]))
            handoff_names = (handoff.name,)
        elif handoff_schemas:
            raise ValueError("unexpected handoff schema")
        return AgentKernel(
            model=model, tools=tools, system_prompt=self.system_prompt,
            response_format=self.output_schema, agent_node_name=self.name,
            handoff_tool_names=handoff_names,
            handoff_workflow=workflow if handoff_names else None,
            runtime_config=runtime_config,
        )


def counter_payload(state: MASState) -> dict[str, Any]:
    return {"llm_payload": {"text": state["case_info"]["text"]}}


def reviewer_payload(state: MASState) -> dict[str, Any]:
    handoffs = [item for item in state.get("handoff_history", []) if item.get("target_agent") == "reviewer_agent"]
    return {
        "llm_payload": {
            "text": state["case_info"]["text"],
            "reported_counts": [
                {"from_agent": item["from_agent"], **item["payload"]} for item in handoffs
            ],
        },
        "metadata": {"handoff_count": len(handoffs)},
    }


class ExactWordCountGrader(BaseGrader):
    def validate_expected(self, expected: Mapping[str, Any]) -> None:
        if set(expected) != {"count"} or type(expected["count"]) is not int:
            raise ValueError("expected exactly one integer count")

    def evaluate(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> GradeDecision:
        answer = WordCountOutput.model_validate(actual)
        passed = answer.count == expected["count"]
        return GradeDecision(passed=passed, score=1.0 if passed else 0.0)

    def aggregate(self, results: Sequence[GradeResult]) -> Mapping[str, Any]:
        attempted = len(results)
        passed = sum(result.passed is True for result in results)
        graded = sum(result.status == GradeStatus.GRADED for result in results)
        return {"attempted": attempted, "graded": graded, "passed": passed,
                "accuracy_all_attempts": passed / attempted if attempted else None}


def register_components(registry: Any) -> None:
    """Register implementations once, using the adjacent YAML as the workflow contract."""
    workflow_path = Path(__file__).with_name("workflow.yaml")
    workflow = WorkflowFileSpec.model_validate(yaml.safe_load(workflow_path.read_text(encoding="utf-8"))).to_workflow()
    registry.register("workflows", "research.word_count_v1", workflow)
    registry.register("tools", "research.count_words_v1", count_words)
    registry.register("schemas", "research.text_input_v1", TextInput)
    registry.register("schemas", "research.count_output_v1", WordCountOutput)
    registry.register("schemas", "research.count_handoff_v1", CountHandoff)
    registry.register("agents", "research.single_v1", WordCountAgent(
        name="baseline", system_prompt="Use count_words and return the word count.",
    ))
    registry.register("agents", "research.counter_v1", WordCountAgent(
        name="counter_agent", system_prompt="Use count_words, then hand the count to reviewer_agent.",
    ))
    registry.register("agents", "research.reviewer_v1", WordCountAgent(
        name="reviewer_agent", system_prompt="Review the counted words and return the final count.",
    ))
    registry.register("payload_builders", "research.counter_payload_v1", counter_payload)
    registry.register("payload_builders", "research.reviewer_payload_v1", reviewer_payload)
    registry.register("graders", "research.exact_word_count_v1", ExactWordCountGrader)
