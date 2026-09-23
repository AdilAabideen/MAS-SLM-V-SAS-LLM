"""An intentionally non-ESI benchmark registered through public toolkit contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from mas_slm_research.agents.definition import AgentDefinition
from mas_slm_research.grading import BaseGrader, GradeDecision, GradeResult, GradeStatus
from mas_slm_research.handoff import HandoffDefinition, create_handoff_tools, define_handoff
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.mas_contract import MASState
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.workflows.definition import WorkflowDefinition


class SumInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numbers: list[int] = Field(min_length=1)


class SumOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: int
    ok: bool = True


class SumHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: int
    method: str


@tool
def sum_numbers(numbers: list[int]) -> int:
    """Add all supplied integers and return the total."""
    return sum(numbers)


HANDOFFS = {
    "adder_agent": define_handoff(
        source_agent="adder_agent", target_agent="final_agent", payload_model=SumHandoff,
        description="Send the computed sum to the final reviewer.", tool_name="send_adder_result",
    ),
    "checker_agent": define_handoff(
        source_agent="checker_agent", target_agent="final_agent", payload_model=SumHandoff,
        description="Send an independent sum check to the final reviewer.", tool_name="send_checker_result",
    ),
}


@dataclass(frozen=True)
class ArithmeticAgent(AgentDefinition):
    name: str
    system_prompt: str
    output_schema: type[BaseModel] = SumOutput

    def build_kernel(
        self, *, model: Any, runtime_config: RuntimeConfig,
        workflow: WorkflowDefinition | None, handoff_schemas: Mapping[str, str], registry: Any,
    ) -> AgentKernel:
        tools = [registry.resolve("tools", "arithmetic.sum_numbers_v1")]
        handoff_names: tuple[str, ...] = ()
        if runtime_config.multi_agent and self.name in HANDOFFS:
            if workflow is None or set(workflow.allowed_targets_for(self.name)) != {"final_agent"}:
                raise ValueError("arithmetic worker requires its declared final-agent route")
            if handoff_schemas != {"final_agent": "arithmetic.sum_handoff_v1"}:
                raise ValueError("arithmetic worker requires its declared handoff schema")
            original = HANDOFFS[self.name]
            handoff = HandoffDefinition(
                source_agent=self.name, target_agent="final_agent",
                payload_model=registry.resolve("schemas", handoff_schemas["final_agent"]),
                description=original.description, tool_name=original.tool_name,
            )
            tools.extend(create_handoff_tools(self.name, [handoff]))
            handoff_names = (handoff.name,)
        elif handoff_schemas:
            raise ValueError("unexpected arithmetic handoff schema")
        return AgentKernel(
            model=model, tools=tools, system_prompt=self.system_prompt,
            response_format=self.output_schema, agent_node_name=self.name,
            handoff_tool_names=handoff_names,
            handoff_workflow=workflow if handoff_names else None,
            runtime_config=runtime_config,
        )


def worker_payload(state: MASState) -> dict[str, Any]:
    return {"llm_payload": {"numbers": list(state["case_info"]["numbers"])}}


def final_payload(state: MASState) -> dict[str, Any]:
    handoffs = [item for item in state.get("handoff_history", []) if item.get("target_agent") == "final_agent"]
    return {
        "llm_payload": {
            "numbers": list(state["case_info"]["numbers"]),
            "independent_results": [
                {"from_agent": item["from_agent"], **item["payload"]} for item in handoffs
            ],
        },
        "metadata": {"handoff_count": len(handoffs)},
    }


class ExactSumGrader(BaseGrader):
    def validate_expected(self, expected: Mapping[str, Any]) -> None:
        if set(expected) != {"total"} or type(expected["total"]) is not int:
            raise ValueError("expected exactly one integer total")

    def evaluate(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> GradeDecision:
        predicted = SumOutput.model_validate(actual)
        passed = predicted.total == expected["total"]
        return GradeDecision(passed=passed, score=1.0 if passed else 0.0)

    def aggregate(self, results: Sequence[GradeResult]) -> Mapping[str, Any]:
        attempted = len(results)
        passed = sum(result.passed is True for result in results)
        graded = sum(result.status == GradeStatus.GRADED for result in results)
        return {"attempted": attempted, "graded": graded, "passed": passed,
                "accuracy_all_attempts": passed / attempted if attempted else None}


def register_components(registry: Any) -> None:
    registry.register("tools", "arithmetic.sum_numbers_v1", sum_numbers)
    registry.register("schemas", "arithmetic.sum_input_v1", SumInput)
    registry.register("schemas", "arithmetic.sum_output_v1", SumOutput)
    registry.register("schemas", "arithmetic.sum_handoff_v1", SumHandoff)
    registry.register("agents", "arithmetic.single_v1", ArithmeticAgent(
        name="baseline", system_prompt="Add the provided integers using the registered tool, then return total.",
    ))
    for role in ("adder_agent", "checker_agent"):
        registry.register("agents", f"arithmetic.{role}_v1", ArithmeticAgent(
            name=role, system_prompt="Independently add the provided integers with the tool, then hand off total and method to final_agent.",
        ))
    registry.register("agents", "arithmetic.final_v1", ArithmeticAgent(
        name="final_agent", system_prompt="Review both independent results and return the final total.",
    ))
    registry.register("payload_builders", "arithmetic.worker_v1", worker_payload)
    registry.register("payload_builders", "arithmetic.final_v1", final_payload)
    registry.register("graders", "arithmetic.exact_sum_v1", ExactSumGrader)
