"""Preserved ESI agent recipes wired to the extracted kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel

from mas_slm_research.agents.definition import AgentDefinition
from mas_slm_research.handoff import HandoffDefinition, create_handoff_tools
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.workflows.definition import WorkflowDefinition

from .doctor.prompt import SYSTEM_PROMPT as DOCTOR_PROMPT
from .doctor.schema import DoctorAgentInput, DoctorAgentOutput
from .esi1.handoffs import HANDOFFS as ESI1_HANDOFFS
from .esi1.prompt import HANDOFF_REQUIREMENTS as ESI1_HANDOFF_PROMPT
from .esi1.prompt import SINGLE_AGENT_OUTPUT_REQUIREMENTS as ESI1_SINGLE_PROMPT
from .esi1.prompt import SYSTEM_PROMPT as ESI1_PROMPT
from .esi1.schema import ES1AgentInput, ES1AgentOutput
from .esi2.handoffs import HANDOFFS as ESI2_HANDOFFS
from .esi2.prompt import HANDOFF_REQUIREMENTS as ESI2_HANDOFF_PROMPT
from .esi2.prompt import SINGLE_AGENT_OUTPUT_REQUIREMENTS as ESI2_SINGLE_PROMPT
from .esi2.prompt import SYSTEM_PROMPT as ESI2_PROMPT
from .esi2.schema import ES2AgentInput, ES2AgentOutput
from .esi345.handoffs import HANDOFFS as ESI345_HANDOFFS
from .esi345.prompt import HANDOFF_REQUIREMENTS as ESI345_HANDOFF_PROMPT
from .esi345.prompt import SINGLE_AGENT_OUTPUT_REQUIREMENTS as ESI345_SINGLE_PROMPT
from .esi345.prompt import SYSTEM_PROMPT as ESI345_PROMPT
from .esi345.schema import ES345AgentInput, ES345AgentOutput
from .single_agent_system.prompt import SYSTEM_PROMPT as BASELINE_PROMPT
from .single_agent_system.schema import SingleAgentInput, SingleAgentOutput, SingleAgentOutputV2
from .vitals.handoffs import HANDOFFS as VITALS_HANDOFFS
from .vitals.prompt import HANDOFF_REQUIREMENTS as VITALS_HANDOFF_PROMPT
from .vitals.prompt import SINGLE_AGENT_OUTPUT_REQUIREMENTS as VITALS_SINGLE_PROMPT
from .vitals.prompt import SYSTEM_PROMPT as VITALS_PROMPT
from .vitals.schema import VitalsAgentInput, VitalsAgentOutput

from mas_slm_research.tools.esi.compute_esi_danger_zone import compute_esi_danger_zone
from mas_slm_research.tools.esi.compute_shock_index import compute_shock_index
from mas_slm_research.tools.esi.create_plan import create_plan
from mas_slm_research.tools.esi.log_thought import log_thought


ESI_TOOLS = {
    "esi.compute_esi_danger_zone_v1": compute_esi_danger_zone,
    "esi.compute_shock_index_v1": compute_shock_index,
    "esi.create_plan_v1": create_plan,
    "esi.log_thought_v1": log_thought,
}


@dataclass(frozen=True)
class ESIAgentDefinition(AgentDefinition):
    name: str
    system_prompt: str
    output_schema: type[BaseModel]
    input_schema: type[BaseModel]
    tool_ids: tuple[str, ...]
    handoffs: tuple[HandoffDefinition, ...] = ()
    single_addon: str | None = None
    multi_addon: str | None = None

    def build_kernel(
        self,
        *,
        model: Any,
        runtime_config: RuntimeConfig,
        workflow: WorkflowDefinition | None,
        handoff_schemas: Mapping[str, str],
        registry: Any,
    ) -> AgentKernel:
        tools = [registry.resolve("tools", identifier) for identifier in self.tool_ids]
        handoff_names = [handoff.name for handoff in self.handoffs]
        if runtime_config.multi_agent:
            if workflow is None:
                raise ValueError(f"{self.name}: MAS construction requires a workflow")
            expected = set(workflow.allowed_targets_for(self.name))
            declared = {handoff.target_agent for handoff in self.handoffs}
            if expected != declared or expected != set(handoff_schemas):
                raise ValueError(f"{self.name}: unsupported handoff route override")
            selected: list[HandoffDefinition] = []
            for handoff in self.handoffs:
                schema_id = handoff_schemas[handoff.target_agent]
                schema = registry.resolve("schemas", schema_id)
                selected.append(HandoffDefinition(
                    source_agent=handoff.source_agent,
                    target_agent=handoff.target_agent,
                    payload_model=schema,
                    description=handoff.description,
                    tool_name=handoff.tool_name,
                ))
            tools.extend(create_handoff_tools(self.name, selected))
        elif handoff_schemas:
            raise ValueError(f"{self.name}: handoff schemas require MAS mode")
        return AgentKernel(
            model=model,
            tools=tools,
            system_prompt=self.system_prompt,
            single_agent_prompt_addon=self.single_addon,
            multi_agent_prompt_addon=self.multi_addon,
            response_format=self.output_schema,
            agent_node_name="agent" if self.name == "single_agent" else self.name,
            handoff_tool_names=handoff_names if runtime_config.multi_agent else (),
            handoff_workflow=workflow if runtime_config.multi_agent and handoff_names else None,
            runtime_config=runtime_config,
        )


BASE_TOOLS = ("esi.create_plan_v1", "esi.log_thought_v1")
VITALS_TOOLS = (
    "esi.compute_esi_danger_zone_v1", "esi.compute_shock_index_v1", *BASE_TOOLS,
)

ESI_AGENTS: dict[str, ESIAgentDefinition] = {
    "esi.single_agent_v1": ESIAgentDefinition(
        name="single_agent", system_prompt=BASELINE_PROMPT,
        input_schema=SingleAgentInput, output_schema=SingleAgentOutput,
        tool_ids=VITALS_TOOLS,
    ),
    "esi.single_agent_v2": ESIAgentDefinition(
        name="single_agent",
        system_prompt=BASELINE_PROMPT + "\n\nFor ESI-1 and ESI-2 final_answer calls, "
        "predicted_resources must be [] (never null); num_resources may be null. "
        "Resource prediction is only relevant on the ESI-3/4/5 pathway.",
        input_schema=SingleAgentInput, output_schema=SingleAgentOutputV2,
        tool_ids=VITALS_TOOLS,
    ),
    "esi.esi1_v1": ESIAgentDefinition(
        name="esi1_agent", system_prompt=ESI1_PROMPT,
        input_schema=ES1AgentInput, output_schema=ES1AgentOutput,
        tool_ids=BASE_TOOLS, handoffs=tuple(ESI1_HANDOFFS),
        single_addon=ESI1_SINGLE_PROMPT, multi_addon=ESI1_HANDOFF_PROMPT,
    ),
    "esi.esi2_v1": ESIAgentDefinition(
        name="esi2_agent", system_prompt=ESI2_PROMPT,
        input_schema=ES2AgentInput, output_schema=ES2AgentOutput,
        tool_ids=BASE_TOOLS, handoffs=tuple(ESI2_HANDOFFS),
        single_addon=ESI2_SINGLE_PROMPT, multi_addon=ESI2_HANDOFF_PROMPT,
    ),
    "esi.esi345_v1": ESIAgentDefinition(
        name="esi345_agent", system_prompt=ESI345_PROMPT,
        input_schema=ES345AgentInput, output_schema=ES345AgentOutput,
        tool_ids=BASE_TOOLS, handoffs=tuple(ESI345_HANDOFFS),
        single_addon=ESI345_SINGLE_PROMPT, multi_addon=ESI345_HANDOFF_PROMPT,
    ),
    "esi.vitals_v1": ESIAgentDefinition(
        name="vitals_agent", system_prompt=VITALS_PROMPT,
        input_schema=VitalsAgentInput, output_schema=VitalsAgentOutput,
        tool_ids=VITALS_TOOLS, handoffs=tuple(VITALS_HANDOFFS),
        single_addon=VITALS_SINGLE_PROMPT, multi_addon=VITALS_HANDOFF_PROMPT,
    ),
    "esi.doctor_v1": ESIAgentDefinition(
        name="doctor_agent", system_prompt=DOCTOR_PROMPT,
        input_schema=DoctorAgentInput, output_schema=DoctorAgentOutput,
        tool_ids=BASE_TOOLS,
    ),
}


ESI_SCHEMAS: dict[str, type[BaseModel]] = {
    "esi.single_agent_input_v1": SingleAgentInput,
    "esi.single_agent_output_v1": SingleAgentOutput,
    "esi.single_agent_output_v2": SingleAgentOutputV2,
    "esi.esi1_agent_input_v1": ES1AgentInput,
    "esi.esi1_agent_output_v1": ES1AgentOutput,
    "esi.esi2_agent_input_v1": ES2AgentInput,
    "esi.esi2_agent_output_v1": ES2AgentOutput,
    "esi.esi345_agent_input_v1": ES345AgentInput,
    "esi.esi345_agent_output_v1": ES345AgentOutput,
    "esi.vitals_agent_input_v1": VitalsAgentInput,
    "esi.vitals_agent_output_v1": VitalsAgentOutput,
    "esi.doctor_agent_input_v1": DoctorAgentInput,
    "esi.doctor_agent_output_v1": DoctorAgentOutput,
}
for handoffs in (ESI1_HANDOFFS, ESI2_HANDOFFS, ESI345_HANDOFFS, VITALS_HANDOFFS):
    for handoff in handoffs:
        source = handoff.source_agent.removesuffix("_agent")
        target = handoff.target_agent.removesuffix("_agent")
        ESI_SCHEMAS[f"esi.{source}_to_{target}_v1"] = handoff.payload_model
