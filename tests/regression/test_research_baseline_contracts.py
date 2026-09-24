"""Preserved ESI contracts checked against the extracted research package."""

from __future__ import annotations

import hashlib
import json

import pytest

from mas_slm_research.agents.esi.definitions import ESI_AGENTS
from mas_slm_research.model_registry import FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.workflows.esi.definition import ESI_MAS
from tests.doubles.fake_provider import FakeChatModel
from tests.fixtures.esi_contract_digests import BASE_TOOLS, HANDOFF_TOOLS, PROMPT_DIGESTS, TOOL_CONTRACT_DIGESTS


@pytest.mark.regression
@pytest.mark.parametrize("agent_name, mode", PROMPT_DIGESTS)
def test_preserved_prompt_and_ordered_tools_match_pre_extraction_fingerprints(agent_name: str, mode: str) -> None:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    definition = next(item for item in ESI_AGENTS.values() if item.name == agent_name)
    schemas = {
        handoff.target_agent: (
            f"esi.{handoff.source_agent.removesuffix('_agent')}_to_"
            f"{handoff.target_agent.removesuffix('_agent')}_v1"
        ) for handoff in definition.handoffs
    } if mode == "multi" else {}
    kernel = definition.build_kernel(
        model=FakeChatModel([]), runtime_config=RuntimeConfig(multi_agent=mode == "multi"),
        workflow=ESI_MAS if mode == "multi" else None,
        handoff_schemas=schemas, registry=registry,
    )
    assert hashlib.sha256(kernel._render_system_prompt().encode()).hexdigest() == PROMPT_DIGESTS[(agent_name, mode)]
    tools = [{"name": tool.name, "description": tool.description,
              "schema": tool.args_schema.model_json_schema() if tool.args_schema else None}
             for tool in kernel.tools]
    contract = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(contract.encode()).hexdigest() == TOOL_CONTRACT_DIGESTS[(agent_name, mode)]
    expected = list(BASE_TOOLS[agent_name])
    expected.extend(HANDOFF_TOOLS[agent_name] if mode == "multi" and agent_name != "doctor_agent" else ["final_answer"])
    assert [tool.name for tool in kernel.tools] == expected
    assert kernel.runtime_config.multi_agent is (mode == "multi")


@pytest.mark.regression
def test_preserved_workflow_routes_and_model_overrides() -> None:
    assert {definition.name for definition in ESI_AGENTS.values()} == set(BASE_TOOLS)
    assert ESI_MAS.start_agents == ("esi1_agent", "vitals_agent")
    assert ESI_MAS.finalizing_agents == ("doctor_agent",)
    assert ESI_MAS.allowed_handoffs == {
        "esi1_agent": ("esi2_agent", "doctor_agent"),
        "esi2_agent": ("esi345_agent", "doctor_agent"),
        "esi345_agent": ("doctor_agent",),
        "vitals_agent": ("doctor_agent",),
        "doctor_agent": (),
    }
    assert ESI_MAS.gates["doctor_gate"].required_sources == ("acuity", "vitals")
    assert ESI_MAS.gates["doctor_gate"].target_node == "doctor_agent"
    assert FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES == {
        "esi1_agent": "esi1-agent-075", "esi2_agent": "esi2-agent-025",
        "esi345_agent": "esi3-agent-075", "vitals_agent": "medgemma-4b-it",
        "doctor_agent": "medgemma-4b-it",
    }
