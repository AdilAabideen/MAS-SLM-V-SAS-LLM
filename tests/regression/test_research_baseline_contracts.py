"""Executable guards for the pre-extraction ESI research contracts.

These digests describe the baseline, not a claim that its prompts or policies are
clinically correct. Full prompt/tool/schema snapshots live in the private
Milestone 01 contract inventory.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from app.agentic.agents.agents import AGENTS
from app.agentic.model_registry import (
    FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES,
    resolve_model_spec,
)
from app.agentic.runtime import AgentRuntime, RuntimeConfig
from app.agentic.workflows.definitions.esi_mas.workflow_definition import ESI_MAS
from tests.doubles.fake_provider import FakeChatModel


PROMPT_DIGESTS = {
    ("vitals_agent", "single"): "ac10fad4509482eb05860491e776cf3204bdebdf11f36035eb32a7af63e6ed77",
    ("vitals_agent", "multi"): "c9655a9e9df4779a98f761152ab624bbe0f567aff9d2b98a56126b1cbcf06136",
    ("doctor_agent", "single"): "064e78293401b857d9b8838b7de47524c0af0ba32e263fceaadbfce40f09fac2",
    ("doctor_agent", "multi"): "064e78293401b857d9b8838b7de47524c0af0ba32e263fceaadbfce40f09fac2",
    ("esi1_agent", "single"): "b5d56aab25b6a4b5468bfbb16565773155aa71ee3a6b63598a29e64a9f8bd5c1",
    ("esi1_agent", "multi"): "c2972e78707debe8bc968c350ae8018fa5ff388062bb3d5539aeda6e784a128e",
    ("esi2_agent", "single"): "b09d295a3bf748410862693587ff41dca1154b5b1af74b78b52d2607d2a92a97",
    ("esi2_agent", "multi"): "b4bb9f8c52189d3bb14997c6e353335657de9b1a19ac2a27504b7ddaf2dbe335",
    ("esi345_agent", "single"): "a45bf7e2429c17ca3e26fdf75bff52721228e138511e99e39867d1cfb29cd2ff",
    ("esi345_agent", "multi"): "f65e2018a95f671eed0169fc89378a4f86f27c90b29445e44e5b169790d28653",
    ("single_agent", "single"): "b6d717aeb1272d46985874ab048a93f4f577f22e2e8bb5d5c21ca4830d22993d",
}

# Ordered names, descriptions, and JSON schemas, serialized with stable JSON keys.
# A digest change must be reviewed against the private Milestone 01 inventory.
TOOL_CONTRACT_DIGESTS = {
    ("vitals_agent", "single"): "89e3b0c494bcf173b7b6eb7b414e220c54504bd3ece61748e0a0c526a6b1a0ec",
    ("vitals_agent", "multi"): "93cada678f448ccd688468d71c263be8205b2deb5771ff3217e9e75252c84b89",
    ("doctor_agent", "single"): "1179004abe2e5ee1bf1e62700b023a4adb735941b8e0d06c44392b2b6ee6046a",
    ("doctor_agent", "multi"): "1179004abe2e5ee1bf1e62700b023a4adb735941b8e0d06c44392b2b6ee6046a",
    ("esi1_agent", "single"): "281c3961ade4e01a2157a754d3a58c6640089c1e1df1fccc4d05d2d89fa79ced",
    ("esi1_agent", "multi"): "1a4d68d15795d82e9e92b5bce66620d60e4afc6e7f2f3397bb66205bbdd11c35",
    ("esi2_agent", "single"): "6ee927cbfc1e2bb336f0fec0c65c62a0b4c756a234bc4ab3c776d20912d4d5ec",
    ("esi2_agent", "multi"): "372c6da221a83985fccfd5a1c530af0edb676849c2b19a79e34a7e396df45d10",
    ("esi345_agent", "single"): "8fbd1c9303301dc86d63f131ca1e6620f10579c502b82b1b5a9a2be9dbf65754",
    ("esi345_agent", "multi"): "1fa1a1746ef79e588b8e5884867bf0cd9a4c62f1922b8d81025908273960c9f5",
    ("single_agent", "single"): "c9e653856a12dd48f45f9045189b9f33eb5435d5b55fd7254403974b31db4791",
}

BASE_TOOLS = {
    "vitals_agent": ["compute_esi_danger_zone", "compute_shock_index", "create_plan", "log_thought"],
    "doctor_agent": ["create_plan", "log_thought"],
    "esi1_agent": ["create_plan", "log_thought"],
    "esi2_agent": ["create_plan", "log_thought"],
    "esi345_agent": ["create_plan", "log_thought"],
    "single_agent": ["compute_esi_danger_zone", "compute_shock_index", "create_plan", "log_thought"],
}

HANDOFF_TOOLS = {
    "vitals_agent": ["finalise_output"],
    "doctor_agent": [],
    "esi1_agent": ["final_esi1_false_handoff_to_esi2_agent", "final_esi1_true_handoff_to_doctor_agent"],
    "esi2_agent": ["final_esi2_false_handoff_to_esi345_agent", "final_esi2_true_handoff_to_doctor_agent"],
    "esi345_agent": ["final_esi345_result_handoff_to_doctor_agent"],
}


@pytest.mark.regression
@pytest.mark.parametrize("agent_name, mode", PROMPT_DIGESTS)
def test_pre_extraction_prompt_and_tool_construction(agent_name: str, mode: str) -> None:
    spec = AGENTS[agent_name]
    runtime = AgentRuntime(
        model_id="gpt-4o-mini",
        model_spec=resolve_model_spec("gpt-4o-mini"),
        model=FakeChatModel([]),
    )
    config = RuntimeConfig(multi_agent=True, print_events=False, persist_events=True) if mode == "multi" else None
    agent = spec.build(runtime, config) if config is not None else spec.build(runtime)

    assert hashlib.sha256(agent._render_system_prompt().encode()).hexdigest() == PROMPT_DIGESTS[(agent_name, mode)]
    tool_contract = [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.args_schema.model_json_schema() if tool.args_schema else None,
        }
        for tool in agent.tools
    ]
    contract_json = json.dumps(tool_contract, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(contract_json.encode()).hexdigest() == TOOL_CONTRACT_DIGESTS[(agent_name, mode)]
    assert [tool.name for tool in spec.tools] == BASE_TOOLS[agent_name]
    expected = list(BASE_TOOLS[agent_name])
    if mode == "multi" and agent_name != "doctor_agent":
        expected.extend(HANDOFF_TOOLS[agent_name])
    else:
        expected.append("final_answer")
    assert [tool.name for tool in agent.tools] == expected
    assert agent.runtime_config.multi_agent is (mode == "multi")


@pytest.mark.regression
def test_pre_extraction_workflow_routes_and_model_overrides() -> None:
    assert set(AGENTS) == set(BASE_TOOLS)
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
        "esi1_agent": "esi1-agent-075",
        "esi2_agent": "esi2-agent-025",
        "esi345_agent": "esi3-agent-075",
        "vitals_agent": "medgemma-4b-it",
        "doctor_agent": "medgemma-4b-it",
    }
