"""Resolved workflows own agent names, routes, and finalization rights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from mas_slm_research.mas_contract import AgentExecutionResult, HandoffEnvelope, HandoffResult
from mas_slm_research.runtime.handoff_policy import HandoffPolicy
from mas_slm_research.workflows.definition import WorkflowDefinition, WorkflowMetadata
from mas_slm_research.workflows.esi.definition import ESI_MAS


def tiny_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        metadata=WorkflowMetadata(workflow_id="review", name="Review", version="1"),
        participating_agents=("researcher", "reviewer"),
        start_agents=("researcher",),
        finalizing_agents=("reviewer",),
        allowed_handoffs={"researcher": ("reviewer",), "reviewer": ()},
    )


@pytest.mark.unit
def test_unrelated_workflow_accepts_its_route_and_rejects_unknown_or_reverse_routes():
    workflow = tiny_workflow()
    handoff = HandoffEnvelope(
        handoff_name="submit", from_agent="researcher", target_agent="reviewer",
        payload_schema="ReviewPayload", payload={"draft": "example"},
    )
    assert handoff.validate_for(workflow) is handoff
    assert AgentExecutionResult(agent_name="researcher", status="handoff", handoff=handoff).validate_for(workflow)
    assert AgentExecutionResult(agent_name="reviewer", status="final", final_output={"accepted": True}).validate_for(workflow)

    with pytest.raises(ValueError, match="Invalid handoff route"):
        HandoffEnvelope(
            handoff_name="reverse", from_agent="reviewer", target_agent="researcher",
            payload_schema="ReviewPayload", payload={},
        ).validate_for(workflow)
    with pytest.raises(ValueError, match="Unknown agent"):
        HandoffEnvelope(
            handoff_name="unknown", from_agent="researcher", target_agent="outsider",
            payload_schema="ReviewPayload", payload={},
        ).validate_for(workflow)
    with pytest.raises(ValueError, match="cannot finalize"):
        AgentExecutionResult(agent_name="researcher", status="final", final_output={}).validate_for(workflow)


@pytest.mark.unit
def test_esi_definition_and_handoff_policy_preserve_declared_routes():
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
    result = HandoffResult(
        handoff_name="to_doctor", from_agent="esi1_agent", target_agent="doctor_agent",
        payload_schema="ESI1Payload", payload={"is_esi1": True},
    )
    policy = HandoffPolicy(handoff_tool_names=["to_doctor"], workflow=ESI_MAS)
    tool_message = ToolMessage(content=result.model_dump_json(), name="to_doctor", tool_call_id="call-1")
    decision = policy.maybe_handoff_from_tool_result({"name": "to_doctor"}, tool_message)
    assert decision.should_handoff is True
    assert decision.envelope.target_agent == "doctor_agent"

    invalid = result.model_copy(update={"target_agent": "vitals_agent"})
    decision = policy.maybe_handoff_from_tool_result(
        {"name": "to_doctor"},
        ToolMessage(content=invalid.model_dump_json(), name="to_doctor", tool_call_id="call-2"),
    )
    assert decision.should_handoff is False
    assert decision.reason == "handoff_tool_result_invalid"


@pytest.mark.unit
def test_extracted_kernel_imports_from_source_tree_without_legacy_app(tmp_path):
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.workflows.esi.definition import ESI_MAS
assert ESI_MAS.allowed_targets_for('esi1_agent') == ('esi2_agent', 'doctor_agent')
assert not any(name == 'app' or name.startswith('app.') for name in sys.modules)
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(source_root)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
