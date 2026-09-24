"""The extracted graph preserves ESI routes with pure tracking."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from mas_slm_research.mas.agent_node_executor import AgentNodeExecutor
from mas_slm_research.mas.execution_strategy import CallableExecutionStrategy
from mas_slm_research.mas.gate_evaluator import GateEvaluator
from mas_slm_research.mas.graph_builder import MASGraphBuilder
from mas_slm_research.mas_contract import AgentExecutionResult, HandoffEnvelope, make_initial_mas_state
from mas_slm_research.mas_tracker import InMemoryMASTracker
from mas_slm_research.workflows.esi.definition import ESI_MAS
from mas_slm_research.workflows.esi.payload_builder import build_pending_agent_payload


CASE = {"chiefcomplaint": "chest pain", "age": 42, "heartrate": 111, "sbp": 96}


def build_graph(execute, tracker):
    return MASGraphBuilder(
        workflow=ESI_MAS,
        agent_executor=AgentNodeExecutor(
            workflow=ESI_MAS,
            strategy=CallableExecutionStrategy(mode="scripted", execute_fn=execute),
            payload_builder=build_pending_agent_payload,
            execution_tracker=tracker,
        ),
        gate_evaluator=GateEvaluator(workflow=ESI_MAS, execution_tracker=tracker),
    ).build()


@pytest.mark.integration
@pytest.mark.parametrize("acuity_path", [
    ("esi1_agent",),
    ("esi1_agent", "esi2_agent"),
    ("esi1_agent", "esi2_agent", "esi345_agent"),
])
def test_extracted_graph_preserves_every_esi_route_gate_and_role_projection(acuity_path):
    tracker = InMemoryMASTracker(workflow=ESI_MAS, mas_run_id="mas-case")
    requests = []

    async def execute(request):
        requests.append(request)
        name = request.agent_name
        assert "llm_payload" in request.pending_agent_payload
        if name == "doctor_agent":
            return AgentExecutionResult(agent_name=name, status="final", final_output={"esi": 2})
        target = "doctor_agent" if name == "vitals_agent" or name == acuity_path[-1] else acuity_path[acuity_path.index(name) + 1]
        return AgentExecutionResult(
            agent_name=name, status="handoff",
            handoff=HandoffEnvelope(
                handoff_name=f"{name}_to_{target}", from_agent=name, target_agent=target,
                payload_schema="test_payload", payload={"source": name},
            ),
        )

    state = asyncio.run(build_graph(execute, tracker).ainvoke(make_initial_mas_state(CASE)))
    assert state["final_output"] == {"esi": 2}
    assert set(state["completed_agents"]) == set(acuity_path) | {"vitals_agent", "doctor_agent"}
    assert sum(request.agent_name == "doctor_agent" for request in requests) == 1
    assert [record.ready for record in tracker.gate_records] == ([True] if len(acuity_path) == 1 else [False, True])
    doctor = next(record for record in tracker.agent_records.values() if record.agent_name == "doctor_agent")
    assert len(doctor.incoming_handoff_ids) == 2
    assert all(tracker.handoff_records[handoff_id].to_agent_run_id == doctor.agent_run_id for handoff_id in doctor.incoming_handoff_ids)
    assert tracker.final_output == {"esi": 2}

    payloads = {request.agent_name: request.pending_agent_payload["llm_payload"] for request in requests}
    for name in acuity_path:
        assert "heart_rate" not in payloads[name]["case_info"]
        assert payloads[name]["case_info"]["chief_complaint"] == CASE["chiefcomplaint"]
    assert payloads["doctor_agent"]["case_info"]["heart_rate"] == CASE["heartrate"]


@pytest.mark.integration
def test_strategy_exception_finishes_child_as_error_without_a_prediction():
    tracker = InMemoryMASTracker(workflow=ESI_MAS, mas_run_id="mas-error")

    async def execute(request):
        if request.agent_name == "esi1_agent":
            raise RuntimeError("provider unavailable")
        return AgentExecutionResult(
            agent_name="vitals_agent", status="handoff",
            handoff=HandoffEnvelope(
                handoff_name="vitals_to_doctor", from_agent="vitals_agent", target_agent="doctor_agent",
                payload_schema="test_payload", payload={},
            ),
        )

    state = asyncio.run(build_graph(execute, tracker).ainvoke(make_initial_mas_state(CASE)))
    assert state["final_output"] is None
    assert "doctor_agent" not in state["completed_agents"]
    esi1 = next(record for record in tracker.agent_records.values() if record.agent_name == "esi1_agent")
    assert esi1.status == "failed"
    assert esi1.finished_at is not None
    assert esi1.output["error"] == "agent_execution_failed"
    assert not any(record.status == "running" for record in tracker.agent_records.values())


@pytest.mark.integration
def test_extracted_graph_imports_without_orm_or_legacy_app(tmp_path):
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mas_slm_research.mas.graph_builder import MASGraphBuilder
from mas_slm_research.mas.agent_node_executor import AgentNodeExecutor
from mas_slm_research.workflows.esi.payload_builder import build_pending_agent_payload
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
assert not any(name == 'app' or name.startswith('app.') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(source_root)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
