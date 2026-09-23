"""In-memory MAS identities, handoffs, gates, and measurements survive without sinks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mas_slm_research.mas_contract import AgentExecutionResult, HandoffEnvelope, make_initial_mas_state
from mas_slm_research.mas_measurements import summarize_agent_measurements
from mas_slm_research.mas_tracker import InMemoryMASTracker
from mas_slm_research.workflows.definition import GateNodeDefinition, SourceDefinition, WorkflowDefinition, WorkflowMetadata


def parallel_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        metadata=WorkflowMetadata(workflow_id="parallel", name="Parallel", version="1"),
        participating_agents=("acuity", "vitals", "doctor"),
        sources={
            "acuity": SourceDefinition(source_id="acuity", name="Acuity", agent_names=("acuity",)),
            "vitals": SourceDefinition(source_id="vitals", name="Vitals", agent_names=("vitals",)),
        },
        start_agents=("acuity", "vitals"),
        finalizing_agents=("doctor",),
        allowed_handoffs={"acuity": ("doctor",), "vitals": ("doctor",), "doctor": ()},
        gates={"doctor_gate": GateNodeDefinition(
            gate_id="doctor_gate", name="Doctor gate", required_sources=("acuity", "vitals"),
            target_node="doctor",
        )},
    )


@pytest.mark.unit
@pytest.mark.parametrize("sink_enabled", [False, True])
def test_parallel_child_and_handoff_ids_remain_correlated_with_or_without_sink(sink_enabled):
    emitted = []
    ids = (f"id-{number}" for number in count(1))
    times = (datetime(2026, 9, 23, tzinfo=timezone.utc) + timedelta(milliseconds=number * 10) for number in count())
    workflow = parallel_workflow()
    tracker = InMemoryMASTracker(
        workflow=workflow, mas_run_id="mas-1", id_factory=lambda: next(ids),
        clock=lambda: next(times), event_sink=emitted.append if sink_enabled else None,
    )
    state = make_initial_mas_state({"case": "demo"})

    starts = {}
    for role in ("acuity", "vitals"):
        starts[role] = tracker.begin_agent_execution(agent_name=role, state=state, pending_agent_payload={"role": role})
    assert starts["acuity"].sequence_index != starts["vitals"].sequence_index

    handoffs = []
    for role in ("acuity", "vitals"):
        result = AgentExecutionResult(
            agent_name=role, status="handoff",
            handoff=HandoffEnvelope(
                handoff_name=f"{role}_to_doctor", from_agent=role, target_agent="doctor",
                payload_schema="TestPayload", payload={"source": role},
            ),
        )
        completed = tracker.complete_agent_execution(tracked=starts[role], result=result)
        handoffs.append(tracker.decorate_handoff_dict(
            handoff_dict=result.handoff.model_dump(), persisted=completed,
        ))

    state["handoff_history"] = handoffs
    outcome = SimpleNamespace(
        ready=True, satisfied_sources=["acuity", "vitals"], missing_sources=[],
        next_target="doctor", handoffs_to_target=handoffs,
    )
    gate_id = tracker.record_gate_evaluation(gate_id="doctor_gate", state=state, outcome=outcome)
    doctor = tracker.begin_agent_execution(agent_name="doctor", state=state, pending_agent_payload={"role": "doctor"})
    tracker.complete_agent_execution(
        tracked=doctor,
        result=AgentExecutionResult(agent_name="doctor", status="final", final_output={"answer": 2}),
    )

    doctor_record = tracker.agent_records[doctor.agent_run_id]
    assert set(doctor_record.incoming_handoff_ids) == set(tracker.handoff_records)
    assert all(record.to_agent_run_id == doctor.agent_run_id for record in tracker.handoff_records.values())
    assert all(record.status == "accepted" and record.latency_ms >= 0 for record in tracker.handoff_records.values())
    assert tracker.gate_records[0].gate_evaluation_id == gate_id
    assert set(tracker.gate_records[0].handoff_ids) == set(tracker.handoff_records)
    assert tracker.final_output == {"answer": 2}
    assert [event["seq"] for event in tracker.events] == list(range(1, len(tracker.events) + 1))
    assert {event["event_type"] for event in tracker.events} >= {
        "agent_started", "handoff_created", "handoff_accepted", "gate_evaluated", "final_output_created",
    }
    assert len(emitted) == (len(tracker.events) if sink_enabled else 0)


@pytest.mark.unit
def test_pure_measurement_counts_match_preserved_sql_aggregation_rules():
    summary = summarize_agent_measurements(
        status="failed", duration_ms=300, error_text="timeout at provider", output={"error": "timeout"},
        llm_calls=[
            {"input_tokens": 10, "output_tokens": 4, "tokens_total": 14, "cost_usd": 0.01},
            {"input_tokens": 5, "output_tokens": 1, "tokens_total": 6, "cost_usd": None},
        ],
        events=[
            {"event_type": "tool_call"}, {"event_type": "tool_call"},
            {"event_type": "tool_result", "status": "error"},
        ],
        reliability_issues=[
            {"issue_code": "text_recovery_used", "severity": "warning"},
            {"issue_code": "final_output_invalid", "severity": "error"},
        ],
        schema_validator=lambda output: (_ for _ in ()).throw(ValueError("invalid")),
    )
    assert summary.failure_reason == "timeout_error"
    assert (summary.llm_call_count, summary.tool_call_count, summary.tool_error_count) == (2, 2, 1)
    assert (summary.reliability_issue_count, summary.reliability_error_count) == (2, 1)
    assert (summary.finalization_failure_count, summary.tool_recovery_failure_count) == (1, 1)
    assert (summary.input_tokens_total, summary.output_tokens_total, summary.tokens_total) == (15, 5, 20)
    assert summary.cost_usd_total == 0.01
    assert summary.schema_valid is False


@pytest.mark.unit
def test_tracker_import_needs_no_sql_or_legacy_app(tmp_path):
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mas_slm_research.mas_tracker import InMemoryMASTracker
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
assert not any(name == 'app' or name.startswith('app.') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(source_root)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.unit
def test_failing_observation_sink_cannot_erase_core_child_record():
    def broken_sink(event):
        raise RuntimeError("observer unavailable")

    tracker = InMemoryMASTracker(
        workflow=parallel_workflow(), mas_run_id="mas-2", event_sink=broken_sink,
    )
    tracked = tracker.begin_agent_execution(
        agent_name="acuity", state=make_initial_mas_state({}), pending_agent_payload={},
    )
    assert tracker.agent_records[tracked.agent_run_id].status == "running"
    assert tracker.events[0]["event_type"] == "agent_started"
    assert tracker.sink_errors == ["observer unavailable"]
