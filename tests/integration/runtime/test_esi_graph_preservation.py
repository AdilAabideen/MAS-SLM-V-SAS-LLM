"""Offline preservation checks for the baseline ESI multi-agent graph."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.agentic.mas.agent_node_executor import AgentNodeExecutor
from app.agentic.mas.execution_strategy import CallableExecutionStrategy
from app.agentic.mas.gate_evaluator import GateEvaluator
from app.agentic.mas.graph_builder import MASGraphBuilder
from app.agentic.mas_contract import AgentExecutionResult, HandoffEnvelope, make_initial_mas_state
from app.agentic.workflows.definitions.esi_mas.workflow_definition import ESI_MAS


CASE = {"chiefcomplaint": "chest pain", "age": 42, "heartrate": 111, "sbp": 96}


@pytest.mark.integration
@pytest.mark.parametrize(
    "acuity_path",
    [
        ("esi1_agent",),
        ("esi1_agent", "esi2_agent"),
        ("esi1_agent", "esi2_agent", "esi345_agent"),
    ],
)
def test_baseline_graph_routes_and_gates(acuity_path):
    """Every baseline acuity branch reaches the doctor only after both sources."""
    requests = []

    async def execute(request):
        name = request.agent_name
        requests.append(request)
        if name == "doctor_agent":
            assert {item["from_agent"] for item in request.state_snapshot["handoff_history"]} >= {
                acuity_path[-1],
                "vitals_agent",
            }
            return AgentExecutionResult(agent_name=name, status="final", final_output={"esi": 2})

        if name == "vitals_agent" or name == acuity_path[-1]:
            target = "doctor_agent"
        else:
            target = acuity_path[acuity_path.index(name) + 1]
        return AgentExecutionResult(
            agent_name=name,
            status="handoff",
            handoff=HandoffEnvelope(
                handoff_name=f"{name}_to_{target}",
                from_agent=name,
                target_agent=target,
                payload_schema="test_payload",
                payload={"source": name},
            ),
        )

    graph = MASGraphBuilder(
        workflow=ESI_MAS,
        agent_executor=AgentNodeExecutor(
            workflow=ESI_MAS,
            strategy=CallableExecutionStrategy(mode="scripted", execute_fn=execute),
        ),
        gate_evaluator=GateEvaluator(workflow=ESI_MAS),
    ).build()
    state = asyncio.run(graph.ainvoke(make_initial_mas_state(CASE)))

    assert state["final_output"] == {"esi": 2}
    assert set(state["completed_agents"]) == set(acuity_path) | {"vitals_agent", "doctor_agent"}
    assert sum(request.agent_name == "doctor_agent" for request in requests) == 1
    gate_events = [event for event in state["execution_trace"] if event["event"] == "doctor_gate"]
    assert [event["ready"] for event in gate_events] == ([True] if len(acuity_path) == 1 else [False, True])
    if len(acuity_path) > 1:
        assert gate_events[0]["missing_sources"] == ["acuity"]

    payloads = {request.agent_name: request.pending_agent_payload["llm_payload"] for request in requests}
    for name in acuity_path:
        case_info = payloads[name]["case_info"]
        assert case_info["chief_complaint"] == CASE["chiefcomplaint"]
        assert "heart_rate" not in case_info
    for name in ("vitals_agent", "doctor_agent"):
        assert payloads[name]["case_info"]["heart_rate"] == CASE["heartrate"]


@pytest.mark.integration
@pytest.mark.parametrize("first_source", ["acuity", "vitals"])
def test_doctor_gate_waits_for_both_handoff_arrival_orders(first_source):
    """The gate waits when either logical source arrives alone."""
    handoffs = {
        "acuity": HandoffEnvelope(
            handoff_name="esi1_to_doctor", from_agent="esi1_agent", target_agent="doctor_agent",
            payload_schema="test_payload", payload={},
        ).model_dump(),
        "vitals": HandoffEnvelope(
            handoff_name="vitals_to_doctor", from_agent="vitals_agent", target_agent="doctor_agent",
            payload_schema="test_payload", payload={},
        ).model_dump(),
    }
    evaluator = GateEvaluator(workflow=ESI_MAS)
    state = make_initial_mas_state(CASE)
    state["handoff_history"] = [handoffs[first_source]]
    waiting = evaluator.evaluate(gate_id="doctor_gate", state=state, persist=False)
    assert waiting.ready is False
    assert waiting.missing_sources == ["vitals" if first_source == "acuity" else "acuity"]
    assert waiting.next_target is None

    other_source = "vitals" if first_source == "acuity" else "acuity"
    state["handoff_history"].append(handoffs[other_source])
    ready = evaluator.evaluate(gate_id="doctor_gate", state=state, persist=False)
    assert ready.ready is True
    assert ready.missing_sources == []
    assert ready.next_target == "doctor_agent"


@pytest.mark.integration
def test_invalid_handoff_route_is_rejected_before_graph_execution():
    """An agent cannot bypass the declared ESI route."""
    with pytest.raises(ValidationError, match="Invalid handoff route"):
        HandoffEnvelope(
            handoff_name="invalid",
            from_agent="vitals_agent",
            target_agent="esi2_agent",
            payload_schema="test_payload",
            payload={},
        )
