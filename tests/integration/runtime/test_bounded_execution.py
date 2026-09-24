"""Corrected policy limits fail closed while preserving trace evidence."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from mas_slm_research.contracts import FailureKind, RunStatus
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.multi_agent import MultiAgentRunner
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.single_agent import SingleAgentRunner
from mas_slm_research.workflows.esi.definition import ESI_MAS
from mas_slm_research.workflows.esi.payload_builder import build_pending_agent_payload
from tests.doubles.fake_provider import FakeChatModel
from tests.integration.runtime.test_multi_agent_runner import (
    identity as multi_identity, output_validator, role_factories,
)
from tests.integration.runtime.test_single_agent_runner import (
    Answer, final_answer, identity as single_identity, lookup_value,
)


def _two_call_kernel(*, max_model_calls=None, max_tool_calls_total=None):
    return AgentKernel(
        model=FakeChatModel([
            AIMessage(content="", tool_calls=[{"id": "lookup", "name": "lookup_value", "args": {"value": "abc"}}]),
            AIMessage(content="", tool_calls=[{"id": "final", "name": "final_answer", "args": {"recommendation": {"value": "abc"}}}]),
        ]),
        tools=[lookup_value, final_answer], response_format=Answer,
        runtime_config=RuntimeConfig(max_model_calls=max_model_calls, max_tool_calls_total=max_tool_calls_total),
    )


@pytest.mark.integration
@pytest.mark.parametrize("limit_name", ["max_model_calls", "max_tool_calls_total"])
def test_single_agent_cumulative_budget_fails_with_counts(limit_name):
    options = {limit_name: 1}
    run = asyncio.run(SingleAgentRunner(
        kernel_factory=lambda: _two_call_kernel(**options), agent_name="baseline",
    ).run_case(identity=single_identity(limit_name), payload="case"))
    assert run.result.status == RunStatus.FAILED
    assert run.result.failure.kind == FailureKind.BUDGET
    assert "used=" in run.result.failure.message and "limit=1" in run.result.failure.message
    assert len(run.llm_calls) == (1 if limit_name == "max_model_calls" else 2)
    assert len(run.tool_calls) == 1
    assert any(event["event_type"] == "runtime_decision" and "budget_exceeded" in str(event.get("payload_json")) for event in run.events)


@pytest.mark.integration
def test_multi_agent_handoff_budget_fails_and_closes_children():
    run = asyncio.run(MultiAgentRunner(
        workflow=ESI_MAS, role_factories=role_factories(("esi1_agent",)),
        payload_builder=build_pending_agent_payload, output_validator=output_validator,
        max_handoffs=1,
    ).run_case(identity=multi_identity("handoff-budget"), case_info={"chiefcomplaint": "pain"}))
    assert run.result.status == RunStatus.FAILED
    assert run.result.failure.kind == FailureKind.BUDGET
    assert "handoffs_budget_exceeded" in run.result.failure.message
    assert run.agent_records and all(record.status != "running" for record in run.agent_records)
    assert len(run.handoff_records) <= 1


@pytest.mark.integration
def test_error_dictionary_cannot_become_a_successful_single_prediction():
    run = asyncio.run(SingleAgentRunner(
        kernel_factory=lambda: AgentKernel(
            model=FakeChatModel([AIMessage(content='{"ok":true,"error":"provider_failed","recommendation":{"value":"wrong"}}')]),
            tools=[], response_format=Answer,
        ), agent_name="baseline",
    ).run_case(identity=single_identity("error-dictionary"), payload="case"))
    assert run.result.status == RunStatus.FAILED
    assert run.result.output is None
    assert run.result.failure.kind in {FailureKind.RUNTIME, FailureKind.VALIDATION}


@pytest.mark.integration
def test_elapsed_budget_terminates_slow_provider():
    class SlowModel(FakeChatModel):
        async def ainvoke(self, messages):
            await asyncio.sleep(0.05)
            return await super().ainvoke(messages)

    run = asyncio.run(SingleAgentRunner(
        kernel_factory=lambda: AgentKernel(
            model=SlowModel([AIMessage(content='{"recommendation":{"value":"late"}}')]),
            response_format=Answer, runtime_config=RuntimeConfig(max_elapsed_seconds=0.001),
        ), agent_name="baseline",
    ).run_case(identity=single_identity("elapsed-budget"), payload="case"))
    assert run.result.status == RunStatus.FAILED
    assert run.result.failure.kind == FailureKind.TIMEOUT


def _slow_role_factories():
    factories = role_factories(("esi1_agent",))
    original = factories["esi1_agent"]

    def build_slow():
        kernel = original()
        provider = kernel.bound_model
        invoke = provider.ainvoke

        async def delayed(messages):
            await asyncio.sleep(0.05)
            return await invoke(messages)

        provider.ainvoke = delayed
        return kernel

    factories["esi1_agent"] = build_slow
    return factories


@pytest.mark.integration
def test_multi_agent_elapsed_budget_closes_started_children():
    run = asyncio.run(MultiAgentRunner(
        workflow=ESI_MAS, role_factories=_slow_role_factories(),
        payload_builder=build_pending_agent_payload, output_validator=output_validator,
        max_elapsed_seconds=0.005,
    ).run_case(identity=multi_identity("multi-timeout"), case_info={"chiefcomplaint": "pain"}))
    assert run.result.status == RunStatus.FAILED
    assert run.result.failure.kind == FailureKind.TIMEOUT
    assert run.agent_records and all(record.status != "running" for record in run.agent_records)


@pytest.mark.integration
def test_multi_agent_cancellation_closes_started_children():
    async def scenario():
        started = asyncio.Event()
        events = []

        def observe(event):
            events.append(event)
            if event.get("event_type") == "agent_started":
                started.set()

        runner = MultiAgentRunner(
            workflow=ESI_MAS, role_factories=_slow_role_factories(),
            payload_builder=build_pending_agent_payload, output_validator=output_validator,
            event_sink=observe,
        )
        task = asyncio.create_task(runner.run_case(
            identity=multi_identity("multi-cancel"), case_info={"chiefcomplaint": "pain"},
        ))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        started_ids = {event.get("agent_run_id") for event in events if event.get("event_type") == "agent_started"}
        failed_ids = {event.get("agent_run_id") for event in events if event.get("event_type") == "agent_completed" and event.get("status") == "failed"}
        assert started_ids <= failed_ids

    asyncio.run(scenario())
