"""Parity checks for the extracted, explicitly injected agent kernel."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agentic.AgentRuntime import AgentKernel as LegacyAgentKernel
from app.agentic.telemetry import token_estimator
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.telemetry import token_estimator as extracted_token_estimator
from tests.doubles.fake_emitters import Collector
from tests.doubles.fake_provider import FakeChatModel


class Answer(BaseModel):
    recommendation: dict


def lookup_value(value: str) -> dict:
    """Return a deterministic value."""
    return {"value": value}


def final_answer(recommendation: dict) -> dict:
    """Return a structured final answer."""
    return {"recommendation": recommendation}


@pytest.fixture(autouse=True)
def offline_token_estimation(monkeypatch):
    monkeypatch.setattr(token_estimator, "tiktoken", None)
    monkeypatch.setattr(extracted_token_estimator, "tiktoken", None)


@pytest.mark.integration
@pytest.mark.parametrize("call_style", ["native", "text", "malformed_then_text"])
def test_extracted_loop_preserves_tool_replay_recovery_and_finalization(call_style):
    if call_style == "native":
        first = AIMessage(content="", tool_calls=[{"id": "lookup", "name": "lookup_value", "args": {"value": "abc"}}])
    elif call_style == "text":
        first = AIMessage(content='{"tool_calls":[{"id":"lookup","name":"lookup_value","arguments":{"value":"abc"}}]}')
    else:
        first = AIMessage(content='{"tool_calls":[{"name":"lookup_value","arguments":')
    script = [first]
    if call_style == "malformed_then_text":
        script.append(AIMessage(content='{"tool_calls":[{"id":"lookup","name":"lookup_value","arguments":{"value":"abc"}}]}'))
    script.append(AIMessage(content="", tool_calls=[{"id": "final", "name": "final_answer", "args": {"recommendation": {"value": "abc"}}}]))

    traces = []
    for kernel_type in (LegacyAgentKernel, AgentKernel):
        events = Collector()
        model = FakeChatModel(list(script))
        agent = kernel_type(
            model=model, tools=[lookup_value, final_answer], response_format=Answer,
            event_handlers=[events],
        )
        agent.set_event_context(run_id="case_1", agent_name="baseline")
        result = asyncio.run(agent.ainvoke("case"))
        traces.append((result, model.messages_seen, events.items))

    legacy_result, legacy_messages, legacy_events = traces[0]
    extracted_result, extracted_messages, extracted_events = traces[1]
    assert extracted_result == legacy_result == {"recommendation": {"value": "abc"}, "ok": True}
    assert len(extracted_messages) == len(legacy_messages) == len(script)
    assert any(isinstance(message, ToolMessage) for message in extracted_messages[-1])
    assert [event["event_type"] for event in extracted_events] == [event["event_type"] for event in legacy_events]
    assert sum(
        event["event_type"] == "runtime_decision"
        and event["payload_json"].get("decision") == "retry_after_malformed_tool_call"
        for event in extracted_events
    ) == (1 if call_style == "malformed_then_text" else 0)


@pytest.mark.integration
def test_extracted_kernel_propagates_provider_error_with_metric():
    metrics = Collector()
    agent = AgentKernel(model=FakeChatModel([RuntimeError("offline provider failure")]), tools=[], llm_call_handlers=[metrics])
    agent.set_event_context(run_id="case_2", agent_name="baseline")
    with pytest.raises(RuntimeError, match="offline provider failure"):
        asyncio.run(agent.ainvoke("case"))
    assert metrics.items[0]["error_text"] == "offline provider failure"


@pytest.mark.integration
def test_extracted_kernel_imports_without_backend_state():
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mas_slm_research.kernel import AgentKernel
assert AgentKernel.__module__ == 'mas_slm_research.kernel'
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
assert not any(name == 'fastapi' or name.startswith('fastapi.') for name in sys.modules)
assert 'app.config' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(source_root)], check=True, capture_output=True, text=True)
