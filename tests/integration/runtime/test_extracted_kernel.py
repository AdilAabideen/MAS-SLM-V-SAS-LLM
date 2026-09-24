"""Parity checks for the extracted, explicitly injected agent kernel."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from mas_slm_research.kernel import AgentKernel
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

    events = Collector()
    model = FakeChatModel(list(script))
    agent = AgentKernel(
        model=model, tools=[lookup_value, final_answer], response_format=Answer,
        event_handlers=[events],
    )
    agent.set_event_context(run_id="case_1", agent_name="baseline")
    result = asyncio.run(agent.ainvoke("case"))

    assert result == {"recommendation": {"value": "abc"}, "ok": True}
    assert len(model.messages_seen) == len(script)
    assert any(isinstance(message, ToolMessage) for message in model.messages_seen[-1])
    assert any(event["event_type"] == "tool_call" for event in events.items)
    assert sum(
        event["event_type"] == "runtime_decision"
        and event["payload_json"].get("decision") == "retry_after_malformed_tool_call"
        for event in events.items
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
