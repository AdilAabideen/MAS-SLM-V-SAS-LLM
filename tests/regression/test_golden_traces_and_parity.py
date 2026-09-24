"""Test Golden Traces And Parity test coverage."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from mas_slm_research.kernel import AgentKernel
from mas_slm_research.protocols.tool_call_recovery import (
    looks_like_malformed_tool_call_content,
    recover_tool_calls_from_content,
)
from mas_slm_research.protocols.tool_protocol import normalize_tool_calls
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from tests.doubles.fake_provider import FakeChatModel


class RegressionOutput(BaseModel):
    recommendation: dict


def final_answer(recommendation: dict) -> dict:
    """Return final regression payload."""
    # Keep the main step clear.
    return {"recommendation": recommendation}


def lookup_value(value: str) -> dict:
    """Return lookup regression payload."""
    # Keep the main step clear.
    return {"recommendation": {"value": value}}


@pytest.mark.regression
@pytest.mark.golden
def test_reg_001_native_tool_call_fixture_normalizes_identically(load_json_fixture):
    """Handle reg 001 native tool call fixture normalizes identically."""
    # Keep the main step clear.
    fixture = load_json_fixture("golden_traces/native_tool_call_success.json")
    calls = normalize_tool_calls(fixture["tool_calls"], allowed_tool_names={"lookup_value"})
    assert calls == [
        {"id": "call_native", "name": "lookup_value", "args": {"value": "abc"}, "type": "tool_call"}
    ]


@pytest.mark.regression
@pytest.mark.golden
def test_reg_002_text_recovered_fixture_replays_identically(load_json_fixture):
    """Handle reg 002 text recovered fixture replays identically."""
    # Keep the main step clear.
    fixture = load_json_fixture("golden_traces/text_recovery_success.json")
    result = recover_tool_calls_from_content(fixture["content"], allowed_tool_names={"lookup_value"})
    assert result.succeeded is True
    assert result.recovered is True
    assert result.calls[0].name == "lookup_value"
    assert result.calls[0].args == {"value": "abc"}


@pytest.mark.regression
@pytest.mark.golden
def test_reg_003_malformed_tool_call_fixture_does_not_crash_and_is_detected(load_json_fixture):
    """Handle reg 003 malformed tool call fixture does not crash and is detected."""
    # Keep the main step clear.
    fixture = load_json_fixture("golden_traces/malformed_tool_call.json")
    result = recover_tool_calls_from_content(fixture["content"], allowed_tool_names={"lookup_value"})
    assert result.succeeded is False
    assert looks_like_malformed_tool_call_content(fixture["content"], allowed_tool_names={"lookup_value"}) is True


@pytest.mark.regression
@pytest.mark.golden
def test_reg_004_native_runtime_trace_yields_stable_final_output():
    """Handle reg 004 native runtime trace yields stable final output."""
    # Keep the main step clear.
    model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "lookup_value", "args": {"value": "native"}}]),
            AIMessage(content="", tool_calls=[{"id": "call_2", "name": "final_answer", "args": {"recommendation": {"value": "native"}}}]),
        ]
    )
    agent = AgentKernel(model=model, tools=[lookup_value, final_answer], response_format=RegressionOutput)
    assert asyncio.run(agent.ainvoke("case")) == {"recommendation": {"value": "native"}, "ok": True}


@pytest.mark.regression
@pytest.mark.golden
def test_reg_005_text_recovery_runtime_trace_yields_stable_final_output():
    """Handle reg 005 text recovery runtime trace yields stable final output."""
    # Keep the main step clear.
    model = FakeChatModel(
        [
            AIMessage(content='{"tool_calls":[{"id":"call_1","name":"lookup_value","arguments":{"value":"text"}}]}'),
            AIMessage(content='{"tool_calls":[{"id":"call_2","name":"final_answer","arguments":{"recommendation":{"value":"text"}}}]}'),
        ]
    )
    agent = AgentKernel(model=model, tools=[lookup_value, final_answer], response_format=RegressionOutput)
    assert asyncio.run(agent.ainvoke("case")) == {"recommendation": {"value": "text"}, "ok": True}


@pytest.mark.regression
@pytest.mark.golden
def test_reg_006_missing_final_output_path_yields_final_output_invalid():
    """Handle reg 006 missing final output path yields final output invalid."""
    # Keep the main step clear.
    model = FakeChatModel([AIMessage(content='{"recommendation":{"value":"plain"}}')])
    agent = AgentKernel(
        model=model,
        tools=[],
        runtime_config=RuntimeConfig(require_final_answer_tool=True, allow_plain_json_final_output=False),
    )
    assert asyncio.run(agent.ainvoke("case"))["error"] == "final_output_invalid"


@pytest.mark.regression
@pytest.mark.wrapper
def test_reg_007_dr7_and_llama_payload_fixtures_have_equivalent_tool_call_semantics(load_json_fixture):
    """Handle reg 007 dr7 and llama payload fixtures have equivalent tool call semantics."""
    # Keep the main step clear.
    dr7 = load_json_fixture("provider_payloads/dr7_native_tool_calls.json")["choices"][0]["message"]["tool_calls"]
    llama = load_json_fixture("provider_payloads/llama_native_tool_calls.json")["choices"][0]["message"]["tool_calls"]
    dr7_call = normalize_tool_calls(dr7, allowed_tool_names={"lookup_value"})[0]
    llama_call = normalize_tool_calls(llama, allowed_tool_names={"lookup_value"})[0]
    assert dr7_call["name"] == llama_call["name"] == "lookup_value"
    assert dr7_call["args"] == llama_call["args"] == {"value": "abc"}
