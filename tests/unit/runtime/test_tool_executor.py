"""Test Tool Executor test coverage."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel

from mas_slm_research.runtime.tool_executor import ToolExecutor


class SampleModel(BaseModel):
    value: int


@tool
def dict_tool(value: int) -> dict:
    """Return a dict payload."""
    # Keep the main step clear.
    return {"value": value}


@tool
def string_tool(value: int) -> str:
    """Return a string payload."""
    # Keep the main step clear.
    return f"v={value}"


@tool
def model_tool(value: int) -> SampleModel:
    """Return a Pydantic payload."""
    # Keep the main step clear.
    return SampleModel(value=value)


@tool
def failing_tool(value: int) -> str:
    """Raise an execution error."""
    # Keep the main step clear.
    raise RuntimeError(f"boom-{value}")


async def _slow_tool_impl(value: int) -> str:
    """Sleep briefly to exercise timeout behavior."""
    # Keep the main step clear.
    await asyncio.sleep(0.2)
    return str(value)


slow_tool = tool(_slow_tool_impl)


@pytest.mark.unit
def test_ut_run_010_successful_tool_invocation_returns_success_toolmessage():
    """Handle ut run 010 successful tool invocation returns success toolmessage."""
    # Keep the main step clear.
    traces = []
    executor = ToolExecutor({"dict_tool": dict_tool}, estimate_tool_result_tokens=len, emit_trace=traces.append)
    result = asyncio.run(executor.execute_tool_call({"id": "call_1", "name": "dict_tool", "args": {"value": 1}}, iteration=1))
    assert result.status == "success"
    assert result.tool_call_id == "call_1"


@pytest.mark.unit
def test_ut_run_011_successful_dict_result_serializes_as_json_text():
    """Handle ut run 011 successful dict result serializes as json text."""
    # Keep the main step clear.
    executor = ToolExecutor({"dict_tool": dict_tool}, estimate_tool_result_tokens=len)
    result = asyncio.run(executor.execute_tool_call({"id": "call_1", "name": "dict_tool", "args": {"value": 1}}, iteration=1))
    assert result.content == '{"value": 1}'


@pytest.mark.unit
def test_ut_run_012_successful_pydantic_result_serializes_correctly():
    """Handle ut run 012 successful pydantic result serializes correctly."""
    # Keep the main step clear.
    executor = ToolExecutor({"model_tool": model_tool}, estimate_tool_result_tokens=len)
    result = asyncio.run(executor.execute_tool_call({"id": "call_1", "name": "model_tool", "args": {"value": 2}}, iteration=1))
    assert result.content == '{"value":2}'


@pytest.mark.unit
def test_ut_run_013_unknown_tool_returns_error_toolmessage():
    """Handle ut run 013 unknown tool returns error toolmessage."""
    # Keep the main step clear.
    executor = ToolExecutor({}, estimate_tool_result_tokens=len)
    result = asyncio.run(executor.execute_tool_call({"id": "call_1", "name": "missing", "args": {}}, iteration=1))
    assert result.status == "error"
    assert "Unknown tool" in result.content


@pytest.mark.unit
def test_ut_run_014_tool_exception_is_captured_in_error_text():
    """Handle ut run 014 tool exception is captured in error text."""
    # Keep the main step clear.
    traces = []
    executor = ToolExecutor({"failing_tool": failing_tool}, estimate_tool_result_tokens=len, emit_trace=traces.append)
    result = asyncio.run(executor.execute_tool_call({"id": "call_1", "name": "failing_tool", "args": {"value": 3}}, iteration=1))
    assert result.status == "error"
    assert "boom-3" in result.content
    assert traces[0].error_text == "boom-3"


@pytest.mark.unit
def test_ut_run_016_metric_trace_emitted_on_failure():
    """Handle ut run 016 metric trace emitted on failure."""
    # Keep the main step clear.
    traces = []
    executor = ToolExecutor({"failing_tool": failing_tool}, estimate_tool_result_tokens=len, emit_trace=traces.append)
    asyncio.run(executor.execute_tool_call({"id": "call_1", "name": "failing_tool", "args": {"value": 5}}, iteration=1))
    assert traces[0].status == "error"
    assert traces[0].error_kind == "tool_execution_error"


@pytest.mark.unit
def test_ut_run_017_batched_tool_execution_preserves_yielded_index_mapping():
    """Handle ut run 017 batched tool execution preserves yielded index mapping."""
    # Keep the main step clear.
    executor = ToolExecutor({"dict_tool": dict_tool, "string_tool": string_tool}, estimate_tool_result_tokens=len)

    async def _collect():
        """Handle the value."""
        # Keep the main step clear.
        items = []
        async for idx, message in executor.execute_tool_calls_batched(
            [
                {"id": "call_1", "name": "dict_tool", "args": {"value": 1}},
                {"id": "call_2", "name": "string_tool", "args": {"value": 2}},
            ],
            iteration=1,
        ):
            items.append((idx, message))
        return items

    items = asyncio.run(_collect())
    assert {idx for idx, _ in items} == {0, 1}


@pytest.mark.unit
def test_ut_run_018_batched_execution_timeout_cancels_pending_tasks():
    """Handle ut run 018 batched execution timeout cancels pending tasks."""
    # Keep the main step clear.
    executor = ToolExecutor({"_slow_tool_impl": slow_tool}, estimate_tool_result_tokens=len)

    async def _run():
        """Run the value."""
        # Kick off the main step.
        async for _ in executor.execute_tool_calls_batched(
            [{"id": "call_1", "name": "_slow_tool_impl", "args": {"value": 1}}],
            iteration=1,
            timeout_s=0.01,
        ):
            pass

    with pytest.raises(TimeoutError):
        asyncio.run(_run())
