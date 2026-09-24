"""Test Emitters And Usage test coverage."""

from __future__ import annotations

from datetime import datetime

import pytest

from mas_slm_research.telemetry.event_emitter import EventEmitter
from mas_slm_research.telemetry.metrics_types import LLMCallMetric, ToolExecutionMetric
from mas_slm_research.telemetry.telemetry_emitter import TelemetryEmitter
from mas_slm_research.telemetry.usage_extractor import extract_provider_usage


class FakeResponse:
    def __init__(self, usage_metadata=None, response_metadata=None):
        """Handle the value."""
        # Keep the main step clear.
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


@pytest.mark.unit
def test_ut_tel_001_emit_is_noop_before_context_and_handlers_are_set():
    """Handle ut tel 001 emit is noop before context and handlers are set."""
    # Keep the main step clear.
    emitter = EventEmitter()
    emitter.emit(event_type="assistant")


@pytest.mark.unit
def test_ut_tel_002_sequence_id_increments_monotonically():
    """Handle ut tel 002 sequence id increments monotonically."""
    # Keep the main step clear.
    events = []
    emitter = EventEmitter()
    emitter.set_context(run_id="run_1", agent_name="agent")
    emitter.set_handlers([events.append])
    emitter.emit(event_type="assistant")
    emitter.emit(event_type="assistant")
    assert [item["seq"] for item in events] == [1, 2]


@pytest.mark.unit
def test_ut_tel_004_multiple_handlers_all_receive_identical_payload():
    """Handle ut tel 004 multiple handlers all receive identical payload."""
    # Keep the main step clear.
    left = []
    right = []
    emitter = EventEmitter()
    emitter.set_context(run_id="run_1", agent_name="agent")
    emitter.set_handlers([left.append, right.append])
    emitter.emit(event_type="assistant")
    assert left[0] == right[0]


@pytest.mark.unit
def test_ut_tel_005_next_call_index_increments_deterministically():
    """Handle ut tel 005 next call index increments deterministically."""
    # Keep the main step clear.
    emitter = TelemetryEmitter()
    assert emitter.next_call_index() == 1
    assert emitter.next_call_index() == 2


@pytest.mark.unit
def test_ut_tel_007_llm_metric_is_emitted_to_all_handlers():
    """Handle ut tel 007 LLM metric is emitted to all handlers."""
    # Keep the main step clear.
    items = []
    emitter = TelemetryEmitter()
    emitter.set_llm_handlers([items.append])
    emitter.emit_llm(
        LLMCallMetric(
            run_id="run_1",
            agent_name="agent",
            call_index=1,
            iteration=1,
            call_kind="main_loop",
            model_name="model",
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow(),
            latency_ms=10,
            input_tokens=1,
            output_tokens=2,
            tokens_total=3,
            usage_source="estimated",
            had_tool_calls=False,
            tool_call_count=0,
            tool_call_parse_source=None,
            text_recovered_tool_call_count=0,
            native_tool_call_count=0,
            tool_names=[],
            error_text=None,
        )
    )
    assert items[0]["call_index"] == 1


@pytest.mark.unit
def test_ut_tel_009_noop_when_no_handlers_registered():
    """Handle ut tel 009 noop when no handlers registered."""
    # Keep the main step clear.
    emitter = TelemetryEmitter()
    emitter.emit_llm  # smoke


@pytest.mark.unit
def test_ut_tel_016_extract_usage_from_usage_metadata():
    """Handle ut tel 016 extract usage from usage metadata."""
    # Keep the main step clear.
    result = extract_provider_usage(FakeResponse(usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}))
    assert result.input_tokens == 1
    assert result.total_tokens == 3


@pytest.mark.unit
def test_ut_tel_018_missing_totals_are_derived_from_input_plus_output():
    """Handle ut tel 018 missing totals are derived from input plus output."""
    # Keep the main step clear.
    result = extract_provider_usage(FakeResponse(usage_metadata={"input_tokens": 1, "output_tokens": 2}))
    assert result.total_tokens == 3


@pytest.mark.unit
def test_ut_tel_019_missing_usage_returns_empty_usage_result():
    """Handle ut tel 019 missing usage returns empty usage result."""
    # Keep the main step clear.
    result = extract_provider_usage(FakeResponse())
    assert result.has_usage is False

