"""Metrics Types module helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMCallMetric:
    run_id: str
    agent_name: str
    call_index: int
    iteration: int
    call_kind: str
    model_name: str | None
    started_at: datetime
    ended_at: datetime
    latency_ms: int
    input_tokens: int
    output_tokens: int | None
    tokens_total: int | None
    usage_source: str
    had_tool_calls: bool
    tool_call_count: int
    tool_call_parse_source: str | None = None
    text_recovered_tool_call_count: int = 0
    native_tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    error_text: str | None = None
    provider_model_id: str | None = None
    request_parameters: dict[str, object] = field(default_factory=dict)
    network_attempts: int = 1
    input_token_source: str = "unknown"
    output_token_source: str = "unknown"


@dataclass
class ToolExecutionMetric:
    run_id: str
    agent_name: str
    iteration: int
    tool_call_id: str
    tool_name: str
    started_at: datetime
    ended_at: datetime
    latency_ms: int
    status: str
    result_char_count: int
    result_estimated_tokens: int
    error_text: str | None = None
