"""Pure per-child measurements formerly calculated during SQL persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


FINALIZATION_CODES = frozenset({
    "final_output_missing", "final_output_unparseable", "final_output_schema_invalid",
    "final_output_invalid", "schema_validation_error",
})
TOOL_RECOVERY_CODES = frozenset({
    "assistant_tool_call_json_unparseable", "assistant_tool_call_recovery_failed",
    "assistant_tool_call_name_not_allowed", "native_tool_parse_failure",
    "text_recovery_used", "text_recovery_failure",
})


@dataclass(frozen=True, kw_only=True)
class AgentMeasurementSummary:
    status: str
    duration_ms: int | None
    failure_reason: str | None
    llm_call_count: int
    tool_call_count: int
    tool_error_count: int
    reliability_issue_count: int
    reliability_error_count: int
    finalization_failure_count: int
    tool_recovery_failure_count: int
    input_tokens_total: int | None
    output_tokens_total: int | None
    tokens_total: int | None
    cost_usd_total: float | None
    schema_valid: bool | None


def summarize_agent_measurements(
    *,
    status: str,
    duration_ms: int | None,
    error_text: str | None,
    output: Mapping[str, Any] | None,
    llm_calls: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    reliability_issues: Sequence[Mapping[str, Any]] = (),
    schema_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> AgentMeasurementSummary:
    """Reproduce baseline counts from in-memory calls/events, with unknown usage explicit."""
    issue_codes = [str(item.get("issue_code") or "") for item in reliability_issues]
    cost_values = [float(item["cost_usd"]) for item in llm_calls if item.get("cost_usd") is not None]
    schema_valid: bool | None = None
    if output is not None and schema_validator is not None:
        try:
            schema_validator(output)
            schema_valid = True
        except Exception:
            schema_valid = False

    failure_reason: str | None = None
    if status == "failed":
        failure_reason = "timeout_error" if "timeout" in (error_text or "").lower() else "provider_error"

    return AgentMeasurementSummary(
        status=status,
        duration_ms=duration_ms,
        failure_reason=failure_reason,
        llm_call_count=len(llm_calls),
        tool_call_count=sum(item.get("event_type") == "tool_call" for item in events),
        tool_error_count=sum(
            item.get("event_type") == "tool_result" and item.get("status") == "error" for item in events
        ),
        reliability_issue_count=len(reliability_issues),
        reliability_error_count=sum(item.get("severity") == "error" for item in reliability_issues),
        finalization_failure_count=sum(code in FINALIZATION_CODES for code in issue_codes),
        tool_recovery_failure_count=sum(code in TOOL_RECOVERY_CODES for code in issue_codes),
        input_tokens_total=sum(int(item.get("input_tokens") or 0) for item in llm_calls) if llm_calls else None,
        output_tokens_total=sum(int(item.get("output_tokens") or 0) for item in llm_calls) if llm_calls else None,
        tokens_total=sum(int(item.get("tokens_total") or 0) for item in llm_calls) if llm_calls else None,
        cost_usd_total=sum(cost_values) if cost_values else None,
        schema_valid=schema_valid,
    )
