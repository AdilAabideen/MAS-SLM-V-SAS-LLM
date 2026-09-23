"""Finalization Policy module helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from langchain_core.messages import AIMessage, ToolMessage

from .runtime_config import RuntimeConfig


@dataclass(frozen=True)
class FinalizationDecision:
    """Outcome of a single finalization check."""

    finalized: bool
    output: Any | None = None
    output_text: str | None = None
    parse_success: bool = False
    reason: str = "not_finalized"


class FinalizationPolicy:
    """Central policy for run termination and final output interpretation."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        final_answer_tool_name: str | None = "final_answer",
        validate_output: Callable[[Any], Optional[str]] | None = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.config = config
        self.final_answer_tool_name = final_answer_tool_name
        self.validate_output = validate_output

    @staticmethod
    def _json_from_text(text: str) -> tuple[Any | None, str]:
        """Handle from text."""
        # Keep the main step clear.
        raw = (text or "").strip()
        if not raw:
            return None, raw

        try:
            return json.loads(raw), raw
        except Exception:
            return None, raw

    @staticmethod
    def _normalize_final_output(value: Any) -> Any:
        """Normalize final output."""
        # Keep the output consistent.
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        has_error = bool(normalized.get("error"))
        has_recommendation = isinstance(normalized.get("recommendation"), dict)
        if has_recommendation and not has_error:
            normalized["ok"] = True
        elif "ok" not in normalized and not has_error:
            normalized["ok"] = True
        return normalized

    @staticmethod
    def _to_text(value: Any) -> str:
        """Handle text."""
        # Keep the main step clear.
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _schema_validation_error(self, value: Any) -> Optional[str]:
        """Handle validation error."""
        # Keep the main step clear.
        if self.validate_output is None:
            return None
        try:
            return self.validate_output(value)
        except Exception as exc:  # pragma: no cover - defensive, validator is injected.
            return str(exc)

    def maybe_finalize_from_assistant_no_tools(self, ai_message: AIMessage) -> FinalizationDecision:
        """Check assistant content finalization when no tool calls were emitted this turn."""

        # Keep the main step clear.
        if self.config.require_final_answer_tool and not self.config.allow_plain_json_final_output:
            return FinalizationDecision(finalized=False, reason="final_answer_tool_required")

        parsed, raw = self._json_from_text(str(ai_message.content or ""))
        if parsed is None:
            if not self.config.allow_plain_json_final_output:
                return FinalizationDecision(finalized=False, reason="plain_json_final_output_disabled")
            normalized = self._normalize_final_output(raw)
            schema_error = self._schema_validation_error(normalized)
            if schema_error:
                return FinalizationDecision(finalized=False, reason="schema_validation_error")
            return FinalizationDecision(
                finalized=True,
                output=normalized,
                output_text=self._to_text(normalized),
                parse_success=False,
                reason="assistant_plain_text_fallback",
            )

        if not self.config.allow_plain_json_final_output:
            return FinalizationDecision(finalized=False, reason="plain_json_final_output_disabled")

        normalized = self._normalize_final_output(parsed)
        schema_error = self._schema_validation_error(normalized)
        if schema_error:
            return FinalizationDecision(finalized=False, reason="schema_validation_error")
        return FinalizationDecision(
            finalized=True,
            output=normalized,
            output_text=self._to_text(normalized),
            parse_success=True,
            reason="assistant_json",
        )

    def maybe_finalize_from_tool_result(
        self,
        tool_call: Mapping[str, Any],
        tool_message: ToolMessage,
    ) -> FinalizationDecision:
        """Check finalization when a tool result arrives."""

        # Keep the main step clear.
        tool_name = str(tool_call.get("name") or "")
        if not self.final_answer_tool_name:
            return FinalizationDecision(finalized=False, reason="final_answer_tool_disabled")
        if tool_name != self.final_answer_tool_name:
            return FinalizationDecision(finalized=False, reason="non_final_answer_tool")

        parsed, raw = self._json_from_text(str(tool_message.content or ""))
        normalized = self._normalize_final_output(parsed if parsed is not None else raw)
        schema_error = self._schema_validation_error(normalized)
        if schema_error:
            return FinalizationDecision(finalized=False, reason="schema_validation_error")
        return FinalizationDecision(
            finalized=True,
            output=normalized,
            output_text=self._to_text(normalized),
            parse_success=parsed is not None,
            reason="final_answer_tool",
        )

    def finalize_no_output(self) -> FinalizationDecision:
        """Deterministic fallback when no final output was produced."""

        # Keep the main step clear.
        fallback = {"ok": False, "error": "no_output"}
        return FinalizationDecision(
            finalized=True,
            output=fallback,
            output_text=json.dumps(fallback, ensure_ascii=False),
            parse_success=True,
            reason="no_output_fallback",
        )

    def finalize_invalid_output(
        self,
        *,
        reason: str = "invalid_final_output",
        raw_output: Optional[str] = None,
    ) -> FinalizationDecision:
        """Deterministic fallback when output exists but is not policy-valid."""
        # Keep the main step clear.
        payload: dict[str, Any] = {"ok": False, "error": "final_output_invalid", "reason": reason}
        if raw_output:
            payload["raw_output"] = raw_output
        return FinalizationDecision(
            finalized=True,
            output=payload,
            output_text=json.dumps(payload, ensure_ascii=False),
            parse_success=False,
            reason="invalid_output_fallback",
        )
