"""Handoff Policy module helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from langchain_core.messages import ToolMessage

from app.agentic.handoff import HandoffResult
from app.agentic.mas_contract import HandoffEnvelope


@dataclass(frozen=True)
class HandoffDecision:
    """Outcome of a single handoff-tool result check."""

    should_handoff: bool
    envelope: Optional[HandoffEnvelope] = None
    parse_success: bool = False
    reason: str = "not_handoff"
    error: Optional[str] = None


class HandoffPolicy:
    """Central policy for interpreting handoff tool results."""

    def __init__(
        self,
        *,
        handoff_tool_names: Sequence[str] | None = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.handoff_tool_names = {
            str(name)
            for name in list(handoff_tool_names or [])
            if isinstance(name, str) and name.strip()
        }

    def is_handoff_tool(self, tool_name: str | None) -> bool:
        """Handle handoff tool."""
        # Keep the main step clear.
        if not tool_name:
            return False
        return str(tool_name) in self.handoff_tool_names

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

    def maybe_handoff_from_tool_result(
        self,
        tool_call: Mapping[str, Any],
        tool_message: ToolMessage,
    ) -> HandoffDecision:
        """Handle handoff from tool result."""
        # Keep the main step clear.
        tool_name = str(tool_call.get("name") or getattr(tool_message, "name", "") or "")
        if not self.is_handoff_tool(tool_name):
            return HandoffDecision(should_handoff=False, reason="non_handoff_tool")

        parsed, raw = self._json_from_text(str(tool_message.content or ""))
        if parsed is None:
            return HandoffDecision(
                should_handoff=False,
                parse_success=False,
                reason="handoff_tool_result_unparseable",
                error=raw or "empty handoff tool result",
            )

        try:
            result = HandoffResult.model_validate(parsed)
            envelope = HandoffEnvelope(
                handoff_name=result.handoff_name,
                from_agent=result.from_agent,
                target_agent=result.target_agent,
                payload_schema=result.payload_schema,
                payload=result.payload,
            )
        except Exception as exc:
            return HandoffDecision(
                should_handoff=False,
                parse_success=True,
                reason="handoff_tool_result_invalid",
                error=str(exc),
            )

        return HandoffDecision(
            should_handoff=True,
            envelope=envelope,
            parse_success=True,
            reason="handoff_tool",
        )
