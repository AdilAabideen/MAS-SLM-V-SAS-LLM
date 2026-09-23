"""Event Emitter module helpers."""

from __future__ import annotations

from typing import Any, Callable, Sequence


class EventEmitter:
    """Emit run events in the existing persistence shape with monotonic sequence IDs."""

    def __init__(self) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self._handlers: list[Callable[[dict[str, Any]], None]] = []
        self._run_id: str | None = None
        self._agent_name: str | None = None
        self._seq: int = 0

    def set_context(self, *, run_id: str, agent_name: str, start_seq: int = 0) -> None:
        """Handle context."""
        # Keep the main step clear.
        self._run_id = str(run_id)
        self._agent_name = str(agent_name)
        self._seq = int(start_seq)

    def set_handlers(self, handlers: Sequence[Callable[[dict[str, Any]], None]] | None) -> None:
        """Handle handlers."""
        # Keep the main step clear.
        self._handlers = list(handlers or [])

    def add_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Handle handler."""
        # Keep the main step clear.
        self._handlers.append(handler)

    def emit(
        self,
        *,
        event_type: str,
        node_name: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        status: str | None = None,
        payload_json: dict[str, Any] | None = None,
        payload_text: str | None = None,
    ) -> None:
        """Emit the value."""
        # Keep events flowing.
        if not self._handlers or self._run_id is None or self._agent_name is None:
            return

        self._seq += 1
        payload = {
            "run_id": self._run_id,
            "agent_name": self._agent_name,
            "seq": self._seq,
            "event_type": event_type,
            "node_name": node_name,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "status": status,
            "payload_json": payload_json,
            "payload_text": payload_text,
        }
        for handler in self._handlers:
            handler(payload)
