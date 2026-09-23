"""Telemetry Emitter module helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Sequence

from .metrics_types import LLMCallMetric, ToolExecutionMetric


class TelemetryEmitter:
    """Emit structured LLM and tool metrics through adapter handlers."""

    def __init__(self) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self._llm_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._tool_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._run_id: str | None = None
        self._agent_name: str | None = None
        self._call_index: int = 0

    def set_context(self, *, run_id: str, agent_name: str) -> None:
        """Handle context."""
        # Keep the main step clear.
        self._run_id = str(run_id)
        self._agent_name = str(agent_name)
        self._call_index = 0

    def set_llm_handlers(self, handlers: Sequence[Callable[[dict[str, Any]], None]] | None) -> None:
        """Handle llm handlers."""
        # Keep the main step clear.
        self._llm_handlers = list(handlers or [])

    def add_llm_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Handle llm handler."""
        # Keep the main step clear.
        self._llm_handlers.append(handler)

    def set_tool_handlers(self, handlers: Sequence[Callable[[dict[str, Any]], None]] | None) -> None:
        """Handle tool handlers."""
        # Keep the main step clear.
        self._tool_handlers = list(handlers or [])

    def add_tool_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Handle tool handler."""
        # Keep the main step clear.
        self._tool_handlers.append(handler)

    def next_call_index(self) -> int:
        """Handle call index."""
        # Keep the main step clear.
        self._call_index += 1
        return self._call_index

    def current_context(self) -> tuple[str | None, str | None]:
        """Handle context."""
        # Keep the main step clear.
        return self._run_id, self._agent_name

    def emit_llm(self, metric: LLMCallMetric) -> None:
        """Emit llm."""
        # Keep events flowing.
        if not self._llm_handlers:
            return
        payload = asdict(metric)
        for handler in self._llm_handlers:
            handler(payload)

    def emit_tool(self, metric: ToolExecutionMetric) -> None:
        """Emit tool."""
        # Keep events flowing.
        if not self._tool_handlers:
            return
        payload = asdict(metric)
        for handler in self._tool_handlers:
            handler(payload)
