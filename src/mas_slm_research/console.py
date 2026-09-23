"""One human-readable stderr renderer for captured case and graph traces."""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from typing import Any, Mapping, TextIO

from .experiment import ExperimentAttempt
from .grading import GradeResult
from .multi_agent import MultiCaseExecution
from .single_agent import SingleCaseExecution


_SENSITIVE = ("api_key", "password", "secret", "authorization", "access_token", "refresh_token")


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "[redacted]" if any(part in str(key).lower() for part in _SENSITIVE)
                else _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


class ConsoleRenderer:
    """Render captured events without modifying result or measurement objects."""

    def __init__(
        self, *, mode: str = "full", color: str = "auto", stream: TextIO | None = None,
        width: int | None = None,
    ) -> None:
        if mode not in {"none", "summary", "events", "full"}:
            raise ValueError(f"unknown console mode {mode!r}")
        if color not in {"auto", "always", "never"}:
            raise ValueError(f"unknown color mode {color!r}")
        self.mode = mode
        self.stream = stream if stream is not None else sys.stderr
        self.width = width or shutil.get_terminal_size((100, 24)).columns
        self.use_color = "NO_COLOR" not in os.environ and (
            color == "always" or color == "auto" and bool(getattr(self.stream, "isatty", lambda: False)())
        )

    def _line(self, message: str, *, color: str | None = None) -> None:
        if self.mode == "none":
            return
        wrapped = textwrap.wrap(message, width=max(40, self.width), subsequent_indent="  ",
                                break_long_words=False, break_on_hyphens=False) or [""]
        for line in wrapped:
            if self.use_color and color:
                print(f"\x1b[{color}m{line}\x1b[0m", file=self.stream)
            else:
                print(line, file=self.stream)

    def _payload(self, value: Any) -> str:
        return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, default=str)

    def _event(self, source: str, event: Mapping[str, Any], *, handoff_targets: Mapping[str, str] | None = None,
               agent_names: Mapping[str, str] | None = None) -> None:
        kind = str(event.get("event_type") or source)
        agent = str(event.get("agent_name") or event.get("from_agent_name")
                    or (agent_names or {}).get(str(event.get("agent_run_id"))) or "workflow")
        prefix = f"  [{agent}] {kind}"
        if kind in {"tool_call", "tool_result"}:
            prefix += f" {event.get('tool_name')}"
        elif kind.startswith("handoff"):
            payload = event.get("payload_json")
            target = payload.get("target_agent") if isinstance(payload, Mapping) else None
            if target is None and isinstance(payload, Mapping) and isinstance(payload.get("handoff"), Mapping):
                target = payload["handoff"].get("target_agent")
            target = target or (handoff_targets or {}).get(str(event.get("handoff_id")))
            prefix += f" -> {target or event.get('to_agent_name') or 'next agent'}"
        elif kind == "gate_evaluated":
            payload = event.get("payload_json")
            if isinstance(payload, Mapping):
                prefix += f" {payload.get('gate_id')} ready={payload.get('ready')}"
        elif source == "model_call":
            prefix += f" tokens={event.get('tokens_total')} usage={event.get('usage_source')}"
        elif source == "tool_measurement":
            prefix += f" {event.get('tool_name')} status={event.get('status')}"
        if event.get("status") is not None and source in {"graph_event", "agent_event"}:
            prefix += f" status={event['status']}"
        self._line(prefix, color="36" if kind.startswith("handoff") else "35" if kind == "gate_evaluated" else None)
        if self.mode == "full":
            payload = event.get("payload_json")
            if payload is not None:
                self._line(f"    payload: {self._payload(payload)}")
            elif source in {"model_call", "tool_measurement"}:
                details = {key: event.get(key) for key in (
                    "call_index", "tool_call_id", "latency_ms", "error_text", "text_recovered_tool_call_count"
                ) if event.get(key) is not None}
                if details:
                    self._line(f"    details: {self._payload(details)}")

    def render_case(
        self, *, system_id: str, case_id: str, repetition: int,
        execution: SingleCaseExecution | MultiCaseExecution | None,
        grade: GradeResult,
    ) -> None:
        if self.mode == "none":
            return
        self._line(f"[{system_id}] case={case_id} repetition={repetition}", color="1;34")
        if self.mode in {"events", "full"} and execution is not None:
            if isinstance(execution, MultiCaseExecution):
                timeline = execution.timeline or tuple({"source": "graph_event", "event": event} for event in execution.events)
                handoff_targets = {item.handoff_id: item.to_agent_name for item in execution.handoff_records}
                agent_names = {item.agent_run_id: item.agent_name for item in execution.agent_records}
            else:
                timeline = tuple({"source": "agent_event", "event": event} for event in execution.events)
                handoff_targets = {}
                agent_names = {}
            for item in timeline:
                event = item.get("event")
                if isinstance(event, Mapping):
                    self._event(str(item.get("source") or "event"), event,
                                handoff_targets=handoff_targets, agent_names=agent_names)
        color = "32" if grade.passed else "31" if grade.passed is False else "33"
        self._line(f"  final: {grade.status.value} passed={grade.passed} score={grade.score}", color=color)

    def render_attempt(self, attempt: ExperimentAttempt) -> None:
        self.render_case(
            system_id=attempt.system_id, case_id=attempt.case_id,
            repetition=attempt.repetition, execution=attempt.execution,
            grade=attempt.grade,
        )
