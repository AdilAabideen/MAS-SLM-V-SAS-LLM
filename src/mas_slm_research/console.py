"""Readable, live stderr progress for single- and multi-agent cases."""

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
        self._handoff_targets: dict[str, str] = {}
        self._agent_names: dict[str, str] = {}
        self._system_id: str | None = None
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
                print(f"\x1b[{color}m{line}\x1b[0m", file=self.stream, flush=True)
            else:
                print(line, file=self.stream, flush=True)

    def _payload(self, value: Any) -> str:
        return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, indent=2, default=str)

    def _json_block(self, value: Any) -> None:
        if self.mode == "none":
            return
        for line in self._payload(value).splitlines():
            print(f"  {line}", file=self.stream, flush=True)

    def _event(self, source: str, event: Mapping[str, Any], *, handoff_targets: Mapping[str, str] | None = None,
               agent_names: Mapping[str, str] | None = None) -> None:
        kind = str(event.get("event_type") or source)
        agent = str(event.get("agent_name") or event.get("from_agent_name")
                    or (agent_names or {}).get(str(event.get("agent_run_id"))) or "workflow")
        payload = event.get("payload_json")
        if kind == "agent_started":
            if self._system_id != "multi":
                self._line(f"Agent: {agent}", color="1;34")
                self._line("")
        elif kind == "tool_call":
            label = f"[Tool call] {event.get('tool_name')}"
            if self._system_id == "multi":
                label = f"Agent: {agent} | {label}"
            self._line(label, color="36")
            if self.mode == "full":
                arguments = payload.get("args") if isinstance(payload, Mapping) else payload
                if arguments is not None:
                    self._json_block(arguments)
            self._line("")
        elif kind == "tool_result" and event.get("status") == "error":
            self._line(f"[Tool error] {event.get('tool_name')}", color="31")
            if self.mode == "full" and payload is not None:
                self._json_block(payload)
            self._line("")
        elif kind == "handoff_created":
            target = payload.get("target_agent") if isinstance(payload, Mapping) else None
            target = target or (handoff_targets or {}).get(str(event.get("handoff_id"))) or "next agent"
            self._line(f"Handoff: {agent} → {target}", color="36")
            self._line("")
        elif kind == "gate_evaluated" and isinstance(payload, Mapping):
            self._line(f"Gate: {payload.get('gate_id')} | ready={payload.get('ready')}", color="35")
            self._line("")
        elif kind in {"error", "agent_failed"}:
            self._line(f"Error: {agent}", color="31")
            if self.mode == "full" and payload is not None:
                self._json_block(payload)
            self._line("")

    def render_case(
        self, *, system_id: str, case_id: str, repetition: int,
        execution: SingleCaseExecution | MultiCaseExecution | None,
        grade: GradeResult,
    ) -> None:
        if self.mode == "none":
            return
        self.start_case(system_id=system_id, case_id=case_id, repetition=repetition)
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
                    self.render_event(str(item.get("source") or "event"), event,
                                      handoff_targets=handoff_targets, agent_names=agent_names)
        self.finish_case(grade, result=execution.result if execution is not None else None)

    def start_case(
        self, *, system_id: str, case_id: str, repetition: int,
        agent_name: str | None = None, agent_definition: str | None = None,
    ) -> None:
        self._handoff_targets.clear()
        self._agent_names.clear()
        self._system_id = system_id
        self._line("")
        label = {"single": "SAS · single-agent system", "multi": "MAS · multi-agent system"}.get(system_id, system_id)
        self._line(f"{label} | Case: {case_id} | Repetition: {repetition}", color="1;34")
        if agent_name:
            detail = f" ({agent_definition})" if agent_definition else ""
            self._line(f"Agent: {agent_name}{detail}")
        self._line("")

    def render_event(
        self, source: str, event: Mapping[str, Any], *,
        handoff_targets: Mapping[str, str] | None = None,
        agent_names: Mapping[str, str] | None = None,
    ) -> None:
        if self.mode in {"events", "full"}:
            run_id = event.get("agent_run_id")
            agent_name = event.get("agent_name")
            if run_id and agent_name:
                self._agent_names[str(run_id)] = str(agent_name)
            handoff_id = event.get("handoff_id")
            payload = event.get("payload_json")
            if handoff_id and isinstance(payload, Mapping) and payload.get("target_agent"):
                self._handoff_targets[str(handoff_id)] = str(payload["target_agent"])
            self._event(
                source, event,
                handoff_targets={**self._handoff_targets, **(handoff_targets or {})},
                agent_names={**self._agent_names, **(agent_names or {})},
            )

    def finish_case(self, grade: GradeResult, *, result: Any = None, expected: Mapping[str, Any] | None = None) -> None:
        if self.mode == "none":
            return
        color = "32" if grade.passed else "31" if grade.passed is False else "33"
        verdict = "PASS" if grade.passed is True else "FAIL" if grade.passed is False else "UNAVAILABLE"
        details = [f"Result: {verdict}"]
        output = getattr(result, "output", None)
        value = getattr(output, "value", None)
        if isinstance(value, Mapping) and value.get("final_esi_level") is not None:
            details.append(f"ESI {value['final_esi_level']}")
        if expected is not None and expected.get("acuity") is not None:
            details.append(f"expected ESI {expected['acuity']}")
        if grade.score is not None:
            details.append(f"score {grade.score}")
        self._line(" | ".join(details), color=color)
        if grade.error:
            self._line(f"Reason: {grade.error}", color="31")
        self._line("")

    def render_attempt(self, attempt: ExperimentAttempt) -> None:
        self.render_case(
            system_id=attempt.system_id, case_id=attempt.case_id,
            repetition=attempt.repetition, execution=attempt.execution,
            grade=attempt.grade,
        )
