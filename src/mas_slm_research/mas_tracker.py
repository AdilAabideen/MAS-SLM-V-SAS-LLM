"""In-memory causal tracking for multi-agent execution, independent of sinks."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from .mas_contract import AgentExecutionResult, HandoffEnvelope, MASState
from .mas_measurements import AgentMeasurementSummary, summarize_agent_measurements
from .workflows.definition import WorkflowDefinition


@dataclass(frozen=True)
class TrackedAgentExecution:
    mas_run_id: str
    agent_run_id: str
    agent_name: str
    sequence_index: int
    incoming_handoff_id: str | None


@dataclass(frozen=True)
class TrackedExecutionOutcome:
    agent_run_id: str
    sequence_index: int
    handoff_id: str | None = None


@dataclass
class AgentRecord:
    agent_run_id: str
    mas_run_id: str
    agent_name: str
    sequence_index: int
    incoming_handoff_ids: tuple[str, ...]
    input_payload: dict[str, Any]
    started_at: datetime
    status: str = "running"
    finished_at: datetime | None = None
    output: dict[str, Any] | None = None
    error_text: str | None = None
    outgoing_handoff_id: str | None = None
    is_final_agent: bool = False
    measurements: AgentMeasurementSummary | None = None


@dataclass
class HandoffRecord:
    handoff_id: str
    mas_run_id: str
    from_agent_run_id: str
    from_agent_name: str
    to_agent_name: str
    handoff_name: str
    payload_schema: str
    payload: dict[str, Any]
    created_at: datetime
    status: str = "created"
    to_agent_run_id: str | None = None
    accepted_at: datetime | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class GateRecord:
    gate_evaluation_id: str
    mas_run_id: str
    gate_id: str
    ready: bool
    satisfied_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]
    next_target: str | None
    handoff_ids: tuple[str, ...]
    created_at: datetime


class InMemoryMASTracker:
    """Own run records and causal IDs even when no event sink is configured."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        mas_run_id: str,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not mas_run_id:
            raise ValueError("mas_run_id must be nonempty")
        self.workflow = workflow
        self.mas_run_id = mas_run_id
        self.event_sink = event_sink
        self.id_factory = id_factory or (lambda: str(uuid4()))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.agent_records: dict[str, AgentRecord] = {}
        self.handoff_records: dict[str, HandoffRecord] = {}
        self.gate_records: list[GateRecord] = []
        self.final_output: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.sink_errors: list[str] = []
        self._sequence = 0
        self._lock = threading.RLock()

    def begin_agent_execution(
        self,
        *,
        agent_name: str,
        state: MASState,
        pending_agent_payload: dict[str, Any],
    ) -> TrackedAgentExecution:
        self.workflow.validate_agent(agent_name)
        with self._lock:
            self._sequence += 1
            agent_run_id = self.id_factory()
            now = self.clock()
            incoming = tuple(
                record.handoff_id for record in self.handoff_records.values()
                if record.to_agent_name == agent_name and record.to_agent_run_id is None
            )
            record = AgentRecord(
                agent_run_id=agent_run_id, mas_run_id=self.mas_run_id,
                agent_name=agent_name, sequence_index=self._sequence,
                incoming_handoff_ids=incoming, input_payload=dict(pending_agent_payload), started_at=now,
            )
            self.agent_records[agent_run_id] = record
            for handoff_id in incoming:
                handoff = self.handoff_records[handoff_id]
                handoff.to_agent_run_id = agent_run_id
                handoff.accepted_at = now
                handoff.status = "accepted"
                handoff.latency_ms = max(0, int((now - handoff.created_at).total_seconds() * 1000))
                self._emit("handoff_accepted", agent_run_id=agent_run_id, handoff_id=handoff_id, status="accepted")
            self._emit(
                "agent_started", agent_run_id=agent_run_id, agent_name=agent_name, status="running",
                payload_json={"sequence_index": self._sequence, "incoming_handoff_ids": list(incoming)},
            )
            return TrackedAgentExecution(
                mas_run_id=self.mas_run_id, agent_run_id=agent_run_id,
                agent_name=agent_name, sequence_index=self._sequence,
                incoming_handoff_id=incoming[-1] if incoming else None,
            )

    def complete_agent_execution(
        self,
        *,
        tracked: TrackedAgentExecution | None,
        result: AgentExecutionResult,
    ) -> TrackedExecutionOutcome | None:
        if tracked is None:
            return None
        result.validate_for(self.workflow)
        with self._lock:
            record = self.agent_records[tracked.agent_run_id]
            if record.status != "running":
                raise ValueError(f"agent run '{tracked.agent_run_id}' already completed")
            record.finished_at = self.clock()
            record.output = dict(result.final_output or {}) if result.status == "final" else dict(result.output)
            record.status = "failed" if result.status == "error" else "succeeded"
            record.error_text = _error_text(result.output) if result.status == "error" else None
            handoff_id: str | None = None
            if result.status == "final":
                record.is_final_agent = True
                self.final_output = dict(result.final_output or {})
                self._emit(
                    "final_output_created", agent_run_id=record.agent_run_id,
                    agent_name=record.agent_name, status="completed", payload_json=self.final_output,
                )
            elif result.handoff is not None:
                handoff_id = self._record_handoff(record, result.handoff)
                record.outgoing_handoff_id = handoff_id
            self._emit(
                "agent_completed", agent_run_id=record.agent_run_id,
                agent_name=record.agent_name, status=record.status,
                payload_json={"sequence_index": record.sequence_index, "result_status": result.status,
                              "handoff_id": handoff_id, "output": record.output},
            )
            return TrackedExecutionOutcome(record.agent_run_id, record.sequence_index, handoff_id)

    def state_updates_for_completion(
        self, *, tracked: TrackedAgentExecution | None, persisted: TrackedExecutionOutcome | None,
    ) -> dict[str, Any]:
        if tracked is None or persisted is None:
            return {}
        update: dict[str, Any] = {
            "current_agent_run_id": persisted.agent_run_id,
            "last_completed_agent_run_id": persisted.agent_run_id,
            "next_sequence_index": persisted.sequence_index + 1,
        }
        if persisted.handoff_id is not None:
            update["last_handoff_id"] = persisted.handoff_id
        return {"execution_context": update}

    def decorate_handoff_dict(
        self, *, handoff_dict: dict[str, Any], persisted: TrackedExecutionOutcome | None,
    ) -> dict[str, Any]:
        decorated = dict(handoff_dict)
        if persisted is not None and persisted.handoff_id is not None:
            decorated["handoff_id"] = persisted.handoff_id
        return decorated

    def record_gate_evaluation(self, *, gate_id: str, state: MASState, outcome: Any) -> str:
        if gate_id not in self.workflow.gates:
            raise ValueError(f"Unknown gate '{gate_id}'")
        with self._lock:
            evaluation_id = self.id_factory()
            handoff_ids = tuple(
                str(item["handoff_id"]) for item in outcome.handoffs_to_target
                if isinstance(item, Mapping) and item.get("handoff_id") is not None
            )
            record = GateRecord(
                gate_evaluation_id=evaluation_id, mas_run_id=self.mas_run_id,
                gate_id=gate_id, ready=bool(outcome.ready),
                satisfied_sources=tuple(outcome.satisfied_sources),
                missing_sources=tuple(outcome.missing_sources),
                next_target=outcome.next_target, handoff_ids=handoff_ids,
                created_at=self.clock(),
            )
            self.gate_records.append(record)
            self._emit(
                "gate_evaluated", gate_evaluation_id=evaluation_id,
                status="ready" if record.ready else "blocked",
                payload_json={"gate_id": gate_id, "ready": record.ready,
                              "satisfied_sources": list(record.satisfied_sources),
                              "missing_sources": list(record.missing_sources),
                              "handoff_ids": list(handoff_ids), "next_target": record.next_target},
            )
            return evaluation_id

    def record_agent_measurements(
        self,
        *,
        agent_run_id: str,
        llm_calls: list[Mapping[str, Any]],
        events: list[Mapping[str, Any]],
        reliability_issues: list[Mapping[str, Any]] | None = None,
        schema_validator: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> AgentMeasurementSummary:
        with self._lock:
            record = self.agent_records[agent_run_id]
            duration_ms = (
                int((record.finished_at - record.started_at).total_seconds() * 1000)
                if record.finished_at is not None else None
            )
            summary = summarize_agent_measurements(
                status=record.status, duration_ms=duration_ms,
                error_text=record.error_text, output=record.output,
                llm_calls=llm_calls, events=events,
                reliability_issues=reliability_issues or (), schema_validator=schema_validator,
            )
            record.measurements = summary
            return summary

    def fail_unfinished(self, *, error_text: str) -> None:
        """Close child records if graph execution stops outside a node outcome."""
        with self._lock:
            for record in self.agent_records.values():
                if record.status != "running":
                    continue
                record.status = "failed"
                record.finished_at = self.clock()
                record.error_text = error_text
                record.output = {"error": "graph_execution_failed", "detail": error_text}
                self._emit(
                    "agent_completed", agent_run_id=record.agent_run_id,
                    agent_name=record.agent_name, status="failed",
                    payload_json={"error": error_text},
                )

    def _record_handoff(self, record: AgentRecord, handoff: HandoffEnvelope) -> str:
        handoff_id = self.id_factory()
        self.handoff_records[handoff_id] = HandoffRecord(
            handoff_id=handoff_id, mas_run_id=self.mas_run_id,
            from_agent_run_id=record.agent_run_id, from_agent_name=record.agent_name,
            to_agent_name=handoff.target_agent, handoff_name=handoff.handoff_name,
            payload_schema=handoff.payload_schema, payload=dict(handoff.payload), created_at=self.clock(),
        )
        self._emit(
            "handoff_created", agent_run_id=record.agent_run_id, agent_name=record.agent_name,
            handoff_id=handoff_id, status="created", payload_json=handoff.model_dump(),
        )
        return handoff_id

    def _emit(self, event_type: str, **details: Any) -> None:
        event = {
            "seq": len(self.events) + 1,
            "mas_run_id": self.mas_run_id,
            "workflow_id": self.workflow.metadata.workflow_id,
            "event_type": event_type,
            **details,
        }
        self.events.append(event)
        if self.event_sink is not None:
            try:
                self.event_sink(dict(event))
            except Exception as exc:
                self.sink_errors.append(str(exc) or type(exc).__name__)


def _error_text(output: Mapping[str, Any]) -> str | None:
    if not output:
        return None
    if output.get("error") is not None:
        detail = output.get("detail")
        return f"{output['error']}: {detail}" if detail else str(output["error"])
    return json.dumps(dict(output), default=str)
