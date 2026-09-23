"""Opt-in, payload-free span records and an optional OTLP HTTP bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from .configuration import TelemetryConfig
from .experiment import ExperimentAttempt, ExperimentPair, ExperimentRun, ExperimentStatus
from .grading import GradeResult
from .multi_agent import MultiCaseExecution
from .single_agent import SingleCaseExecution


@dataclass(frozen=True, kw_only=True)
class TraceEvent:
    name: str
    attributes: Mapping[str, str | int | bool]


@dataclass(frozen=True, kw_only=True)
class SpanRecord:
    span_id: str
    parent_id: str | None
    name: str
    started_at: str
    ended_at: str
    attributes: Mapping[str, str | int | bool]
    events: tuple[TraceEvent, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TraceOutcome:
    enabled: bool
    spans: tuple[SpanRecord, ...]
    warnings: tuple[str, ...] = ()


class SpanSink(Protocol):
    def export(self, spans: Sequence[SpanRecord]) -> None: ...


def _id(*parts: object) -> str:
    return str(uuid5(NAMESPACE_URL, "/".join(str(part) for part in parts)))


def _instant(value: Any, *, fallback: str) -> str:
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return fallback


def _events(attempt: ExperimentAttempt, agent_ids: Mapping[str, str]) -> tuple[dict[str, list[TraceEvent]], list[TraceEvent]]:
    by_agent: dict[str, list[TraceEvent]] = {span_id: [] for span_id in agent_ids.values()}
    system_events: list[TraceEvent] = []
    execution = attempt.execution
    if execution is None:
        return by_agent, system_events
    timeline = execution.timeline if isinstance(execution, MultiCaseExecution) else tuple(
        {"source": "agent_event", "event": event} for event in execution.events
    )
    for item in timeline:
        event = item.get("event")
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("event_type") or "")
        payload = event.get("payload_json")
        decision = payload.get("decision") if isinstance(payload, Mapping) else None
        if not (kind.startswith("handoff") or kind == "gate_evaluated" or kind == "runtime_decision"):
            continue
        attributes: dict[str, str | int | bool] = {
            "case_id": attempt.case_id, "system_id": attempt.system_id,
        }
        for key in ("agent_name", "agent_run_id", "handoff_id", "gate_evaluation_id", "status"):
            value = event.get(key)
            if isinstance(value, (str, int, bool)):
                attributes[key] = value
        if isinstance(decision, str):
            attributes["decision"] = decision
        if kind == "gate_evaluated" and isinstance(payload, Mapping):
            if isinstance(payload.get("gate_id"), str):
                attributes["gate_id"] = payload["gate_id"]
            if isinstance(payload.get("ready"), bool):
                attributes["ready"] = payload["ready"]
        record = TraceEvent(name=kind, attributes=attributes)
        agent_id = str(event.get("agent_run_id") or event.get("run_id") or "")
        if kind == "runtime_decision" and agent_id in agent_ids:
            by_agent[agent_ids[agent_id]].append(record)
        else:
            system_events.append(record)
    return by_agent, system_events


def build_span_records(run: ExperimentRun) -> tuple[SpanRecord, ...]:
    """Reconstruct one experiment/case/system/agent/model/tool tree after execution."""
    root_id = _id(run.experiment_id, "experiment")
    spans: list[SpanRecord] = [SpanRecord(
        span_id=root_id, parent_id=None, name="experiment", started_at=run.started_at,
        ended_at=run.ended_at,
        attributes={"experiment_id": run.experiment_id, "status": run.status.value},
    )]
    for pair in run.pairs:
        pair_attempts = tuple(item for item in (pair.single, pair.multi) if item is not None)
        if not pair_attempts:
            continue
        case_id = _id(run.experiment_id, pair.case_id, pair.repetition, "case")
        spans.append(SpanRecord(
            span_id=case_id, parent_id=root_id, name="case",
            started_at=min(item.started_at for item in pair_attempts),
            ended_at=max(item.ended_at for item in pair_attempts),
            attributes={"case_id": pair.case_id, "repetition": pair.repetition},
        ))
        for attempt in pair_attempts:
            system_id = _id(attempt.result.identity.run_id, "system")
            execution = attempt.execution
            agent_ids: dict[str, str] = {}
            if isinstance(execution, MultiCaseExecution):
                agent_ids = {item.agent_run_id: _id(item.agent_run_id, "agent")
                             for item in execution.agent_records}
            elif execution is not None:
                agent_ids = {attempt.result.identity.run_id: _id(attempt.result.identity.run_id, "agent")}
            agent_events, system_events = _events(attempt, agent_ids)
            spans.append(SpanRecord(
                span_id=system_id, parent_id=case_id, name=f"system.{attempt.system_id}",
                started_at=attempt.started_at, ended_at=attempt.ended_at,
                attributes={"case_id": attempt.case_id, "repetition": attempt.repetition,
                            "system_id": attempt.system_id, "run_id": attempt.result.identity.run_id,
                            "status": attempt.result.status.value},
                events=tuple(system_events),
            ))
            if isinstance(execution, MultiCaseExecution):
                for agent in execution.agent_records:
                    span_id = agent_ids[agent.agent_run_id]
                    spans.append(SpanRecord(
                        span_id=span_id, parent_id=system_id, name=f"agent.{agent.agent_name}",
                        started_at=_instant(agent.started_at, fallback=attempt.started_at),
                        ended_at=_instant(agent.finished_at, fallback=attempt.ended_at),
                        attributes={"agent_name": agent.agent_name, "agent_run_id": agent.agent_run_id,
                                    "status": agent.status}, events=tuple(agent_events[span_id]),
                    ))
            elif execution is not None:
                span_id = agent_ids[attempt.result.identity.run_id]
                agent_name = str(execution.llm_calls[0].get("agent_name") or "single_agent") if execution.llm_calls else "single_agent"
                spans.append(SpanRecord(
                    span_id=span_id, parent_id=system_id, name=f"agent.{agent_name}",
                    started_at=attempt.started_at, ended_at=attempt.ended_at,
                    attributes={"agent_name": agent_name, "run_id": attempt.result.identity.run_id},
                    events=tuple(agent_events[span_id]),
                ))
            if execution is None:
                continue
            for index, call in enumerate(execution.llm_calls, 1):
                run_id = str(call.get("run_id") or attempt.result.identity.run_id)
                spans.append(SpanRecord(
                    span_id=_id(attempt.result.identity.run_id, "model", run_id, index),
                    parent_id=agent_ids.get(run_id, system_id), name="model.call",
                    started_at=_instant(call.get("started_at"), fallback=attempt.started_at),
                    ended_at=_instant(call.get("ended_at"), fallback=attempt.ended_at),
                    attributes={"agent_name": str(call.get("agent_name") or ""),
                                "run_id": run_id, "call_index": index,
                                "usage_source": str(call.get("usage_source") or "unknown")},
                ))
            for index, call in enumerate(execution.tool_calls, 1):
                run_id = str(call.get("run_id") or attempt.result.identity.run_id)
                spans.append(SpanRecord(
                    span_id=_id(attempt.result.identity.run_id, "tool", run_id, index),
                    parent_id=agent_ids.get(run_id, system_id), name="tool.call",
                    started_at=_instant(call.get("started_at"), fallback=attempt.started_at),
                    ended_at=_instant(call.get("ended_at"), fallback=attempt.ended_at),
                    attributes={"agent_name": str(call.get("agent_name") or ""),
                                "run_id": run_id, "tool_name": str(call.get("tool_name") or ""),
                                "status": str(call.get("status") or "unknown")},
                ))
    ids = [span.span_id for span in spans]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trace span ID")
    known = set(ids)
    if any(span.parent_id not in known for span in spans if span.parent_id is not None):
        raise ValueError("trace span has an unknown parent")
    return tuple(spans)


def _nanoseconds(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1_000_000_000)


class OTLPHTTPSpanSink:
    """Late-imported HTTP/protobuf bridge; no SDK or collector required when off."""

    def __init__(self, *, endpoint: str, headers: Mapping[str, str] | None = None) -> None:
        self.endpoint = endpoint
        self.headers = dict(headers or {})

    def export(self, spans: Sequence[SpanRecord]) -> None:
        from opentelemetry.context import Context
        from opentelemetry.trace import set_span_in_context
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        memory = InMemorySpanExporter()
        provider = TracerProvider(resource=Resource.create({"service.name": "mas-slm-research"}))
        provider.add_span_processor(SimpleSpanProcessor(memory))
        tracer = provider.get_tracer("mas_slm_research.tracing")
        parents: dict[str, Any] = {}
        for record in spans:
            context = set_span_in_context(parents[record.parent_id]) if record.parent_id else Context()
            span = tracer.start_span(
                record.name, context=context, attributes=dict(record.attributes),
                start_time=_nanoseconds(record.started_at),
            )
            parents[record.span_id] = span
            for event in record.events:
                span.add_event(event.name, attributes=dict(event.attributes))
            span.end(end_time=_nanoseconds(record.ended_at))
        exporter = OTLPSpanExporter(endpoint=self.endpoint, headers=self.headers)
        try:
            result = exporter.export(memory.get_finished_spans())
            if result != SpanExportResult.SUCCESS:
                raise RuntimeError(f"OTLP export returned {result}")
        finally:
            exporter.shutdown()
            provider.shutdown()


def trace_experiment(
    run: ExperimentRun, *, config: TelemetryConfig,
    environment: Mapping[str, str] | None = None, sink: SpanSink | None = None,
) -> TraceOutcome:
    """Export when enabled; surface exporter errors without altering the run."""
    if not config.enabled:
        return TraceOutcome(enabled=False, spans=())
    spans = build_span_records(run)
    warnings: list[str] = []
    if sink is None and config.exporter == "otlp_http":
        env = environment if environment is not None else os.environ
        endpoint = env.get(config.endpoint_env or "")
        if not endpoint:
            warnings.append("OTLP endpoint environment reference is unavailable")
        else:
            headers: dict[str, str] = {}
            for header, env_name in config.header_env.items():
                value = env.get(env_name)
                if not value:
                    warnings.append(f"OTLP header environment reference {env_name!r} is unavailable")
                else:
                    headers[header] = value
            if not warnings:
                sink = OTLPHTTPSpanSink(endpoint=endpoint, headers=headers)
    if sink is not None:
        try:
            sink.export(spans)
        except ModuleNotFoundError:
            warnings.append("trace export failed: optional OpenTelemetry dependency unavailable")
        except Exception as exc:
            warnings.append(f"trace export failed: {type(exc).__name__}")
    return TraceOutcome(enabled=True, spans=spans, warnings=tuple(warnings))


def trace_case_execution(
    execution: SingleCaseExecution | MultiCaseExecution, *, grade: GradeResult,
    config: TelemetryConfig, environment: Mapping[str, str] | None = None,
    sink: SpanSink | None = None,
) -> TraceOutcome:
    """Use the same span tree for a CLI single-case run."""
    if not config.enabled:
        return TraceOutcome(enabled=False, spans=())
    result = execution.result
    identity = result.identity
    ended = datetime.now(timezone.utc)
    started = ended - timedelta(seconds=result.timing.wall_seconds)
    attempt = ExperimentAttempt(
        sequence=1, case_id=identity.case_id, repetition=identity.repetition,
        system_id=identity.system_id, started_at=started.isoformat(), ended_at=ended.isoformat(),
        model_choices={}, runtime_policies={}, execution=execution, result=result, grade=grade,
    )
    pair = ExperimentPair(
        case_id=identity.case_id, repetition=identity.repetition,
        single=attempt if identity.system_id == "single" else None,
        multi=attempt if identity.system_id == "multi" else None,
    )
    run = ExperimentRun(
        experiment_id=identity.experiment_id, status=ExperimentStatus.COMPLETED,
        schedule="single_case", dataset_path="", started_at=started.isoformat(),
        ended_at=ended.isoformat(), attempts=(attempt,), pairs=(pair,),
    )
    return trace_experiment(run, config=config, environment=environment, sink=sink)
