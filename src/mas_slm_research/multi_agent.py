"""Execute one multi-agent case with fresh roles and in-memory accounting."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .contracts import (
    CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming,
    TokenUsage, UsageSource, ValidatedOutput,
)
from .kernel import AgentKernel
from .runtime.budget import BudgetExceeded
from .mas.agent_node_executor import AgentNodeExecutor
from .mas.execution_strategy import ExecutionRequest
from .mas.gate_evaluator import GateEvaluator
from .mas.graph_builder import MASGraphBuilder
from .mas_contract import AgentExecutionResult, HandoffEnvelope, MASState, make_initial_mas_state
from .mas_tracker import AgentRecord, GateRecord, HandoffRecord, InMemoryMASTracker
from .workflows.definition import WorkflowDefinition


RoleFactory = Callable[[], AgentKernel]
PayloadBuilder = Callable[[str, MASState], dict[str, Any]]
OutputValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class _ChildTrace:
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class MultiCaseExecution:
    result: CaseResult
    agent_records: tuple[AgentRecord, ...]
    handoff_records: tuple[HandoffRecord, ...]
    gate_records: tuple[GateRecord, ...]
    events: tuple[dict[str, Any], ...]
    llm_calls: tuple[dict[str, Any], ...]
    tool_calls: tuple[dict[str, Any], ...]
    sink_errors: tuple[str, ...]
    timeline: tuple[dict[str, Any], ...] = ()


class _KernelRoleStrategy:
    mode = "kernel"

    def __init__(self, *, workflow: WorkflowDefinition, roles: Mapping[str, AgentKernel], timeline: list[dict[str, Any]], max_handoffs: int | None = None) -> None:
        self.workflow = workflow
        self.roles = roles
        self.child_traces: dict[str, _ChildTrace] = {}
        self.timeline = timeline
        self.max_handoffs = max_handoffs
        self.handoffs_used = 0

    async def execute(self, request: ExecutionRequest) -> AgentExecutionResult:
        role = request.agent_name
        kernel = self.roles[role]
        context = dict(request.state_snapshot.get("execution_context") or {})
        child_id = str(context.get("current_agent_run_id") or "")
        if not child_id:
            raise ValueError("tracked child run ID is required before role execution")
        trace = _ChildTrace()
        self.child_traces[child_id] = trace
        kernel.set_event_context(run_id=child_id, agent_name=role)
        def record_event(event: dict[str, Any]) -> None:
            trace.events.append(event)
            self.timeline.append({"source": "agent_event", "event": event})

        def record_llm(call: dict[str, Any]) -> None:
            trace.llm_calls.append(call)
            self.timeline.append({"source": "model_call", "event": call})

        def record_tool(call: dict[str, Any]) -> None:
            trace.tool_calls.append(call)
            self.timeline.append({"source": "tool_measurement", "event": call})

        kernel.add_event_handler(record_event)
        kernel.add_llm_call_handler(record_llm)
        kernel.add_tool_call_handler(record_tool)
        invocation = kernel.ainvoke(request.pending_agent_payload["llm_payload"])
        raw = await asyncio.wait_for(invocation, timeout=kernel.runtime_config.max_elapsed_seconds) if kernel.runtime_config.max_elapsed_seconds else await invocation

        if isinstance(raw, Mapping) and isinstance(raw.get("handoff"), Mapping):
            handoff = HandoffEnvelope.model_validate(raw["handoff"]).validate_for(self.workflow)
            self.handoffs_used += 1
            if self.max_handoffs is not None and self.handoffs_used > self.max_handoffs:
                raise BudgetExceeded(counter="handoffs", used=self.handoffs_used, limit=self.max_handoffs)
            output = raw.get("output")
            return AgentExecutionResult(
                agent_name=role, status="handoff",
                output=dict(output) if isinstance(output, Mapping) else {},
                handoff=handoff,
            )
        if isinstance(raw, Mapping) and raw.get("ok") is True and raw.get("error") is None:
            return AgentExecutionResult(agent_name=role, status="final", final_output=dict(raw), output=dict(raw))
        if isinstance(raw, Mapping):
            return AgentExecutionResult(agent_name=role, status="error", output=dict(raw))
        return AgentExecutionResult(
            agent_name=role, status="error",
            output={"error": "agent_output_invalid", "detail": str(raw)},
        )


class MultiAgentRunner:
    """Build one role registry and tracked LangGraph execution per case."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        role_factories: Mapping[str, RoleFactory],
        payload_builder: PayloadBuilder,
        output_validator: OutputValidator,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        max_handoffs: int | None = None,
        max_elapsed_seconds: float | None = None,
    ) -> None:
        expected = set(workflow.participating_agents)
        actual = set(role_factories)
        if actual != expected:
            raise ValueError(f"role factories must exactly match workflow agents; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        self.workflow = workflow
        self.role_factories = dict(role_factories)
        self.payload_builder = payload_builder
        self.output_validator = output_validator
        self.event_sink = event_sink
        if max_handoffs is not None and (isinstance(max_handoffs, bool) or not isinstance(max_handoffs, int) or max_handoffs < 1):
            raise ValueError("max_handoffs must be a positive integer")
        if max_elapsed_seconds is not None and (
            isinstance(max_elapsed_seconds, bool) or not isinstance(max_elapsed_seconds, (int, float))
            or not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0
        ):
            raise ValueError("max_elapsed_seconds must be positive")
        self.max_handoffs = max_handoffs
        self.max_elapsed_seconds = max_elapsed_seconds

    async def run_case(self, *, identity: RunIdentity, case_info: Mapping[str, Any]) -> MultiCaseExecution:
        started = time.perf_counter()
        timeline: list[dict[str, Any]] = []

        def record_graph_event(event: dict[str, Any]) -> None:
            timeline.append({"source": "graph_event", "event": event})
            if self.event_sink is not None:
                self.event_sink(event)

        tracker = InMemoryMASTracker(
            workflow=self.workflow, mas_run_id=identity.run_id, event_sink=record_graph_event,
        )
        strategy: _KernelRoleStrategy | None = None
        graph_error: Exception | None = None
        final_output: Any = None

        try:
            roles = {name: self.role_factories[name]() for name in self.workflow.participating_agents}
            for name, kernel in roles.items():
                if not isinstance(kernel, AgentKernel):
                    raise TypeError(f"role '{name}' factory must return an extracted AgentKernel")
                if not kernel.runtime_config.persist_events:
                    raise ValueError(f"role '{name}' must emit in-memory measurements")
            strategy = _KernelRoleStrategy(workflow=self.workflow, roles=roles, timeline=timeline, max_handoffs=self.max_handoffs)
            graph = MASGraphBuilder(
                workflow=self.workflow,
                agent_executor=AgentNodeExecutor(
                    workflow=self.workflow, strategy=strategy,
                    payload_builder=self.payload_builder, execution_tracker=tracker,
                ),
                gate_evaluator=GateEvaluator(workflow=self.workflow, execution_tracker=tracker),
            ).build()
            invocation = graph.ainvoke(make_initial_mas_state(
                dict(case_info), execution_context={
                    "mas_run_id": identity.run_id,
                    "workflow_id": self.workflow.metadata.workflow_id,
                    "workflow_version": self.workflow.metadata.version,
                },
            ))
            state = await asyncio.wait_for(invocation, timeout=self.max_elapsed_seconds) if self.max_elapsed_seconds else await invocation
            final_output = state.get("final_output")
        except asyncio.CancelledError:
            tracker.fail_unfinished(error_text="experiment_cancelled")
            raise
        except Exception as exc:
            graph_error = exc
            tracker.fail_unfinished(error_text=str(exc) or type(exc).__name__)

        traces = strategy.child_traces if strategy is not None else {}
        for child_id, record in tracker.agent_records.items():
            trace = traces.get(child_id, _ChildTrace())
            tracker.record_agent_measurements(
                agent_run_id=child_id, llm_calls=trace.llm_calls, events=trace.events,
            )

        all_llm_calls = tuple(call for trace in traces.values() for call in trace.llm_calls)
        all_tool_calls = tuple(call for trace in traces.values() for call in trace.tool_calls)
        output: ValidatedOutput | None = None
        failure: RunFailure | None = None
        failed_children = [record for record in tracker.agent_records.values() if record.status == "failed"]
        if graph_error is not None:
            if isinstance(graph_error, BudgetExceeded):
                kind = FailureKind.BUDGET
            elif isinstance(graph_error, TimeoutError):
                kind = FailureKind.TIMEOUT
            elif any(call.get("error_text") for call in all_llm_calls):
                kind = FailureKind.PROVIDER
            else:
                kind = FailureKind.RUNTIME
            failure = RunFailure(kind=kind, message=str(graph_error) or type(graph_error).__name__)
        elif failed_children:
            provider_messages = [str(call["error_text"]) for call in all_llm_calls if call.get("error_text")]
            provider_error = bool(provider_messages)
            reason = provider_messages[0] if provider_error else (failed_children[0].error_text or "child_execution_failed")
            if "_budget_exceeded" in reason:
                kind = FailureKind.BUDGET
            elif "TimeoutError" in reason or "run_timeout_exceeded" in reason:
                kind = FailureKind.TIMEOUT
            elif provider_error:
                kind = FailureKind.PROVIDER
            elif any(call.get("error_text") for call in all_tool_calls):
                kind = FailureKind.TOOL
            else:
                kind = FailureKind.RUNTIME
            failure = RunFailure(kind=kind, message=reason)
        elif not isinstance(final_output, Mapping):
            failure = RunFailure(kind=FailureKind.VALIDATION, message="final_output_missing")
        else:
            try:
                output = ValidatedOutput(value=self.output_validator(final_output))
            except Exception as exc:
                failure = RunFailure(kind=FailureKind.VALIDATION, message=str(exc) or "final_output_invalid")

        child_duration_ms = [
            record.measurements.duration_ms for record in tracker.agent_records.values()
            if record.measurements is not None and record.measurements.duration_ms is not None
        ]
        result = CaseResult(
            identity=identity,
            status=RunStatus.COMPLETED if output is not None else RunStatus.FAILED,
            output=output, failure=failure,
            usage=_aggregate_usage(all_llm_calls),
            timing=RunTiming(
                wall_seconds=time.perf_counter() - started,
                child_seconds_sum=sum(child_duration_ms) / 1000.0 if child_duration_ms else None,
            ),
        )
        return MultiCaseExecution(
            result=result,
            agent_records=tuple(tracker.agent_records.values()),
            handoff_records=tuple(tracker.handoff_records.values()),
            gate_records=tuple(tracker.gate_records),
            events=tuple(tracker.events),
            llm_calls=all_llm_calls,
            tool_calls=all_tool_calls,
            sink_errors=tuple(tracker.sink_errors),
            timeline=tuple({"sequence": index, **entry} for index, entry in enumerate(timeline, 1)),
        )


def _aggregate_usage(llm_calls: tuple[dict[str, Any], ...]) -> TokenUsage:
    if not llm_calls:
        return TokenUsage()
    source = UsageSource.PROVIDER if all(call.get("usage_source") == "provider" for call in llm_calls) else UsageSource.ESTIMATED
    return TokenUsage(
        source=source,
        input_tokens=sum(int(call.get("input_tokens") or 0) for call in llm_calls),
        output_tokens=sum(int(call.get("output_tokens") or 0) for call in llm_calls),
        total_tokens=sum(int(call.get("tokens_total") or 0) for call in llm_calls),
    )
