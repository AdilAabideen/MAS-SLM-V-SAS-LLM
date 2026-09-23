"""Execute one single-agent case with in-memory measurements."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .contracts import (
    CaseResult,
    FailureKind,
    RunFailure,
    RunIdentity,
    RunStatus,
    RunTiming,
    TokenUsage,
    UsageSource,
    ValidatedOutput,
)
from .kernel import AgentKernel


OutputValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, kw_only=True)
class SingleCaseExecution:
    """Terminal result plus the raw in-memory trace for one attempt."""

    result: CaseResult
    events: tuple[dict[str, Any], ...]
    llm_calls: tuple[dict[str, Any], ...]
    tool_calls: tuple[dict[str, Any], ...]


class SingleAgentRunner:
    """Create a fresh configured kernel for every case attempt."""

    def __init__(
        self,
        *,
        kernel_factory: Callable[[], AgentKernel],
        agent_name: str,
        output_validator: OutputValidator | None = None,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("agent_name must be nonempty")
        self.kernel_factory = kernel_factory
        self.agent_name = agent_name
        self.output_validator = output_validator

    async def run_case(self, *, identity: RunIdentity, payload: Any) -> SingleCaseExecution:
        events: list[dict[str, Any]] = []
        llm_calls: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        started = time.perf_counter()
        output: ValidatedOutput | None = None
        failure: RunFailure | None = None

        try:
            kernel = self.kernel_factory()
            if not isinstance(kernel, AgentKernel):
                raise TypeError("kernel_factory must return an extracted AgentKernel")
            if not kernel.runtime_config.persist_events:
                raise ValueError("kernel must emit in-memory measurements; use no external reporting sink")
            kernel.set_event_context(run_id=identity.run_id, agent_name=self.agent_name)
            kernel.add_event_handler(events.append)
            kernel.add_llm_call_handler(llm_calls.append)
            kernel.add_tool_call_handler(tool_calls.append)
            raw = await kernel.ainvoke(payload)

            if not isinstance(raw, Mapping) or raw.get("ok") is not True or raw.get("error") is not None:
                code = raw.get("error") if isinstance(raw, Mapping) else None
                kind = FailureKind.VALIDATION if code == "final_output_invalid" else FailureKind.RUNTIME
                detail = raw.get("reason") if isinstance(raw, Mapping) else None
                failure = RunFailure(kind=kind, message=str(detail or code or "nonterminal_agent_output"))
            elif kernel.response_format is None and self.output_validator is None:
                failure = RunFailure(kind=FailureKind.VALIDATION, message="output_validator_required")
            else:
                try:
                    value = self.output_validator(raw) if self.output_validator is not None else raw
                    output = ValidatedOutput(value=value)
                except Exception as exc:
                    failure = RunFailure(kind=FailureKind.VALIDATION, message=str(exc) or "output_validation_failed")
        except TimeoutError as exc:
            failure = RunFailure(kind=FailureKind.TIMEOUT, message=str(exc) or "run_timeout_exceeded")
        except Exception as exc:
            kind = FailureKind.PROVIDER if llm_calls and llm_calls[-1].get("error_text") else FailureKind.RUNTIME
            failure = RunFailure(kind=kind, message=str(exc) or type(exc).__name__)

        result = CaseResult(
            identity=identity,
            status=RunStatus.COMPLETED if output is not None else RunStatus.FAILED,
            output=output,
            failure=failure,
            usage=_summarize_usage(llm_calls),
            timing=RunTiming(
                wall_seconds=time.perf_counter() - started,
                child_seconds_sum=_summed_child_seconds(llm_calls, tool_calls),
            ),
        )
        return SingleCaseExecution(
            result=result,
            events=tuple(events),
            llm_calls=tuple(llm_calls),
            tool_calls=tuple(tool_calls),
        )


def _summarize_usage(llm_calls: list[dict[str, Any]]) -> TokenUsage:
    if not llm_calls:
        return TokenUsage()
    sources = {item.get("usage_source") for item in llm_calls}
    source = UsageSource.PROVIDER if sources == {"provider"} else UsageSource.ESTIMATED
    return TokenUsage(
        source=source,
        input_tokens=sum(int(item.get("input_tokens") or 0) for item in llm_calls),
        output_tokens=sum(int(item.get("output_tokens") or 0) for item in llm_calls),
        total_tokens=sum(int(item.get("tokens_total") or 0) for item in llm_calls),
    )


def _summed_child_seconds(llm_calls: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> float | None:
    children = [*llm_calls, *tool_calls]
    if not children:
        return None
    return sum(float(item.get("latency_ms") or 0) for item in children) / 1000.0
