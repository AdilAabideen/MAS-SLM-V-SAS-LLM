"""Derive paired outcomes and auditable totals from recorded attempts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .configuration import LoadedConfiguration
from .contracts import RunStatus
from .experiment import ExperimentAttempt, ExperimentRun
from .grading import GradeStatus, GraderLike, aggregate_grades, require_grader


@dataclass(frozen=True, kw_only=True)
class PriceRate:
    """Explicit research estimate in USD per 1,000 input/output tokens."""

    input_per_1k: float
    output_per_1k: float

    def __post_init__(self) -> None:
        for name in ("input_per_1k", "output_per_1k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative price")


@dataclass(frozen=True, kw_only=True)
class AttemptMeasurements:
    run_id: str
    llm_calls: int
    tool_calls: int
    text_recovered_tool_calls: int
    malformed_retry_decisions: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_sources: Mapping[str, int]
    cost_usd_estimate: float | None
    workflow_wall_seconds: float
    child_seconds_sum: float | None
    network_attempts: int = 0


@dataclass(frozen=True, kw_only=True)
class SystemSummary:
    system_id: str
    attempted: int
    valid_completions: int
    graded: int
    passed: int
    grader_errors: int
    task_score_all_attempts: float | None
    task_score_graded_only: float | None
    graded_only_denominator: int
    failure_reasons: Mapping[str, int]
    llm_calls: int
    tool_calls: int
    text_recovered_tool_calls: int
    malformed_retry_decisions: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_sources: Mapping[str, int]
    cost_usd_estimate: float | None
    workflow_wall_seconds: float
    grader_summary: Mapping[str, Any] | None
    network_attempts: int = 0


@dataclass(frozen=True, kw_only=True)
class PairedOutcome:
    case_id: str
    repetition: int
    single_run_id: str | None
    multi_run_id: str | None
    single_score: float | None
    multi_score: float | None
    winner: str | None


@dataclass(frozen=True, kw_only=True)
class ComparisonReport:
    experiment_id: str
    experiment_status: str
    attempt_measurements: tuple[AttemptMeasurements, ...]
    systems: Mapping[str, SystemSummary]
    pairs: tuple[PairedOutcome, ...]
    single_wins: int
    multi_wins: int
    ties: int
    uncomparable_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configured_prices(loaded: LoadedConfiguration) -> dict[str, PriceRate]:
    """Use only explicitly selected model-catalog prices, never model-name guesses."""
    spec = loaded.experiment
    selected = {spec.sas.agent: loaded.models[spec.sas.model]}
    for role, alias in spec.mas.agents.items():
        model_name = spec.mas.model_overrides.get(role) or spec.agents[alias].model or spec.mas.default_model
        selected[role] = loaded.models[model_name]
    prices: dict[str, PriceRate] = {}
    for role, model in selected.items():
        if model.catalog is None:
            continue
        catalog = loaded.registry.resolve("models", model.catalog)
        pricing = getattr(catalog, "pricing", None)
        if (pricing is not None and getattr(pricing, "input_per_1k", None) is not None
                and getattr(pricing, "output_per_1k", None) is not None):
            prices[role] = PriceRate(
                input_per_1k=pricing.input_per_1k, output_per_1k=pricing.output_per_1k,
            )
    return prices


def _sum_known(values: list[int | None]) -> int | None:
    return sum(values) if all(value is not None for value in values) else None


def _sum_known_cost(values: list[float | None]) -> float | None:
    return sum(values) if all(value is not None for value in values) else None


def _measure(attempt: ExperimentAttempt, prices: Mapping[str, PriceRate]) -> AttemptMeasurements:
    execution = attempt.execution
    llm_calls = tuple(execution.llm_calls) if execution is not None else ()
    tool_calls = tuple(execution.tool_calls) if execution is not None else ()
    sources: dict[str, int] = {}
    costs: list[float | None] = []
    for call in llm_calls:
        source = str(call.get("usage_source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        role = str(call.get("agent_name") or "")
        rate = prices.get(role)
        input_tokens, output_tokens = call.get("input_tokens"), call.get("output_tokens")
        if (rate is None or int(call.get("network_attempts") or 1) > 1
                or not isinstance(input_tokens, int) or isinstance(input_tokens, bool)
                or not isinstance(output_tokens, int) or isinstance(output_tokens, bool)):
            costs.append(None)
        else:
            costs.append((input_tokens * rate.input_per_1k + output_tokens * rate.output_per_1k) / 1000)
    events = tuple(execution.events) if execution is not None else ()
    return AttemptMeasurements(
        run_id=attempt.result.identity.run_id,
        llm_calls=len(llm_calls), tool_calls=len(tool_calls),
        text_recovered_tool_calls=sum(int(call.get("text_recovered_tool_call_count") or 0) for call in llm_calls),
        malformed_retry_decisions=sum(
            event.get("event_type") == "runtime_decision"
            and isinstance(event.get("payload_json"), Mapping)
            and event["payload_json"].get("decision") == "retry_after_malformed_tool_call"
            for event in events
        ),
        input_tokens=None if any(int(call.get("network_attempts") or 1) > 1 for call in llm_calls) else _sum_known([call.get("input_tokens") for call in llm_calls]),
        output_tokens=None if any(int(call.get("network_attempts") or 1) > 1 for call in llm_calls) else _sum_known([call.get("output_tokens") for call in llm_calls]),
        total_tokens=None if any(int(call.get("network_attempts") or 1) > 1 for call in llm_calls) else _sum_known([call.get("tokens_total") for call in llm_calls]),
        usage_sources=sources,
        cost_usd_estimate=_sum_known_cost(costs),
        workflow_wall_seconds=attempt.result.timing.wall_seconds,
        child_seconds_sum=attempt.result.timing.child_seconds_sum,
        network_attempts=sum(int(call.get("network_attempts") or 1) for call in llm_calls),
    )


def _validate_records(run: ExperimentRun) -> None:
    """Reject reports whose pair view disagrees with the authoritative attempts."""
    by_key: dict[tuple[str, int, str], ExperimentAttempt] = {}
    for sequence, attempt in enumerate(run.attempts, 1):
        if attempt.sequence != sequence:
            raise ValueError("experiment attempt sequence is not contiguous")
        if attempt.result.identity.system_id != attempt.system_id or attempt.grade.identity != attempt.result.identity:
            raise ValueError("experiment attempt result/grade identity mismatch")
        key = (attempt.case_id, attempt.repetition, attempt.system_id)
        if attempt.system_id not in ("single", "multi") or key in by_key:
            raise ValueError(f"invalid or duplicate experiment attempt {key}")
        by_key[key] = attempt
    paired_keys: set[tuple[str, int, str]] = set()
    for pair in run.pairs:
        for arm, attempt in (("single", pair.single), ("multi", pair.multi)):
            if attempt is None:
                continue
            key = (pair.case_id, pair.repetition, arm)
            if key in paired_keys or by_key.get(key) is not attempt:
                raise ValueError(f"pair does not match experiment attempt {key}")
            paired_keys.add(key)
    if paired_keys != set(by_key):
        raise ValueError("paired outcomes omit experiment attempts")


def compare_experiment(
    run: ExperimentRun, *, grader: GraderLike | None = None,
    prices_by_role: Mapping[str, PriceRate] | None = None,
    recorded_grader_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> ComparisonReport:
    """Compute all denominators from case attempts; never drop a failed pair."""
    _validate_records(run)
    grader = require_grader(grader) if grader is not None else None
    prices = prices_by_role or {}
    measurements = tuple(_measure(attempt, prices) for attempt in run.attempts)
    summaries: dict[str, SystemSummary] = {}
    for arm in ("single", "multi"):
        selected = [(attempt, metric) for attempt, metric in zip(run.attempts, measurements)
                    if attempt.system_id == arm]
        attempts = [attempt for attempt, _ in selected]
        metrics = [metric for _, metric in selected]
        attempted = len(attempts)
        graded = [attempt.grade for attempt in attempts if attempt.grade.status == GradeStatus.GRADED]
        failure_reasons: dict[str, int] = {}
        usage_sources: dict[str, int] = {}
        for attempt, metric in selected:
            if attempt.result.status == RunStatus.FAILED:
                assert attempt.result.failure is not None
                key = attempt.result.failure.kind.value
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
            for source, count in metric.usage_sources.items():
                usage_sources[source] = usage_sources.get(source, 0) + count
        scores = [grade.score or 0.0 for grade in (attempt.grade for attempt in attempts)]
        summaries[arm] = SystemSummary(
            system_id=arm, attempted=attempted,
            valid_completions=sum(attempt.result.status == RunStatus.COMPLETED for attempt in attempts),
            graded=len(graded), passed=sum(grade.passed is True for grade in graded),
            grader_errors=sum(attempt.grade.status == GradeStatus.GRADER_ERROR for attempt in attempts),
            task_score_all_attempts=sum(scores) / attempted if attempted else None,
            task_score_graded_only=sum(grade.score or 0.0 for grade in graded) / len(graded) if graded else None,
            graded_only_denominator=len(graded), failure_reasons=failure_reasons,
            llm_calls=sum(item.llm_calls for item in metrics),
            tool_calls=sum(item.tool_calls for item in metrics),
            text_recovered_tool_calls=sum(item.text_recovered_tool_calls for item in metrics),
            malformed_retry_decisions=sum(item.malformed_retry_decisions for item in metrics),
            input_tokens=_sum_known([item.input_tokens for item in metrics]),
            output_tokens=_sum_known([item.output_tokens for item in metrics]),
            total_tokens=_sum_known([item.total_tokens for item in metrics]),
            usage_sources=usage_sources,
            cost_usd_estimate=_sum_known_cost([item.cost_usd_estimate for item in metrics]),
            workflow_wall_seconds=sum(item.workflow_wall_seconds for item in metrics),
            grader_summary=(aggregate_grades(grader, [attempt.grade for attempt in attempts]) if grader
                            else dict(recorded_grader_summaries[arm])
                            if recorded_grader_summaries and arm in recorded_grader_summaries else None),
            network_attempts=sum(item.network_attempts for item in metrics),
        )
    pairs: list[PairedOutcome] = []
    for pair in run.pairs:
        single = pair.single
        multi = pair.multi
        single_score = single.grade.score if single and single.grade.status == GradeStatus.GRADED else None
        multi_score = multi.grade.score if multi and multi.grade.status == GradeStatus.GRADED else None
        winner = None
        if single_score is not None and multi_score is not None:
            winner = "single" if single_score > multi_score else "multi" if multi_score > single_score else "tie"
        pairs.append(PairedOutcome(
            case_id=pair.case_id, repetition=pair.repetition,
            single_run_id=single.result.identity.run_id if single else None,
            multi_run_id=multi.result.identity.run_id if multi else None,
            single_score=single_score, multi_score=multi_score, winner=winner,
        ))
    return ComparisonReport(
        experiment_id=run.experiment_id, experiment_status=run.status.value,
        attempt_measurements=measurements, systems=summaries, pairs=tuple(pairs),
        single_wins=sum(pair.winner == "single" for pair in pairs),
        multi_wins=sum(pair.winner == "multi" for pair in pairs),
        ties=sum(pair.winner == "tie" for pair in pairs),
        uncomparable_pairs=sum(pair.winner is None for pair in pairs),
    )
