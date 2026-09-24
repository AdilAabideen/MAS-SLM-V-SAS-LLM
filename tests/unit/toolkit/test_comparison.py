"""Hand-calculated comparison denominators, measurements, and unknowns."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mas_slm_research.comparison import PriceRate, compare_experiment, configured_prices
from mas_slm_research.configuration import load_configuration
from mas_slm_research.contracts import (
    CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming, ValidatedOutput,
)
from mas_slm_research.experiment import (
    ExperimentAttempt, ExperimentPair, ExperimentRun, ExperimentStatus,
)
from mas_slm_research.grading import BaseGrader, GradeResult, GradeStatus
from mas_slm_research.registry import ComponentRegistry, register_builtin_components


def _attempt(case_id, arm, sequence, score, *, failure=False, calls=(), tools=(), events=(), wall=0.5, child=1.2):
    identity = RunIdentity(experiment_id="fixture", system_id=arm, case_id=case_id,
                           repetition=1, run_id=f"{case_id}-{arm}")
    if failure:
        result = CaseResult(identity=identity, status=RunStatus.FAILED,
                            failure=RunFailure(kind=FailureKind.PROVIDER, message="provider unavailable"),
                            timing=RunTiming(wall_seconds=wall, child_seconds_sum=child))
        grade = GradeResult(identity=identity, status=GradeStatus.EXECUTION_FAILED,
                            passed=False, score=0.0, error="provider unavailable",
                            execution_failure_kind=FailureKind.PROVIDER)
    else:
        result = CaseResult(identity=identity, status=RunStatus.COMPLETED,
                            output=ValidatedOutput(value={"answer": score}),
                            timing=RunTiming(wall_seconds=wall, child_seconds_sum=child))
        grade = GradeResult(identity=identity, status=GradeStatus.GRADED,
                            passed=score == 1.0, score=score)
    return ExperimentAttempt(
        sequence=sequence, case_id=case_id, repetition=1, system_id=arm,
        started_at="2026-09-23T00:00:00+00:00", ended_at="2026-09-23T00:00:01+00:00",
        model_choices={}, runtime_policies={},
        execution=SimpleNamespace(llm_calls=calls, tool_calls=tools, events=events),
        result=result, grade=grade,
    )


def _call(role="agent", *, input_tokens=100, output_tokens=20, source="provider", repair=0):
    return {"agent_name": role, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "tokens_total": None if input_tokens is None or output_tokens is None else input_tokens + output_tokens,
            "usage_source": source, "text_recovered_tool_call_count": repair}


def test_hand_computed_totals_keep_failures_unknown_usage_and_child_time_separate() -> None:
    sas = [
        _attempt("a", "single", 1, 1.0, calls=(_call(repair=1),), tools=({},)),
        _attempt("b", "single", 2, 0.0, calls=(_call(source="estimated"),)),
        _attempt("c", "single", 3, 0.0, failure=True, calls=(_call(input_tokens=None),)),
        _attempt("d", "single", 4, 1.0),
    ]
    mas = [
        _attempt("a", "multi", 5, 0.0, calls=(_call(),), events=({"event_type": "runtime_decision", "payload_json": {"decision": "retry_after_malformed_tool_call"}},)),
        _attempt("b", "multi", 6, 1.0, calls=(_call(),)),
        _attempt("c", "multi", 7, 1.0, calls=(_call(),)),
        _attempt("d", "multi", 8, 1.0, calls=(_call(),)),
    ]
    run = ExperimentRun(
        experiment_id="fixture", status=ExperimentStatus.COMPLETED, schedule="system_major",
        dataset_path="synthetic.jsonl", started_at="start", ended_at="end",
        attempts=tuple(sas + mas),
        pairs=tuple(ExperimentPair(case_id=case_id, repetition=1, single=sas[index], multi=mas[index])
                    for index, case_id in enumerate("abcd")),
    )
    report = compare_experiment(run, prices_by_role={"agent": PriceRate(input_per_1k=0.001, output_per_1k=0.002)})
    single = report.systems["single"]
    multi = report.systems["multi"]
    assert (single.attempted, single.valid_completions, single.graded, single.passed) == (4, 3, 3, 2)
    assert (single.task_score_all_attempts, single.task_score_graded_only, single.graded_only_denominator) == (0.5, 2 / 3, 3)
    assert single.failure_reasons == {"provider": 1}
    assert (single.llm_calls, single.tool_calls, single.text_recovered_tool_calls) == (3, 1, 1)
    assert single.usage_sources == {"provider": 2, "estimated": 1}
    assert single.input_tokens is None and single.cost_usd_estimate is None
    assert single.workflow_wall_seconds == 2.0
    assert report.attempt_measurements[0].child_seconds_sum == 1.2
    assert multi.cost_usd_estimate == 4 * (100 * 0.001 + 20 * 0.002) / 1000
    assert multi.malformed_retry_decisions == 1
    assert (report.single_wins, report.multi_wins, report.ties, report.uncomparable_pairs) == (1, 1, 1, 1)
    assert report.pairs[2].single_run_id == "c-single" and report.pairs[2].winner is None
    assert len(report.to_dict()["attempt_measurements"]) == len(run.attempts)


def test_missing_price_is_unknown_not_free() -> None:
    attempt = _attempt("a", "single", 1, 1.0, calls=(_call(),))
    run = ExperimentRun(
        experiment_id="fixture", status=ExperimentStatus.CANCELLED, schedule="case_major",
        dataset_path="synthetic.jsonl", started_at="start", ended_at="end",
        attempts=(attempt,), pairs=(ExperimentPair(case_id="a", repetition=1, single=attempt, multi=None),),
    )
    report = compare_experiment(run)
    assert report.systems["single"].cost_usd_estimate is None
    assert report.systems["multi"].attempted == 0
    assert report.systems["multi"].task_score_all_attempts is None
    assert report.uncomparable_pairs == 1


def test_only_explicit_catalog_pricing_is_used() -> None:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    example = Path(__file__).resolve().parents[3] / "examples" / "esi" / "experiment.yaml"
    loaded = load_configuration(example, registry=registry, environment={
        "BASELINE_MODEL_ID": "gpt-4o", "BASELINE_API_KEY": "test-only",
        "BASELINE_AZURE_ENDPOINT": "https://azure.invalid", "BASELINE_AZURE_API_VERSION": "2024-02-01",
        "SPECIALIST_MODEL_ID": "medgemma-4b-it", "SPECIALIST_API_KEY": "test-only",
        "SPECIALIST_BASE_URL": "http://localhost:9999/v1",
    })
    assert configured_prices(loaded) == {}


def test_grader_summary_receives_failures_and_inconsistent_pairs_are_rejected() -> None:
    attempt = _attempt("a", "single", 1, 0.0, failure=True)
    run = ExperimentRun(
        experiment_id="fixture", status=ExperimentStatus.CANCELLED, schedule="case_major",
        dataset_path="synthetic.jsonl", started_at="start", ended_at="end",
        attempts=(attempt,), pairs=(ExperimentPair(case_id="a", repetition=1, single=attempt, multi=None),),
    )
    class Grader(BaseGrader):
        def validate_expected(self, expected): pass
        def evaluate(self, expected, actual): raise AssertionError("not used")
        def aggregate(self, results): return {"statuses": [item.status.value for item in results]}
    assert compare_experiment(run, grader=Grader()).systems["single"].grader_summary == {
        "statuses": ["execution_failed"]}
    with pytest.raises(ValueError, match="omit"):
        compare_experiment(replace(run, pairs=(ExperimentPair(case_id="a", repetition=1, single=None, multi=None),)))
