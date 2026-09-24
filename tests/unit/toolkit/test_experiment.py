"""Offline scheduling, pairing, failure retention, and preflight checks."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas_slm_research.configuration import load_configuration
from mas_slm_research.configured_systems import build_configured_systems
from mas_slm_research.contracts import (
    CaseResult, FailureKind, RunFailure, RunStatus, RunTiming, ValidatedOutput,
)
from mas_slm_research.dataset import DatasetError, load_configured_dataset
from mas_slm_research.experiment import ExperimentStatus, run_configured_experiment, run_experiment
from mas_slm_research.grading import GradeDecision
from mas_slm_research.registry import ComponentRegistry, register_builtin_components


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "esi" / "experiment.yaml"
ENV = {
    "BASELINE_MODEL_ID": "large-test", "BASELINE_API_KEY": "test-only",
    "BASELINE_AZURE_ENDPOINT": "https://azure.invalid",
    "BASELINE_AZURE_API_VERSION": "2024-02-01",
    "SPECIALIST_MODEL_ID": "small-test", "SPECIALIST_API_KEY": "test-only",
    "SPECIALIST_BASE_URL": "http://localhost:9999/v1",
}


class _Grader:
    def validate_expected(self, expected):
        assert isinstance(expected["acuity"], int)

    def evaluate(self, expected, actual):
        return GradeDecision(passed=expected["acuity"] == actual["final_esi_level"], score=float(expected["acuity"] == actual["final_esi_level"]))

    def aggregate(self, results):
        return {"attempted": len(results)}


class _Runner:
    def __init__(self, arm, calls, failed=()):
        self.arm, self.calls, self.failed = arm, calls, set(failed)

    async def run_case(self, *, identity, payload=None, case_info=None):
        facts = payload if self.arm == "single" else case_info
        self.calls.append((self.arm, identity.case_id, identity.repetition, facts))
        assert "acuity" not in facts and "expected" not in facts
        if (self.arm, identity.case_id, identity.repetition) in self.failed:
            result = CaseResult(identity=identity, status=RunStatus.FAILED,
                                failure=RunFailure(kind=FailureKind.PROVIDER, message="scripted provider failure"),
                                timing=RunTiming(wall_seconds=0.1))
        else:
            result = CaseResult(identity=identity, status=RunStatus.COMPLETED,
                                output=ValidatedOutput(value={"final_esi_level": 1}),
                                timing=RunTiming(wall_seconds=0.1))
        return SimpleNamespace(result=result, llm_calls=(), tool_calls=())


def test_failing_case_start_reporter_does_not_fail_execution() -> None:
    loaded, dataset, systems = _setup()
    calls = []
    systems = replace(systems, sas_runner=_Runner("single", calls), mas_runner=_Runner("multi", calls))

    def broken_start(*_args):
        raise RuntimeError("console closed")

    run = asyncio.run(run_experiment(
        loaded, dataset=dataset, systems=systems, grader=_Grader(),
        on_case_start=broken_start,
    ))
    assert all(attempt.result.status == RunStatus.COMPLETED for attempt in run.attempts)
    assert len(run.reporting_errors) == len(run.attempts)


def _setup():
    registry = ComponentRegistry()
    register_builtin_components(registry)
    loaded = load_configuration(EXAMPLE, registry=registry, environment=ENV)
    dataset = load_configured_dataset(loaded)
    systems = build_configured_systems(loaded, model_factory=lambda model, role: object())
    return loaded, dataset, systems


def test_system_major_pairs_all_repetitions_and_keeps_failed_arm() -> None:
    loaded, dataset, systems = _setup()
    loaded = replace(loaded, experiment=loaded.experiment.model_copy(update={"repetitions": 2}))
    calls = []
    systems = replace(systems,
        sas_runner=_Runner("single", calls, {("single", dataset.cases[0].case_id, 1)}),
        mas_runner=_Runner("multi", calls))
    run = asyncio.run(run_experiment(loaded, dataset=dataset, systems=systems,
                                     grader=_Grader(), experiment_id="offline"))
    assert run.status == ExperimentStatus.COMPLETED
    assert len(run.attempts) == len(dataset.cases) * 2 * 2
    assert len(run.pairs) == len(dataset.cases) * 2
    assert [item.system_id for item in run.attempts] == ["single"] * 6 + ["multi"] * 6
    assert all(pair.single and pair.multi for pair in run.pairs)
    assert run.pairs[0].single.grade.status.value == "execution_failed"
    assert run.pairs[0].multi.grade.status.value == "graded"
    assert [call[3] for call in calls[:6]] == [call[3] for call in calls[6:]]
    assert run.attempts[0].model_choices["baseline"]["model_id"] == "large-test"
    assert run.attempts[6].model_choices["esi1_agent"]["model_id"] == "small-test"
    assert all(item.started_at <= item.ended_at for item in run.attempts)


def test_case_major_and_explicit_cancellation_keep_partial_pair() -> None:
    loaded, dataset, systems = _setup()
    loaded = replace(loaded, experiment=loaded.experiment.model_copy(update={"schedule": "case_major"}))
    calls = []
    systems = replace(systems, sas_runner=_Runner("single", calls), mas_runner=_Runner("multi", calls))
    run = asyncio.run(run_experiment(
        loaded, dataset=dataset, systems=systems, grader=_Grader(),
        cancel_requested=lambda: len(calls) >= 1,
    ))
    assert run.status == ExperimentStatus.CANCELLED
    assert len(run.attempts) == 1
    assert run.pairs[0].single is not None and run.pairs[0].multi is None
    assert len(run.pairs) == len(dataset.cases)


def test_unexpected_runner_exception_still_records_a_failed_attempt() -> None:
    loaded, dataset, systems = _setup()
    class ExplodingRunner:
        async def run_case(self, **kwargs):
            raise RuntimeError("unexpected adapter error")
    calls = []
    systems = replace(systems, sas_runner=ExplodingRunner(), mas_runner=_Runner("multi", calls))
    run = asyncio.run(run_experiment(loaded, dataset=dataset, systems=systems, grader=_Grader()))
    assert len(run.attempts) == len(dataset.cases) * 2
    assert all(item.result.failure.message == "unexpected adapter error" for item in run.attempts[:3])
    assert all(item.result.status == RunStatus.COMPLETED for item in run.attempts[3:])


def test_dataset_error_occurs_before_model_construction(tmp_path: Path) -> None:
    loaded, _, _ = _setup()
    bad = tmp_path / "cases.jsonl"
    bad.write_text('{"case_id":"x","input":{},"expected":{"acuity":9}}\n', encoding="utf-8")
    loaded = replace(loaded, dataset_path=bad)
    made = []
    with pytest.raises(DatasetError):
        asyncio.run(run_configured_experiment(
            loaded, model_factory=lambda model, role: made.append(role),
        ))
    assert made == []


def test_inflight_cancellation_is_recorded_and_stops_scheduling() -> None:
    loaded, dataset, systems = _setup()
    class CancellingRunner:
        async def run_case(self, **kwargs):
            raise asyncio.CancelledError()
    systems = replace(systems, sas_runner=CancellingRunner())
    run = asyncio.run(run_experiment(loaded, dataset=dataset, systems=systems, grader=_Grader()))
    assert run.status == ExperimentStatus.CANCELLED
    assert len(run.attempts) == 1
    assert run.attempts[0].cancelled
    assert run.attempts[0].result.failure.message == "experiment_cancelled"


def test_direct_scheduler_validates_all_labels_before_first_call() -> None:
    loaded, dataset, systems = _setup()
    bad = replace(dataset.cases[-1], expected={"acuity": "invalid"})
    dataset = replace(dataset, cases=(*dataset.cases[:-1], bad))
    calls = []
    systems = replace(systems, sas_runner=_Runner("single", calls), mas_runner=_Runner("multi", calls))
    with pytest.raises(AssertionError):
        asyncio.run(run_experiment(loaded, dataset=dataset, systems=systems, grader=_Grader()))
    assert calls == []


def test_failing_reporting_observer_cannot_change_results() -> None:
    loaded, dataset, systems = _setup()
    calls = []
    systems = replace(systems, sas_runner=_Runner("single", calls), mas_runner=_Runner("multi", calls))
    def fails(attempt):
        raise RuntimeError("display unavailable")
    run = asyncio.run(run_experiment(loaded, dataset=dataset, systems=systems,
                                     grader=_Grader(), on_attempt=fails))
    assert run.status == ExperimentStatus.COMPLETED
    assert len(run.attempts) == len(dataset.cases) * 2
    assert all(item.result.status == RunStatus.COMPLETED for item in run.attempts)
    assert len(run.reporting_errors) == len(run.attempts)
