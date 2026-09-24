"""Grader extension and ESI final-task scoring are independent of execution."""

from __future__ import annotations

import pytest

from mas_slm_research.evaluation.legacy_single_agent_acuity import SingleAgentAcuityEvaluator as LegacyESI
from mas_slm_research.evaluation.legacy_esi1 import ES1AcuityEvaluator
from mas_slm_research.evaluation.legacy_doctor import DoctorAlwaysPassEvaluator
from mas_slm_research.contracts import (
    CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming, ValidatedOutput,
)
from mas_slm_research.evaluation.esi_final_acuity import ESIFinalAcuityGrader, LEGACY_DOCTOR_DIAGNOSTIC
from mas_slm_research.evaluation.diagnostics import (
    aggregate_legacy_diagnostics, evaluate_legacy_diagnostic,
)
from mas_slm_research.grading import (
    BaseGrader, GradeDecision, GradeStatus, aggregate_grades, grade_case, require_grader,
)
from mas_slm_research.registry import ComponentRegistry, register_builtin_components


def _case(value=None, *, failed=False, system="single") -> CaseResult:
    identity = RunIdentity(
        experiment_id="toy", system_id=system, case_id="c1", repetition=1,
        run_id=f"{system}-{'failed' if failed else 'complete'}",
    )
    if failed:
        return CaseResult(
            identity=identity, status=RunStatus.FAILED,
            failure=RunFailure(kind=FailureKind.PROVIDER, message="provider unavailable"),
            timing=RunTiming(wall_seconds=0.1),
        )
    return CaseResult(
        identity=identity, status=RunStatus.COMPLETED,
        output=ValidatedOutput(value=value or {}), timing=RunTiming(wall_seconds=0.1),
    )


class ToyGrader(BaseGrader):
    def validate_expected(self, expected):
        if set(expected) != {"answer"}:
            raise ValueError("answer label required")

    def evaluate(self, expected, actual):
        if actual.get("explode"):
            raise RuntimeError("grader crashed")
        passed = actual.get("answer") == expected["answer"]
        return GradeDecision(passed=passed, score=float(passed), diagnostics={"toy": True})

    def aggregate(self, results):
        return {"attempts": len(results), "correct": sum(item.passed is True for item in results)}


def test_external_subclass_and_structural_grader_distinguish_outcomes() -> None:
    grader = require_grader(ToyGrader)
    correct = grade_case(grader, expected={"answer": "yes"}, result=_case({"answer": "yes"}))
    incorrect = grade_case(grader, expected={"answer": "yes"}, result=_case({"answer": "no"}))
    missing = grade_case(grader, expected={"answer": "yes"}, result=_case({}))
    execution = grade_case(grader, expected={"answer": "yes"}, result=_case(failed=True))
    grader_error = grade_case(grader, expected={"answer": "yes"}, result=_case({"explode": True}))
    label_error = grade_case(grader, expected={"wrong": 1}, result=_case({"answer": "yes"}))

    assert [item.status for item in (correct, incorrect, missing)] == [GradeStatus.GRADED] * 3
    assert [item.passed for item in (correct, incorrect, missing)] == [True, False, False]
    assert execution.status == GradeStatus.EXECUTION_FAILED
    assert execution.execution_failure_kind == FailureKind.PROVIDER
    assert execution.score == 0
    assert grader_error.status == GradeStatus.GRADER_ERROR
    assert grader_error.score is None
    assert label_error.status == GradeStatus.GRADER_ERROR
    assert aggregate_grades(grader, [correct, incorrect, execution, grader_error]) == {
        "attempts": 4, "correct": 1,
    }

    class Structural:
        validate_expected = ToyGrader.validate_expected
        evaluate = ToyGrader.evaluate
        aggregate = ToyGrader.aggregate

    assert grade_case(require_grader(Structural()), expected={"answer": "x"}, result=_case({"answer": "x"})).passed
    class Incomplete:
        validate_expected = ToyGrader.validate_expected
        evaluate = ToyGrader.evaluate

    with pytest.raises(TypeError, match="aggregate"):
        require_grader(Incomplete())


@pytest.mark.parametrize("actual", [
    {"final_esi_level": 2}, {"final_esi_level": 3},
    {"final_esi_level": "invalid"}, {},
])
def test_esi_grader_matches_preserved_final_acuity_evaluator(actual) -> None:
    grader = ESIFinalAcuityGrader()
    legacy = LegacyESI().evaluate({"acuity": 2}, actual, agent_status="succeeded")
    result = grade_case(grader, expected={"acuity": 2}, result=_case(actual, system="multi"))
    assert result.status == GradeStatus.GRADED
    assert result.passed == legacy.passed
    assert result.score == legacy.score
    assert result.diagnostics["legacy_diff"] == legacy.diff_json
    assert result.diagnostics["legacy_metrics"] == legacy.metrics_json


def test_esi_final_score_uses_both_systems_and_labels_doctor_placeholder() -> None:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    grader = require_grader(registry.resolve("graders", "esi.final_acuity_v1"))
    grades = [
        grade_case(grader, expected={"acuity": 2}, result=_case({"final_esi_level": 2}, system="single")),
        grade_case(grader, expected={"acuity": 2}, result=_case({"final_esi_level": 3}, system="multi")),
        grade_case(grader, expected={"acuity": 2}, result=_case(failed=True, system="multi")),
    ]
    summary = aggregate_grades(grader, grades)
    assert summary["attempted"] == 3
    assert summary["graded"] == 2
    assert summary["passed"] == 1
    assert summary["execution_failed"] == 1
    assert summary["accuracy_all_attempts"] == 0.3333
    assert summary["accuracy_graded"] == 0.5
    assert summary["doctor_diagnostic"] == LEGACY_DOCTOR_DIAGNOSTIC
    assert summary["doctor_diagnostic"]["headline_final_task_grader"] is False


def test_esi_invalid_label_is_not_a_wrong_prediction() -> None:
    result = grade_case(ESIFinalAcuityGrader(), expected={"acuity": 9}, result=_case({"final_esi_level": 2}))
    assert result.status == GradeStatus.GRADER_ERROR
    assert result.passed is None
    assert "expected-label validation" in result.error


def test_legacy_specialist_and_doctor_diagnostics_stay_outside_headline_grade() -> None:
    specialist = ES1AcuityEvaluator()
    diagnostic = evaluate_legacy_diagnostic(
        specialist, agent_name="esi1_agent", expected={"acuity": 1},
        actual={"is_esi1": True}, agent_status="succeeded",
    )
    assert diagnostic.passed is True
    assert diagnostic.placeholder is False
    assert aggregate_legacy_diagnostics(specialist, [diagnostic])["diagnostic_only"] is True

    doctor = DoctorAlwaysPassEvaluator()
    placeholder = evaluate_legacy_diagnostic(
        doctor, agent_name="doctor_agent", expected={}, actual=None, agent_status="failed",
    )
    assert placeholder.passed is True  # Preserved legacy behavior, not task accuracy.
    assert placeholder.placeholder is True
    assert aggregate_legacy_diagnostics(doctor, [placeholder])["placeholder"] is True

    final_grade = grade_case(
        ESIFinalAcuityGrader(), expected={"acuity": 1}, result=_case(failed=True, system="multi"),
    )
    assert final_grade.status == GradeStatus.EXECUTION_FAILED
    assert final_grade.passed is False
