"""Grader extension and ESI final-task scoring are independent of execution."""

from __future__ import annotations

import pytest

from mas_slm_research.contracts import (
    CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming, ValidatedOutput,
)
from mas_slm_research.evaluation.esi_final_acuity import ESIFinalAcuityGrader
from mas_slm_research.grading import (
    BaseGrader, GradeDecision, GradeStatus, grade_case,
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


def test_grader_distinguishes_outcomes_and_allows_custom_summary() -> None:
    grader = ToyGrader()
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
    assert grader.aggregate([correct, incorrect, execution, grader_error]) == {
        "attempts": 4, "correct": 1,
    }


def test_minimal_grader_uses_default_validation_and_summary() -> None:
    class MinimalGrader(BaseGrader):
        def evaluate(self, expected, actual):
            passed = actual.get("answer") == expected["answer"]
            return GradeDecision(passed=passed, score=float(passed))

    grader = MinimalGrader()
    correct = grade_case(grader, expected={"answer": "yes"}, result=_case({"answer": "yes"}))
    failed = grade_case(grader, expected={"answer": "yes"}, result=_case(failed=True))
    invalid_label = grade_case(grader, expected={}, result=_case({"answer": "yes"}))
    assert grader.aggregate([correct, failed, invalid_label]) == {
        "attempted": 3, "graded": 1, "passed": 1, "execution_failed": 1,
        "grader_errors": 1, "accuracy_all_attempts": 0.3333, "accuracy_graded": 1.0,
    }


@pytest.mark.parametrize(("actual", "passed", "predicted", "invalid"), [
    ({"final_esi_level": 2}, True, 2, False),
    ({"final_esi_level": 3}, False, 3, False),
    ({"final_esi_level": "2"}, True, 2, False),
    ({"acuity": 2}, True, 2, False),
    ({"final_esi_level": "invalid"}, False, None, True),
    ({"final_esi_level": 6}, False, 6, True),
    ({}, False, None, True),
])
def test_esi_grader_scores_final_acuity_for_either_system(actual, passed, predicted, invalid) -> None:
    grader = ESIFinalAcuityGrader()
    result = grade_case(grader, expected={"acuity": 2}, result=_case(actual, system="multi"))
    assert result.status == GradeStatus.GRADED
    assert result.passed is passed
    assert result.score == float(passed)
    assert result.diagnostics == {
        "expected_acuity": 2,
        "predicted_acuity": predicted,
        "invalid_prediction": invalid,
    }


def test_esi_final_score_uses_both_systems() -> None:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    grader = registry.resolve("graders", "esi.final_acuity_v1")
    grades = [
        grade_case(grader, expected={"acuity": 2}, result=_case({"final_esi_level": 2}, system="single")),
        grade_case(grader, expected={"acuity": 2}, result=_case({"final_esi_level": 3}, system="multi")),
        grade_case(grader, expected={"acuity": 2}, result=_case(failed=True, system="multi")),
    ]
    summary = grader.aggregate(grades)
    assert summary["attempted"] == 3
    assert summary["graded"] == 2
    assert summary["passed"] == 1
    assert summary["execution_failed"] == 1
    assert summary["accuracy_all_attempts"] == 0.3333
    assert summary["accuracy_graded"] == 0.5


def test_esi_invalid_label_is_not_a_wrong_prediction() -> None:
    result = grade_case(ESIFinalAcuityGrader(), expected={"acuity": 9}, result=_case({"final_esi_level": 2}))
    assert result.status == GradeStatus.GRADER_ERROR
    assert result.passed is None
    assert "expected-label validation" in result.error


def test_failed_multi_run_is_not_graded_as_a_correct_final_answer() -> None:
    final_grade = grade_case(
        ESIFinalAcuityGrader(), expected={"acuity": 1}, result=_case(failed=True, system="multi"),
    )
    assert final_grade.status == GradeStatus.EXECUTION_FAILED
    assert final_grade.passed is False
