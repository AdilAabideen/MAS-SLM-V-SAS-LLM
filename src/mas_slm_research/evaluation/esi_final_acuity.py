"""Shared final-ESI grader for both SAS and MAS research arms."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mas_slm_research.grading import BaseGrader, GradeDecision, GradeResult, GradeStatus

from .legacy_single_agent_acuity import SingleAgentAcuityEvaluator


LEGACY_DOCTOR_DIAGNOSTIC = {
    "id": "doctor_always_pass",
    "status": "placeholder_diagnostic_only",
    "headline_final_task_grader": False,
}


class ESIFinalAcuityGrader(BaseGrader):
    """Apply the preserved exact-acuity rule to either system's final output.

    The legacy doctor always-pass evaluator is a placeholder diagnostic and
    must never be used as the comparison's final-task accuracy.
    """

    label_name = "esi.final_acuity_v1"

    def __init__(self) -> None:
        self._legacy = SingleAgentAcuityEvaluator()

    def validate_expected(self, expected: Mapping[str, Any]) -> None:
        self._legacy.validate_expected(dict(expected))

    def evaluate(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> GradeDecision:
        verdict = self._legacy.evaluate(dict(expected), dict(actual), agent_status="succeeded")
        return GradeDecision(
            passed=verdict.passed, score=verdict.score,
            diagnostics={"legacy_diff": verdict.diff_json, "legacy_metrics": verdict.metrics_json},
        )

    def aggregate(self, results: Sequence[GradeResult]) -> Mapping[str, Any]:
        records = tuple(results)
        attempted = len(records)
        passed = sum(item.passed is True for item in records)
        graded = sum(item.status == GradeStatus.GRADED for item in records)
        execution_failed = sum(item.status == GradeStatus.EXECUTION_FAILED for item in records)
        grader_errors = sum(item.status == GradeStatus.GRADER_ERROR for item in records)
        return {
            "label": self.label_name,
            "attempted": attempted,
            "graded": graded,
            "passed": passed,
            "execution_failed": execution_failed,
            "grader_errors": grader_errors,
            "accuracy_all_attempts": round(passed / attempted, 4) if attempted else None,
            "accuracy_graded": round(passed / graded, 4) if graded else None,
            "doctor_diagnostic": dict(LEGACY_DOCTOR_DIAGNOSTIC),
        }
