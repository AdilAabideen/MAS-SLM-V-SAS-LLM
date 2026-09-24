"""Shared final-ESI grader for both SAS and MAS research arms."""

from __future__ import annotations

from typing import Any, Mapping

from mas_slm_research.grading import BaseGrader, GradeDecision


def _acuity(value: Any) -> int | None:
    """Accept the integer-like final levels supported by the original grader."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


class ESIFinalAcuityGrader(BaseGrader):
    """Score either system by comparing its final ESI level with the case label."""

    label_name = "esi.final_acuity_v1"

    def validate_expected(self, expected: Mapping[str, Any]) -> None:
        if set(expected) != {"acuity"}:
            raise ValueError("expected must only contain: acuity")
        acuity = expected["acuity"]
        if not isinstance(acuity, int) or not 1 <= acuity <= 5:
            raise ValueError("expected.acuity must be an integer between 1 and 5")

    def evaluate(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> GradeDecision:
        value = actual.get("final_esi_level", actual.get("acuity"))
        predicted = _acuity(value)
        valid = predicted is not None and 1 <= predicted <= 5
        passed = valid and predicted == expected["acuity"]
        return GradeDecision(
            passed=passed,
            score=1.0 if passed else 0.0,
            diagnostics={
                "expected_acuity": expected["acuity"],
                "predicted_acuity": predicted,
                "invalid_prediction": not valid,
            },
        )
