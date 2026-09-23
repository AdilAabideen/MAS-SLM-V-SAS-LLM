"""Evaluator module helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from mas_slm_research.evaluation.legacy_types import EvalResult


def _safe_div(num: int, denom: int) -> Optional[float]:
    """Handle div."""
    # Keep the main step clear.
    return None if denom == 0 else (num / denom)


def _round_or_none(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    """Handle or none."""
    # Keep the main step clear.
    return None if value is None else round(value, ndigits)


def _coerce_to_int(value: Any) -> Optional[int]:
    """Handle to int."""
    # Keep the main step clear.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


class SingleAgentAcuityEvaluator:
    """Exact-acuity evaluator for the single-agent system."""

    label_name = "single_agent_final_acuity"

    def validate_expected(self, expected_json: Dict[str, Any]) -> None:
        """Validate expected."""
        # Fail fast on bad input.
        if set(expected_json.keys()) != {"acuity"}:
            raise ValueError("expected_json must only contain: acuity")
        acuity = expected_json.get("acuity")
        if not isinstance(acuity, int):
            raise ValueError("expected_json.acuity must be an integer")
        if acuity < 1 or acuity > 5:
            raise ValueError("expected_json.acuity must be between 1 and 5")

    def _prediction(self, actual_json: Dict[str, Any]) -> Optional[int]:
        """Handle the value."""
        # Keep the main step clear.
        if "final_esi_level" in actual_json:
            return _coerce_to_int(actual_json.get("final_esi_level"))
        if "acuity" in actual_json:
            return _coerce_to_int(actual_json.get("acuity"))
        return None

    def evaluate(
        self,
        expected_json: Dict[str, Any],
        actual_json: Optional[Dict[str, Any]],
        *,
        agent_status: str,
    ) -> EvalResult:
        """Handle the value."""
        # Keep the main step clear.
        self.validate_expected(expected_json)
        expected_acuity = int(expected_json["acuity"])

        if agent_status != "succeeded" or actual_json is None:
            return EvalResult(
                passed=False,
                score=0.0,
                diff_json={
                    "error": "exec_failed_or_missing_output",
                    "expected_acuity": expected_acuity,
                    "agent_status": agent_status,
                },
                metrics_json={
                    "label": self.label_name,
                    "expected_acuity": expected_acuity,
                    "actual_acuity": None,
                    "exact_match": False,
                    "exec_failed": True,
                    "invalid_pred": False,
                },
            )

        actual_acuity = self._prediction(actual_json)
        if actual_acuity is None or actual_acuity < 1 or actual_acuity > 5:
            return EvalResult(
                passed=False,
                score=0.0,
                diff_json={
                    "error": "missing_or_invalid_prediction",
                    "expected_acuity": expected_acuity,
                    "actual_acuity": actual_acuity,
                },
                metrics_json={
                    "label": self.label_name,
                    "expected_acuity": expected_acuity,
                    "actual_acuity": actual_acuity,
                    "exact_match": False,
                    "exec_failed": False,
                    "invalid_pred": True,
                },
            )

        exact_match = actual_acuity == expected_acuity
        return EvalResult(
            passed=exact_match,
            score=1.0 if exact_match else 0.0,
            diff_json=(
                {}
                if exact_match
                else {
                    "expected_acuity": expected_acuity,
                    "actual_acuity": actual_acuity,
                }
            ),
            metrics_json={
                "label": self.label_name,
                "expected_acuity": expected_acuity,
                "actual_acuity": actual_acuity,
                "exact_match": exact_match,
                "exec_failed": False,
                "invalid_pred": False,
            },
        )

    def aggregate(self, results: Sequence[EvalResult]) -> Dict[str, Any]:
        """Handle the value."""
        # Keep the main step clear.
        total = 0
        passed = 0
        exec_failed = 0
        invalid_pred = 0

        for result in results:
            metrics = result.metrics_json or {}
            if metrics.get("exec_failed") is True:
                exec_failed += 1
            if metrics.get("invalid_pred") is True:
                invalid_pred += 1
            if metrics.get("expected_acuity") is not None:
                total += 1
            if result.passed:
                passed += 1

        failed = total - passed
        accuracy = _safe_div(passed, total)
        return {
            "label": self.label_name,
            "n_eval": total,
            "passed": passed,
            "failed": failed,
            "accuracy": _round_or_none(accuracy),
            "excluded": {
                "exec_failed": exec_failed,
                "invalid_pred": invalid_pred,
            },
        }
