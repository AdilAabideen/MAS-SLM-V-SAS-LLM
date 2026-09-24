"""Evaluator module helpers."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence

from .legacy_types import EvalResult


Verdict = Literal["pass", "warning", "fail"]


def _safe_div(num: int, denom: int) -> Optional[float]:
    """Handle div."""
    # Keep the main step clear.
    return None if denom == 0 else (num / denom)


def _round_or_none(v: Optional[float], ndigits: int = 4) -> Optional[float]:
    """Handle or none."""
    # Keep the main step clear.
    return None if v is None else round(v, ndigits)


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


class ESI345AcuityEvaluator:
    """
    ESI-3/4/5 evaluator with acuity + resource-use grading.

    Verdicts:
    - pass: acuity and resources both match
    - warning: acuity matches but resources mismatch
    - fail: acuity mismatch, invalid prediction, or execution failure
    """

    label_name = "esi345_acuity_resources"

    def validate_expected(self, expected_json: Dict[str, Any]) -> None:
        """Validate expected."""
        # Fail fast on bad input.
        if set(expected_json.keys()) != {"acuity", "resources_used"}:
            raise ValueError("expected_json must only contain: acuity, resources_used")
        acuity = expected_json.get("acuity")
        resources_used = expected_json.get("resources_used")
        if not isinstance(acuity, int):
            raise ValueError("expected_json.acuity must be an integer")
        if not isinstance(resources_used, int):
            raise ValueError("expected_json.resources_used must be an integer")
        if acuity < 1 or acuity > 5:
            raise ValueError("expected_json.acuity must be between 1 and 5")
        if resources_used < 0:
            raise ValueError("expected_json.resources_used must be >= 0")

    def _targets(self, expected_json: Dict[str, Any]) -> tuple[int, int]:
        """Handle the value."""
        # Keep the main step clear.
        self.validate_expected(expected_json)
        return int(expected_json["acuity"]), int(expected_json["resources_used"])

    def _prediction(self, actual_json: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
        """Handle the value."""
        # Keep the main step clear.
        acuity_pred = _coerce_to_int(actual_json.get("esi_level"))
        resources_pred = _coerce_to_int(actual_json.get("num_resources"))
        if resources_pred is not None and resources_pred < 0:
            resources_pred = None
        return acuity_pred, resources_pred

    def _build_result(
        self,
        *,
        verdict: Verdict,
        score: float,
        expected_acuity: int,
        expected_resources: int,
        actual_acuity: Optional[int],
        actual_resources: Optional[int],
        exec_failed: bool,
        invalid_pred: bool,
        agent_status: str,
        error: Optional[str] = None,
    ) -> EvalResult:
        """Build result."""
        # Build the next value.
        acuity_match = actual_acuity is not None and (actual_acuity == expected_acuity)
        resources_match = actual_resources is not None and (actual_resources == expected_resources)
        warning = verdict == "warning"
        passed = verdict in {"pass", "warning"}

        diff_json: Dict[str, Any] = {
            "verdict": verdict,
            "warning": warning,
            "expected_acuity": expected_acuity,
            "expected_resources_used": expected_resources,
            "actual_acuity": actual_acuity,
            "actual_resources_used": actual_resources,
            "acuity_match": acuity_match,
            "resources_match": resources_match,
            "agent_status": agent_status,
        }
        if error:
            diff_json["error"] = error

        metrics_json = {
            "label": self.label_name,
            "evaluation_status": verdict,
            "warning": warning,
            "expected_acuity": expected_acuity,
            "expected_resources_used": expected_resources,
            "actual_acuity": actual_acuity,
            "actual_resources_used": actual_resources,
            "acuity_match": acuity_match,
            "resources_match": resources_match,
            "exec_failed": exec_failed,
            "invalid_pred": invalid_pred,
        }
        return EvalResult(
            passed=passed,
            score=score,
            diff_json=diff_json,
            metrics_json=metrics_json,
        )

    def evaluate(
        self,
        expected_json: Dict[str, Any],
        actual_json: Optional[Dict[str, Any]],
        *,
        agent_status: str,
    ) -> EvalResult:
        """Handle the value."""
        # Keep the main step clear.
        expected_acuity, expected_resources = self._targets(expected_json)

        if agent_status != "succeeded" or actual_json is None:
            return self._build_result(
                verdict="fail",
                score=0.0,
                expected_acuity=expected_acuity,
                expected_resources=expected_resources,
                actual_acuity=None,
                actual_resources=None,
                exec_failed=True,
                invalid_pred=False,
                agent_status=agent_status,
                error="exec_failed_or_missing_output",
            )

        actual_acuity, actual_resources = self._prediction(actual_json)
        if actual_acuity is None or actual_resources is None:
            return self._build_result(
                verdict="fail",
                score=0.0,
                expected_acuity=expected_acuity,
                expected_resources=expected_resources,
                actual_acuity=actual_acuity,
                actual_resources=actual_resources,
                exec_failed=False,
                invalid_pred=True,
                agent_status=agent_status,
                error="missing_or_invalid_prediction",
            )

        acuity_match = actual_acuity == expected_acuity
        resources_match = actual_resources == expected_resources

        if acuity_match and resources_match:
            verdict: Verdict = "pass"
            score = 1.0
        elif acuity_match and (not resources_match):
            verdict = "warning"
            score = 0.5
        else:
            verdict = "fail"
            score = 0.0

        return self._build_result(
            verdict=verdict,
            score=score,
            expected_acuity=expected_acuity,
            expected_resources=expected_resources,
            actual_acuity=actual_acuity,
            actual_resources=actual_resources,
            exec_failed=False,
            invalid_pred=False,
            agent_status=agent_status,
        )

    def aggregate(self, results: Sequence[EvalResult]) -> Dict[str, Any]:
        """Handle the value."""
        # Keep the main step clear.
        pass_count = 0
        warning_count = 0
        fail_count = 0
        exec_failed = 0
        invalid_pred = 0
        acuity_correct = 0
        resources_correct = 0
        hard_fail_count = 0
        soft_pass_count = 0

        for r in results:
            m = r.metrics_json or {}
            status = str(m.get("evaluation_status") or "")
            if status == "pass":
                pass_count += 1
                soft_pass_count += 1
            elif status == "warning":
                warning_count += 1
                soft_pass_count += 1
            else:
                fail_count += 1
                hard_fail_count += 1

            if m.get("acuity_match") is True:
                acuity_correct += 1
            if m.get("resources_match") is True:
                resources_correct += 1
            if m.get("exec_failed") is True:
                exec_failed += 1
            if m.get("invalid_pred") is True:
                invalid_pred += 1

        n = len(results)
        warning_rate = _safe_div(warning_count, n)
        hard_fail_rate = _safe_div(hard_fail_count, n)
        soft_pass_rate = _safe_div(soft_pass_count, n)
        acuity_accuracy = _safe_div(acuity_correct, n)
        resource_accuracy = _safe_div(resources_correct, n)

        return {
            "label": self.label_name,
            "pass_count": pass_count,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "n_eval": n,
            "soft_pass_rate": _round_or_none(soft_pass_rate),
            "warning_rate": _round_or_none(warning_rate),
            "hard_fail_rate": _round_or_none(hard_fail_rate),
            "acuity_accuracy": _round_or_none(acuity_accuracy),
            "resource_accuracy": _round_or_none(resource_accuracy),
            "excluded": {
                "exec_failed": exec_failed,
                "invalid_pred": invalid_pred,
            },
        }
