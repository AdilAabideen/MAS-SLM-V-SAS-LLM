"""Evaluator module helpers."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence

from .legacy_types import EvalResult


Confusion = Literal["tp", "tn", "fp", "fn"]


def _safe_div(num: int, denom: int) -> Optional[float]:
    """Handle div."""
    # Keep the main step clear.
    return None if denom == 0 else (num / denom)


def _round_or_none(v: Optional[float], ndigits: int = 4) -> Optional[float]:
    """Handle or none."""
    # Keep the main step clear.
    return None if v is None else round(v, ndigits)


def _coerce_to_bool(value: Any) -> Optional[bool]:
    """Handle to bool."""
    # Keep the main step clear.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "esi1", "esi-1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "not_esi1", "not-esi1", "not esi1"}:
            return False
    return None


class ES1AcuityEvaluator:
    """
    Binary ESI-1 evaluator derived from expected acuity.

    Ground truth rule:
    - acuity == 1  -> expected is_esi1 = True
    - acuity != 1  -> expected is_esi1 = False
    """

    label_name = "is_esi1_from_acuity"

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

    def _y_true(self, expected_json: Dict[str, Any]) -> bool:
        """Handle true."""
        # Keep the main step clear.
        self.validate_expected(expected_json)
        return int(expected_json["acuity"]) == 1

    def _y_pred(self, actual_json: Dict[str, Any]) -> Optional[bool]:
        # Primary output contract requested by user.
        """Handle pred."""
        if "is_esi1" in actual_json:
            return _coerce_to_bool(actual_json.get("is_esi1"))

        # Backward-compatible fallback if old field still appears.
        if "provisional_esi" in actual_json:
            return _coerce_to_bool(actual_json.get("provisional_esi"))

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
        y_true = self._y_true(expected_json)
        expected_acuity = int(expected_json["acuity"])

        if agent_status != "succeeded" or actual_json is None:
            return EvalResult(
                passed=False,
                score=0.0,
                diff_json={
                    "error": "exec_failed_or_missing_output",
                    "agent_status": agent_status,
                    "expected_acuity": expected_acuity,
                    "expected_is_esi1": y_true,
                },
                metrics_json={
                    "label": self.label_name,
                    "expected_acuity": expected_acuity,
                    "y_true": y_true,
                    "y_pred": None,
                    "confusion": None,
                    "exec_failed": True,
                    "invalid_pred": False,
                },
            )

        y_pred = self._y_pred(actual_json)
        if y_pred is None:
            return EvalResult(
                passed=False,
                score=0.0,
                diff_json={
                    "error": "missing_or_invalid_prediction",
                    "expected_acuity": expected_acuity,
                    "expected_is_esi1": y_true,
                },
                metrics_json={
                    "label": self.label_name,
                    "expected_acuity": expected_acuity,
                    "y_true": y_true,
                    "y_pred": None,
                    "confusion": None,
                    "exec_failed": False,
                    "invalid_pred": True,
                },
            )

        passed = y_pred == y_true
        confusion: Confusion
        if y_true and y_pred:
            confusion = "tp"
        elif (not y_true) and (not y_pred):
            confusion = "tn"
        elif (not y_true) and y_pred:
            confusion = "fp"
        else:
            confusion = "fn"

        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            diff_json=(
                {}
                if passed
                else {
                    "expected_acuity": expected_acuity,
                    "expected_is_esi1": y_true,
                    "actual_is_esi1": y_pred,
                }
            ),
            metrics_json={
                "label": self.label_name,
                "expected_acuity": expected_acuity,
                "y_true": y_true,
                "y_pred": y_pred,
                "confusion": confusion,
                "exec_failed": False,
                "invalid_pred": False,
            },
        )

    def aggregate(self, results: Sequence[EvalResult]) -> Dict[str, Any]:
        """Handle the value."""
        # Keep the main step clear.
        tp = tn = fp = fn = 0
        exec_failed = 0
        invalid_pred = 0
        other_excluded = 0

        for r in results:
            m = r.metrics_json or {}
            if m.get("exec_failed") is True:
                exec_failed += 1
                continue
            if m.get("invalid_pred") is True or m.get("y_pred") is None:
                invalid_pred += 1
                continue
            c = m.get("confusion")
            if c == "tp":
                tp += 1
            elif c == "tn":
                tn += 1
            elif c == "fp":
                fp += 1
            elif c == "fn":
                fn += 1
            else:
                other_excluded += 1

        n = tp + tn + fp + fn
        accuracy = _safe_div(tp + tn, n)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        specificity = _safe_div(tn, tn + fp)

        return {
            "label": self.label_name,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_eval": n,
            "accuracy": _round_or_none(accuracy),
            "precision": _round_or_none(precision),
            "recall": _round_or_none(recall),
            "f1": _round_or_none(f1),
            "specificity": _round_or_none(specificity),
            "excluded": {
                "exec_failed": exec_failed,
                "invalid_pred": invalid_pred,
                "other": other_excluded,
            },
        }
