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


class VitalsUptriageEvaluator:
    label_name = "recommendation.consider_uptriage"

    def validate_expected(self, expected_json: Dict[str, Any]) -> None:
        """Validate expected."""
        # Fail fast on bad input.
        if set(expected_json.keys()) != {"recommendation"}:
            raise ValueError(
                "expected_json must only contain: recommendation.consider_uptriage"
            )
        rec = expected_json.get("recommendation")
        if not isinstance(rec, dict) or set(rec.keys()) != {"consider_uptriage"}:
            raise ValueError(
                "expected_json.recommendation must only contain: consider_uptriage"
            )
        if not isinstance(rec.get("consider_uptriage"), bool):
            raise ValueError(
                "expected_json.recommendation.consider_uptriage must be boolean"
            )

    def _y_true(self, expected_json: Dict[str, Any]) -> bool:
        """Handle true."""
        # Keep the main step clear.
        self.validate_expected(expected_json)
        return bool(expected_json["recommendation"]["consider_uptriage"])

    def _y_pred(self, actual_json: Dict[str, Any]) -> Optional[bool]:
        """Handle pred."""
        # Keep the main step clear.
        recommendation = actual_json.get("recommendation")
        if isinstance(recommendation, dict):
            val = recommendation.get("consider_uptriage")
        else:
            val = actual_json.get("consider_uptriage")
        return val if isinstance(val, bool) else None

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

        if agent_status != "succeeded" or actual_json is None:
            return EvalResult(
                passed=False,
                score=0.0,
                diff_json={"error": "exec_failed_or_missing_output", "agent_status": agent_status},
                metrics_json={
                    "label": self.label_name,
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
                diff_json={"error": "missing_or_invalid_prediction"},
                metrics_json={
                    "label": self.label_name,
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
            diff_json={} if passed else {"expected": y_true, "actual": y_pred},
            metrics_json={
                "label": self.label_name,
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
