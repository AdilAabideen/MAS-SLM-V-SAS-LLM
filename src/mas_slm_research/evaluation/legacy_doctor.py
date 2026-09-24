"""Evaluator module helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .legacy_types import EvalResult


class DoctorAlwaysPassEvaluator:
    """Temporary evaluator for the doctor agent.

    This intentionally passes every run so the doctor agent can participate in the
    existing evaluation pipeline before a real benchmark contract is defined.
    """

    label_name = "doctor_always_pass"

    def validate_expected(self, expected_json: Dict[str, Any]) -> None:
        # Intentionally permissive for now.
        """Validate expected."""
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
        return EvalResult(
            passed=True,
            score=1.0,
            diff_json={},
            metrics_json={
                "label": self.label_name,
                "agent_status": agent_status,
                "had_output": actual_json is not None,
                "always_pass": True,
            },
        )

    def aggregate(self, results: Sequence[EvalResult]) -> Dict[str, Any]:
        """Handle the value."""
        # Keep the main step clear.
        total = len(list(results))
        return {
            "label": self.label_name,
            "n_eval": total,
            "passed": total,
            "failed": 0,
            "pass_rate": 1.0,
            "always_pass": True,
        }
