"""Optional legacy specialist diagnostics, never headline final-task grades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .legacy_types import EvalResult


@dataclass(frozen=True, kw_only=True)
class AgentDiagnostic:
    agent_name: str
    label: str
    passed: bool | None
    score: float | None
    diff: Mapping[str, Any]
    metrics: Mapping[str, Any]
    error: str | None = None
    placeholder: bool = False


def evaluate_legacy_diagnostic(
    evaluator: Any,
    *,
    agent_name: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any] | None,
    agent_status: str,
) -> AgentDiagnostic:
    """Adapt the old expected/actual/status signature into a separate record."""
    label = str(getattr(evaluator, "label_name", type(evaluator).__name__))
    placeholder = label == "doctor_always_pass"
    try:
        evaluator.validate_expected(dict(expected))
        verdict = evaluator.evaluate(
            dict(expected), dict(actual) if actual is not None else None,
            agent_status=agent_status,
        )
        if not isinstance(verdict.passed, bool):
            raise TypeError("legacy evaluator passed must be a boolean")
        return AgentDiagnostic(
            agent_name=agent_name, label=label,
            passed=verdict.passed, score=verdict.score,
            diff=dict(verdict.diff_json), metrics=dict(verdict.metrics_json),
            placeholder=placeholder,
        )
    except Exception as exc:
        return AgentDiagnostic(
            agent_name=agent_name, label=label, passed=None, score=None,
            diff={}, metrics={}, error=str(exc) or type(exc).__name__,
            placeholder=placeholder,
        )


def aggregate_legacy_diagnostics(evaluator: Any, diagnostics: Sequence[AgentDiagnostic]) -> Mapping[str, Any]:
    """Use the evaluator's original aggregate only for matching diagnostics."""
    label = str(getattr(evaluator, "label_name", type(evaluator).__name__))
    mismatched = [item.label for item in diagnostics if item.label != label]
    if mismatched:
        raise ValueError(f"diagnostic labels do not match {label!r}: {mismatched}")
    valid = [
        EvalResult(
            passed=item.passed, score=item.score,
            diff_json=dict(item.diff), metrics_json=dict(item.metrics),
        )
        for item in diagnostics if item.error is None
    ]
    result = evaluator.aggregate(valid)
    if not isinstance(result, Mapping):
        raise TypeError("legacy diagnostic aggregate must return a mapping")
    return {**dict(result), "diagnostic_only": True, "placeholder": label == "doctor_always_pass"}
