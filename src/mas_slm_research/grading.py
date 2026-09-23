"""Common case-grading contract separate from agent execution."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import CaseResult, FailureKind, RunIdentity, RunStatus


class GradeStatus(str, Enum):
    GRADED = "graded"
    EXECUTION_FAILED = "execution_failed"
    GRADER_ERROR = "grader_error"


@dataclass(frozen=True, kw_only=True)
class GradeDecision:
    """A task grader's judgment for one successfully completed output."""

    passed: bool
    score: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("score must be a finite number")
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        object.__setattr__(self, "diagnostics", json.loads(json.dumps(dict(self.diagnostics), allow_nan=False)))


@dataclass(frozen=True, kw_only=True)
class GradeResult:
    """One outcome per case attempt, including execution and grader failures."""

    identity: RunIdentity
    status: GradeStatus
    passed: bool | None
    score: float | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_failure_kind: FailureKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, GradeStatus):
            raise TypeError("status must be a GradeStatus")
        if self.status == GradeStatus.GRADED:
            if (not isinstance(self.passed, bool) or self.score is None or self.error is not None
                    or self.execution_failure_kind is not None):
                raise ValueError("graded result requires a judgment and no error")
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
                raise ValueError("graded score must be finite")
            if not 0 <= self.score <= 1:
                raise ValueError("graded score must be between 0 and 1")
        elif self.status == GradeStatus.EXECUTION_FAILED:
            if self.passed is not False or self.score != 0 or not self.error or self.execution_failure_kind is None:
                raise ValueError("execution failure requires false/zero and classified error")
        elif self.status == GradeStatus.GRADER_ERROR:
            if (self.passed is not None or self.score is not None or not self.error
                    or self.execution_failure_kind is not None):
                raise ValueError("grader error requires unknown judgment and error")
        object.__setattr__(self, "diagnostics", json.loads(json.dumps(dict(self.diagnostics), allow_nan=False)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "status": self.status.value,
            "passed": self.passed,
            "score": self.score,
            "diagnostics": dict(self.diagnostics),
            "error": self.error,
            "execution_failure_kind": self.execution_failure_kind.value if self.execution_failure_kind else None,
        }


@runtime_checkable
class GraderLike(Protocol):
    """Structural protocol for registered graders, including external objects."""

    def validate_expected(self, expected: Mapping[str, Any]) -> None: ...

    def evaluate(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> GradeDecision: ...

    def aggregate(self, results: Sequence[GradeResult]) -> Mapping[str, Any]: ...


class BaseGrader(ABC):
    """Subclass this to supply a new task's label, case, and summary policy."""

    @abstractmethod
    def validate_expected(self, expected: Mapping[str, Any]) -> None:
        """Raise if a label cannot be graded."""

    @abstractmethod
    def evaluate(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> GradeDecision:
        """Grade only a completed, validated system output."""

    @abstractmethod
    def aggregate(self, results: Sequence[GradeResult]) -> Mapping[str, Any]:
        """Aggregate case-grade records without losing failed attempts."""


def require_grader(value: Any) -> GraderLike:
    """Accept subclasses and compatible objects while rejecting incomplete hooks."""
    if isinstance(value, type):
        value = value()
    for method in ("validate_expected", "evaluate", "aggregate"):
        if not callable(getattr(value, method, None)):
            raise TypeError(f"grader must implement {method}()")
    return value


def grade_case(grader: GraderLike, *, expected: Mapping[str, Any], result: CaseResult) -> GradeResult:
    """Adapt a common SAS/MAS result into a distinct grade outcome."""
    try:
        grader.validate_expected(expected)
    except Exception as exc:
        return GradeResult(
            identity=result.identity, status=GradeStatus.GRADER_ERROR,
            passed=None, score=None, error=f"expected-label validation: {exc}",
        )
    if result.status == RunStatus.FAILED:
        assert result.failure is not None
        return GradeResult(
            identity=result.identity, status=GradeStatus.EXECUTION_FAILED,
            passed=False, score=0.0, error=result.failure.message,
            execution_failure_kind=result.failure.kind,
        )
    assert result.output is not None
    try:
        decision = grader.evaluate(expected, result.output.to_dict())
        if not isinstance(decision, GradeDecision):
            raise TypeError("evaluate() must return GradeDecision")
    except Exception as exc:
        return GradeResult(
            identity=result.identity, status=GradeStatus.GRADER_ERROR,
            passed=None, score=None, error=f"grader evaluation: {exc}",
        )
    return GradeResult(
        identity=result.identity, status=GradeStatus.GRADED,
        passed=decision.passed, score=decision.score, diagnostics=decision.diagnostics,
    )


def aggregate_grades(grader: GraderLike, results: Sequence[GradeResult]) -> Mapping[str, Any]:
    """Use the registered grader aggregation hook on all recorded attempts."""
    summary = grader.aggregate(results)
    if not isinstance(summary, Mapping):
        raise TypeError("aggregate() must return a mapping")
    return dict(summary)
