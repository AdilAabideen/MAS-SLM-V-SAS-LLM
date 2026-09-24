"""Pure terminal result contracts shared by single- and multi-agent runs.

These values describe one case attempt. They deliberately have no provider,
database, evaluator, or reporting dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class FailureKind(str, Enum):
    PROVIDER = "provider"
    TOOL = "tool"
    PROTOCOL = "protocol"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    BUDGET = "budget"
    CANCELLED = "cancelled"
    RUNTIME = "runtime"


class UsageSource(str, Enum):
    UNKNOWN = "unknown"
    PROVIDER = "provider"
    ESTIMATED = "estimated"


@dataclass(frozen=True, kw_only=True)
class RunIdentity:
    """Stable identity for one case, system, and repetition attempt."""

    experiment_id: str
    system_id: str
    case_id: str
    repetition: int
    run_id: str

    def __post_init__(self) -> None:
        for name in ("experiment_id", "system_id", "case_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if isinstance(self.repetition, bool) or not isinstance(self.repetition, int) or self.repetition < 1:
            raise ValueError("repetition must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "system_id": self.system_id,
            "case_id": self.case_id,
            "repetition": self.repetition,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, kw_only=True)
class ValidatedOutput:
    """A JSON-compatible output already checked by the caller's schema."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping):
            raise TypeError("validated output must be a mapping")
        # Snapshot the validated payload so later caller mutations cannot alter a result.
        object.__setattr__(self, "value", json.loads(json.dumps(dict(self.value), allow_nan=False)))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.value)


@dataclass(frozen=True, kw_only=True)
class RunFailure:
    kind: FailureKind
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FailureKind):
            raise TypeError("failure kind must be a FailureKind")
        if not self.message:
            raise ValueError("failure message must be nonempty")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "message": self.message}


@dataclass(frozen=True, kw_only=True)
class TokenUsage:
    """Unknown token counts are None; measured zero is the integer 0."""

    source: UsageSource = UsageSource.UNKNOWN
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, UsageSource):
            raise TypeError("usage source must be a UsageSource")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            count = getattr(self, name)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, kw_only=True)
class RunTiming:
    """Elapsed wall time and sum of child durations are distinct measurements."""

    wall_seconds: float
    child_seconds_sum: float | None = None

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "child_seconds_sum"):
            duration = getattr(self, name)
            if duration is not None and (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number or None")

    def to_dict(self) -> dict[str, float | None]:
        return {"wall_seconds": self.wall_seconds, "child_seconds_sum": self.child_seconds_sum}


@dataclass(frozen=True, kw_only=True)
class CaseResult:
    """Exactly one validated completion or classified failure for a case run."""

    identity: RunIdentity
    status: RunStatus
    timing: RunTiming
    usage: TokenUsage = field(default_factory=TokenUsage)
    output: ValidatedOutput | None = None
    failure: RunFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.status == RunStatus.COMPLETED and (self.output is None or self.failure is not None):
            raise ValueError("completed result requires validated output and no failure")
        if self.status == RunStatus.FAILED and (self.failure is None or self.output is not None):
            raise ValueError("failed result requires a classified failure and no output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "status": self.status.value,
            "output": self.output.to_dict() if self.output is not None else None,
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "usage": self.usage.to_dict(),
            "timing": self.timing.to_dict(),
        }
