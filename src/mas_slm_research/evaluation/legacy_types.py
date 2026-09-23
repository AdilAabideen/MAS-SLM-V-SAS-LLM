from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    passed: bool
    score: float
    diff_json: dict[str, Any]
    metrics_json: dict[str, Any]
