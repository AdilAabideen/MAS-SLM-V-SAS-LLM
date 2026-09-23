"""Pure result contracts must work without the legacy backend."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mas_slm_research import (
    CaseResult,
    FailureKind,
    RunFailure,
    RunIdentity,
    RunStatus,
    RunTiming,
    TokenUsage,
    UsageSource,
    ValidatedOutput,
)


def identity() -> RunIdentity:
    return RunIdentity(experiment_id="comparison", system_id="sas", case_id="case-1", repetition=1, run_id="run-1")


@pytest.mark.unit
def test_completed_and_failed_attempts_serialize_with_distinct_usage_and_timing():
    output_data = {"esi": 2}
    complete = CaseResult(
        identity=identity(),
        status=RunStatus.COMPLETED,
        output=ValidatedOutput(value=output_data),
        usage=TokenUsage(source=UsageSource.PROVIDER, input_tokens=0, output_tokens=4, total_tokens=4),
        timing=RunTiming(wall_seconds=1.0, child_seconds_sum=1.5),
    )
    output_data["esi"] = 5
    failed = CaseResult(
        identity=identity(),
        status=RunStatus.FAILED,
        failure=RunFailure(kind=FailureKind.PROVIDER, message="unavailable"),
        timing=RunTiming(wall_seconds=0.4),
    )

    complete_json = json.loads(json.dumps(complete.to_dict()))
    failed_json = json.loads(json.dumps(failed.to_dict()))
    assert complete_json["output"] == {"esi": 2}
    assert complete_json["failure"] is None
    assert complete_json["usage"] == {
        "source": "provider", "input_tokens": 0, "output_tokens": 4, "total_tokens": 4,
    }
    assert complete_json["timing"] == {"wall_seconds": 1.0, "child_seconds_sum": 1.5}
    assert failed_json["status"] == "failed"
    assert failed_json["output"] is None
    assert failed_json["failure"] == {"kind": "provider", "message": "unavailable"}
    assert failed_json["usage"]["total_tokens"] is None


@pytest.mark.unit
def test_terminal_result_cannot_confuse_failure_with_validated_completion():
    with pytest.raises(ValueError, match="validated output"):
        CaseResult(identity=identity(), status=RunStatus.COMPLETED, timing=RunTiming(wall_seconds=0.1))
    with pytest.raises(ValueError, match="classified failure"):
        CaseResult(identity=identity(), status=RunStatus.FAILED, timing=RunTiming(wall_seconds=0.1))


@pytest.mark.unit
def test_package_imports_in_isolated_stdlib_only_process():
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from mas_slm_research import CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming, ValidatedOutput
identity = RunIdentity(experiment_id='e', system_id='s', case_id='c', repetition=1, run_id='r')
result = CaseResult(
    identity=identity,
    status=RunStatus.COMPLETED,
    timing=RunTiming(wall_seconds=0.0),
    output=ValidatedOutput(value={'answer': 1}),
)
failure = CaseResult(
    identity=identity,
    status=RunStatus.FAILED,
    timing=RunTiming(wall_seconds=0.2),
    failure=RunFailure(kind=FailureKind.PROVIDER, message='offline'),
)
print(json.dumps({'result': result.to_dict(), 'failure': failure.to_dict(), 'backend_loaded': any(
    name == 'sqlalchemy' or name.startswith('sqlalchemy.') or
    name == 'fastapi' or name.startswith('fastapi.') for name in sys.modules
)}))
"""
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(source_root)],
        capture_output=True, text=True, check=True,
    )
    imported = json.loads(process.stdout)
    assert imported["backend_loaded"] is False
    assert imported["result"]["output"] == {"answer": 1}
    assert imported["failure"]["failure"] == {"kind": "provider", "message": "offline"}
