"""Portable, incremental research artifacts and inference-free recomputation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .comparison import ComparisonReport, PriceRate, compare_experiment
from .configuration import LoadedConfiguration
from .contracts import (
    CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming,
    TokenUsage, UsageSource, ValidatedOutput,
)
from .dataset import LoadedDataset
from .experiment import ExperimentAttempt, ExperimentPair, ExperimentRun, ExperimentStatus
from .grading import GradeResult, GradeStatus
from .multi_agent import MultiCaseExecution


ARTIFACT_VERSION = 1
_SENSITIVE_KEYS = ("api_key", "password", "secret", "authorization", "access_token", "refresh_token")


class ArtifactError(ValueError):
    """A research artifact is incomplete, inconsistent, or would be overwritten."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


def _scrub(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "[redacted]" if any(part in str(key).lower() for part in _SENSITIVE_KEYS)
                else _scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scrub(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
    return value


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_revision(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return revision or None, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _package_version(root: Path) -> str | None:
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return data["project"]["version"]
    except (OSError, KeyError, ValueError):
        return None


def _call_facts(call: Mapping[str, Any]) -> dict[str, Any]:
    return {key: call.get(key) for key in (
        "run_id", "agent_name", "call_index", "tool_call_id", "tool_name", "status",
        "input_tokens", "output_tokens", "tokens_total", "usage_source",
        "provider_model_id", "request_parameters", "network_attempts",
        "input_token_source", "output_token_source",
        "text_recovered_tool_call_count", "latency_ms",
    ) if key in call}


def _comparison_events(execution: Any) -> list[dict[str, Any]]:
    if execution is None:
        return []
    return [{"event_type": "runtime_decision", "payload_json": {"decision": "retry_after_malformed_tool_call"}}
            for event in execution.events
            if event.get("event_type") == "runtime_decision"
            and isinstance(event.get("payload_json"), Mapping)
            and event["payload_json"].get("decision") == "retry_after_malformed_tool_call"]


def _attempt_record(attempt: ExperimentAttempt) -> dict[str, Any]:
    execution = attempt.execution
    return {
        "sequence": attempt.sequence, "case_id": attempt.case_id,
        "repetition": attempt.repetition, "system_id": attempt.system_id,
        "started_at": attempt.started_at, "ended_at": attempt.ended_at,
        "model_choices": attempt.model_choices, "runtime_policies": attempt.runtime_policies,
        "cancelled": attempt.cancelled,
        "result": attempt.result.to_dict(), "grade": attempt.grade.to_dict(),
        "measurements": {
            "llm_calls": [_call_facts(call) for call in execution.llm_calls] if execution else [],
            "tool_calls": [_call_facts(call) for call in execution.tool_calls] if execution else [],
            "events": _comparison_events(execution),
        },
    }


def _event_records(attempt: ExperimentAttempt) -> list[dict[str, Any]]:
    execution = attempt.execution
    if execution is None:
        return []
    timeline = execution.timeline if isinstance(execution, MultiCaseExecution) else tuple(
        {"source": "agent_event", "event": event} for event in execution.events
    )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(timeline, 1):
        event = item.get("event")
        if not isinstance(event, Mapping):
            continue
        records.append({
            "run_id": attempt.result.identity.run_id, "sequence": index,
            "source": item.get("source"), "event_type": event.get("event_type"),
            "agent_name": event.get("agent_name"), "agent_run_id": event.get("agent_run_id"),
            "handoff_id": event.get("handoff_id"), "gate_evaluation_id": event.get("gate_evaluation_id"),
            "tool_name": event.get("tool_name"), "tool_call_id": event.get("tool_call_id"),
            "status": event.get("status"),
        })
    return records


class ArtifactWriter:
    """Create a new directory and flush every completed attempt immediately."""

    def __init__(
        self, *, directory: Path, loaded: LoadedConfiguration, dataset: LoadedDataset,
        experiment_id: str, prices_by_role: Mapping[str, PriceRate], include_events: bool = False,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.directory = directory
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactError(f"artifact directory already exists: {directory}") from exc
        self.results_path = directory / "results.jsonl"
        self.events_path = directory / "events.jsonl" if include_events else None
        self._results = self.results_path.open("w", encoding="utf-8")
        self._events = self.events_path.open("w", encoding="utf-8") if self.events_path else None
        self._seen: set[tuple[str, int, str]] = set()
        env = environment if environment is not None else os.environ
        secret_names = [model.api_key_env for model in loaded.models.values() if model.api_key_env]
        secret_names.extend(loaded.experiment.telemetry.header_env.values())
        self._secrets = tuple(sorted({env[name] for name in secret_names
                                      if name in env and len(env[name]) >= 4}, key=len, reverse=True))
        root = Path(__file__).resolve().parents[2]
        revision, dirty = _source_revision(root)
        snapshot_path = directory / "resolved-config.json"
        _atomic_json(snapshot_path, _scrub(loaded.safe_snapshot(), self._secrets))
        self.manifest: dict[str, Any] = {
            "artifact_version": ARTIFACT_VERSION,
            "state": "in_progress", "experiment_id": experiment_id,
            "experiment_name": loaded.experiment.name,
            "schedule": loaded.experiment.schedule,
            "repetitions": loaded.experiment.repetitions,
            "started_at": datetime.now(timezone.utc).isoformat(), "ended_at": None,
            "dataset_path": str(dataset.path),
            "pair_keys": [[case.case_id, repetition] for case in dataset.cases
                          for repetition in range(1, loaded.experiment.repetitions + 1)],
            "recorded_grader_summaries": None,
            "prices_by_role": {role: {"input_per_1k": rate.input_per_1k,
                                       "output_per_1k": rate.output_per_1k}
                               for role, rate in prices_by_role.items()},
            "source_revision": revision, "source_dirty": dirty,
            "package_version": _package_version(root),
            "hashes": {
                "experiment_yaml": _digest(loaded.experiment_path),
                "workflow_yaml": _digest(loaded.workflow_path),
                "dataset": _digest(dataset.path),
                "resolved-config.json": _digest(snapshot_path),
                "results.jsonl": _digest(self.results_path),
            },
            "attempts_written": 0,
            "events_included": include_events,
        }
        if self.events_path:
            self.manifest["hashes"]["events.jsonl"] = _digest(self.events_path)
        self._manifest()

    def _manifest(self) -> None:
        _atomic_json(self.directory / "manifest.json", _scrub(self.manifest, self._secrets))

    def record_attempt(self, attempt: ExperimentAttempt) -> None:
        key = (attempt.case_id, attempt.repetition, attempt.system_id)
        if key in self._seen or attempt.sequence != len(self._seen) + 1:
            raise ArtifactError(f"duplicate or out-of-order attempt {key}")
        if attempt.result.identity.experiment_id != self.manifest["experiment_id"]:
            raise ArtifactError("attempt experiment ID differs from artifact manifest")
        self._results.write(_json(_scrub(_attempt_record(attempt), self._secrets)) + "\n")
        self._results.flush()
        os.fsync(self._results.fileno())
        if self._events is not None:
            for event in _event_records(attempt):
                self._events.write(_json(_scrub(event, self._secrets)) + "\n")
            self._events.flush()
            os.fsync(self._events.fileno())
            assert self.events_path is not None
            self.manifest["hashes"]["events.jsonl"] = _digest(self.events_path)
        self._seen.add(key)
        self.manifest["attempts_written"] = len(self._seen)
        self.manifest["hashes"]["results.jsonl"] = _digest(self.results_path)
        self._manifest()

    def mark_interrupted(self) -> None:
        self.manifest["state"] = "interrupted"
        self.manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        self._manifest()
        self.close()

    def finalize(self, run: ExperimentRun, report: ComparisonReport) -> None:
        if len(run.attempts) != len(self._seen):
            raise ArtifactError("saved attempt count differs from experiment")
        self.manifest["state"] = "completed" if run.status == ExperimentStatus.COMPLETED else "cancelled"
        self.manifest["started_at"] = run.started_at
        self.manifest["ended_at"] = run.ended_at
        self.manifest["recorded_grader_summaries"] = _scrub({
            arm: summary.grader_summary for arm, summary in report.systems.items()
            if summary.grader_summary is not None
        }, self._secrets)
        self._manifest()
        recomputed = summarize_artifacts(self.directory)
        if recomputed.to_dict() != report.to_dict():
            raise ArtifactError("offline recomputation differs from terminal comparison")
        summary_path = self.directory / "summary.json"
        _atomic_json(summary_path, _scrub(report.to_dict(), self._secrets))
        csv_path = self.directory / "comparison.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "case_id", "repetition", "single_run_id", "multi_run_id",
                "single_score", "multi_score", "winner",
            ))
            writer.writeheader()
            for pair in report.pairs:
                writer.writerow(_scrub(vars(pair), self._secrets))
        self.manifest["hashes"].update({
            "summary.json": _digest(summary_path), "comparison.csv": _digest(csv_path),
        })
        self._manifest()
        self.close()

    def close(self) -> None:
        if not self._results.closed:
            self._results.close()
        if self._events is not None and not self._events.closed:
            self._events.close()


def _result(value: Mapping[str, Any]) -> CaseResult:
    identity = RunIdentity(**value["identity"])
    failure = value.get("failure")
    output = value.get("output")
    usage = value["usage"]
    return CaseResult(
        identity=identity, status=RunStatus(value["status"]),
        output=ValidatedOutput(value=output) if output is not None else None,
        failure=RunFailure(kind=FailureKind(failure["kind"]), message=failure["message"])
        if failure is not None else None,
        usage=TokenUsage(source=UsageSource(usage["source"]),
                         input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                         total_tokens=usage["total_tokens"]),
        timing=RunTiming(**value["timing"]),
    )


def _grade(value: Mapping[str, Any]) -> GradeResult:
    kind = value.get("execution_failure_kind")
    return GradeResult(
        identity=RunIdentity(**value["identity"]), status=GradeStatus(value["status"]),
        passed=value["passed"], score=value["score"], diagnostics=value.get("diagnostics", {}),
        error=value.get("error"), execution_failure_kind=FailureKind(kind) if kind else None,
    )


def _read_manifest(directory: Path, *, verify_derived: bool = False) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{directory}: invalid manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ArtifactError(f"{directory}: unsupported artifact version")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise ArtifactError(f"{directory}: manifest hashes must be an object")
    allowed_files = {"experiment_yaml", "workflow_yaml", "dataset", "resolved-config.json",
                     "results.jsonl", "events.jsonl", "summary.json", "comparison.csv"}
    for filename, expected in hashes.items():
        if filename not in allowed_files or not isinstance(expected, str) or len(expected) != 64:
            raise ArtifactError(f"{directory}: invalid manifest hash entry {filename!r}")
        if filename in {"experiment_yaml", "workflow_yaml", "dataset"}:
            continue  # Source files may be absent after the artifact is moved.
        if filename in {"summary.json", "comparison.csv"} and not verify_derived:
            continue  # Recompute from authoritative records even if derived files are absent.
        path = directory / filename
        if not path.is_file() or _digest(path) != expected:
            raise ArtifactError(f"{directory}: hash mismatch for {filename}")
    return manifest


def summarize_artifacts(directory: Path, *, verify_derived: bool = False) -> ComparisonReport:
    """Recompute generic totals from saved per-attempt records, without inference."""
    manifest = _read_manifest(directory, verify_derived=verify_derived)
    attempts: list[ExperimentAttempt] = []
    try:
        with (directory / "results.jsonl").open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    measurement = row["measurements"]
                    result = _result(row["result"])
                    grade = _grade(row["grade"])
                    attempts.append(ExperimentAttempt(
                        sequence=row["sequence"], case_id=row["case_id"],
                        repetition=row["repetition"], system_id=row["system_id"],
                        started_at=row["started_at"], ended_at=row["ended_at"],
                        model_choices=row["model_choices"], runtime_policies=row["runtime_policies"],
                        execution=SimpleNamespace(
                            llm_calls=measurement["llm_calls"],
                            tool_calls=measurement["tool_calls"], events=measurement["events"],
                        ),
                        result=result, grade=grade, cancelled=row.get("cancelled", False),
                    ))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ArtifactError(f"{directory}/results.jsonl:{line_number}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise ArtifactError(f"{directory}/results.jsonl: {exc}") from exc
    if len(attempts) != manifest.get("attempts_written"):
        raise ArtifactError("results count differs from manifest")
    indexed = {(item.case_id, item.repetition, item.system_id): item for item in attempts}
    if len(indexed) != len(attempts):
        raise ArtifactError("duplicate attempt identity in results")
    pairs = tuple(ExperimentPair(
        case_id=case_id, repetition=repetition,
        single=indexed.get((case_id, repetition, "single")),
        multi=indexed.get((case_id, repetition, "multi")),
    ) for case_id, repetition in manifest["pair_keys"])
    state = manifest["state"]
    if state == "completed" and any(pair.single is None or pair.multi is None for pair in pairs):
        raise ArtifactError("completed artifact has an unattempted pair arm")
    run = ExperimentRun(
        experiment_id=manifest["experiment_id"],
        status=ExperimentStatus.COMPLETED if state == "completed" else ExperimentStatus.CANCELLED,
        schedule=manifest["schedule"], dataset_path=manifest["dataset_path"],
        started_at=manifest["started_at"], ended_at=manifest["ended_at"] or manifest["started_at"],
        attempts=tuple(attempts), pairs=pairs,
    )
    prices = {role: PriceRate(**rate) for role, rate in manifest["prices_by_role"].items()}
    report = compare_experiment(
        run, prices_by_role=prices,
        recorded_grader_summaries=manifest.get("recorded_grader_summaries"),
    )
    if state in {"in_progress", "interrupted"}:
        from dataclasses import replace
        report = replace(report, experiment_status=state)
    return report
