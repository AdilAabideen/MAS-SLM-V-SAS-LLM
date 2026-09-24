"""Sequential, paired research scheduling over prevalidated case data."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from .configuration import LoadedConfiguration
from .configured_systems import ConfiguredSystems, ModelFactory, _runtime_config, build_configured_systems
from .contracts import CaseResult, FailureKind, RunFailure, RunIdentity, RunStatus, RunTiming
from .dataset import DatasetCase, LoadedDataset, load_configured_dataset
from .grading import GradeResult, GraderLike, grade_case, require_grader
from .multi_agent import MultiCaseExecution
from .single_agent import SingleCaseExecution


class ExperimentStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, kw_only=True)
class ExperimentAttempt:
    sequence: int
    case_id: str
    repetition: int
    system_id: str
    started_at: str
    ended_at: str
    model_choices: Mapping[str, Mapping[str, Any]]
    runtime_policies: Mapping[str, Mapping[str, Any]]
    execution: SingleCaseExecution | MultiCaseExecution | None
    result: CaseResult
    grade: GradeResult
    cancelled: bool = False

    @property
    def pair_key(self) -> tuple[str, int]:
        return self.case_id, self.repetition


@dataclass(frozen=True, kw_only=True)
class ExperimentPair:
    case_id: str
    repetition: int
    single: ExperimentAttempt | None
    multi: ExperimentAttempt | None


@dataclass(frozen=True, kw_only=True)
class ExperimentRun:
    experiment_id: str
    status: ExperimentStatus
    schedule: str
    dataset_path: str
    started_at: str
    ended_at: str
    attempts: tuple[ExperimentAttempt, ...]
    pairs: tuple[ExperimentPair, ...]
    reporting_errors: tuple[str, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _choices(loaded: LoadedConfiguration, systems: ConfiguredSystems, arm: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if arm == "single":
        aliases = {loaded.experiment.sas.agent: loaded.experiment.sas.agent}
        models = {loaded.experiment.sas.agent: systems.sas_model}
    else:
        aliases = loaded.experiment.mas.agents
        models = systems.mas_models
    choices = {
        role: {"provider": models[role].provider, "model_id": models[role].model_id,
               "catalog": models[role].catalog, "temperature": models[role].temperature,
               "max_tokens": models[role].max_tokens}
        for role in aliases
    }
    policies = {
        role: asdict(_runtime_config(loaded.experiment.agents[alias], multi_agent=arm == "multi"))
        for role, alias in aliases.items()
    }
    return choices, policies


def _schedule(cases: tuple[DatasetCase, ...], repetitions: int, mode: str) -> tuple[tuple[str, DatasetCase, int], ...]:
    if mode == "system_major":
        return tuple((arm, case, repetition)
                     for arm in ("single", "multi")
                     for case in cases for repetition in range(1, repetitions + 1))
    if mode == "case_major":
        return tuple((arm, case, repetition)
                     for case in cases for repetition in range(1, repetitions + 1)
                     for arm in ("single", "multi"))
    raise ValueError(f"unsupported experiment schedule {mode!r}")


def _pairs(cases: tuple[DatasetCase, ...], repetitions: int, attempts: tuple[ExperimentAttempt, ...]) -> tuple[ExperimentPair, ...]:
    indexed: dict[tuple[str, int, str], ExperimentAttempt] = {}
    for attempt in attempts:
        key = (attempt.case_id, attempt.repetition, attempt.system_id)
        if key in indexed:
            raise ValueError(f"duplicate experiment attempt {key}")
        indexed[key] = attempt
    return tuple(ExperimentPair(
        case_id=case.case_id, repetition=repetition,
        single=indexed.get((case.case_id, repetition, "single")),
        multi=indexed.get((case.case_id, repetition, "multi")),
    ) for case in cases for repetition in range(1, repetitions + 1))


async def run_experiment(
    loaded: LoadedConfiguration, *, dataset: LoadedDataset, systems: ConfiguredSystems,
    grader: GraderLike, experiment_id: str | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    on_attempt: Callable[[ExperimentAttempt], None] | None = None,
) -> ExperimentRun:
    """Run all selected cases sequentially, retaining each failed arm and grade."""
    grader = require_grader(grader)
    if not dataset.cases:
        raise ValueError("experiment dataset must contain cases")
    for case in dataset.cases:
        grader.validate_expected(case.expected_label())
    identifier = experiment_id or str(uuid.uuid4())
    if not identifier.strip():
        raise ValueError("experiment_id must be nonempty")
    started_at = _utc_now()
    attempts: list[ExperimentAttempt] = []
    reporting_errors: list[str] = []
    status = ExperimentStatus.COMPLETED
    for arm, case, repetition in _schedule(dataset.cases, loaded.experiment.repetitions, loaded.experiment.schedule):
        if cancel_requested is not None and cancel_requested():
            status = ExperimentStatus.CANCELLED
            break
        identity = RunIdentity(
            experiment_id=identifier, system_id=arm, case_id=case.case_id,
            repetition=repetition, run_id=f"{identifier}:{arm}:{case.case_id}:{repetition}",
        )
        sequence = len(attempts) + 1
        model_choices, runtime_policies = _choices(loaded, systems, arm)
        attempt_start = _utc_now()
        attempt_clock = time.perf_counter()
        execution: SingleCaseExecution | MultiCaseExecution | None = None
        cancelled = False
        try:
            facts = case.agent_input()
            if arm == "single":
                execution = await systems.sas_runner.run_case(identity=identity, payload=facts)
            else:
                execution = await systems.mas_runner.run_case(identity=identity, case_info=facts)
            result = execution.result
            if result.identity != identity:
                raise ValueError("runner returned a mismatched run identity")
        except asyncio.CancelledError:
            cancelled = True
            status = ExperimentStatus.CANCELLED
            result = CaseResult(
                identity=identity, status=RunStatus.FAILED,
                failure=RunFailure(kind=FailureKind.CANCELLED, message="experiment_cancelled"),
                timing=RunTiming(wall_seconds=time.perf_counter() - attempt_clock),
            )
        except Exception as exc:
            result = CaseResult(
                identity=identity, status=RunStatus.FAILED,
                failure=RunFailure(kind=FailureKind.RUNTIME, message=str(exc) or type(exc).__name__),
                timing=RunTiming(wall_seconds=time.perf_counter() - attempt_clock),
            )
        grade = grade_case(grader, expected=case.expected_label(), result=result)
        attempts.append(ExperimentAttempt(
            sequence=sequence, case_id=case.case_id, repetition=repetition,
            system_id=arm, started_at=attempt_start, ended_at=_utc_now(),
            model_choices=model_choices, runtime_policies=runtime_policies,
            execution=execution, result=result, grade=grade, cancelled=cancelled,
        ))
        if on_attempt is not None:
            try:
                on_attempt(attempts[-1])
            except Exception as exc:
                reporting_errors.append(f"attempt {sequence}: {type(exc).__name__}: {exc}")
        if cancelled:
            break
    recorded = tuple(attempts)
    return ExperimentRun(
        experiment_id=identifier, status=status, schedule=loaded.experiment.schedule,
        dataset_path=str(dataset.path), started_at=started_at, ended_at=_utc_now(),
        attempts=recorded,
        pairs=_pairs(dataset.cases, loaded.experiment.repetitions, recorded),
        reporting_errors=tuple(reporting_errors),
    )


async def run_configured_experiment(
    loaded: LoadedConfiguration, *, model_factory: ModelFactory | None = None,
    experiment_id: str | None = None, cancel_requested: Callable[[], bool] | None = None,
    on_attempt: Callable[[ExperimentAttempt], None] | None = None,
) -> ExperimentRun:
    """Perform all dataset preflight checks before building or invoking models."""
    dataset = load_configured_dataset(loaded)
    grader = require_grader(loaded.registry.resolve("graders", loaded.experiment.grader))
    systems = build_configured_systems(loaded, model_factory=model_factory)
    return await run_experiment(
        loaded, dataset=dataset, systems=systems, grader=grader,
        experiment_id=experiment_id, cancel_requested=cancel_requested,
        on_attempt=on_attempt,
    )
