"""Registered, upfront-validated case data with labels kept outside agent input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from .configuration import LoadedConfiguration
from .grading import BaseGrader


class DatasetError(ValueError):
    """A dataset cannot be used for an experiment as declared."""


def _json_copy(value: Any, *, context: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{context}: must contain finite JSON-compatible values") from exc


@dataclass(frozen=True, kw_only=True)
class DatasetCase:
    case_id: str
    input: Mapping[str, Any]
    expected: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    split: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip() or self.case_id != self.case_id.strip():
            raise DatasetError("case_id must be a nonempty, trimmed string")
        for field_name in ("input", "expected", "metadata"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise DatasetError(f"{field_name} must be an object")
            copied = _json_copy(dict(value), context=field_name)
            object.__setattr__(self, field_name, MappingProxyType(copied))
        if not self.expected:
            raise DatasetError("expected must contain at least one label")
        if "expected" in self.input or "expected_json" in self.input:
            raise DatasetError("input must not contain expected labels")
        if self.split is not None and (not isinstance(self.split, str) or not self.split.strip()):
            raise DatasetError("split must be a nonempty string when supplied")
        if not isinstance(self.tags, (tuple, list)) or any(
            not isinstance(tag, str) or not tag.strip() for tag in self.tags
        ):
            raise DatasetError("tags must contain nonempty strings")
        if len(set(self.tags)) != len(self.tags):
            raise DatasetError("tags must be unique")
        object.__setattr__(self, "tags", tuple(self.tags))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DatasetCase":
        allowed = {"case_id", "input", "expected", "metadata", "split", "tags"}
        unknown = set(record) - allowed
        if unknown:
            raise DatasetError(f"unknown case fields: {sorted(unknown)}")
        missing = {"case_id", "input", "expected"} - set(record)
        if missing:
            raise DatasetError(f"missing case fields: {sorted(missing)}")
        return cls(
            case_id=record["case_id"], input=record["input"], expected=record["expected"],
            metadata=record.get("metadata", {}), split=record.get("split"),
            tags=record.get("tags", ()),
        )

    def agent_input(self) -> dict[str, Any]:
        """Return only case facts, copied so agent code cannot mutate the dataset."""
        return _json_copy(dict(self.input), context=f"case {self.case_id}.input")

    def expected_label(self) -> dict[str, Any]:
        """Return labels only to a grader or preflight validator."""
        return _json_copy(dict(self.expected), context=f"case {self.case_id}.expected")


@dataclass(frozen=True, kw_only=True)
class LoadedDataset:
    path: Path
    loader_id: str
    cases: tuple[DatasetCase, ...]

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)


DatasetLoader = Callable[[Path], Iterable[Mapping[str, Any] | DatasetCase]]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Read local JSONL without silently skipping malformed or duplicate keys."""
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                except (json.JSONDecodeError, DatasetError) as exc:
                    raise DatasetError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(record, Mapping):
                    raise DatasetError(f"{path}:{line_number}: each row must be an object")
                rows.append(record)
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"{path}: {exc}") from exc
    return rows


def normalize_esi_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize documented ESI aliases without reading or inventing labels."""
    from .agents.esi.single_agent_system.schema import SingleAgentInput

    aliases = {
        "arrivaltransport": "arrival_transport", "arrival": "arrival_transport", "transfer": "arrival_transport",
        "chief complaint": "chiefcomplaint", "chief_complaint": "chiefcomplaint",
        "triage case": "tiragecase", "triage_case": "tiragecase",
    }
    data = dict(value)
    for alias, canonical in aliases.items():
        if alias in data:
            if canonical in data:
                raise DatasetError(f"ESI input contains both {alias!r} and {canonical!r}")
            data[canonical] = data.pop(alias)
    forbidden = {"acuity", "final_esi_level", "resources_used", "expected", "expected_json"} & set(data)
    if forbidden:
        raise DatasetError(f"ESI input contains target labels: {sorted(forbidden)}")
    unknown = set(data) - set(SingleAgentInput.model_fields)
    if unknown:
        raise DatasetError(f"unknown ESI input fields: {sorted(unknown)}")
    for optional in (
        "pain", "age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    ):
        data.setdefault(optional, None)
    try:
        return SingleAgentInput.model_validate(data).model_dump()
    except Exception as exc:
        raise DatasetError(f"invalid ESI case input: {exc}") from exc


def load_esi_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Normalize ESI case aliases while preserving labels in their own field."""
    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(load_jsonl(path), 1):
        if not isinstance(record.get("input"), Mapping):
            raise DatasetError(f"{path}:case {index}: input must be an object")
        try:
            normalized.append({**record, "input": normalize_esi_input(record["input"])})
        except DatasetError as exc:
            raise DatasetError(f"{path}:case {index}: {exc}") from exc
    return normalized


def load_dataset(
    *,
    path: Path,
    loader_id: str,
    loader: DatasetLoader,
    grader: BaseGrader,
    split: str | None = None,
    case_ids: Sequence[str] | None = None,
) -> LoadedDataset:
    """Load every record and validate selected labels/IDs before inference."""
    try:
        raw_rows = tuple(loader(path))
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"{path}: loader {loader_id!r} failed: {exc}") from exc
    seen: set[str] = set()
    parsed: list[DatasetCase] = []
    for index, raw in enumerate(raw_rows, 1):
        try:
            case = raw if isinstance(raw, DatasetCase) else DatasetCase.from_record(raw)
        except (DatasetError, TypeError) as exc:
            raise DatasetError(f"{path}:case {index}: {exc}") from exc
        if case.case_id in seen:
            raise DatasetError(f"{path}:case {index}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        if (split is not None and case.split != split) or (
            case_ids is not None and case.case_id not in case_ids
        ):
            continue
        try:
            grader.validate_expected(case.expected_label())
        except Exception as exc:
            raise DatasetError(f"{path}:case {index} ({case.case_id}): invalid expected label: {exc}") from exc
        parsed.append(case)
    if case_ids is not None:
        missing = set(case_ids) - seen
        if missing:
            raise DatasetError(f"{path}: selected case IDs not found: {sorted(missing)}")
    if not parsed:
        raise DatasetError(f"{path}: selection contains no cases")
    return LoadedDataset(path=path, loader_id=loader_id, cases=tuple(parsed))


def load_configured_dataset(
    loaded: LoadedConfiguration, *, split: str | None = None,
    case_ids: Sequence[str] | None = None,
) -> LoadedDataset:
    """Resolve the selected loader and grader and validate the dataset now."""
    loader = loaded.registry.resolve("dataset_loaders", loaded.experiment.dataset.loader)
    grader = loaded.registry.resolve("graders", loaded.experiment.grader)
    return load_dataset(
        path=loaded.dataset_path, loader_id=loaded.experiment.dataset.loader,
        loader=loader, grader=grader, split=split, case_ids=case_ids,
    )
