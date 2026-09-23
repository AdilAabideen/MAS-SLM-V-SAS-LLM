"""Load and validate split, explicit experiment/workflow YAML before inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .registry import ComponentRegistry
from .workflows.definition import WorkflowDefinition


class ConfigurationError(ValueError):
    """A configuration file is invalid; the message includes its location."""


class _FrozenDict(dict):
    """A JSON-serializable mapping whose contents cannot be changed."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("configuration mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


class _StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_StrictSpec):
    provider: str = Field(min_length=1)
    model_env: str | None = None
    model_id: str | None = None
    catalog: str | None = None
    api_key_env: str | None = None
    base_url_env: str | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_model_identity(self) -> "ModelConfig":
        if sum(value is not None for value in (self.model_env, self.model_id, self.catalog)) != 1:
            raise ValueError("exactly one of model_env, model_id, or catalog is required")
        return self


class RuntimePolicyConfig(_StrictSpec):
    max_tool_calls_per_turn: int | None = Field(default=None, ge=1)
    require_final_answer_tool: bool | None = None
    allow_text_tool_recovery: bool | None = None
    malformed_tool_retry_enabled: bool | None = None
    max_malformed_tool_retries_per_tool: int | None = Field(default=None, ge=0)
    allow_plain_json_final_output: bool | None = None
    drop_extra_tool_calls: bool | None = None


class AgentConfig(_StrictSpec):
    definition: str = Field(min_length=1)
    model: str | None = None
    runtime: RuntimePolicyConfig | None = None


class SASConfig(_StrictSpec):
    agent: str = Field(min_length=1)
    model: str = Field(min_length=1)


class MASConfig(_StrictSpec):
    default_model: str = Field(min_length=1)
    agents: dict[str, str] = Field(min_length=1)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    workflow: str = Field(min_length=1)


class DatasetConfig(_StrictSpec):
    loader: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ReportingConfig(_StrictSpec):
    console: Literal["full", "summary", "none"] = "full"
    color: Literal["auto", "always", "never"] = "auto"
    output_directory: str | None = None


class ExperimentSpec(_StrictSpec):
    version: Literal[1]
    name: str = Field(min_length=1)
    extensions: tuple[str, ...] = ()
    models: dict[str, ModelConfig] = Field(min_length=1)
    agents: dict[str, AgentConfig] = Field(min_length=1)
    sas: SASConfig
    mas: MASConfig
    dataset: DatasetConfig
    grader: str = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1)
    schedule: Literal["system_major", "case_major"] = "system_major"
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


class WorkflowMetadataSpec(_StrictSpec):
    workflow_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str | None = None


class SourceSpec(_StrictSpec):
    source_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    agent_names: tuple[str, ...] = ()
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GateSpec(_StrictSpec):
    gate_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    required_sources: tuple[str, ...] = ()
    incoming_from: tuple[str, ...] = ()
    target_node: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PayloadSpec(_StrictSpec):
    builder: str = Field(min_length=1)
    input_schema: str | None = None


class WorkflowFileSpec(_StrictSpec):
    version: Literal[1]
    definition: str | None = None
    metadata: WorkflowMetadataSpec
    participating_agents: tuple[str, ...] = Field(min_length=1)
    sources: dict[str, SourceSpec] = Field(default_factory=dict)
    start_agents: tuple[str, ...] = Field(min_length=1)
    finalizing_agents: tuple[str, ...] = Field(min_length=1)
    allowed_handoffs: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    gates: dict[str, GateSpec] = Field(default_factory=dict)
    agent_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    workflow_metadata: dict[str, Any] = Field(default_factory=dict)
    payloads: dict[str, PayloadSpec] = Field(min_length=1)
    handoff_schemas: dict[str, dict[str, str]] = Field(default_factory=dict)

    def to_workflow(self) -> WorkflowDefinition:
        data = self.model_dump(exclude={"version", "definition", "payloads", "handoff_schemas"})
        return WorkflowDefinition.model_validate(data)


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    provider: str
    model_id: str
    model_env: str | None
    catalog: str | None
    api_key_env: str | None
    base_url_env: str | None
    temperature: float | None
    max_tokens: int | None


@dataclass(frozen=True)
class LoadedConfiguration:
    experiment_path: Path
    workflow_path: Path
    dataset_path: Path
    output_directory: Path | None
    experiment: ExperimentSpec
    workflow_file: WorkflowFileSpec
    workflow: WorkflowDefinition
    models: Mapping[str, ResolvedModel]
    registry: ComponentRegistry

    def safe_snapshot(self) -> dict[str, Any]:
        """Return a serializable, credential-free view of resolved choices."""
        return {
            "experiment_path": str(self.experiment_path),
            "workflow_path": str(self.workflow_path),
            "dataset_path": str(self.dataset_path),
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "experiment": self.experiment.model_dump(mode="json"),
            "workflow": self.workflow_file.model_dump(mode="json"),
            "resolved_models": {name: vars(model).copy() for name, model in self.models.items()},
            "registered_ids": self.registry.inventory(),
        }


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConfigurationError(
                f"unhashable YAML key at line {key_node.start_mark.line + 1}"
            ) from exc
        if duplicate:
            raise ConfigurationError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        parsed = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ConfigurationError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigurationError(f"{path}: root must be a YAML mapping")
    return parsed


def _parse_spec(model: type[_StrictSpec], raw: dict[str, Any], path: Path) -> Any:
    try:
        parsed = model.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "root"
        raise ConfigurationError(f"{path}: {location}: {first['msg']}") from exc
    return _freeze(parsed)


def _freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        for field in type(value).model_fields:
            object.__setattr__(value, field, _freeze(getattr(value, field)))
        return value
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _require_registered(
    registry: ComponentRegistry, kind: str, identifier: str, path: Path, field: str
) -> Any:
    try:
        return registry.resolve(kind, identifier)  # type: ignore[arg-type]
    except (KeyError, ValueError) as exc:
        raise ConfigurationError(f"{path}: {field}: {exc}") from exc


def _env_value(name: str, environment: Mapping[str, str], path: Path, field: str) -> str:
    if not name.isidentifier():
        raise ConfigurationError(f"{path}: {field}: invalid environment variable name {name!r}")
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}: {field}: environment variable {name!r} is missing or blank")
    return value


def _validate_routes(workflow: WorkflowDefinition, source: Path) -> None:
    names = set(workflow.participating_agents)
    if set(workflow.allowed_handoffs) != names:
        raise ConfigurationError(
            f"{source}: allowed_handoffs: expected one route list per participating agent"
        )
    starts = set(workflow.start_agents)
    finals = set(workflow.finalizing_agents)
    if not starts or not finals:
        raise ConfigurationError(f"{source}: start_agents/finalizing_agents: both must be nonempty")
    if len(starts) != len(workflow.start_agents) or len(finals) != len(workflow.finalizing_agents):
        raise ConfigurationError(f"{source}: start_agents/finalizing_agents: duplicate agent")
    visited: set[str] = set()
    visiting: set[str] = set()

    def walk(agent: str) -> None:
        if agent in visiting:
            raise ConfigurationError(f"{source}: allowed_handoffs: unbounded cycle at {agent!r}")
        if agent in visited:
            return
        visiting.add(agent)
        for target in workflow.allowed_targets_for(agent):
            walk(target)
        visiting.remove(agent)
        visited.add(agent)

    for agent in names:
        walk(agent)

    def reachable(start: str) -> set[str]:
        found = {start}
        pending = [start]
        while pending:
            for target in workflow.allowed_targets_for(pending.pop()):
                if target not in found:
                    found.add(target)
                    pending.append(target)
        return found

    reached = set().union(*(reachable(start) for start in starts))
    if finals - reached:
        raise ConfigurationError(
            f"{source}: finalizing_agents: unreachable finalizers {sorted(finals - reached)}"
        )
    if names - reached:
        raise ConfigurationError(
            f"{source}: start_agents: unreachable agents {sorted(names - reached)}"
        )
    for start in starts:
        if not reachable(start) & finals:
            raise ConfigurationError(
                f"{source}: finalizing_agents: no finalizer reachable from start {start!r}"
            )
    for gate_id, gate in workflow.gates.items():
        target = gate.target_node
        if target is None or target not in names or not gate.required_sources:
            raise ConfigurationError(
                f"{source}: gates.{gate_id}: an agent target and required_sources are required"
            )
        for source_id in gate.required_sources:
            members = workflow.source_agents(source_id)
            if not any(target in reachable(member) for member in members):
                raise ConfigurationError(
                    f"{source}: gates.{gate_id}.required_sources: {source_id!r} cannot reach {target!r}"
                )


def _validate_payloads(spec: WorkflowFileSpec, workflow: WorkflowDefinition, registry: ComponentRegistry, path: Path) -> None:
    names = set(workflow.participating_agents)
    if set(spec.payloads) != names:
        raise ConfigurationError(f"{path}: payloads: expected a builder entry for each participating agent")
    for role, payload in spec.payloads.items():
        _require_registered(registry, "payload_builders", payload.builder, path, f"payloads.{role}.builder")
        if payload.input_schema:
            _require_registered(registry, "schemas", payload.input_schema, path, f"payloads.{role}.input_schema")
    expected = {(source, target) for source, targets in workflow.allowed_handoffs.items() for target in targets}
    actual = {(source, target) for source, targets in spec.handoff_schemas.items() for target in targets}
    if actual != expected:
        raise ConfigurationError(
            f"{path}: handoff_schemas: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for source, targets in spec.handoff_schemas.items():
        for target, schema_id in targets.items():
            _require_registered(
                registry, "schemas", schema_id, path, f"handoff_schemas.{source}.{target}"
            )


def load_configuration(
    experiment_path: str | Path,
    *,
    registry: ComponentRegistry,
    environment: Mapping[str, str] | None = None,
) -> LoadedConfiguration:
    """Parse both files, resolve references, and reject invalid runs before model calls."""
    path = Path(experiment_path).expanduser().resolve()
    experiment = _parse_spec(ExperimentSpec, _read_yaml(path), path)
    for module_name in experiment.extensions:
        try:
            registry.load_module(module_name)
        except Exception as exc:
            raise ConfigurationError(f"{path}: extensions.{module_name}: {exc}") from exc
    workflow_path = (path.parent / experiment.mas.workflow).resolve()
    workflow_file = _parse_spec(WorkflowFileSpec, _read_yaml(workflow_path), workflow_path)
    try:
        workflow = workflow_file.to_workflow()
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "workflow"
        raise ConfigurationError(f"{workflow_path}: {location}: {first['msg']}") from exc
    _validate_routes(workflow, workflow_path)
    _validate_payloads(workflow_file, workflow, registry, workflow_path)
    if workflow_file.definition:
        registered = _require_registered(registry, "workflows", workflow_file.definition, workflow_path, "definition")
        if isinstance(registered, WorkflowDefinition) and registered.model_dump() != workflow.model_dump():
            raise ConfigurationError(
                f"{workflow_path}: definition: declared workflow differs from registered {workflow_file.definition!r}"
            )

    model_names = set(experiment.models)
    agents = experiment.agents
    if experiment.sas.agent not in agents:
        raise ConfigurationError(f"{path}: sas.agent: unknown agent {experiment.sas.agent!r}")
    if experiment.sas.model not in model_names:
        raise ConfigurationError(f"{path}: sas.model: unknown model {experiment.sas.model!r}")
    sas_declared_model = agents[experiment.sas.agent].model
    if sas_declared_model and sas_declared_model != experiment.sas.model:
        raise ConfigurationError(f"{path}: sas.model: conflicts with agents.{experiment.sas.agent}.model")
    if experiment.mas.default_model not in model_names:
        raise ConfigurationError(f"{path}: mas.default_model: unknown model {experiment.mas.default_model!r}")
    roles = set(workflow.participating_agents)
    if set(experiment.mas.agents) != roles:
        raise ConfigurationError(
            f"{path}: mas.agents: missing={sorted(roles - set(experiment.mas.agents))}, "
            f"extra={sorted(set(experiment.mas.agents) - roles)}"
        )
    if set(experiment.mas.model_overrides) - roles:
        raise ConfigurationError(f"{path}: mas.model_overrides: unknown workflow roles")
    for role, agent_id in experiment.mas.agents.items():
        if agent_id not in agents:
            raise ConfigurationError(f"{path}: mas.agents.{role}: unknown agent {agent_id!r}")
    for name, agent in agents.items():
        _require_registered(registry, "agents", agent.definition, path, f"agents.{name}.definition")
        if agent.model is not None and agent.model not in model_names:
            raise ConfigurationError(f"{path}: agents.{name}.model: unknown model {agent.model!r}")
    for role, model_name in experiment.mas.model_overrides.items():
        if model_name not in model_names:
            raise ConfigurationError(f"{path}: mas.model_overrides.{role}: unknown model {model_name!r}")
    _require_registered(registry, "dataset_loaders", experiment.dataset.loader, path, "dataset.loader")
    _require_registered(registry, "graders", experiment.grader, path, "grader")

    env = environment if environment is not None else os.environ
    resolved: dict[str, ResolvedModel] = {}
    for name, model in experiment.models.items():
        _require_registered(registry, "providers", model.provider, path, f"models.{name}.provider")
        catalog_spec = None
        if model.catalog:
            catalog_spec = _require_registered(registry, "models", model.catalog, path, f"models.{name}.catalog")
            if getattr(catalog_spec, "provider", model.provider) != model.provider:
                raise ConfigurationError(f"{path}: models.{name}.catalog: provider conflicts with catalog")
        model_id = (
            _env_value(model.model_env, env, path, f"models.{name}.model_env")
            if model.model_env else model.model_id or getattr(catalog_spec, "id", None)
        )
        if not model_id:
            raise ConfigurationError(f"{path}: models.{name}: no model ID resolved")
        for field, env_name in (("api_key_env", model.api_key_env), ("base_url_env", model.base_url_env)):
            if env_name:
                _env_value(env_name, env, path, f"models.{name}.{field}")
        resolved[name] = ResolvedModel(
            name=name, provider=model.provider, model_id=model_id,
            model_env=model.model_env, catalog=model.catalog,
            api_key_env=model.api_key_env, base_url_env=model.base_url_env,
            temperature=model.temperature, max_tokens=model.max_tokens,
        )
    return LoadedConfiguration(
        experiment_path=path,
        workflow_path=workflow_path,
        dataset_path=(path.parent / experiment.dataset.path).resolve(),
        output_directory=(path.parent / experiment.reporting.output_directory).resolve()
        if experiment.reporting.output_directory else None,
        experiment=experiment,
        workflow_file=workflow_file,
        workflow=workflow,
        models=MappingProxyType(resolved),
        registry=registry,
    )
