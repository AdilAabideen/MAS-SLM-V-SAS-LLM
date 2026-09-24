"""Construct case runners from validated registrations and split configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from pydantic import BaseModel

from .agents.definition import AgentDefinition
from .configuration import AgentConfig, LoadedConfiguration, ResolvedModel
from .kernel import AgentKernel
from .mas_contract import MASState
from .multi_agent import MultiAgentRunner
from .runtime.runtime_config import RuntimeConfig
from .single_agent import SingleAgentRunner
from .workflows.definition import WorkflowDefinition


ModelFactory = Callable[[ResolvedModel, str], Any]


@dataclass(frozen=True)
class ConfiguredSystems:
    sas_runner: SingleAgentRunner
    mas_runner: MultiAgentRunner
    sas_model: ResolvedModel
    mas_models: Mapping[str, ResolvedModel]
    workflow: WorkflowDefinition


def _runtime_config(agent: AgentConfig, *, multi_agent: bool) -> RuntimeConfig:
    values = asdict(RuntimeConfig(multi_agent=multi_agent))
    if agent.runtime:
        values.update(agent.runtime.model_dump(exclude_none=True))
    values["multi_agent"] = multi_agent
    values["persist_events"] = True
    values["print_events"] = False
    return RuntimeConfig(**values)


def _kernel_from_definition(
    definition: Any,
    *,
    model: Any,
    runtime_config: RuntimeConfig,
    workflow: WorkflowDefinition | None,
    handoff_schemas: Mapping[str, str],
    registry: Any,
) -> AgentKernel:
    kwargs = dict(
        model=model, runtime_config=runtime_config, workflow=workflow,
        handoff_schemas=handoff_schemas, registry=registry,
    )
    if isinstance(definition, type) and issubclass(definition, AgentDefinition):
        definition = definition()
    if callable(getattr(definition, "build_kernel", None)):
        kernel = definition.build_kernel(**kwargs)
    elif callable(definition):
        kernel = definition(**kwargs)
    else:
        raise TypeError("registered agent definition must be callable or implement build_kernel")
    if not isinstance(kernel, AgentKernel):
        raise TypeError("registered agent definition must build an extracted AgentKernel")
    return kernel


def _validator_for(definition: Any) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    schema = getattr(definition, "output_schema", None)
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise ValueError("finalizing agent definition must expose a Pydantic output_schema")

    def validate(value: Mapping[str, Any]) -> Mapping[str, Any]:
        return schema.model_validate(value).model_dump()

    return validate


def _scoped_state(role: str, state: MASState) -> MASState:
    scoped = dict(state)
    pending = state.get("pending_handoff")
    if not isinstance(pending, dict) or pending.get("target_agent") != role:
        pending = None
        for candidate in reversed(list(state.get("handoff_history") or [])):
            if isinstance(candidate, dict) and candidate.get("target_agent") == role:
                pending = candidate
                break
    scoped["pending_handoff"] = dict(pending) if isinstance(pending, dict) else None
    return scoped  # type: ignore[return-value]


def build_configured_systems(
    loaded: LoadedConfiguration,
    *,
    model_factory: ModelFactory | None = None,
    environment: Mapping[str, str] | None = None,
) -> ConfiguredSystems:
    """Resolve all effective roles now; instantiate models on each case attempt.

    ``model_factory`` is an optional offline/test injection taking the resolved
    model and role name. The default uses the explicitly registered provider.
    """
    spec = loaded.experiment
    registry = loaded.registry
    environment = environment if environment is not None else os.environ
    selected_workflow = loaded.workflow
    if loaded.workflow_file.definition:
        registered = registry.resolve("workflows", loaded.workflow_file.definition)
        selected_workflow = registered() if callable(registered) else registered
        if not isinstance(selected_workflow, WorkflowDefinition):
            raise TypeError("registered workflow must return WorkflowDefinition")
        if selected_workflow.model_dump() != loaded.workflow.model_dump():
            raise ValueError("registered workflow differs from validated declarative workflow")

    def construct_model(model: ResolvedModel, role: str) -> Any:
        if model_factory is not None:
            return model_factory(model, role)
        provider = registry.resolve("providers", model.provider)
        return provider(model, environment)

    sas_agent = spec.agents[spec.sas.agent]
    sas_definition = registry.resolve("agents", sas_agent.definition)
    sas_model = loaded.models[spec.sas.model]

    def build_sas_kernel() -> AgentKernel:
        return _kernel_from_definition(
            sas_definition,
            model=construct_model(sas_model, spec.sas.agent),
            runtime_config=_runtime_config(sas_agent, multi_agent=False),
            workflow=None, handoff_schemas={}, registry=registry,
        )

    sas_runner = SingleAgentRunner(
        kernel_factory=build_sas_kernel, agent_name=spec.sas.agent,
        output_validator=_validator_for(sas_definition),
    )

    role_definitions: dict[str, Any] = {}
    role_factories: dict[str, Callable[[], AgentKernel]] = {}
    role_models: dict[str, ResolvedModel] = {}
    for role, alias in spec.mas.agents.items():
        agent = spec.agents[alias]
        definition = registry.resolve("agents", agent.definition)
        declared_role = getattr(definition, "name", role)
        if declared_role != role:
            raise ValueError(
                f"mas.agents.{role}: definition {agent.definition!r} declares role {declared_role!r}"
            )
        model_name = spec.mas.model_overrides.get(role) or agent.model or spec.mas.default_model
        effective_model = loaded.models[model_name]
        handoff_schemas = loaded.workflow_file.handoff_schemas.get(role, {})
        role_definitions[role] = definition
        role_models[role] = effective_model

        def build_role(
            role: str = role,
            agent: AgentConfig = agent,
            definition: Any = definition,
            effective_model: ResolvedModel = effective_model,
            handoff_schemas: Mapping[str, str] = handoff_schemas,
        ) -> AgentKernel:
            return _kernel_from_definition(
                definition,
                model=construct_model(effective_model, role),
                runtime_config=_runtime_config(agent, multi_agent=True),
                workflow=selected_workflow, handoff_schemas=handoff_schemas,
                registry=registry,
            )

        role_factories[role] = build_role

    finalizers = selected_workflow.finalizing_agents
    validators = {role: _validator_for(role_definitions[role]) for role in finalizers}

    def validate_final(value: Mapping[str, Any]) -> Mapping[str, Any]:
        errors: list[str] = []
        for role in finalizers:
            try:
                return validators[role](value)
            except Exception as exc:
                errors.append(f"{role}: {exc}")
        raise ValueError("final output matched no finalizer schema: " + "; ".join(errors))

    payload_builders = {
        role: registry.resolve("payload_builders", payload.builder)
        for role, payload in loaded.workflow_file.payloads.items()
    }
    def build_payload(role: str, state: MASState) -> dict[str, Any]:
        payload = payload_builders[role](_scoped_state(role, state))
        if not isinstance(payload, dict) or not isinstance(payload.get("llm_payload"), dict):
            raise ValueError(f"payload builder for {role!r} must return a dict with llm_payload")
        return payload

    mas_runner = MultiAgentRunner(
        workflow=selected_workflow,
        role_factories=role_factories,
        payload_builder=build_payload,
        output_validator=validate_final,
        max_handoffs=spec.mas.max_handoffs,
        max_elapsed_seconds=spec.mas.max_elapsed_seconds,
    )
    return ConfiguredSystems(
        sas_runner=sas_runner, mas_runner=mas_runner, sas_model=sas_model,
        mas_models=MappingProxyType(role_models), workflow=selected_workflow,
    )
