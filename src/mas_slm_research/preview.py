"""Inspect the effective SAS/MAS construction without provider inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .configuration import LoadedConfiguration, ResolvedModel
from .configured_systems import build_configured_systems
from .model_registry import FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES, ModelSpec


class _PreviewModel:
    """Allow the real kernel to bind tools while forbidding any inference."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_PreviewModel":
        return self

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("configuration preview must never invoke a model")


@dataclass(frozen=True)
class ConfigurationPreview:
    concise: dict[str, Any]
    details: dict[str, Any]

    def render(self, *, details: bool = False) -> str:
        """Return an explicit, copyable JSON view for CLI and research review."""
        return json.dumps(self.details if details else self.concise, indent=2, ensure_ascii=False)


def _catalog_spec(loaded: LoadedConfiguration, model: ResolvedModel) -> ModelSpec | None:
    identifier = model.catalog or model.model_id
    try:
        value = loaded.registry.resolve("models", identifier)
    except KeyError:
        return None
    return value if isinstance(value, ModelSpec) else None


def _model_view(loaded: LoadedConfiguration, model: ResolvedModel, role: str) -> dict[str, Any]:
    catalog = _catalog_spec(loaded, model)
    if catalog is None and model.provider in {"openai", "azure_openai", "dr7", "vllm"}:
        catalog = ModelSpec(id=model.model_id, provider=model.provider)
    base_provider_id = catalog.provider_model_id if catalog and catalog.provider_model_id else model.model_id
    provider_id = base_provider_id
    if model.provider == "vllm" and model.model_id == "medgemma-4b-it-Finetuned":
        provider_id = FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES.get(role, base_provider_id)
    temperature = model.temperature if model.temperature is not None else (
        catalog.default_temperature if catalog else None
    )
    max_tokens = model.max_tokens if model.max_tokens is not None else (
        catalog.max_tokens if catalog else None
    )
    if model.provider == "vllm":
        actual_decoding: dict[str, Any] | None = (
            {"temperature": 0, "max_tokens": 250 if max_tokens is not None else None,
             "top_k": 1, "top_p": 1, "seed": 42, "policy": "legacy_vllm_request_v1"}
            if model.request_policy == "legacy_v1"
            else {"temperature": temperature, "max_tokens": max_tokens, "policy": "configured_v2"}
        )
    elif model.provider == "dr7":
        actual_decoding = {"temperature": temperature, "max_tokens": max_tokens}
    elif model.provider in {"openai", "azure_openai"}:
        actual_decoding = {"temperature": temperature, "max_tokens": max_tokens,
                           "policy": model.request_policy}
    else:
        actual_decoding = None
    return {
        "configuration_model": model.name,
        "provider": model.provider,
        "model_id": model.model_id,
        "request_policy": model.request_policy,
        "provider_model_id": provider_id,
        "catalog": model.catalog,
        "model_env": model.model_env,
        "api_key_env": model.api_key_env,
        "base_url_env": model.base_url_env,
        "api_version_env": model.api_version_env,
        "configured_decoding": {"temperature": model.temperature, "max_tokens": model.max_tokens},
        "effective_model_settings": {"temperature": temperature, "max_tokens": max_tokens},
        "actual_request_decoding": actual_decoding,
    }


def _kernel_view(kernel: Any, *, include_details: bool) -> dict[str, Any]:
    prompt = kernel._render_system_prompt()
    tools = [{
        "name": tool.name,
        "description": tool.description,
        "schema": tool.args_schema.model_json_schema() if tool.args_schema else None,
    } for tool in kernel.tools]
    view = {
        "agent_node": kernel.agent_node_name,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "ordered_tool_names": [tool["name"] for tool in tools],
        "response_schema": kernel.response_format.__name__
        if isinstance(kernel.response_format, type) else None,
        "runtime_policy": kernel.runtime_config.to_dict(),
        "handoff_tool_names": list(kernel._handoff_tool_names),
    }
    if include_details:
        view["assembled_prompt"] = prompt
        view["tools"] = tools
    return view


def inspect_configuration(loaded: LoadedConfiguration) -> ConfigurationPreview:
    """Build every configured kernel with an inert model, then describe it."""
    systems = build_configured_systems(
        loaded, model_factory=lambda model, role: _PreviewModel(),
    )
    sas_kernel = systems.sas_runner.kernel_factory()
    role_kernels = {
        role: factory() for role, factory in systems.mas_runner.role_factories.items()
    }
    spec = loaded.experiment
    sas_alias = spec.sas.agent
    sas_agent = spec.agents[sas_alias]

    def sas_view(*, details: bool) -> dict[str, Any]:
        return {
            "agent_alias": sas_alias,
            "definition": sas_agent.definition,
            "model_source": "sas.model",
            "model": _model_view(loaded, systems.sas_model, sas_alias),
            **_kernel_view(sas_kernel, include_details=details),
        }

    def mas_view(*, details: bool) -> dict[str, Any]:
        roles: dict[str, Any] = {}
        for role, alias in spec.mas.agents.items():
            agent = spec.agents[alias]
            source = (
                "mas.model_overrides" if role in spec.mas.model_overrides
                else f"agents.{alias}.model" if agent.model else "mas.default_model"
            )
            payload = loaded.workflow_file.payloads[role]
            roles[role] = {
                "agent_alias": alias,
                "definition": agent.definition,
                "model_source": source,
                "model": _model_view(loaded, systems.mas_models[role], role),
                "payload_builder": payload.builder,
                "input_schema": payload.input_schema,
                "outgoing_handoff_schemas": dict(loaded.workflow_file.handoff_schemas.get(role, {})),
                **_kernel_view(role_kernels[role], include_details=details),
            }
        return {
            "workflow_id": systems.workflow.metadata.workflow_id,
            "workflow_version": systems.workflow.metadata.version,
            "selected_definition": loaded.workflow_file.definition,
            "start_agents": list(systems.workflow.start_agents),
            "finalizing_agents": list(systems.workflow.finalizing_agents),
            "allowed_handoffs": {role: list(targets) for role, targets in systems.workflow.allowed_handoffs.items()},
            "source_groups": {
                source: list(item.agent_names) for source, item in systems.workflow.sources.items()
            },
            "gates": {
                gate: {
                    "required_sources": list(item.required_sources),
                    "incoming_from": list(systems.workflow.gate_incoming_nodes(gate)),
                    "target_node": item.target_node,
                    "metadata": dict(item.metadata),
                }
                for gate, item in systems.workflow.gates.items()
            },
            "roles": roles,
        }

    common = {
        "version": spec.version,
        "name": spec.name,
        "paths": {
            "experiment": str(loaded.experiment_path),
            "workflow": str(loaded.workflow_path),
            "dataset": str(loaded.dataset_path),
            "output_directory": str(loaded.output_directory) if loaded.output_directory else None,
        },
        "dataset_loader": spec.dataset.loader,
        "grader": spec.grader,
        "schedule": spec.schedule,
        "repetitions": spec.repetitions,
        "registered_components": loaded.registry.inventory(),
        "model_precedence": {
            "sas": "sas.model (agent model must agree if declared)",
            "mas": ["mas.model_overrides.<role>", "agents.<alias>.model", "mas.default_model"],
        },
    }
    return ConfigurationPreview(
        concise={**common, "sas": sas_view(details=False), "mas": mas_view(details=False)},
        details={**common, "sas": sas_view(details=True), "mas": mas_view(details=True)},
    )
