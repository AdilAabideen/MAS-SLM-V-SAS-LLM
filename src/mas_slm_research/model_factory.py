"""Connect resolved model entries to registered provider factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .configuration import ResolvedModel
from .model_registry import (
    ModelSpec, ProviderSettings, build_model_from_spec, get_registered_model_spec,
)


ProviderFactory = Callable[[ResolvedModel, Mapping[str, str]], Any]


def builtin_provider_factory(provider_id: str) -> ProviderFactory:
    """Return a factory for a built-in provider adapter."""
    if provider_id not in {"openai", "azure_openai", "openai_api", "dr7", "vllm"}:
        raise ValueError(f"Unknown built-in provider {provider_id!r}")

    def build(model: ResolvedModel, environment: Mapping[str, str]) -> Any:
        if model.provider != provider_id:
            raise ValueError(f"Provider factory {provider_id!r} cannot build {model.provider!r}")
        if provider_id == "openai_api":
            return _build_standard_openai_model(model, environment)
        try:
            original = get_registered_model_spec(model.catalog or model.model_id)
        except KeyError:
            original = ModelSpec(id=model.model_id, provider=provider_id)
        if original.provider != provider_id and {original.provider, provider_id} != {"openai", "azure_openai"}:
            raise ValueError(f"Model {model.model_id!r} conflicts with provider {provider_id!r}")
        effective = original.model_copy(update={
            "provider": provider_id,
            "default_temperature": model.temperature if model.temperature is not None else original.default_temperature,
            "max_tokens": model.max_tokens if model.max_tokens is not None else original.max_tokens,
            "request_policy": model.request_policy,
        })
        key = environment.get(model.api_key_env) if model.api_key_env else None
        endpoint = environment.get(model.base_url_env) if model.base_url_env else None
        version = environment.get(model.api_version_env) if model.api_version_env else None
        settings = ProviderSettings(
            azure_endpoint=endpoint, azure_api_key=key, azure_api_version=version,
            dr7_base_url=endpoint, dr7_api_key=key,
            vllm_base_url=endpoint, vllm_api_key=key,
        )
        return build_model_from_spec(effective, settings)

    return build


def _build_standard_openai_model(model: ResolvedModel, environment: Mapping[str, str]) -> Any:
    """Use the standard OpenAI API without inheriting historical Azure catalog defaults."""
    key = environment.get(model.api_key_env) if model.api_key_env else None
    if not key or not key.strip():
        raise ValueError("OpenAI API key must be supplied through api_key_env")

    from langchain_openai import ChatOpenAI

    options: dict[str, Any] = {"model": model.model_id, "api_key": key}
    if model.temperature is not None:
        options["temperature"] = model.temperature
    if model.max_tokens is not None:
        options["max_tokens"] = model.max_tokens
    return ChatOpenAI(**options)
