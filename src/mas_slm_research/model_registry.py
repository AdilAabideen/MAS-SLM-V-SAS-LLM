"""Model Registry module helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field



ModelProvider = Literal["openai", "dr7", "vllm"]


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None
    unit: str = "per_1k_tokens"


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    provider: ModelProvider
    model_category: str = "default"
    provider_model_id: Optional[str] = None
    description: Optional[str] = None
    context_length: Optional[int] = None
    max_tokens: Optional[int] = None
    pricing: Optional[ModelPricing] = None
    capabilities: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    supports_tools: bool = True
    default_temperature: float = 0.7


FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES: dict[str, str] = {
    "esi1_agent": "esi1-agent-075",
    "esi2_agent": "esi2-agent-025",
    "esi345_agent": "esi3-agent-075",
    "vitals_agent": "medgemma-4b-it",
    "doctor_agent": "medgemma-4b-it",
}


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    # OpenAI (known defaults)
    "gpt-4o-mini": ModelSpec(
        id="gpt-4o-mini",
        provider="openai",
        model_category="default",
        description="OpenAI GPT-4o Mini",
        supports_tools=True,
        default_temperature=0.7,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
    ),
    "gpt-4o": ModelSpec(
        id="gpt-4o",
        provider="openai",
        model_category="default",
        description="OpenAI GPT-4o",
        supports_tools=True,
        default_temperature=0.7,
        pricing=ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
    ),
     "gpt-5": ModelSpec(
        id="gpt-5",
        provider="openai",
        model_category="default",
        description="OpenAI GPT-5",
        supports_tools=True,
        default_temperature=0.7,
        pricing=ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
    ),
    "medgemma-4b-it": ModelSpec(
        id="medgemma-4b-it",
        provider="dr7",
        model_category="default",
        description=(
            "MedGemma 4B Instruction Tuned - Optimized for medical text understanding and generation"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
     "gemma-3-4b-it": ModelSpec(
        id="gemma-3-4b-it",
        provider="vllm",
        model_category="default",
        description=(
            "Gemma 4B Instruction Tuned - Optimized for text understanding and generation"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
    "medgemma-4b-it-vllm": ModelSpec(
        id="medgemma-4b-it-vllm",
        provider="vllm",
        model_category="default",
        provider_model_id="medgemma-4b-it",
        description=(
            "MedGemma 4B Instruction Tuned - Optimized for medical text understanding and generation Q_8 Quantization"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0.7,
    ),
    "medgemma-4b-it-ESI1": ModelSpec(
        id="medgemma-4b-it-ESI1",
        provider="vllm",
        model_category="default",
        provider_model_id="esi1-agent",
        description=(
            "MedGemma 4B Instruction Tuned - Optimized for medical text understanding and generation Q_8 Quantization"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
    "medgemma-4b-it-ESI2": ModelSpec(
        id="medgemma-4b-it-ESI2",
        provider="vllm",
        model_category="default",
        provider_model_id="esi2-agent",
        description=(
            "MedGemma 4B Instruction Tuned - Optimized for medical text understanding and generation Q_8 Quantization"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
    "medgemma-4b-it-ESI345": ModelSpec(
        id="medgemma-4b-it-ESI345",
        provider="vllm",
        model_category="default",
        provider_model_id="esi3-agent",
        description=(
            "MedGemma 4B Instruction Tuned - Optimized for medical text understanding and generation Q_8 Quantization"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
    "medgemma-4b-it-vllm-tool": ModelSpec(
        id="medgemma-4b-it-vllm-tool",
        provider="vllm",
        model_category="default",
        provider_model_id="medgemma-tool-050",
        description=(
            "MedGemma 4B Instruction Tuned - Optimized for medical text understanding and generation Q_8 Quantization"
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
    "medgemma-4b-it-Finetuned": ModelSpec(
        id="medgemma-4b-it-Finetuned",
        provider="vllm",
        model_category="default",
        provider_model_id="medgemma-tool",
        description=(
            "MedGemma 4B finetuned multi-agent routing entry served through the shared vLLM endpoint."
        ),
        context_length=8192,
        max_tokens=4096,
        pricing=ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        capabilities=[
            "medical_text_analysis",
            "symptom_analysis",
            "medical_qa",
            "clinical_reasoning",
        ],
        languages=["en", "zh"],
        supports_tools=True,
        default_temperature=0,
    ),
}


_MODEL_ID_ALIASES: dict[str, str] = {
    "medgemma-4b-it-llama": "medgemma-4b-it-vllm",
    "medgemma-4b-it-llama-tool": "medgemma-4b-it-vllm-tool",
}


def list_registered_models() -> list[ModelSpec]:
    """List registered models."""
    # Read the current list.
    return sorted(_MODEL_REGISTRY.values(), key=lambda s: s.id)


def get_registered_model_spec(model_id: str) -> ModelSpec:
    """Return registered model spec."""
    # Read the current value.
    return _MODEL_REGISTRY[model_id]


def resolve_model_spec(model_id: str) -> ModelSpec:
    """
    Resolve a model id to a spec.

    - If the model is registered, return the registered spec.
    - Otherwise, treat it as an OpenAI model id (no metadata).
    """
    # Pick the needed value.
    canonical_model_id = _MODEL_ID_ALIASES.get(model_id, model_id)
    spec = _MODEL_REGISTRY.get(canonical_model_id)
    if spec is not None:
        return spec

    return ModelSpec(
        id=model_id,
        provider="openai",
        description=None,
        supports_tools=True,
        default_temperature=0.7,
    )


def validate_model_for_agent(
    *,
    model_id: str,
    agent_name: str,
    requires_tools: bool,
) -> ModelSpec:
    """Validate model for agent."""
    # Fail fast on bad input.
    spec = resolve_model_spec(model_id)
    if requires_tools and not spec.supports_tools:
        raise ValueError(
            f"Model '{model_id}' does not support tool calling, but agent '{agent_name}' requires tools."
        )
    return spec


@dataclass(frozen=True, kw_only=True)
class ProviderSettings:
    """Caller-supplied provider connection values; no backend settings import."""

    azure_endpoint: str | None = None
    azure_api_key: str | None = field(default=None, repr=False)
    azure_api_version: str | None = None
    azure_deployment: str | None = None
    dr7_base_url: str | None = None
    dr7_api_key: str | None = field(default=None, repr=False)
    dr7_rate_limit_max_retries: int = 2
    dr7_rate_limit_backoff_initial_s: float = 10.0
    dr7_rate_limit_backoff_multiplier: float = 2.0
    dr7_rate_limit_backoff_max_s: float = 40.0
    vllm_base_url: str | None = None
    vllm_api_key: str | None = field(default=None, repr=False)
    vllm_serialize_requests: bool = False
    vllm_timeout_s: float = 60.0


def build_registered_model(model_id: str, provider_settings: ProviderSettings) -> BaseChatModel:
    """Build a preserved registered model from explicit connection settings."""
    spec = resolve_model_spec(model_id)
    if spec.provider == "openai":
        return _build_azure_model(spec, provider_settings)
    if spec.provider == "dr7":
        return _build_dr7_model(spec, provider_settings)
    if spec.provider == "vllm":
        return build_vllm_model(spec, provider_settings)
    raise RuntimeError(f"Unsupported model provider: {spec.provider}")


def _build_azure_model(spec: ModelSpec, connection: ProviderSettings) -> BaseChatModel:
    if not connection.azure_api_key or not connection.azure_endpoint or not connection.azure_api_version:
        raise ValueError("Azure endpoint, API key, and API version must be supplied")
    from langchain_openai import AzureChatOpenAI

    kwargs: dict[str, Any] = {
        "azure_deployment": spec.provider_model_id or connection.azure_deployment or spec.id,
        "azure_endpoint": connection.azure_endpoint,
        "api_version": connection.azure_api_version,
        "temperature": spec.default_temperature,
    }
    parameters = inspect.signature(AzureChatOpenAI).parameters
    if "api_key" in parameters:
        kwargs["api_key"] = connection.azure_api_key
    elif "openai_api_key" in parameters:
        kwargs["openai_api_key"] = connection.azure_api_key
    return AzureChatOpenAI(**kwargs)


def _build_dr7_model(spec: ModelSpec, connection: ProviderSettings) -> BaseChatModel:
    if not connection.dr7_api_key or not connection.dr7_base_url:
        raise ValueError("Dr7 base URL and API key must be supplied")
    from mas_slm_research.providers.medgemma_medical_chat import MedGemmaMedicalChatModel

    return MedGemmaMedicalChatModel(
        model=spec.provider_model_id or spec.id,
        base_url=connection.dr7_base_url,
        api_key=connection.dr7_api_key,
        temperature=spec.default_temperature,
        max_tokens=spec.max_tokens,
        rate_limit_max_retries=connection.dr7_rate_limit_max_retries,
        rate_limit_backoff_initial_s=connection.dr7_rate_limit_backoff_initial_s,
        rate_limit_backoff_multiplier=connection.dr7_rate_limit_backoff_multiplier,
        rate_limit_backoff_max_s=connection.dr7_rate_limit_backoff_max_s,
    )


def build_vllm_model(spec: ModelSpec, connection: ProviderSettings) -> BaseChatModel:
    if not connection.vllm_base_url:
        raise ValueError("vLLM base URL must be supplied")
    from mas_slm_research.providers.vllm_chat import VLLMChat

    overrides = FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES if spec.id == "medgemma-4b-it-Finetuned" else {}
    return VLLMChat(
        model=spec.provider_model_id or spec.id,
        base_url=connection.vllm_base_url,
        api_key=connection.vllm_api_key or "",
        agent_model_id_overrides=overrides,
        serialize_requests=connection.vllm_serialize_requests,
        temperature=spec.default_temperature,
        max_tokens=spec.max_tokens,
        timeout_s=connection.vllm_timeout_s,
    )
