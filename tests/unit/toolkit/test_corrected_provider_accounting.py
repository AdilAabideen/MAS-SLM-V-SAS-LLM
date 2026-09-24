"""Configured request policy and auditable provider usage contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
import langchain_openai
from langchain_core.messages import AIMessage, HumanMessage

from mas_slm_research.comparison import PriceRate, _measure
from mas_slm_research.contracts import RunStatus, RunTiming
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.model_registry import ModelSpec, ProviderSettings, build_model_from_spec
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.single_agent import SingleAgentRunner
from mas_slm_research.telemetry.usage_extractor import extract_provider_usage
from tests.doubles.fake_provider import FakeChatModel
from tests.integration.runtime.test_single_agent_runner import Answer, identity


class _Response:
    status_code = 200
    headers = {}
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "complete"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


def test_azure_provider_has_clear_registered_alias_without_guessing_standard_openai():
    registry = ComponentRegistry()
    register_builtin_components(registry)
    assert registry.resolve("providers", "azure_openai")
    with pytest.raises(ValueError, match="Azure endpoint"):
        build_model_from_spec(ModelSpec(id="deployment", provider="azure_openai"), ProviderSettings())

    captured = {}

    def fake_azure(*, api_key, **kwargs):
        captured.update(kwargs)
        captured["api_key"] = api_key
        return object()

    from pytest import MonkeyPatch
    with MonkeyPatch.context() as patcher:
        patcher.setattr(langchain_openai, "AzureChatOpenAI", fake_azure)
        build_model_from_spec(
            ModelSpec(id="deployment", provider="azure_openai", default_temperature=0.4, max_tokens=321),
            ProviderSettings(azure_endpoint="https://azure.invalid", azure_api_key="secret", azure_api_version="2024-02-01"),
        )
    assert captured["temperature"] == 0.4
    assert captured["max_tokens"] == 321
    assert captured["azure_deployment"] == "deployment"


def test_configured_vllm_uses_declared_decoding_and_exposes_actual_request(monkeypatch):
    sent = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers, json):
            sent.append(json)
            return _Response()

    monkeypatch.setattr(httpx, "Client", Client)
    model = build_model_from_spec(
        ModelSpec(id="research-checkpoint", provider="vllm", default_temperature=0.35,
                  max_tokens=789, request_policy="configured_v2"),
        ProviderSettings(vllm_base_url="https://local.invalid/v1"),
    )
    response = model._generate([HumanMessage(content="case")]).generations[0].message
    assert sent[0]["model"] == "research-checkpoint"
    assert sent[0]["temperature"] == 0.35
    assert sent[0]["max_tokens"] == 789
    assert not {"top_k", "top_p", "seed"} & sent[0].keys()
    assert response.response_metadata["provider_model_id"] == "research-checkpoint"
    assert response.response_metadata["request_parameters"] == {"temperature": 0.35, "max_tokens": 789}
    usage = extract_provider_usage(response)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (7, 3, 10)


def test_partial_and_nested_usage_never_invent_zero():
    partial = AIMessage(content="done", response_metadata={"token_usage": {"prompt_tokens": 9}})
    nested = AIMessage(content="done", response_metadata={"raw": {"usage": {"prompt_tokens": 4, "completion_tokens": 2}}})
    assert extract_provider_usage(partial).input_tokens == 9
    assert extract_provider_usage(partial).output_tokens is None
    assert (extract_provider_usage(nested).input_tokens, extract_provider_usage(nested).total_tokens) == (4, 6)

    model = FakeChatModel([AIMessage(
        content='{"recommendation":{"value":"ok"}}',
        response_metadata={"token_usage": {"prompt_tokens": 9}},
    )])
    run = asyncio.run(SingleAgentRunner(
        kernel_factory=lambda: AgentKernel(model=model, response_format=Answer), agent_name="baseline",
    ).run_case(identity=identity("partial-usage"), payload="case"))
    assert run.result.status == RunStatus.COMPLETED
    assert run.llm_calls[0]["usage_source"] == "mixed"
    assert run.llm_calls[0]["input_tokens"] == 9
    assert run.llm_calls[0]["input_token_source"] == "provider"
    assert run.llm_calls[0]["output_token_source"] == "estimated"


def test_runner_marks_retry_usage_unknown_but_counts_network_attempts():
    model = FakeChatModel([AIMessage(
        content='{"recommendation":{"value":"ok"}}',
        response_metadata={
            "provider_model_id": "checkpoint-2", "request_parameters": {"temperature": 0.2},
            "token_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            "network_attempts": 2,
        },
    )])
    run = asyncio.run(SingleAgentRunner(
        kernel_factory=lambda: AgentKernel(model=model, response_format=Answer), agent_name="baseline",
    ).run_case(identity=identity("retry-usage"), payload="case"))
    assert run.result.status == RunStatus.COMPLETED
    assert run.llm_calls[0]["network_attempts"] == 2
    assert run.llm_calls[0]["provider_model_id"] == "checkpoint-2"
    assert run.llm_calls[0]["request_parameters"] == {"temperature": 0.2}
    assert run.result.usage.input_tokens is None
    assert run.result.usage.total_tokens is None

    attempt = SimpleNamespace(
        execution=SimpleNamespace(llm_calls=run.llm_calls, tool_calls=run.tool_calls, events=run.events),
        result=SimpleNamespace(identity=run.result.identity, timing=RunTiming(wall_seconds=0.1)),
    )
    measurement = _measure(attempt, {"baseline": PriceRate(input_per_1k=0.1, output_per_1k=0.2)})
    assert measurement.llm_calls == 1
    assert measurement.network_attempts == 2
    assert measurement.input_tokens is None
    assert measurement.total_tokens is None
    assert measurement.cost_usd_estimate is None
