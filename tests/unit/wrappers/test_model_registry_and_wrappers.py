"""Test Model Registry And Wrappers test coverage."""

from __future__ import annotations

import json
import threading
import time

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from mas_slm_research.model_registry import (
    FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES,
    ModelSpec,
    ProviderSettings,
    build_vllm_model,
    list_registered_models,
    resolve_model_spec,
)
from mas_slm_research.providers.medgemma_medical_chat import MedGemmaMedicalChatModel
from mas_slm_research.providers.vllm_chat import VLLMChat


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        """Handle the value."""
        # Keep the main step clear.
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        """Handle the value."""
        # Keep the main step clear.
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeClient:
    def __init__(self, response, recorder):
        """Handle the value."""
        # Keep the main step clear.
        self.response = response
        self.recorder = recorder

    def __enter__(self):
        """Handle the value."""
        # Keep the main step clear.
        return self

    def __exit__(self, exc_type, exc, tb):
        """Handle the value."""
        # Keep the main step clear.
        return False

    def post(self, url, headers, json):
        """Handle the value."""
        # Keep the main step clear.
        self.recorder.append({"url": url, "headers": headers, "json": json})
        return self.response


class SequencedFakeClient:
    def __init__(self, responses, recorder):
        """Handle the value."""
        # Keep the main step clear.
        self.responses = list(responses)
        self.recorder = recorder

    def __enter__(self):
        """Handle the value."""
        # Keep the main step clear.
        return self

    def __exit__(self, exc_type, exc, tb):
        """Handle the value."""
        # Keep the main step clear.
        return False

    def post(self, url, headers, json):
        """Handle the value."""
        # Keep the main step clear.
        self.recorder.append({"url": url, "headers": headers, "json": json})
        if not self.responses:
            raise AssertionError("No fake responses remaining")
        return self.responses.pop(0)


@tool
def lookup_value(value: str) -> dict:
    """Return a lookup payload."""
    # Keep the main step clear.
    return {"value": value}


def _patched_client(monkeypatch, response, recorder):
    """Handle client."""
    # Keep the main step clear.
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda timeout: FakeClient(response, recorder))


def _patched_sequenced_client(monkeypatch, responses, recorder):
    """Handle sequenced client."""
    # Keep the main step clear.
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda timeout: SequencedFakeClient(responses, recorder))


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_001_dr7_wrapper_builds_bearer_auth_header(monkeypatch, load_json_fixture):
    """Handle ut wrp 001 dr7 wrapper builds bearer auth header."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/dr7_native_tool_calls.json")), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test", api_key="secret")
    model._generate([HumanMessage(content="hi")], tools=[])
    assert recorded[0]["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.unit
@pytest.mark.wrapper
@pytest.mark.parametrize("configured_url", [
    "https://dr7.test/api/v1/medical",
    "https://dr7.test/api/v1/medical/",
    "https://dr7.test/api/v1/medical/chat/completions",
    "https://dr7.test/api/v1/medical/chat/completions/",
])
def test_dr7_accepts_base_or_complete_endpoint_without_appending_twice(
    monkeypatch, load_json_fixture, configured_url,
):
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/dr7_native_tool_calls.json")), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url=configured_url, api_key="secret")
    model._generate([HumanMessage(content="hi")], tools=[])
    assert recorded[0]["url"] == "https://dr7.test/api/v1/medical/chat/completions"


@pytest.mark.unit
@pytest.mark.wrapper
def test_dr7_html_error_reports_route_without_dumping_page(monkeypatch):
    recorded = []
    html = "<!DOCTYPE html><html><head><title>Not found</title></head><body>" + "page" * 1000
    _patched_client(monkeypatch, FakeResponse(status_code=404, text=html, headers={"content-type": "text/html"}), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test/api/v1/medical", api_key="secret")
    with pytest.raises(RuntimeError, match="Dr7 API error 404") as caught:
        model._generate([HumanMessage(content="hi")], tools=[])
    assert "check DR7_BASE_URL" in str(caught.value)
    assert "<!DOCTYPE" not in str(caught.value)
    assert len(str(caught.value)) < 200


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_003_dr7_wrapper_injects_tool_instruction_when_tools_bound(monkeypatch, load_json_fixture):
    """Handle ut wrp 003 dr7 wrapper injects tool instruction when tools bound."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/dr7_native_tool_calls.json")), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test", api_key="secret")
    model._generate([HumanMessage(content="hi")], tools=[{"function": {"name": "lookup_value", "description": "", "parameters": {}}}], tool_choice="any")
    assert "<tool_rules>" in recorded[0]["json"]["messages"][0]["content"]


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_004_dr7_wrapper_parses_provider_native_tool_calls(monkeypatch, load_json_fixture):
    """Handle ut wrp 004 dr7 wrapper parses provider native tool calls."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/dr7_native_tool_calls.json")), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test", api_key="secret")
    result = model._generate([HumanMessage(content="hi")], tools=[{"function": {"name": "lookup_value", "description": "", "parameters": {}}}])
    assert result.generations[0].message.tool_calls[0]["name"] == "lookup_value"


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_006_dr7_wrapper_raises_on_http_error(monkeypatch):
    """Handle ut wrp 006 dr7 wrapper raises on http error."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(status_code=500, payload={"detail": "bad"}, text="bad"), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test", api_key="secret")
    with pytest.raises(RuntimeError):
        model._generate([HumanMessage(content="hi")], tools=[])


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_007_dr7_wrapper_raises_on_invalid_json_body(monkeypatch):
    """Handle ut wrp 007 dr7 wrapper raises on invalid json body."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=ValueError("bad"), text="oops"), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test", api_key="secret")
    with pytest.raises(RuntimeError):
        model._generate([HumanMessage(content="hi")], tools=[])


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_008_dr7_wrapper_raises_on_missing_choices_message(monkeypatch):
    """Handle ut wrp 008 dr7 wrapper raises on missing choices message."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload={"choices": []}), recorded)
    model = MedGemmaMedicalChatModel(model="medgemma-4b-it", base_url="https://dr7.test", api_key="secret")
    with pytest.raises(RuntimeError):
        model._generate([HumanMessage(content="hi")], tools=[])


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_009_dr7_wrapper_retries_once_on_429_then_succeeds(monkeypatch, load_json_fixture):
    """Handle ut wrp 009 dr7 wrapper retries once on 429 then succeeds."""
    # Keep the main step clear.
    recorded = []
    sleeps = []
    _patched_sequenced_client(
        monkeypatch,
        [
            FakeResponse(
                status_code=429,
                payload={"error": {"message": "rate limited"}},
                text='{"error":{"message":"rate limited"}}',
            ),
            FakeResponse(payload=load_json_fixture("provider_payloads/dr7_native_tool_calls.json")),
        ],
        recorded,
    )
    monkeypatch.setattr("mas_slm_research.providers.medgemma_medical_chat.time.sleep", lambda seconds: sleeps.append(seconds))
    model = MedGemmaMedicalChatModel(
        model="medgemma-4b-it",
        base_url="https://dr7.test",
        api_key="secret",
        rate_limit_max_retries=2,
        rate_limit_backoff_initial_s=10.0,
        rate_limit_backoff_multiplier=2.0,
        rate_limit_backoff_max_s=40.0,
    )

    result = model._generate([HumanMessage(content="hi")], tools=[{"function": {"name": "lookup_value", "description": "", "parameters": {}}}])

    assert result.generations[0].message.tool_calls[0]["name"] == "lookup_value"
    assert sleeps == [10.0]
    assert len(recorded) == 2
    assert result.generations[0].message.response_metadata["network_attempts"] == 2


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_010_llama_wrapper_includes_auth_header_when_api_key_provided(monkeypatch, load_json_fixture):
    """Handle ut wrp 010 llama wrapper includes auth header when api key provided."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/llama_native_tool_calls.json")), recorded)
    model = VLLMChat(model="medgemma-4b-it", base_url="https://llama.test", api_key="secret")
    model._generate([HumanMessage(content="hi")], tools=[])
    assert recorded[0]["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_011_llama_wrapper_accepts_full_chat_completions_url(monkeypatch, load_json_fixture):
    """Handle ut wrp 011 llama wrapper accepts full chat completions url."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/llama_native_tool_calls.json")), recorded)
    model = VLLMChat(
        model="medgemma-4b-it",
        base_url="https://llama.test/v1/chat/completions",
    )
    model._generate([HumanMessage(content="hi")], tools=[])
    assert recorded[0]["url"] == "https://llama.test/v1/chat/completions"


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_011a_llama_wrapper_overrides_provider_model_by_agent_name(monkeypatch, load_json_fixture):
    """Handle ut wrp 011a llama wrapper overrides provider model by agent name."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/llama_native_tool_calls.json")), recorded)
    model = VLLMChat(
        model="medgemma-tool",
        base_url="https://llama.test",
        agent_model_id_overrides={
            "esi1_agent": "esi1-agent",
            "esi2_agent": "esi2-agent",
        },
    )
    model._generate([HumanMessage(content="hi")], tools=[], agent_name="esi2_agent")
    assert recorded[0]["json"]["model"] == "esi2-agent"


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_015_llama_wrapper_parses_tool_calls_when_present(monkeypatch, load_json_fixture):
    """Handle ut wrp 015 llama wrapper parses tool calls when present."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(payload=load_json_fixture("provider_payloads/llama_native_tool_calls.json")), recorded)
    model = VLLMChat(model="medgemma-4b-it", base_url="https://llama.test")
    result = model._generate([HumanMessage(content="hi")], tools=[{"function": {"name": "lookup_value", "description": "", "parameters": {}}}])
    assert result.generations[0].message.tool_calls[0]["name"] == "lookup_value"


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_017_llama_wrapper_raises_on_http_error(monkeypatch):
    """Handle ut wrp 017 llama wrapper raises on http error."""
    # Keep the main step clear.
    recorded = []
    _patched_client(monkeypatch, FakeResponse(status_code=500, payload={"detail": "bad"}, text="bad"), recorded)
    model = VLLMChat(model="medgemma-4b-it", base_url="https://llama.test")
    with pytest.raises(RuntimeError):
        model._generate([HumanMessage(content="hi")], tools=[])


@pytest.mark.unit
def test_ut_wrp_019_medgemma_default_registry_entry_uses_dr7():
    """Handle ut wrp 019 medgemma default registry entry uses dr7."""
    # Keep the main step clear.
    spec = resolve_model_spec("medgemma-4b-it")
    assert spec.provider == "dr7"


@pytest.mark.unit
def test_ut_wrp_020_medgemma_vllm_alias_keeps_provider_model_id():
    """Handle ut wrp 020 medgemma vllm alias keeps provider model id."""
    # Keep the main step clear.
    spec = resolve_model_spec("medgemma-4b-it-vllm")
    model = build_vllm_model(spec, ProviderSettings(vllm_base_url="https://llama.test", vllm_serialize_requests=False))
    assert isinstance(model, VLLMChat)
    assert model.model == "medgemma-4b-it"
    assert model.serialize_requests is False


@pytest.mark.unit
def test_ut_wrp_020a_finetuned_vllm_registry_applies_agent_model_overrides():
    """Handle ut wrp 020a finetuned vllm registry applies agent model overrides."""
    # Keep the main step clear.
    spec = resolve_model_spec("medgemma-4b-it-Finetuned")
    model = build_vllm_model(spec, ProviderSettings(vllm_base_url="https://llama.test"))
    assert isinstance(model, VLLMChat)
    assert model.model == "medgemma-tool"
    assert model.agent_model_id_overrides == FINETUNED_MULTI_AGENT_MODEL_ID_OVERRIDES


@pytest.mark.unit
def test_ut_wrp_022_vllm_registry_threads_serialize_flag():
    """Handle ut wrp 022 vllm registry threads serialize flag."""
    # Keep the main step clear.
    spec = resolve_model_spec("medgemma-4b-it-vllm")
    model = build_vllm_model(spec, ProviderSettings(
        vllm_base_url="https://llama.test", vllm_serialize_requests=True, vllm_timeout_s=123.0,
    ))
    assert isinstance(model, VLLMChat)
    assert model.serialize_requests is True
    assert model.timeout_s == 123.0


@pytest.mark.unit
@pytest.mark.wrapper
def test_ut_wrp_023_vllm_wrapper_serializes_requests_when_enabled(monkeypatch, load_json_fixture):
    """Handle ut wrp 023 vllm wrapper serializes requests when enabled."""
    # Keep the main step clear.
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    start_gate = threading.Event()

    class SlowFakeClient:
        def __enter__(self):
            """Handle the value."""
            # Keep the main step clear.
            return self

        def __exit__(self, exc_type, exc, tb):
            """Handle the value."""
            # Keep the main step clear.
            return False

        def post(self, url, headers, json):
            """Handle the value."""
            # Keep the main step clear.
            nonlocal active, max_active
            start_gate.wait(timeout=1.0)
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return FakeResponse(payload=load_json_fixture("provider_payloads/llama_native_tool_calls.json"))

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda timeout: SlowFakeClient())
    model = VLLMChat(
        model="medgemma-4b-it",
        base_url="https://llama.test",
        serialize_requests=True,
    )

    errors: list[Exception] = []

    def _worker():
        """Handle the value."""
        # Keep the main step clear.
        try:
            model._generate([HumanMessage(content="hi")], tools=[])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    start_gate.set()
    for thread in threads:
        thread.join(timeout=1.0)

    assert errors == []
    assert max_active == 1


@pytest.mark.unit
def test_ut_wrp_019_resolve_unknown_model_id_as_openai_provider_fallback():
    """Handle ut wrp 019 resolve unknown model id as openai provider fallback."""
    # Keep the main step clear.
    spec = resolve_model_spec("future-model")
    assert spec.provider == "openai"


@pytest.mark.unit
def test_ut_wrp_024_registered_models_list_is_stable_and_sorted():
    """Handle ut wrp 024 registered models list is stable and sorted."""
    # Keep the main step clear.
    ids = [spec.id for spec in list_registered_models()]
    assert ids == sorted(ids)


@pytest.mark.unit
def test_ut_wrp_025_explicit_custom_vllm_model_is_supported():
    """A researcher can select a custom vLLM checkpoint with explicit settings."""
    model = build_vllm_model(
        ModelSpec(id="custom-vllm", provider="vllm"),
        ProviderSettings(vllm_base_url="https://llama.test"),
    )
    assert isinstance(model, VLLMChat)
    assert model.model == "custom-vllm"


@pytest.mark.unit
def test_ut_wrp_026_legacy_llama_model_id_alias_resolves_to_vllm_spec():
    """Handle ut wrp 026 legacy llama model id alias resolves to vllm spec."""
    # Keep the main step clear.
    spec = resolve_model_spec("medgemma-4b-it-llama")
    assert spec.id == "medgemma-4b-it-vllm"
    assert spec.provider == "vllm"
