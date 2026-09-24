"""Mock HTTP parity for the extracted provider adapters and explicit settings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from mas_slm_research.model_registry import ProviderSettings, build_registered_model
from mas_slm_research.configuration import ResolvedModel
from mas_slm_research.model_factory import builtin_provider_factory
from mas_slm_research.providers.medgemma_medical_chat import MedGemmaMedicalChatModel
from mas_slm_research.providers.vllm_chat import VLLMChat


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.text = json.dumps(payload)
        self.headers = {}

    def json(self):
        return self.payload


class RecordingClient:
    def __init__(self, response, requests):
        self.response = response
        self.requests = requests

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers, json):
        self.requests.append({"url": url, "headers": headers, "body": json})
        return self.response


def test_standard_openai_factory_uses_selected_model_and_key_without_azure(monkeypatch):
    import langchain_openai

    observed = []
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: observed.append(kwargs) or object())
    model = ResolvedModel(
        name="baseline", provider="openai_api", model_id="gpt-4o",
        model_env="BASELINE_MODEL_ID", catalog=None, api_key_env="OPENAI_API_KEY",
        base_url_env=None, api_version_env=None, temperature=None, max_tokens=None,
    )
    factory = builtin_provider_factory("openai_api")
    factory(model, {"OPENAI_API_KEY": "test-only-key"})
    assert observed == [{"model": "gpt-4o", "api_key": "test-only-key"}]

    configured = ResolvedModel(**{**vars(model), "temperature": 0.2, "max_tokens": 300})
    factory(configured, {"OPENAI_API_KEY": "test-only-key"})
    assert observed[-1] == {
        "model": "gpt-4o", "api_key": "test-only-key", "temperature": 0.2, "max_tokens": 300,
    }
    with pytest.raises(ValueError, match="OpenAI API key"):
        factory(model, {})
    with pytest.raises(ValueError, match="OpenAI API key"):
        factory(model, {"OPENAI_API_KEY": "  "})


def test_standard_openai_factory_does_not_replace_historical_azure_provider(monkeypatch):
    import langchain_openai

    observed = []
    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", lambda **kwargs: observed.append(kwargs) or object())
    model = ResolvedModel(
        name="baseline", provider="openai", model_id="gpt-4o",
        model_env=None, catalog=None, api_key_env="BASELINE_API_KEY",
        base_url_env="BASELINE_AZURE_ENDPOINT", api_version_env="BASELINE_AZURE_API_VERSION",
        temperature=None, max_tokens=None,
    )
    builtin_provider_factory("openai")(model, {
        "BASELINE_API_KEY": "test-only-key",
        "BASELINE_AZURE_ENDPOINT": "https://azure.invalid",
        "BASELINE_AZURE_API_VERSION": "2024-02-01",
    })
    assert observed[0]["azure_endpoint"] == "https://azure.invalid"


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["dr7", "vllm"])
def test_registered_provider_request_and_parse(provider, monkeypatch, load_json_fixture):
    response = FakeResponse(load_json_fixture(f"provider_payloads/{'dr7' if provider == 'dr7' else 'llama'}_native_tool_calls.json"))
    recorded = []
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda timeout: RecordingClient(response, recorded))
    if provider == "dr7":
        connection = ProviderSettings(dr7_base_url="https://dr7.test", dr7_api_key="private-key")
        extracted = build_registered_model("medgemma-4b-it", connection)
        assert isinstance(extracted, MedGemmaMedicalChatModel)
        arguments = {}
    else:
        connection = ProviderSettings(vllm_base_url="https://llama.test/v1", vllm_api_key="private-key")
        extracted = build_registered_model("medgemma-4b-it-Finetuned", connection)
        assert isinstance(extracted, VLLMChat)
        arguments = {"agent_name": "esi2_agent"}

    tools = [{"function": {"name": "lookup_value", "description": "Lookup", "parameters": {}}}]
    extracted_output = extracted._generate([HumanMessage(content="case")], tools=tools, tool_choice="any", **arguments)

    assert extracted_output.generations[0].message.tool_calls
    assert len(recorded) == 1
    assert recorded[0]["body"]["messages"][0]["content"].startswith("<tool_rules>")
    assert "private-key" not in json.dumps(recorded[0]["body"])
    assert recorded[0]["headers"]["Authorization"] == "Bearer private-key"
    assert "private-key" not in repr(connection)
    if provider == "vllm":
        assert recorded[0]["body"]["model"] == "esi2-agent-025"
        assert recorded[0]["body"]["max_tokens"] == 250
        assert recorded[0]["body"]["temperature"] == 0


@pytest.mark.unit
def test_extracted_provider_settings_are_required_for_builders():
    with pytest.raises(ValueError, match="Dr7 base URL and API key"):
        build_registered_model("medgemma-4b-it", ProviderSettings())
    with pytest.raises(ValueError, match="vLLM base URL"):
        build_registered_model("medgemma-4b-it-Finetuned", ProviderSettings())


@pytest.mark.unit
def test_provider_and_registry_modules_import_without_backend_settings():
    source_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from mas_slm_research.model_registry import ProviderSettings, resolve_model_spec
from mas_slm_research.providers.medgemma_medical_chat import MedGemmaMedicalChatModel
from mas_slm_research.providers.vllm_chat import VLLMChat
assert resolve_model_spec('medgemma-4b-it').provider == 'dr7'
assert 'app.config' not in sys.modules
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(source_root)], check=True, capture_output=True, text=True)
