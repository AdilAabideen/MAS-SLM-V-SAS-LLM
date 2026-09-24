"""Test Token Estimator test coverage."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from mas_slm_research.telemetry.token_estimator import TokenEstimator


def test_offline_default_never_initializes_a_downloadable_encoding(monkeypatch):
    from mas_slm_research.telemetry import token_estimator

    class DownloadingTokenizer:
        def get_encoding(self, name):
            raise AssertionError("tokenizer network path was entered")

    monkeypatch.setattr(token_estimator, "tiktoken", DownloadingTokenizer())
    assert TokenEstimator().estimate_text_tokens("abcdefgh") == 2


def test_opted_in_tokenizer_failure_falls_back_to_characters(monkeypatch):
    from mas_slm_research.telemetry import token_estimator

    class UnavailableTokenizer:
        def get_encoding(self, name):
            raise ConnectionError("offline encoding cache")

    monkeypatch.setattr(token_estimator, "tiktoken", UnavailableTokenizer())
    estimator = TokenEstimator(prefer_tiktoken=True)
    assert estimator.estimate_text_tokens("abcdefgh") == 2
    assert estimator.estimate_text_tokens("abcd") == 1


@pytest.mark.unit
def test_ut_tel_010_empty_text_returns_zero_tokens():
    """Handle ut tel 010 empty text returns zero tokens."""
    # Keep the main step clear.
    assert TokenEstimator().estimate_text_tokens("") == 0


@pytest.mark.unit
def test_ut_tel_011_fallback_estimator_returns_positive_value_for_non_empty_text():
    """Handle ut tel 011 fallback estimator returns positive value for non empty text."""
    # Keep the main step clear.
    estimator = TokenEstimator(chars_per_token_fallback=4)
    estimator._encoder_checked = True
    estimator._encoder = None
    assert estimator.estimate_text_tokens("abcd") == 1


@pytest.mark.unit
def test_ut_tel_012_message_serialization_is_deterministic_for_same_input():
    """Handle ut tel 012 message serialization is deterministic for same input."""
    # Keep the main step clear.
    estimator = TokenEstimator()
    msg = AIMessage(content="hello")
    assert estimator.serialize_message_for_estimation(msg) == estimator.serialize_message_for_estimation(msg)


@pytest.mark.unit
def test_ut_tel_013_ai_output_serialization_includes_tool_calls():
    """Handle ut tel 013 ai output serialization includes tool calls."""
    # Keep the main step clear.
    estimator = TokenEstimator()
    msg = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool_a", "args": {"x": 1}}])
    assert '"tool_calls"' in estimator.serialize_ai_output_for_estimation(msg)


@pytest.mark.unit
def test_ut_tel_014_tool_call_string_arguments_are_normalized_in_token_serialization():
    """Handle ut tel 014 tool call string arguments are normalized in token serialization."""
    # Keep the main step clear.
    estimator = TokenEstimator()
    msg = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [{"id": "call_1", "function": {"name": "tool_a", "arguments": "{\"x\": 1}"}}]},
    )
    serialized = estimator.serialize_ai_output_for_estimation(msg)
    assert '"provider_tool_calls"' in serialized
    assert '"x":1' in serialized


@pytest.mark.unit
def test_ut_tel_015_tool_messages_include_tool_metadata_in_serialization():
    """Handle ut tel 015 tool messages include tool metadata in serialization."""
    # Keep the main step clear.
    estimator = TokenEstimator()
    serialized = estimator.serialize_message_for_estimation(
        ToolMessage(content="done", tool_call_id="call_1", name="tool_a", status="success")
    )
    assert '"tool_call_id":"call_1"' in serialized
    assert '"status":"success"' in serialized
