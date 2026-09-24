"""Test Tool Protocol test coverage."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mas_slm_research.protocols.protocol_types import ToolCallParseSource
from mas_slm_research.protocols.tool_protocol import normalize_tool_calls_typed


@pytest.mark.unit
def test_ut_pro_001_normalize_openai_style_raw_tool_call():
    """Handle ut pro 001 normalize openai style raw tool call."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed(
        [{"id": "call_1", "function": {"name": "tool_a", "arguments": "{\"x\": 1}"}}],
        source=ToolCallParseSource.FUNCTION_CALL,
    )
    assert len(calls) == 1
    assert calls[0].name == "tool_a"
    assert calls[0].args == {"x": 1}
    assert calls[0].id == "call_1"
    assert calls[0].source == ToolCallParseSource.FUNCTION_CALL


@pytest.mark.unit
def test_ut_pro_004_parse_string_arguments_into_dict():
    """Handle ut pro 004 parse string arguments into dict."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed([{"id": "call_4", "name": "tool_a", "arguments": "{\"x\": 4}"}])
    assert calls[0].args == {"x": 4}


@pytest.mark.unit
def test_ut_pro_005_wrap_scalar_argument_payload_into_input():
    """Handle ut pro 005 wrap scalar argument payload into input."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed([{"id": "call_5", "name": "tool_a", "arguments": "[1, 2]"}])
    assert calls[0].args == {"input": [1, 2]}


@pytest.mark.unit
def test_ut_pro_006_invalid_json_argument_string_falls_back_safely():
    """Handle ut pro 006 invalid json argument string falls back safely."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed([{"id": "call_6", "name": "tool_a", "arguments": "{bad"}])
    assert calls[0].args == {"input": "{bad"}


@pytest.mark.unit
def test_ut_pro_007_reject_tool_with_missing_name():
    """Handle ut pro 007 reject tool with missing name."""
    # Keep the main step clear.
    assert normalize_tool_calls_typed([{"id": "call_7", "arguments": {"x": 1}}]) == []


@pytest.mark.unit
def test_ut_pro_008_reject_unknown_tool_when_allow_list_present():
    """Handle ut pro 008 reject unknown tool when allow list present."""
    # Keep the main step clear.
    assert normalize_tool_calls_typed(
        [{"id": "call_8", "name": "bad_tool", "arguments": {}}],
        allowed_tool_names={"good_tool"},
    ) == []


@pytest.mark.unit
def test_ut_pro_009_preserve_allowed_tool_when_allow_list_present():
    """Handle ut pro 009 preserve allowed tool when allow list present."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed(
        [{"id": "call_9", "name": "good_tool", "arguments": {}}],
        allowed_tool_names={"good_tool"},
    )
    assert len(calls) == 1
    assert calls[0].name == "good_tool"


@pytest.mark.unit
def test_ut_pro_010_generate_id_when_missing():
    """Handle ut pro 010 generate id when missing."""
    # Keep the main step clear.
    with patch("mas_slm_research.protocols.tool_protocol.uuid.uuid4") as mocked_uuid:
        mocked_uuid.return_value.hex = "abcdef1234567890"
        calls = normalize_tool_calls_typed([{"name": "tool_a", "arguments": {}}])
    assert calls[0].id == "call_abcdef123456"


@pytest.mark.unit
def test_ut_pro_011_deduplicate_repeated_ids():
    """Handle ut pro 011 deduplicate repeated ids."""
    # Keep the main step clear.
    with patch("mas_slm_research.protocols.tool_protocol.uuid.uuid4") as mocked_uuid:
        mocked_uuid.return_value.hex = "deduped1234567890"
        calls = normalize_tool_calls_typed(
            [
                {"id": "call_dup", "name": "tool_a", "arguments": {}},
                {"id": "call_dup", "name": "tool_b", "arguments": {}},
            ]
    )
    assert calls[0].id == "call_dup"
    assert calls[1].id == "call_deduped12345"


@pytest.mark.unit
def test_ut_pro_012_preserve_source_metadata():
    """Handle ut pro 012 preserve source metadata."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed(
        [{"id": "call_12", "name": "tool_a", "arguments": {}}],
        source=ToolCallParseSource.TEXT_JSON,
        recovered=True,
    )
    assert calls[0].source == ToolCallParseSource.TEXT_JSON
    assert calls[0].recovered is True


@pytest.mark.unit
def test_ut_pro_013_accept_tool_name_alias():
    """Handle ut pro 013 accept tool name alias."""
    # Keep the main step clear.
    calls = normalize_tool_calls_typed(
        [{"id": "call_13", "tool_name": "tool_a", "arguments": {"x": 13}}],
        source=ToolCallParseSource.TEXT_JSON,
        recovered=True,
    )
    assert len(calls) == 1
    assert calls[0].name == "tool_a"
    assert calls[0].args == {"x": 13}
