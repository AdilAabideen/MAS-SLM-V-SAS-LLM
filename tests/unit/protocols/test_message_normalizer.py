"""Test Message Normalizer test coverage."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mas_slm_research.protocols.message_normalizer import (
    normalize_chat_messages,
    render_ai_message_for_provider,
    render_ai_tool_calls_json,
    render_tool_message_as_user_content,
    to_provider_messages,
)


@pytest.mark.unit
def test_ut_pro_024_merge_multiple_system_messages_into_one():
    """Handle ut pro 024 merge multiple system messages into one."""
    # Keep the main step clear.
    result = normalize_chat_messages(
        [
            {"role": "system", "content": "one"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "two"},
        ]
    )
    assert result[0] == {"role": "system", "content": "one\n\ntwo"}


@pytest.mark.unit
def test_ut_pro_025_merge_consecutive_user_messages():
    """Handle ut pro 025 merge consecutive user messages."""
    # Keep the main step clear.
    result = normalize_chat_messages(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    )
    assert result == [{"role": "user", "content": "a\n\nb"}]


@pytest.mark.unit
def test_ut_pro_027_preserve_mixed_non_system_ordering():
    """Handle ut pro 027 preserve mixed non system ordering."""
    # Keep the main step clear.
    result = normalize_chat_messages(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    )
    assert [item["role"] for item in result] == ["user", "assistant", "user"]


@pytest.mark.unit
def test_ut_pro_028_render_tool_message_into_provider_compatible_user_text():
    """Handle ut pro 028 render tool message into provider compatible user text."""
    # Keep the main step clear.
    msg = ToolMessage(content="done", tool_call_id="call_1", name="tool_a", status="success")
    rendered = render_tool_message_as_user_content(msg)
    assert "tool_a" in rendered
    assert "call_1" in rendered
    assert "done" in rendered


@pytest.mark.unit
def test_ut_pro_030_render_ai_message_with_both_content_and_tool_calls():
    """Handle ut pro 030 render ai message with both content and tool calls."""
    # Keep the main step clear.
    msg = AIMessage(content="thinking", tool_calls=[{"id": "call_1", "name": "tool_a", "args": {"x": 1}}])
    rendered = render_ai_message_for_provider(msg)
    assert "thinking" in rendered
    assert '"tool_calls"' in rendered


@pytest.mark.unit
def test_ut_pro_031_convert_mixed_langchain_messages_to_provider_messages():
    """Handle ut pro 031 convert mixed langchain messages to provider messages."""
    # Keep the main step clear.
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
        ToolMessage(content="done", tool_call_id="call_1", name="tool_a", status="success"),
    ]
    result = to_provider_messages(
        messages,
        allow_tool_messages=True,
        tool_message_error="nope",
        unsupported_type_label="test",
    )
    assert [item["role"] for item in result] == ["system", "user", "assistant", "user"]


@pytest.mark.unit
def test_ut_pro_032_reject_unsupported_message_type_in_provider_conversion():
    """Handle ut pro 032 reject unsupported message type in provider conversion."""
    # Keep the main step clear.
    with pytest.raises(ValueError):
        to_provider_messages(
            [object()],
            allow_tool_messages=False,
            tool_message_error="nope",
            unsupported_type_label="test",
        )
