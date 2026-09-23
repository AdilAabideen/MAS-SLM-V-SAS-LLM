"""Protocols package exports."""

from .protocol_types import (
    AllowedToolNames,
    NormalizedToolCall,
    ToolCallParseResult,
    ToolCallParseSource,
)
from .message_normalizer import (
    normalize_chat_messages,
    render_ai_message_for_provider,
    render_ai_tool_calls_json,
    render_tool_message_as_user_content,
)
from .tool_call_recovery import (
    extract_tool_calls_with_priority,
    looks_like_malformed_tool_call_content,
    recover_from_fenced_json_text,
    recover_from_jsonl_text,
    recover_from_partial_json_text,
    recover_from_raw_json_text,
    recover_from_tool_calls_array_text,
    recover_tool_calls_from_content,
    to_legacy_recovery_output,
)
from .message_normalizer import to_provider_messages
from .tool_protocol import (
    build_tool_instruction,
    coerce_bound_tools,
    extract_allowed_tool_names,
    inject_tool_instruction,
    normalize_tool_calls,
    normalize_tool_calls_typed,
)
from .prompt_protocol import (
    build_system_prompt,
    join_prompt_sections,
    normalize_prompt_block,
)

__all__ = [
    "AllowedToolNames",
    "NormalizedToolCall",
    "ToolCallParseResult",
    "ToolCallParseSource",
    "normalize_chat_messages",
    "render_ai_message_for_provider",
    "render_ai_tool_calls_json",
    "render_tool_message_as_user_content",
    "to_provider_messages",
    "extract_tool_calls_with_priority",
    "looks_like_malformed_tool_call_content",
    "recover_from_fenced_json_text",
    "recover_from_jsonl_text",
    "recover_from_partial_json_text",
    "recover_from_raw_json_text",
    "recover_from_tool_calls_array_text",
    "recover_tool_calls_from_content",
    "to_legacy_recovery_output",
    "build_tool_instruction",
    "build_system_prompt",
    "coerce_bound_tools",
    "extract_allowed_tool_names",
    "inject_tool_instruction",
    "join_prompt_sections",
    "normalize_prompt_block",
    "normalize_tool_calls",
    "normalize_tool_calls_typed",
]
