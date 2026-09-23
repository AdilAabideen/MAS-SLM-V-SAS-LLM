"""Tool Call Recovery module helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from .protocol_types import (
    AllowedToolNames,
    ToolCallParseResult,
    ToolCallParseSource,
)
from .tool_protocol import normalize_tool_calls_typed


def _extract_raw_calls(parsed: Any) -> list[Any]:
    """Extract raw calls."""
    # Pull out the needed value.
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        return list(parsed["tool_calls"])
    if isinstance(parsed, dict) and isinstance(parsed.get("name", parsed.get("tool_name")), str):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def recover_from_raw_json_text(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> ToolCallParseResult:
    """Handle from raw json text."""
    # Keep the main step clear.
    stripped = (content or "").strip()
    if not stripped:
        return ToolCallParseResult()

    try:
        parsed = json.loads(stripped)
    except Exception:
        return ToolCallParseResult()

    raw_calls = _extract_raw_calls(parsed)
    calls = normalize_tool_calls_typed(
        raw_calls,
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.TEXT_JSON,
        recovered=True,
    )
    return ToolCallParseResult(
        calls=calls,
        succeeded=bool(calls),
        source=ToolCallParseSource.TEXT_JSON,
        recovered=bool(calls),
    )


def recover_from_fenced_json_text(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> ToolCallParseResult:
    """Handle from fenced json text."""
    # Keep the main step clear.
    stripped = (content or "").strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return ToolCallParseResult()

    block = stripped[3:-3].strip()
    if block.lower().startswith("json"):
        block = block[4:].strip()

    try:
        parsed = json.loads(block)
    except Exception:
        return ToolCallParseResult()

    raw_calls = _extract_raw_calls(parsed)
    calls = normalize_tool_calls_typed(
        raw_calls,
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.TEXT_FENCED_JSON,
        recovered=True,
    )
    return ToolCallParseResult(
        calls=calls,
        succeeded=bool(calls),
        source=ToolCallParseSource.TEXT_FENCED_JSON,
        recovered=bool(calls),
    )


def recover_from_jsonl_text(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> ToolCallParseResult:
    """Handle from jsonl text."""
    # Keep the main step clear.
    stripped = (content or "").strip()
    if not stripped:
        return ToolCallParseResult()

    all_lines_parsed = True
    saw_non_empty_line = False
    raw_calls: list[Any] = []

    for ln in stripped.splitlines():
        line = ln.strip()
        if not line:
            continue
        saw_non_empty_line = True
        try:
            parsed_line = json.loads(line)
        except Exception:
            all_lines_parsed = False
            continue
        raw_calls.extend(_extract_raw_calls(parsed_line))

    if not saw_non_empty_line:
        return ToolCallParseResult()

    calls = normalize_tool_calls_typed(
        raw_calls,
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.TEXT_JSONL,
        recovered=True,
    )
    return ToolCallParseResult(
        calls=calls,
        succeeded=bool(calls),
        source=ToolCallParseSource.TEXT_JSONL,
        recovered=bool(calls),
        all_lines_parsed=all_lines_parsed,
    )


def _extract_balanced_json_segment(
    text: str,
    *,
    open_char: str,
    close_char: str,
    start_index: int,
) -> str | None:
    """Extract first balanced JSON-like segment from `start_index`."""
    # Pull out the needed value.
    if start_index < 0 or start_index >= len(text):
        return None
    if text[start_index] != open_char:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start_index, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            depth += 1
            continue
        if ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_index : i + 1]

    return None


def _extract_tool_call_objects_from_unbalanced_array(text: str, *, array_start: int) -> list[Any]:
    """
    Recover one or more balanced object entries from an unbalanced tool_calls array.

    This targets outputs like:
    {"tool_calls":[{"id":"call_1","name":"x","arguments":{...}}}

    where the outer array/object wrapper is truncated but the inner call object(s)
    are still structurally intact.
    """
    # Pull out the needed value.
    if array_start < 0 or array_start >= len(text) or text[array_start] != "[":
        return []

    i = array_start + 1
    recovered: list[Any] = []
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text):
            break
        if text[i] == "]":
            break
        if text[i] != "{":
            break

        obj_segment = _extract_balanced_json_segment(
            text,
            open_char="{",
            close_char="}",
            start_index=i,
        )
        if not obj_segment:
            break

        try:
            recovered.append(json.loads(obj_segment))
        except Exception:
            break

        i += len(obj_segment)

    return recovered


def recover_from_partial_json_text(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> ToolCallParseResult:
    """
    Recover from partial JSON by extracting first balanced object and parsing it.

    Useful when output contains valid tool-call JSON plus trailing garbage.
    """
    # Keep the main step clear.
    stripped = (content or "").strip()
    if not stripped:
        return ToolCallParseResult()

    first_obj_start = stripped.find("{")
    if first_obj_start < 0:
        return ToolCallParseResult()

    candidate = _extract_balanced_json_segment(
        stripped,
        open_char="{",
        close_char="}",
        start_index=first_obj_start,
    )
    if not candidate:
        return ToolCallParseResult()

    try:
        parsed = json.loads(candidate)
    except Exception:
        return ToolCallParseResult()

    raw_calls = _extract_raw_calls(parsed)
    calls = normalize_tool_calls_typed(
        raw_calls,
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.TEXT_PARTIAL_JSON,
        recovered=True,
    )
    return ToolCallParseResult(
        calls=calls,
        succeeded=bool(calls),
        source=ToolCallParseSource.TEXT_PARTIAL_JSON,
        recovered=bool(calls),
    )


def recover_from_tool_calls_array_text(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> ToolCallParseResult:
    """
    Recover by extracting a balanced `tool_calls` array and wrapping it as JSON.

    Useful when the model emits malformed object wrappers but the `tool_calls` array
    itself is still syntactically valid.
    """
    # Keep the main step clear.
    stripped = (content or "").strip()
    if not stripped:
        return ToolCallParseResult()

    key_match = re.search(r'"tool_calls"\s*:', stripped)
    if not key_match:
        return ToolCallParseResult()

    array_start = stripped.find("[", key_match.end())
    if array_start < 0:
        return ToolCallParseResult()

    array_segment = _extract_balanced_json_segment(
        stripped,
        open_char="[",
        close_char="]",
        start_index=array_start,
    )
    raw_calls: list[Any]
    if array_segment:
        candidate = f'{{"tool_calls":{array_segment}}}'
        try:
            parsed = json.loads(candidate)
        except Exception:
            return ToolCallParseResult()
        raw_calls = _extract_raw_calls(parsed)
    else:
        raw_calls = _extract_tool_call_objects_from_unbalanced_array(
            stripped,
            array_start=array_start,
        )
        if not raw_calls:
            return ToolCallParseResult()

    calls = normalize_tool_calls_typed(
        raw_calls,
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.TEXT_TOOL_CALLS_ARRAY,
        recovered=True,
    )
    return ToolCallParseResult(
        calls=calls,
        succeeded=bool(calls),
        source=ToolCallParseSource.TEXT_TOOL_CALLS_ARRAY,
        recovered=bool(calls),
    )


def recover_tool_calls_from_content(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> ToolCallParseResult:
    """Handle tool calls from content."""
    # Keep the main step clear.
    fenced_result = recover_from_fenced_json_text(content, allowed_tool_names=allowed_tool_names)
    if fenced_result.succeeded:
        return fenced_result

    raw_result = recover_from_raw_json_text(content, allowed_tool_names=allowed_tool_names)
    if raw_result.succeeded:
        return raw_result

    partial_result = recover_from_partial_json_text(content, allowed_tool_names=allowed_tool_names)
    if partial_result.succeeded:
        return partial_result

    tool_calls_array_result = recover_from_tool_calls_array_text(
        content,
        allowed_tool_names=allowed_tool_names,
    )
    if tool_calls_array_result.succeeded:
        return tool_calls_array_result

    jsonl_result = recover_from_jsonl_text(content, allowed_tool_names=allowed_tool_names)
    if jsonl_result.succeeded:
        return jsonl_result

    return ToolCallParseResult()


def looks_like_malformed_tool_call_content(
    content: str,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> bool:
    """
    Heuristic detector for malformed tool-call intent.

    Returns True only when content strongly suggests tool-call intent but could not be
    parsed/recovered into normalized calls.
    """
    # Keep the main step clear.
    text = (content or "").strip()
    if not text:
        return False

    if recover_tool_calls_from_content(text, allowed_tool_names=allowed_tool_names).succeeded:
        return False

    lowered = text.lower()
    if '"tool_calls"' in lowered or "'tool_calls'" in lowered:
        return True

    if "arguments" in lowered or '"name"' in lowered:
        if allowed_tool_names:
            lowered_names = {str(n).lower() for n in allowed_tool_names}
            if any(name in lowered for name in lowered_names):
                return True

    return False


def extract_tool_calls_with_priority(
    message: AIMessage,
    *,
    allowed_tool_names: AllowedToolNames = None,
    allow_text_recovery: bool = True,
) -> ToolCallParseResult:
    """Parse tool calls using a deterministic global priority order.

    Priority:
    1) native `message.tool_calls`
    2) provider metadata `additional_kwargs.tool_calls`
    3) provider metadata `additional_kwargs.function_call`
    4) text recovery from fenced/raw JSON
    5) text recovery from JSONL
    """

    # Pull out the needed value.
    native_calls = normalize_tool_calls_typed(
        list(getattr(message, "tool_calls", []) or []),
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.NATIVE_TOOL_CALLS,
        recovered=False,
    )
    if native_calls:
        return ToolCallParseResult(
            calls=native_calls,
            succeeded=True,
            source=ToolCallParseSource.NATIVE_TOOL_CALLS,
            recovered=False,
        )

    additional = getattr(message, "additional_kwargs", {}) or {}
    additional_tool_calls = additional.get("tool_calls")
    metadata_calls = normalize_tool_calls_typed(
        additional_tool_calls,
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.PROVIDER_METADATA,
        recovered=False,
    )
    if metadata_calls:
        return ToolCallParseResult(
            calls=metadata_calls,
            succeeded=True,
            source=ToolCallParseSource.PROVIDER_METADATA,
            recovered=False,
        )

    function_call = additional.get("function_call")
    function_calls = normalize_tool_calls_typed(
        [{"function": function_call}] if isinstance(function_call, dict) else [],
        allowed_tool_names=allowed_tool_names,
        source=ToolCallParseSource.FUNCTION_CALL,
        recovered=False,
    )
    if function_calls:
        return ToolCallParseResult(
            calls=function_calls,
            succeeded=True,
            source=ToolCallParseSource.FUNCTION_CALL,
            recovered=False,
        )

    if not allow_text_recovery:
        return ToolCallParseResult()

    content = str(getattr(message, "content", "") or "")
    recovered = recover_tool_calls_from_content(content, allowed_tool_names=allowed_tool_names)
    if recovered.succeeded:
        return recovered
    return ToolCallParseResult()


def to_legacy_recovery_output(
    result: ToolCallParseResult,
) -> tuple[list[dict[str, Any]], bool]:
    """Handle legacy recovery output."""
    # Keep the main step clear.
    calls = [
        {
            "id": call.id,
            "name": call.name,
            "args": call.args,
            "type": "tool_call",
        }
        for call in result.calls
    ]
    if not calls:
        return [], False

    if result.source == ToolCallParseSource.TEXT_JSONL and result.all_lines_parsed is not None:
        return calls, bool(result.all_lines_parsed)

    return calls, True
