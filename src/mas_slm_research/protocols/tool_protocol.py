"""Tool Protocol module helpers."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional, Sequence

from .protocol_types import AllowedToolNames, NormalizedToolCall, ToolCallParseSource


def build_tool_instruction(
    tools: Sequence[dict[str, Any]],
    tool_choice: Optional[str],
    multi_agent: bool = False,
    handoff_names: Optional[list[str]] = None,
    *,
    final_answer_tool_name: str = "final_answer",
    highlight_final_answer: bool = True,
) -> str:
    """Build a canonical prompt block describing tool-call contract and schemas."""
    # Build the next value.
    handoff_names = list(handoff_names or [])
    prompt_tools: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", {})
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        prompt_tools.append(
            {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )

    parts = [
        "<tool_rules> - When tools are available, you MUST follow this tool-calling contract.",
        " - Return a SINGLE JSON object (no markdown, no extra text).",
        " - Tool-call format (exact):",
        ' - {"tool_calls":[{"id":"call_<unique_id>","name":"<tool_name>","arguments":{...}}]}',
        " - Do NOT output multiple JSON objects. If you need multiple tool calls, put them in the single tool_calls array.",
    ]

    if multi_agent and handoff_names:
        parts.extend(
            [
                " - When you are ready to transfer control, call exactly one matching handoff tool with the full payload.",
                f" - Available handoff tools: {', '.join(handoff_names)}",
                " - Tool call ids must be unique per call.",
            ]
        )
    else:
        parts.extend(
            [
                f" - When you are ready to finalize, call the {final_answer_tool_name} tool with the full final payload.",
                " - Tool call ids must be unique per call.",
            ]
        )

    if tool_choice == "any":
        parts.append("You must call at least one tool (tool call is required).")
    elif tool_choice and tool_choice not in {"auto", "none"}:
        parts.append(f'You must call the tool "{tool_choice}".')
    else:
        parts.append("If no tool is needed, respond with normal assistant text (not JSON).")

    parts.append("</tool_rules> \n<available_tools>")

    for tool in prompt_tools:
        name = str(tool.get("name", ""))
        desc = str(tool.get("description", ""))
        parameters = tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {}
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        if not multi_agent or len(handoff_names) <= 0:
            is_final_answer_tool = highlight_final_answer and name == final_answer_tool_name
            header = (
                "FINAL ANSWER TOOL -- CALL THIS WHEN YOU WANT TO OUTPUT THE FINAL ANSWER"
                if is_final_answer_tool
                else "TOOL"
            )
        else:
            is_handoff_tool = multi_agent and name in handoff_names
            header = (
                "HANDOFF TOOL - CALL THIS WHEN YOU WANT TO HANDOFF TO ANOTHER AGENT "
                if is_handoff_tool
                else "TOOL"
            )

        parts.append(
            "\n".join(
                [
                    "",
                    f"[{header}]",
                    f"TOOL NAME: {name}",
                    f"TOOL DESCRIPTION: {desc}",
                    f"TOOL PARAMETERS (arguments schema): {json.dumps(properties, ensure_ascii=False)}",
                    f"TOOL REQUIRED PARAMETERS: {json.dumps(required, ensure_ascii=False)}",
                ]
            )
        )

    parts.append("</available_tools>")
    return "\n".join(parts)


def coerce_bound_tools(bound_tools: Any) -> list[dict[str, Any]]:
    """Keep only dict-like bound tools for prompt/tool-call processing."""
    # Keep the main step clear.
    if not isinstance(bound_tools, list):
        return []
    return [tool for tool in bound_tools if isinstance(tool, dict)]


def extract_allowed_tool_names(tools: Sequence[dict[str, Any]]) -> AllowedToolNames:
    """Extract an allow-list of tool names from OpenAI-style tool schemas."""
    # Pull out the needed value.
    allowed_tool_names: AllowedToolNames = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool.get("function"), dict)
    }
    return {name for name in allowed_tool_names if isinstance(name, str) and name} or None


def inject_tool_instruction(
    messages: list[dict[str, str]],
    *,
    tools: Sequence[dict[str, Any]],
    tool_choice: Optional[str],
    multi_agent: bool = False,
    handoff_names: Optional[list[str]] = None,
    final_answer_tool_name: str = "final_answer",
    highlight_final_answer: bool = True,
) -> list[dict[str, str]]:
    """Inject tool instruction block into the leading system message."""
    # Keep the main step clear.
    if not tools:
        return messages

    tool_instruction = build_tool_instruction(
        tools,
        tool_choice,
        multi_agent=multi_agent,
        handoff_names=handoff_names,
        final_answer_tool_name=final_answer_tool_name,
        highlight_final_answer=highlight_final_answer,
    )

    if messages and messages[0].get("role") == "system":
        existing = (messages[0].get("content") or "").strip()
        messages[0]["content"] = f"{existing}\n\n{tool_instruction}" if existing else tool_instruction
        return messages
    return [{"role": "system", "content": tool_instruction}, *messages]


def normalize_tool_calls(
    raw_tool_calls: Any,
    *,
    allowed_tool_names: AllowedToolNames = None,
) -> list[dict[str, Any]]:
    """Legacy dict API for normalized tool calls.

    Keep this wrapper for compatibility while internal call-sites migrate to
    `normalize_tool_calls_typed`.
    """
    # Keep the output consistent.
    typed_calls = normalize_tool_calls_typed(
        raw_tool_calls,
        allowed_tool_names=allowed_tool_names,
    )
    return [
        {
            "id": call.id,
            "name": call.name,
            "args": call.args,
            "type": "tool_call",
        }
        for call in typed_calls
    ]


def normalize_tool_calls_typed(
    raw_tool_calls: Any,
    *,
    allowed_tool_names: AllowedToolNames = None,
    source: ToolCallParseSource = ToolCallParseSource.UNKNOWN,
    recovered: bool = False,
) -> list[NormalizedToolCall]:
    """Normalize provider/tool-call payloads into provider-agnostic dataclasses."""
    # Keep the output consistent.
    if not isinstance(raw_tool_calls, list):
        return []

    normalized: list[NormalizedToolCall] = []
    seen_ids: set[str] = set()
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, dict):
            continue

        raw_function = raw_call.get("function")
        if isinstance(raw_function, dict):
            name = raw_function.get("name", raw_function.get("tool_name"))
            raw_args = raw_function.get("arguments", {})
        else:
            name = raw_call.get("name", raw_call.get("tool_name"))
            raw_args = raw_call.get("arguments", raw_call.get("args", {}))

        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        if allowed_tool_names is not None and name not in allowed_tool_names:
            continue

        args: dict[str, Any]
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
                args = parsed_args if isinstance(parsed_args, dict) else {"input": parsed_args}
            except Exception:
                args = {"input": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        if call_id in seen_ids:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        seen_ids.add(call_id)

        normalized.append(
            NormalizedToolCall(
                id=call_id,
                name=name,
                args=args,
                source=source,
                recovered=recovered,
            )
        )

    return normalized
