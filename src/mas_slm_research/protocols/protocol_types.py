"""Protocol Types module helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Set


AllowedToolNames = Optional[Set[str]]


class ToolCallParseSource(str, Enum):
    """Origin category for a parsed tool call."""

    NATIVE_TOOL_CALLS = "native_tool_calls"
    PROVIDER_METADATA = "provider_metadata"
    FUNCTION_CALL = "function_call"
    TEXT_JSON = "text_json"
    TEXT_FENCED_JSON = "text_fenced_json"
    TEXT_JSONL = "text_jsonl"
    TEXT_PARTIAL_JSON = "text_partial_json"
    TEXT_TOOL_CALLS_ARRAY = "text_tool_calls_array"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormalizedToolCall:
    """Provider-agnostic normalized tool call payload."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    # Per-call provenance enables mixed-source batches to be represented.
    source: ToolCallParseSource = ToolCallParseSource.UNKNOWN
    # True when this call came from textual/heuristic recovery.
    recovered: bool = False


@dataclass(frozen=True)
class ToolCallParseResult:
    """Result bundle returned by tool-call parsing/recovery."""

    calls: list[NormalizedToolCall] = field(default_factory=list)
    # True when parsing produced at least one normalized tool call.
    succeeded: bool = False
    # Optional batch-level summary. Use `None` when calls have mixed provenance.
    source: ToolCallParseSource | None = ToolCallParseSource.UNKNOWN
    recovered: bool | None = False
    # For JSONL parsing, indicates whether every non-empty line parsed.
    all_lines_parsed: bool | None = None
