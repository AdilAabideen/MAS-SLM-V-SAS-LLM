"""Runtime Types module helpers."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union
try:
    from typing import TypeAlias
except ImportError:  # Python < 3.10
    from typing_extensions import TypeAlias

from langchain_core.messages import AIMessage, BaseMessage

from app.agentic.protocols import NormalizedToolCall


# JSON-like runtime payload aliases
JSONScalar: TypeAlias = Union[str, int, float, bool, None]
JSONValue: TypeAlias = Union[JSONScalar, Dict[str, "JSONValue"], List["JSONValue"]]

# Stream mode aliases used by hand-rolled runtime emitters
StreamMode: TypeAlias = Literal["updates", "values"]
StreamModesInput: TypeAlias = Union[Sequence[StreamMode], StreamMode, None]

# Lightweight callback payload aliases
TelemetryContext: TypeAlias = Tuple[Optional[str], Optional[str]]
NormalizedToolCallDict: TypeAlias = Dict[str, Any]
ToolCallList: TypeAlias = List[NormalizedToolCallDict]
TypedToolCallList: TypeAlias = List[NormalizedToolCall]
# Safe migration alias: accept either legacy dict shape or typed dataclass.
ToolCallItem: TypeAlias = Union[NormalizedToolCallDict, NormalizedToolCall]
ToolCallItems: TypeAlias = List[ToolCallItem]


class BoundModel(Protocol):
    """Model contract required by AgentRunner."""
    def ainvoke(self, messages: list[BaseMessage]) -> Awaitable[AIMessage]: ...


class CurrentTelemetryContext(Protocol):
    """Return run-level telemetry identity."""
    def __call__(self) -> TelemetryContext: ...


class RenderSystemPrompt(Protocol):
    """Render current system prompt."""
    def __call__(self) -> str: ...


class PayloadToHumanContent(Protocol):
    """Convert runtime payload into user message text."""
    def __call__(self, payload: Any) -> str: ...


class InvokeWithTelemetry(Protocol):
    """Invoke model with telemetry instrumentation."""

    def __call__(
        self,
        *,
        call_kind: str,
        iteration: int,
        messages: list[BaseMessage],
        invoke_fn: Callable[[], Awaitable[AIMessage]],
    ) -> Awaitable[AIMessage]: ...


class NormalizeToolCall(Protocol):
    """Normalize one raw tool call object."""
    def __call__(self, obj: Mapping[str, Any]) -> NormalizedToolCallDict | None: ...


class RecoverToolCalls(Protocol):
    """Recover tool calls from assistant text."""
    def __call__(self, content: str) -> ToolCallList: ...


class BuildAIMessageWithToolCalls(Protocol):
    """Rebuild an AIMessage with normalized tool calls."""

    def __call__(
        self,
        source: AIMessage,
        tool_calls: ToolCallItems,
        *,
        extra_additional_kwargs: Mapping[str, Any] | None = None,
    ) -> AIMessage: ...


class LimitToolCalls(Protocol):
    """Split retained vs dropped tool calls."""
    def __call__(self, tool_calls: ToolCallItems) -> tuple[ToolCallItems, ToolCallItems]: ...


class JSONFromText(Protocol):
    """Parse JSON from text; return parsed value and raw stripped text."""
    def __call__(self, text: str) -> tuple[Any | None, str]: ...


class ParsedToPayloadJSON(Protocol):
    """Convert parsed object into event payload JSON shape."""
    def __call__(self, parsed: Any) -> dict[str, Any] | None: ...


class EmitEvent(Protocol):
    """Emit one runtime event."""

    def __call__(
        self,
        *,
        event_type: str,
        node_name: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        status: str | None = None,
        payload_json: dict[str, Any] | None = None,
        payload_text: str | None = None,
    ) -> None: ...


class ValuesStateBuilder(Protocol):
    """Build values-stream state payload."""

    def __call__(
        self,
        messages: list[BaseMessage],
        iteration: int,
        done: bool,
        output_json: Any,
        handoff_json: Any = None,
    ) -> dict[str, Any]: ...
