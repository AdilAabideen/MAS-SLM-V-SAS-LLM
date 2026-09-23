"""Short Term Memory module helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agentic.protocols import to_provider_messages
from app.agentic.telemetry import TokenEstimator


@dataclass(frozen=True)
class ShortTermMemoryConfig:
    """Policy toggles for short-term memory contents."""

    include_final_assistant_output: bool = False
    include_raw_provider_debug: bool = False
    verbose: bool = False
    log_token_estimates: bool = True
    log_message_preview: bool = True
    log_message_max_chars: int = 4000
    log_prefix: str = "[short_term_memory]"
    on_message_appended: Callable[[dict[str, Any]], None] | None = None


class ShortTermMemory:
    """
    Explicit short-term memory state contract for runtime context accumulation.

    Contract:
    - Include assistant tool-call messages.
    - Include tool results.
    - Include final assistant output only when configured.
    - Strip raw provider debug payloads unless explicitly configured.
    """

    _RAW_PROVIDER_DEBUG_KEYS = (
        "raw_tool_text",
        "raw_provider_content",
        "provider_debug",
        "provider_raw",
    )

    def __init__(
        self,
        *,
        config: ShortTermMemoryConfig | None = None,
        token_estimator: TokenEstimator | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.config = config or ShortTermMemoryConfig()
        self._messages: list[BaseMessage] = []
        self._token_estimator = token_estimator or TokenEstimator()
        self._log_fn = log_fn or print

    def append_assistant_tool_call(self, message: AIMessage) -> AIMessage:
        """Append an assistant message that contains one or more tool calls."""
        # Keep the next value explicit.
        tool_calls = list(getattr(message, "tool_calls", []) or [])
        if not tool_calls:
            raise ValueError("append_assistant_tool_call requires AIMessage.tool_calls to be non-empty.")

        cloned = self._clone_ai_message(message)
        self._messages.append(cloned)
        self._emit_append_event(kind="assistant_tool_call", message=cloned)
        return cloned

    def append_tool_result(self, message: ToolMessage) -> ToolMessage:
        """Append a tool result message."""
        # Keep the next value explicit.
        cloned = self._clone_tool_message(message)
        self._messages.append(cloned)
        self._emit_append_event(kind="tool_result", message=cloned)
        return cloned

    def append_final_assistant(self, message: AIMessage) -> AIMessage | None:
        """
        Optionally append final assistant output.

        Returns appended message when enabled, else returns None.
        """
        # Keep the next value explicit.
        if not self.config.include_final_assistant_output:
            return None

        cloned = self._clone_ai_message(message)
        self._messages.append(cloned)
        self._emit_append_event(kind="final_assistant", message=cloned)
        return cloned

    def messages(self) -> list[BaseMessage]:
        """Return a shallow copy of current short-term memory messages."""
        # Keep the main step clear.
        return list(self._messages)

    def clear(self) -> None:
        """Clear short-term memory state."""
        # Keep the main step clear.
        self._messages.clear()
        if self.config.verbose:
            self._log_fn(f"{self.config.log_prefix} cleared messages=0")

    def __len__(self) -> int:
        """Handle the value."""
        # Keep the main step clear.
        return len(self._messages)

    def _clone_ai_message(self, message: AIMessage) -> AIMessage:
        """Handle ai message."""
        # Keep the main step clear.
        additional_kwargs = self._sanitize_additional_kwargs(
            dict(getattr(message, "additional_kwargs", {}) or {})
        )
        return AIMessage(
            content=str(getattr(message, "content", "") or ""),
            tool_calls=list(getattr(message, "tool_calls", []) or []),
            additional_kwargs=additional_kwargs,
            response_metadata=getattr(message, "response_metadata", {}),
            id=getattr(message, "id", None),
            name=getattr(message, "name", None),
        )

    @staticmethod
    def _clone_tool_message(message: ToolMessage) -> ToolMessage:
        """Handle tool message."""
        # Keep the main step clear.
        kwargs: dict[str, Any] = {
            "content": str(getattr(message, "content", "") or ""),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "name": getattr(message, "name", None),
            "status": getattr(message, "status", None),
        }
        if hasattr(message, "artifact"):
            kwargs["artifact"] = getattr(message, "artifact")
        return ToolMessage(**kwargs)

    def _sanitize_additional_kwargs(self, additional_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Handle additional kwargs."""
        # Keep the main step clear.
        if self.config.include_raw_provider_debug:
            return dict(additional_kwargs)

        sanitized = dict(additional_kwargs)
        for key in self._RAW_PROVIDER_DEBUG_KEYS:
            sanitized.pop(key, None)
        return sanitized

    def _emit_append_event(self, *, kind: str, message: BaseMessage) -> None:
        """Emit append event."""
        # Keep events flowing.
        if not (self.config.verbose or self.config.on_message_appended is not None):
            return

        msg_tokens: int | None = None
        total_tokens: int | None = None
        if self.config.log_token_estimates:
            msg_tokens = self._token_estimator.estimate_messages_tokens([message])
            total_tokens = self._token_estimator.estimate_messages_tokens(self._messages)

        payload = {
            "event": "short_term_memory_message_appended",
            "kind": kind,
            "role": self._message_role(message),
            "message_count": len(self._messages),
            "message_tokens_estimate": msg_tokens,
            "short_term_memory_tokens_estimate": total_tokens,
            "provider_preview": self._provider_preview(message),
        }

        if self.config.verbose:
            self._log_fn(
                f"{self.config.log_prefix} append kind={kind} role={payload['role']} "
                f"messages={payload['message_count']} msg_tokens={msg_tokens} total_tokens={total_tokens}"
            )
            if self.config.log_message_preview and payload["provider_preview"] is not None:
                preview = str(payload["provider_preview"])
                max_chars = max(1, int(self.config.log_message_max_chars))
                if len(preview) > max_chars:
                    preview = preview[:max_chars] + "…(truncated)"
                self._log_fn(f"{self.config.log_prefix} message {preview}")

        if self.config.on_message_appended is not None:
            self.config.on_message_appended(payload)

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        """Handle role."""
        # Keep the main step clear.
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, ToolMessage):
            return "tool"
        return type(message).__name__.lower()

    @staticmethod
    def _provider_preview(message: BaseMessage) -> str | None:
        """Render one-message preview in provider message format."""
        # Keep the main step clear.
        try:
            rendered = to_provider_messages(
                [message],
                allow_tool_messages=True,
                tool_message_error="tool messages enabled for preview rendering",
                unsupported_type_label="short-term-memory-preview",
            )
        except Exception:
            return None

        if not rendered:
            return None
        role = str(rendered[0].get("role") or "")
        content = str(rendered[0].get("content") or "")
        return f"role={role} content={content}"
