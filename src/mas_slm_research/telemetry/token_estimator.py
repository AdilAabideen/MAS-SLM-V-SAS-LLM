"""Token Estimator module helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency at runtime
    tiktoken = None


class TokenEstimator:
    """Token estimation helper with canonical serialization for telemetry fallback paths."""

    DEFAULT_ENCODING = "cl100k_base"
    CHARS_PER_TOKEN_FALLBACK = 4

    def __init__(
        self,
        *,
        encoding_name: str = DEFAULT_ENCODING,
        chars_per_token_fallback: int = CHARS_PER_TOKEN_FALLBACK,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.encoding_name = str(encoding_name or self.DEFAULT_ENCODING)
        self.chars_per_token_fallback = max(1, int(chars_per_token_fallback))
        self._encoder: Any | None = None
        self._encoder_checked = False

    def _get_encoder(self) -> Any | None:
        """Lazily initialize and cache a tokenizer encoder when available."""
        # Read the current value.
        if self._encoder_checked:
            return self._encoder

        self._encoder_checked = True
        if tiktoken is None:
            self._encoder = None
            return None

        try:
            self._encoder = tiktoken.get_encoding(self.encoding_name)
        except (LookupError, ValueError, TypeError):
            self._encoder = None
        return self._encoder

    def estimate_text_tokens(self, text: str) -> int:
        """Handle text tokens."""
        # Keep the main step clear.
        content = text or ""
        if not content:
            return 0

        encoder = self._get_encoder()
        if encoder is not None:
            try:
                return len(encoder.encode(content))
            except (ValueError, TypeError):
                pass

        return max(1, len(content) // self.chars_per_token_fallback)

    def estimate_messages_tokens(self, messages: list[BaseMessage]) -> int:
        """Handle messages tokens."""
        # Keep the main step clear.
        serialized = [self.serialize_message_for_estimation(msg) for msg in messages]
        return self.estimate_text_tokens("\n".join(serialized))

    def estimate_ai_output_tokens(self, msg: AIMessage) -> int:
        """Handle ai output tokens."""
        # Keep the main step clear.
        return self.estimate_text_tokens(self.serialize_ai_output_for_estimation(msg))

    def estimate_tool_result_tokens(self, content: str) -> int:
        """Handle tool result tokens."""
        # Keep the main step clear.
        return self.estimate_text_tokens(content)

    def estimate_jsonable_output_tokens(self, value: Any) -> int:
        """Handle jsonable output tokens."""
        # Keep the main step clear.
        return self.estimate_text_tokens(self._json_dumps(self._to_jsonable(value)))

    def serialize_message_for_estimation(self, message: BaseMessage) -> str:
        """Handle message for estimation."""
        # Keep the main step clear.
        canonical = self._canonical_message(message)
        return self._json_dumps(canonical)

    def serialize_ai_output_for_estimation(self, message: AIMessage) -> str:
        """Handle ai output for estimation."""
        # Keep the main step clear.
        canonical: dict[str, Any] = {
            "content": self._normalize_content(getattr(message, "content", "")),
        }
        canonical.update(self._canonical_ai_extras(message))

        return self._json_dumps(canonical)

    @staticmethod
    def _json_dumps(value: Any) -> str:
        """Handle dumps."""
        # Keep the main step clear.
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _normalize_content(content: Any) -> Any:
        """Normalize content."""
        # Keep the output consistent.
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            normalized_parts: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    normalized_parts.append({"type": "text", "text": item})
                    continue

                if not isinstance(item, Mapping):
                    continue

                part_type = str(item.get("type") or "")
                if part_type == "text":
                    normalized_parts.append({"type": "text", "text": str(item.get("text") or "")})
                elif part_type in {"image_url", "input_image"}:
                    url = item.get("image_url")
                    if isinstance(url, Mapping):
                        url = url.get("url")
                    normalized_parts.append({"type": part_type, "url": str(url or "")})
                elif part_type:
                    normalized_parts.append({"type": part_type})

            return normalized_parts

        return str(content)

    def _canonical_message(self, message: BaseMessage) -> dict[str, Any]:
        """Handle message."""
        # Keep the main step clear.
        role = self._message_role(message)
        payload: dict[str, Any] = {
            "role": role,
            "content": self._normalize_content(getattr(message, "content", "")),
        }

        if isinstance(message, AIMessage):
            payload.update(self._canonical_ai_extras(message))

        if isinstance(message, ToolMessage):
            payload.update(
                {
                    "name": getattr(message, "name", None),
                    "status": getattr(message, "status", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                }
            )

        return payload

    def _canonical_ai_extras(self, message: AIMessage) -> dict[str, Any]:
        """Canonicalize AI-specific fields shared across serializers."""
        # Keep the main step clear.
        extras: dict[str, Any] = {}

        tool_calls = self._canonical_tool_calls(list(getattr(message, "tool_calls", []) or []))
        if tool_calls:
            extras["tool_calls"] = tool_calls

        fn = self._extract_function_call_fields(getattr(message, "additional_kwargs", {}) or {})
        if fn:
            extras.update(fn)

        return extras

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        """Handle role."""
        # Keep the main step clear.
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, ToolMessage):
            return "tool"
        if isinstance(message, AIMessage):
            return "assistant"
        return type(message).__name__.lower()

    @staticmethod
    def _canonical_tool_calls(raw_calls: list[Any]) -> list[dict[str, Any]]:
        """Handle tool calls."""
        # Keep the main step clear.
        canonical: list[dict[str, Any]] = []
        for item in raw_calls:
            if not isinstance(item, Mapping):
                continue

            name = item.get("name")
            args = item.get("args", item.get("arguments", {}))
            call_id = item.get("id") or item.get("tool_call_id")

            if not name and isinstance(item.get("function"), Mapping):
                fn = item["function"]
                name = fn.get("name")
                args = fn.get("arguments", args)

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = {"raw_arguments": args}

            if not isinstance(args, Mapping):
                args = {"value": args}

            if name:
                canonical.append(
                    {
                        "id": str(call_id) if call_id is not None else None,
                        "name": str(name),
                        "args": dict(args),
                    }
                )

        return canonical

    def _extract_function_call_fields(self, additional_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Extract function call fields."""
        # Pull out the needed value.
        fields: dict[str, Any] = {}

        function_call = additional_kwargs.get("function_call")
        if isinstance(function_call, Mapping):
            fields["function_call"] = {
                "name": function_call.get("name"),
                "arguments": function_call.get("arguments"),
            }

        tool_calls = additional_kwargs.get("tool_calls")
        if isinstance(tool_calls, list):
            canonical = self._canonical_tool_calls(tool_calls)
            if canonical:
                fields["provider_tool_calls"] = canonical

        return fields

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Handle jsonable."""
        # Keep the main step clear.
        if isinstance(value, BaseModel):
            return value.model_dump()
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return value.model_dump()
            except (AttributeError, TypeError, ValueError):
                pass
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
