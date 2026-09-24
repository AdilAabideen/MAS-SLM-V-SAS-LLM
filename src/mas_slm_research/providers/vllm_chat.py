"""Vllm Chat ORM models."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Callable, Optional, Sequence, Union

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from mas_slm_research.protocols import (
    AllowedToolNames,
    coerce_bound_tools,
    extract_allowed_tool_names,
    inject_tool_instruction,
    normalize_chat_messages,
    to_provider_messages,
    normalize_tool_calls,
)

class _SerialRequestQueue:
    def __init__(self) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0

    @contextmanager
    def acquire(self):
        """Handle the value."""
        # Keep the main step clear.
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket:
                self._condition.wait()

        try:
            yield
        finally:
            with self._condition:
                self._serving_ticket += 1
                self._condition.notify_all()


_SERIAL_QUEUE_GUARD = threading.Lock()
_SERIAL_QUEUES_BY_BASE_URL: dict[str, _SerialRequestQueue] = {}


def _queue_key_for(base_url: str) -> str:
    """Handle key for."""
    # Keep the main step clear.
    return str(base_url or "").rstrip("/")


def _serial_queue_for(base_url: str) -> _SerialRequestQueue:
    """Handle queue for."""
    # Keep the main step clear.
    key = _queue_key_for(base_url)
    with _SERIAL_QUEUE_GUARD:
        queue = _SERIAL_QUEUES_BY_BASE_URL.get(key)
        if queue is None:
            queue = _SerialRequestQueue()
            _SERIAL_QUEUES_BY_BASE_URL[key] = queue
        return queue


class VLLMChat(BaseChatModel):
    """
    Minimal LangChain ChatModel wrapper around LLama Server chat completions endpoint.

    Notes:
    - v1 only supports non-streaming responses (`stream=false`).
    - Llama CPP function calling is not available; `bind_tools()` is emulated
      through a JSON tool-call contract so LangGraph agents can run tool loops.
    """

    model: str = Field(description="Llama server model id (e.g. 'medgemma-4b-it').")
    base_url: str = Field(
        default="https://aa4its07ztlqwx-8000.proxy.runpod.net/v1",
        description="Base URL for llama-server OpenAI-compatible API.",
    )
    api_key: Optional[str] = Field(
        default="",
        description="Optional API key. Local llama-server usually does not require one.",
        repr=False,
    )
    agent_model_id_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-agent override map for the outgoing provider model id.",
    )
    serialize_requests: bool = Field(
        default=False,
        description="When true, queue llama-server requests serially per backend URL.",
    )
    temperature: float = 0
    max_tokens: Optional[int] = 250
    timeout_s: float = 60.0

    def _llm_type(self) -> str:
        """Handle type."""
        # Keep the main step clear.
        return "vllm-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Handle params."""
        # Keep the main step clear.
        return {
            "model": self.model,
            "base_url": self.base_url,
        }

    def _resolve_provider_model(self, **kwargs: Any) -> str:
        """Resolve provider model."""
        # Pick the needed value.
        agent_name = str(kwargs.get("agent_name") or "").strip()
        if agent_name and self.agent_model_id_overrides:
            override = self.agent_model_id_overrides.get(agent_name)
            if isinstance(override, str) and override.strip():
                return override.strip()
        return self.model

    def _normalize_llama_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Normalize provider messages via shared protocol helper."""
        # Keep the output consistent.
        return normalize_chat_messages(messages)

    def _build_chat_completions_url(self) -> str:
        """Accept either a base API URL or a full chat completions endpoint."""
        # Build the next value.
        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return normalized + "/chat/completions"

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """
        Bind tools for LangChain/LangGraph compatibility.

        Llama Server does not natively support function-calling in this wrapper; bound
        tool schemas are used to instruct the model to emit JSON tool calls that are then
        converted into `AIMessage.tool_calls`.
        """
        # Keep the main step clear.
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # LangGraph's `create_react_agent` binds tools with no tool_choice; for this wrapper we
        # default to requiring a tool call so it doesn't "answer in one go" without
        # producing `AIMessage.tool_calls`.
        if tool_choice is None:
            tool_choice = "any"
        return self.bind(
            tools=formatted_tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Handle the value."""
        # Keep the main step clear.
        serialize_requests = bool(kwargs.get("serialize_requests", self.serialize_requests))
        if not serialize_requests:
            return self._generate_once(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        with _serial_queue_for(self.base_url).acquire():
            return self._generate_once(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    def _generate_once(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Handle once."""
        # Keep the main step clear.
        bound_tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        multi_agent = bool(kwargs.get("multi_agent"))
        handoff_names = list(kwargs.get("handoff_names") or [])
        tools = coerce_bound_tools(bound_tools)

        llama_messages = to_provider_messages(
            messages,
            allow_tool_messages=bool(tools),
            tool_message_error="VLLMChat does not support tool calling (ToolMessage present).",
            unsupported_type_label="Llama Server",
        )
        llama_messages = inject_tool_instruction(
            llama_messages,
            tools=tools,
            tool_choice=tool_choice,
            multi_agent=multi_agent,
            handoff_names=handoff_names,
            final_answer_tool_name="final_answer",
            highlight_final_answer=not multi_agent,
        )
        llama_messages = self._normalize_llama_messages(llama_messages)
        provider_model = self._resolve_provider_model(**kwargs)


        payload: dict[str, Any] = {
            "model": provider_model,
            "messages": llama_messages,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 42,
            "stream": False,
        }

        if self.max_tokens is not None:
            payload["max_tokens"] = 250
        if stop:
            payload["stop"] = stop

        url = self._build_chat_completions_url()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
        except Exception as e:
            raise RuntimeError(f"Llama Server request failed: {e}") from e

        if resp.status_code >= 400:
            detail = (resp.text or "").strip()
            if len(detail) > 5000:
                detail = detail[:5000] + "…(truncated)"
            raise RuntimeError(f"Llama Server API error {resp.status_code}: {detail}")

        try:
            data = resp.json()
        except Exception as e:
            text = (resp.text or "").strip()
            raise RuntimeError(f"Llama Server returned invalid JSON: {e}; body={text[:2000]}") from e

        try:
            choice_msg = data["choices"][0]["message"]
        except Exception as e:
            raise RuntimeError(
                f"Llama Server response missing choices/message/content: {data}"
            ) from e

        content = ""
        if isinstance(choice_msg, dict):
            content = (choice_msg.get("content") or "").strip()

        allowed_tool_names: AllowedToolNames = extract_allowed_tool_names(tools)

        tool_calls: list[dict[str, Any]] = []
        if tools and isinstance(choice_msg, dict):
            tool_calls = normalize_tool_calls(
                choice_msg.get("tool_calls", choice_msg.get("content")),
                allowed_tool_names=allowed_tool_names,
            )

        message = AIMessage(content=content, tool_calls=tool_calls)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation], llm_output={"raw": data})
