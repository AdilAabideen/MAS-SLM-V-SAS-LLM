"""Medgemma Medical Chat ORM models."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence, Union

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from mas_slm_research.protocols import (
    AllowedToolNames,
    coerce_bound_tools,
    extract_allowed_tool_names,
    inject_tool_instruction,
    normalize_chat_messages,
    to_provider_messages,
    normalize_tool_calls,
)


class MedGemmaMedicalChatModel(BaseChatModel):
    """
    Minimal LangChain ChatModel wrapper around Dr7's medical chat completions endpoint.

    Notes:
    - v1 only supports non-streaming responses (`stream=false`).
    - Dr7-native function calling is not available; `bind_tools()` is emulated
      through a JSON tool-call contract so LangGraph agents can run tool loops.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(description="Dr7 model id (e.g. 'medgemma-4b-it').")
    base_url: str = Field(description="Base URL, e.g. 'https://dr7.ai/api/v1/medical'.")
    api_key: str = Field(description="Dr7 API key (Bearer token).", repr=False)

    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_s: float = 60.0
    rate_limit_max_retries: int = 2
    rate_limit_backoff_initial_s: float = 10.0
    rate_limit_backoff_multiplier: float = 2.0
    rate_limit_backoff_max_s: float = 40.0

    def _llm_type(self) -> str:
        """Handle type."""
        # Keep the main step clear.
        return "dr7-medical-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Handle params."""
        # Keep the main step clear.
        return {"model": self.model, "base_url": self.base_url}

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """
        Bind tools for LangChain/LangGraph compatibility.

        Dr7 does not natively support function-calling in this wrapper; bound tool
        schemas are used to instruct the model to emit JSON tool calls that are then
        converted into `AIMessage.tool_calls`.
        """
        # Keep the main step clear.
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # LangGraph's `create_react_agent` binds tools with no tool_choice; for Dr7 we
        # default to requiring a tool call so it doesn't "answer in one go" without
        # producing `AIMessage.tool_calls`.
        if tool_choice is None:
            tool_choice = "any"
        return self.bind(
            tools=formatted_tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def _normalize_dr7_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Normalize provider messages via shared protocol helper."""
        # Keep the output consistent.
        return normalize_chat_messages(messages)

    def _compute_rate_limit_backoff_s(self, attempt_index: int, retry_after_header: Any = None) -> float:
        """Compute rate limit backoff s."""
        # Derive the needed value.
        if retry_after_header is not None:
            try:
                retry_after_s = float(retry_after_header)
            except (TypeError, ValueError):
                retry_after_s = None
            else:
                if retry_after_s >= 0:
                    return retry_after_s

        base_delay = max(0.0, float(self.rate_limit_backoff_initial_s))
        if base_delay == 0.0:
            return 0.0

        multiplier = max(1.0, float(self.rate_limit_backoff_multiplier))
        computed = base_delay * (multiplier ** max(0, attempt_index))
        max_delay = max(0.0, float(self.rate_limit_backoff_max_s))
        if max_delay > 0.0:
            return min(computed, max_delay)
        return computed

    def _post_with_rate_limit_retry(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        """Handle with rate limit retry."""
        # Keep the main step clear.
        max_retries = max(0, int(self.rate_limit_max_retries))
        with httpx.Client(timeout=self.timeout_s) as client:
            attempt_index = 0
            while True:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 429 or attempt_index >= max_retries:
                    return resp

                delay_s = self._compute_rate_limit_backoff_s(
                    attempt_index,
                    retry_after_header=getattr(resp, "headers", {}).get("Retry-After"),
                )
                if delay_s > 0:
                    time.sleep(delay_s)
                attempt_index += 1


    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Handle the value."""
        # Keep the main step clear.
        bound_tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        multi_agent = bool(kwargs.get("multi_agent"))
        handoff_names = list(kwargs.get("handoff_names") or [])
        tools = coerce_bound_tools(bound_tools)

        dr7_messages = to_provider_messages(
            messages,
            allow_tool_messages=bool(tools),
            tool_message_error="MedGemmaMedicalChatModel does not support tool calling (ToolMessage present).",
            unsupported_type_label="Dr7",
        )
        dr7_messages = inject_tool_instruction(
            dr7_messages,
            tools=tools,
            tool_choice=tool_choice,
            multi_agent=multi_agent,
            handoff_names=handoff_names,
            final_answer_tool_name="final_answer",
            highlight_final_answer=not multi_agent,
        )
        dr7_messages = self._normalize_dr7_messages(dr7_messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": dr7_messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._post_with_rate_limit_retry(
                url=url,
                headers=headers,
                payload=payload,
            )
        except Exception as e:
            raise RuntimeError(f"Dr7 request failed: {e}") from e

        if resp.status_code >= 400:
            detail = (resp.text or "").strip()
            if len(detail) > 5000:
                detail = detail[:5000] + "…(truncated)"
            raise RuntimeError(f"Dr7 API error {resp.status_code}: {detail}")

        try:
            data = resp.json()
        except Exception as e:
            text = (resp.text or "").strip()
            raise RuntimeError(f"Dr7 returned invalid JSON: {e}; body={text[:2000]}") from e

        try:
            choice_msg = data["choices"][0]["message"]
        except Exception as e:
            raise RuntimeError(f"Dr7 response missing choices/message/content: {data}") from e

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
