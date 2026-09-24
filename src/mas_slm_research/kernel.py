"""Handrolledagent module helpers."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Any, AsyncGenerator, Callable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool as lc_tool
from pydantic import BaseModel, ConfigDict

from mas_slm_research.protocols import (
    build_system_prompt,
    extract_tool_calls_with_priority,
    looks_like_malformed_tool_call_content,
)
from mas_slm_research.runtime.agent_runner import AgentRunner
from mas_slm_research.runtime.finalization_policy import FinalizationPolicy
from mas_slm_research.runtime.handoff_policy import HandoffPolicy
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.runtime.tool_executor import ToolExecutionTrace, ToolExecutor
from mas_slm_research.telemetry import (
    EventEmitter,
    LLMCallMetric,
    TelemetryEmitter,
    TokenEstimator,
    ToolExecutionMetric,
)
from mas_slm_research.telemetry.usage_extractor import extract_provider_usage
from mas_slm_research.workflows.definition import WorkflowDefinition


class AgentKernel:
    """Lightweight agent kernel with create_react_agent-like stream outputs.

    Stream yields `(mode, data)` tuples where:
    - mode == "updates" -> {node_name: {"messages": [message, ...]}}
    - mode == "values" -> full evolving state dict
    """

    def __init__(
        self,
        model: Any,
        tools: Sequence[Any] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        *,
        single_agent_prompt_addon: str | None = None,
        multi_agent_prompt_addon: str | None = None,
        prompt_extra_sections: Sequence[str] | None = None,
        final_answer_tool_name: str | None = "final_answer",
        llm_kwargs: dict[str, Any] | None = None,
        agent_node_name: str = "agent",
        tools_node_name: str = "tools",
        event_handlers: Sequence[Callable[[dict[str, Any]], None]] | None = None,
        llm_call_handlers: Sequence[Callable[[dict[str, Any]], None]] | None = None,
        tool_call_handlers: Sequence[Callable[[dict[str, Any]], None]] | None = None,
        handoff_tool_names: Sequence[str] | None = None,
        handoff_workflow: WorkflowDefinition | None = None,
        max_tool_calls: int = 2,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.model = self._build_model(model=model, llm_kwargs=llm_kwargs)
        self.tools: list[BaseTool] = self._coerce_tools(tools or [])

        self.system_prompt = (
            system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else "You are a helpful assistant."
        )
        self.single_agent_prompt_addon = (
            single_agent_prompt_addon.strip()
            if isinstance(single_agent_prompt_addon, str) and single_agent_prompt_addon.strip()
            else None
        )
        self.multi_agent_prompt_addon = (
            multi_agent_prompt_addon.strip()
            if isinstance(multi_agent_prompt_addon, str) and multi_agent_prompt_addon.strip()
            else None
        )
        self.prompt_extra_sections = [
            str(section).strip()
            for section in list(prompt_extra_sections or [])
            if isinstance(section, str) and section.strip()
        ]
        self.response_format = response_format
        self.final_answer_tool_name = final_answer_tool_name

        if not isinstance(max_tool_calls, int):
            raise TypeError("max_tool_calls must be an integer.")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1.")
        if runtime_config is not None and not isinstance(runtime_config, RuntimeConfig):
            raise TypeError("runtime_config must be a RuntimeConfig instance when provided.")
        self.runtime_config = runtime_config or RuntimeConfig(
            max_tool_calls_per_turn=max_tool_calls,
            require_final_answer_tool=True,
            allow_text_tool_recovery=True,
            allow_plain_json_final_output=True,
            drop_extra_tool_calls=True,
        )

        if self._should_add_final_answer_tool(handoff_tool_names=handoff_tool_names):
            existing = {t.name for t in self.tools}
            if self.final_answer_tool_name not in existing:
                self.tools.append(self._build_final_answer_tool())

        self.tools_by_name: dict[str, BaseTool] = {t.name: t for t in self.tools}
        self.agent_node_name = agent_node_name
        self.tools_node_name = tools_node_name
        self.max_tool_calls = int(self.runtime_config.max_tool_calls_per_turn)
        self._token_estimator = TokenEstimator()
        self._events = EventEmitter()
        self._telemetry = TelemetryEmitter()
        self._finalization_policy = FinalizationPolicy(
            config=self.runtime_config,
            final_answer_tool_name=self.final_answer_tool_name,
            validate_output=self._schema_validation_error_for_output,
        )
        self._handoff_tool_names = list(handoff_tool_names or [])
        self._handoff_policy = HandoffPolicy(handoff_tool_names=handoff_tool_names, workflow=handoff_workflow)
        self._tool_executor = ToolExecutor(
            tools_by_name=self.tools_by_name,
            estimate_tool_result_tokens=self._token_estimator.estimate_tool_result_tokens,
            emit_trace=self._emit_tool_execution_trace,
        )

        self.set_event_handlers(event_handlers)
        self.set_llm_call_handlers(llm_call_handlers)
        self.set_tool_call_handlers(tool_call_handlers)

        if self.tools:
            self.bound_model = self._bind_model_tools()
        else:
            self.bound_model = self.model
        self._agent_runner = AgentRunner(
            bound_model=self.bound_model,
            runtime_config=self.runtime_config,
            agent_node_name=self.agent_node_name,
            tools_node_name=self.tools_node_name,
            finalization_policy=self._finalization_policy,
            handoff_policy=self._handoff_policy,
            tool_executor=self._tool_executor,
            current_telemetry_context=self._telemetry.current_context,
            render_system_prompt=self._render_system_prompt,
            payload_to_human_content=self._payload_to_human_content,
            ainvoke_with_telemetry=self._ainvoke_with_telemetry,
            ai_message_with_tool_calls=self._ai_message_with_tool_calls,
            limit_tool_calls=self._limit_tool_calls,
            json_from_text=self._json_from_text,
            parsed_to_payload_json=self._parsed_to_payload_json,
            emit_event=self._emit_event,
            values_state=self._values_state,
        )

    @staticmethod
    def _build_model(model: Any, llm_kwargs: dict[str, Any] | None) -> Any:
        """Use an explicitly constructed provider; never read backend settings."""
        if isinstance(model, str) or not callable(getattr(model, "ainvoke", None)):
            raise TypeError("model must be an injected chat model with ainvoke")
        if llm_kwargs:
            raise ValueError("configure provider options on the injected model")
        return model

    @staticmethod
    def _coerce_tools(raw_tools: Sequence[Any]) -> list[BaseTool]:
        """Handle tools."""
        # Keep the main step clear.
        normalized: list[BaseTool] = []
        for item in raw_tools:
            if isinstance(item, BaseTool):
                normalized.append(item)
            elif callable(item):
                normalized.append(lc_tool(item))
            else:
                raise TypeError(f"Unsupported tool type: {type(item)!r}")
        return normalized

    def _supports_runtime_tool_binding_hints(self) -> bool:
        """
        Return whether this model wrapper explicitly supports extra bind-time runtime hints.

        These hints are consumed by the custom Dr7 and llama wrappers, but should not be
        forwarded to generic LangChain providers such as ChatOpenAI.
        """
        # Keep the main step clear.
        if bool(getattr(self.model, "supports_runtime_tool_binding_hints", False)):
            return True

        llm_type_getter = getattr(self.model, "_llm_type", None)
        if not callable(llm_type_getter):
            return False

        try:
            llm_type = str(llm_type_getter() or "").strip().lower()
        except Exception:
            return False

        return llm_type in {"dr7-medical-chat", "vllm-chat"}

    def _bind_model_tools(self) -> Any:
        """
        Bind tools to the underlying model.

        Some lightweight test doubles only accept `(tools, tool_choice=...)`, while
        the production wrappers accept extra runtime hints such as `agent_name`,
        `multi_agent`, and `handoff_names`.
        """
        # Keep the main step clear.
        if not self._supports_runtime_tool_binding_hints():
            return self.model.bind_tools(
                self.tools,
                tool_choice="any",
            )

        try:
            return self.model.bind_tools(
                self.tools,
                tool_choice="any",
                agent_name=self.agent_node_name,
                multi_agent=bool(self.runtime_config.multi_agent),
                handoff_names=self._handoff_tool_names,
            )
        except TypeError:
            return self.model.bind_tools(
                self.tools,
                tool_choice="any",
            )

    @staticmethod
    def _fallback_final_answer_schema() -> type[BaseModel]:
        """Handle final answer schema."""
        # Keep the main step clear.
        class FinalAnswerPayload(BaseModel):
            model_config = ConfigDict(extra="allow")

        return FinalAnswerPayload

    def _should_add_final_answer_tool(self, *, handoff_tool_names: Sequence[str] | None) -> bool:
        """Handle add final answer tool."""
        # Keep the main step clear.
        if not self.final_answer_tool_name or self.response_format is None:
            return False

        has_handoff_tools = bool(list(handoff_tool_names or []))
        if (
            self.runtime_config.multi_agent
            and self.runtime_config.disable_final_answer_tool_when_handoff_tools_present
            and has_handoff_tools
        ):
            return False

        return True

    def _build_final_answer_tool(self) -> BaseTool:
        """Build final answer tool."""
        # Build the next value.
        if not self.final_answer_tool_name:
            raise ValueError("final_answer_tool_name must be set to build final_answer tool.")

        strict_schema = isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel)
        schema_model = self.response_format if strict_schema else self._fallback_final_answer_schema()
        tool_name = self.final_answer_tool_name

        @lc_tool(tool_name, args_schema=schema_model)
        def _final_answer(**kwargs: Any) -> dict[str, Any]:
            """Signal completion and return the final structured payload."""
            # Keep the main step clear.
            if not kwargs:
                raise ValueError("final_answer requires a non-empty JSON object.")

            if strict_schema:
                validated = schema_model.model_validate(kwargs)
                return validated.model_dump()

            return dict(kwargs)

        return _final_answer

    def _schema_validation_error_for_output(self, value: Any) -> str | None:
        """Validate final output against configured response format when strict schema is available."""
        # Keep the main step clear.
        if not (isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel)):
            return None
        try:
            self.response_format.model_validate(value)
            return None
        except Exception as exc:
            return str(exc)

    def _render_system_prompt(self) -> str:
        """Handle system prompt."""
        # Keep the main step clear.
        return build_system_prompt(
            self.system_prompt,
            multi_agent_addon=self.multi_agent_prompt_addon,
            single_agent_addon=self.single_agent_prompt_addon,
            multi_agent=bool(self.runtime_config.multi_agent),
            extra_sections=self.prompt_extra_sections,
        )

    @staticmethod
    def _payload_to_human_content(payload: Any) -> str:
        """Handle to human content."""
        # Keep the main step clear.
        if isinstance(payload, str):
            return payload

        if isinstance(payload, Mapping) and isinstance(payload.get("llm_payload"), Mapping):
            return json.dumps(payload["llm_payload"], indent=2, default=str, ensure_ascii=False)

        if isinstance(payload, Mapping) and isinstance(payload.get("input"), str):
            return str(payload["input"])

        if isinstance(payload, Mapping) and isinstance(payload.get("messages"), list):
            for item in reversed(payload["messages"]):
                if isinstance(item, tuple) and len(item) == 2 and str(item[0]).lower() == "user":
                    return str(item[1])
                if isinstance(item, Mapping) and str(item.get("role", "")).lower() == "user":
                    return str(item.get("content", ""))
                if isinstance(item, HumanMessage):
                    return str(item.content or "")

        return json.dumps(payload, indent=2, default=str, ensure_ascii=False)

    @staticmethod
    def _json_from_text(text: str) -> tuple[Any | None, str]:
        """Handle from text."""
        # Keep the main step clear.
        raw = (text or "").strip()
        if not raw:
            return None, raw

        try:
            return json.loads(raw), raw
        except Exception:
            return None, raw

    @staticmethod
    def _ai_message_with_tool_calls(
        source: AIMessage,
        tool_calls: list[dict[str, Any]],
        *,
        extra_additional_kwargs: Mapping[str, Any] | None = None,
    ) -> AIMessage:
        """Handle message with tool calls."""
        # Keep the main step clear.
        additional_kwargs = dict(getattr(source, "additional_kwargs", {}) or {})
        if extra_additional_kwargs:
            additional_kwargs.update(dict(extra_additional_kwargs))
        return AIMessage(
            content="",
            tool_calls=list(tool_calls),
            additional_kwargs=additional_kwargs,
            response_metadata=getattr(source, "response_metadata", {}),
            id=getattr(source, "id", None),
            name=getattr(source, "name", None),
        )

    def _limit_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Handle tool calls."""
        # Keep the main step clear.
        if len(tool_calls) <= self.max_tool_calls:
            return tool_calls, []
        return tool_calls[: self.max_tool_calls], tool_calls[self.max_tool_calls :]

    def _emit_tool_execution_trace(self, trace: ToolExecutionTrace) -> None:
        """Emit tool execution trace."""
        # Keep events flowing.
        if self.runtime_config.print_events:
            status = trace.status or "unknown"
            print(
                "[agent-tool] name={name} status={status} latency_ms={latency_ms}".format(
                    name=trace.tool_name,
                    status=status,
                    latency_ms=trace.latency_ms,
                ),
                flush=True,
            )

        if not self.runtime_config.persist_events:
            return

        if not trace.run_id or not trace.agent_name:
            return

        self._telemetry.emit_tool(
            ToolExecutionMetric(
                run_id=trace.run_id,
                agent_name=trace.agent_name,
                iteration=trace.iteration,
                tool_call_id=trace.tool_call_id,
                tool_name=trace.tool_name,
                started_at=trace.started_at,
                ended_at=trace.ended_at,
                latency_ms=trace.latency_ms,
                status=trace.status,
                result_char_count=trace.result_char_count,
                result_estimated_tokens=trace.result_estimated_tokens,
                error_text=trace.error_text,
            )
        )

    @staticmethod
    def _values_state(
        messages: list[BaseMessage],
        iteration: int,
        done: bool,
        output_json: Any,
        handoff_json: Any = None,
    ) -> dict[str, Any]:
        """Handle state."""
        # Keep the main step clear.
        return {
            "messages": list(messages),
            "iteration": iteration,
            "done": done,
            "output": output_json,
            "handoff": handoff_json,
        }

    def set_event_context(self, *, run_id: str, agent_name: str, start_seq: int = 0) -> None:
        """Handle event context."""
        # Keep the main step clear.
        self._events.set_context(run_id=run_id, agent_name=agent_name, start_seq=start_seq)
        self._telemetry.set_context(run_id=run_id, agent_name=agent_name)

    def set_event_handlers(self, handlers: Sequence[Callable[[dict[str, Any]], None]] | None) -> None:
        """Handle event handlers."""
        # Keep the main step clear.
        self._events.set_handlers(handlers)

    def add_event_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Handle event handler."""
        # Keep the main step clear.
        self._events.add_handler(handler)

    def set_llm_call_handlers(self, handlers: Sequence[Callable[[dict[str, Any]], None]] | None) -> None:
        """Handle llm call handlers."""
        # Keep the main step clear.
        self._telemetry.set_llm_handlers(handlers)

    def add_llm_call_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Handle llm call handler."""
        # Keep the main step clear.
        self._telemetry.add_llm_handler(handler)

    def set_tool_call_handlers(self, handlers: Sequence[Callable[[dict[str, Any]], None]] | None) -> None:
        """Handle tool call handlers."""
        # Keep the main step clear.
        self._telemetry.set_tool_handlers(handlers)

    def add_tool_call_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Handle tool call handler."""
        # Keep the main step clear.
        self._telemetry.add_tool_handler(handler)

    @staticmethod
    def _parsed_to_payload_json(parsed: Any) -> dict[str, Any] | None:
        """Handle to payload json."""
        # Keep the main step clear.
        if parsed is None:
            return None
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    async def _ainvoke_with_telemetry(
        self,
        *,
        call_kind: str,
        iteration: int,
        messages: list[BaseMessage],
        invoke_fn: Callable[[], Any],
    ) -> Any:
        """Handle with telemetry."""
        # Keep the main step clear.
        started_at = datetime.utcnow()
        t0 = time.perf_counter()
        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "model", None)
        run_id, agent_name = self._telemetry.current_context()

        try:
            response = await invoke_fn()

            ended_at = datetime.utcnow()
            latency_ms = int((time.perf_counter() - t0) * 1000)

            usage = extract_provider_usage(response)
            response_meta = getattr(response, "response_metadata", {}) or {}
            if not isinstance(response_meta, Mapping):
                response_meta = {}
            provider_model_id = response_meta.get("provider_model_id")
            request_parameters = response_meta.get("request_parameters")
            request_parameters = dict(request_parameters) if isinstance(request_parameters, Mapping) else {}
            network_attempts = response_meta.get("network_attempts", 1)
            if not isinstance(network_attempts, int) or isinstance(network_attempts, bool) or network_attempts < 1:
                network_attempts = 1
            in_tok = usage.input_tokens
            out_tok = usage.output_tokens
            input_token_source = "provider" if in_tok is not None else "estimated"
            output_token_source = "provider" if out_tok is not None else "estimated"
            if in_tok is None:
                in_tok = self._token_estimator.estimate_messages_tokens(messages)
            if out_tok is None:
                if isinstance(response, AIMessage):
                    out_tok = self._token_estimator.estimate_ai_output_tokens(response)
                else:
                    out_tok = self._token_estimator.estimate_jsonable_output_tokens(response)
            tot_tok = usage.total_tokens if input_token_source == output_token_source == "provider" and usage.total_tokens is not None else in_tok + out_tok
            usage_source = (
                "provider" if input_token_source == output_token_source == "provider"
                else "estimated" if input_token_source == output_token_source == "estimated"
                else "mixed"
            )

            tool_calls: list[dict[str, Any]] = []
            parse_source: str | None = None
            text_recovered_tool_call_count = 0
            native_tool_call_count = 0
            malformed_tool_call_detected = False
            if isinstance(response, AIMessage):
                allowed_tool_names = set(self.tools_by_name.keys()) or None
                parse_result = extract_tool_calls_with_priority(
                    response,
                    allowed_tool_names=allowed_tool_names,
                    allow_text_recovery=self.runtime_config.allow_text_tool_recovery,
                )
                tool_calls = [
                    {
                        "id": call.id,
                        "name": call.name,
                        "args": dict(call.args),
                    }
                    for call in parse_result.calls
                ]
                parse_source = (
                    parse_result.source.value
                    if parse_result.succeeded and getattr(parse_result, "source", None) is not None
                    else None
                )
                text_recovered_tool_call_count = sum(1 for call in parse_result.calls if call.recovered)
                native_tool_call_count = max(0, len(tool_calls) - text_recovered_tool_call_count)
                malformed_tool_call_detected = (
                    not parse_result.succeeded
                    and looks_like_malformed_tool_call_content(
                        str(getattr(response, "content", "") or ""),
                        allowed_tool_names=allowed_tool_names,
                    )
                )
                response = AIMessage(
                    content=str(getattr(response, "content", "") or ""),
                    tool_calls=tool_calls,
                    additional_kwargs={
                        **(getattr(response, "additional_kwargs", {}) or {}),
                        "tool_call_parse_source": parse_source,
                        "tool_call_recovered": bool(parse_result.recovered),
                        "tool_call_parse_all_lines_parsed": parse_result.all_lines_parsed,
                        "malformed_tool_call_detected": malformed_tool_call_detected,
                    },
                    response_metadata=getattr(response, "response_metadata", {}),
                    id=getattr(response, "id", None),
                    name=getattr(response, "name", None),
                )
            tool_names = [str(tc.get("name", "")) for tc in tool_calls if tc.get("name")]

            if run_id and agent_name:
                call_index = self._telemetry.next_call_index()
                metric = LLMCallMetric(
                    run_id=run_id,
                    agent_name=agent_name,
                    call_index=call_index,
                    iteration=iteration,
                    call_kind=call_kind,
                    model_name=str(provider_model_id or model_name) if provider_model_id or model_name else None,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    tokens_total=int(tot_tok),
                    usage_source=str(usage_source or "estimated"),
                    had_tool_calls=bool(tool_calls),
                    tool_call_count=len(tool_calls),
                    tool_call_parse_source=parse_source,
                    text_recovered_tool_call_count=text_recovered_tool_call_count,
                    native_tool_call_count=native_tool_call_count,
                    tool_names=tool_names,
                    error_text=None,
                    provider_model_id=str(provider_model_id) if provider_model_id else None,
                    request_parameters=request_parameters,
                    network_attempts=network_attempts,
                    input_token_source=input_token_source,
                    output_token_source=output_token_source,
                )
                self._emit_llm_metric(metric)

            return response
        except Exception as exc:
            normalized_exc: Exception
            if isinstance(exc, TimeoutError):
                normalized_exc = TimeoutError(str(exc) or "run_timeout_exceeded")
            else:
                normalized_exc = exc

            ended_at = datetime.utcnow()
            latency_ms = int((time.perf_counter() - t0) * 1000)
            in_tok = self._token_estimator.estimate_messages_tokens(messages)
            network_attempts = getattr(normalized_exc, "network_attempts", 1)
            if not isinstance(network_attempts, int) or network_attempts < 1:
                network_attempts = 1

            if run_id and agent_name:
                call_index = self._telemetry.next_call_index()
                metric = LLMCallMetric(
                    run_id=run_id,
                    agent_name=agent_name,
                    call_index=call_index,
                    iteration=iteration,
                    call_kind=call_kind,
                    model_name=str(model_name) if model_name else None,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    input_tokens=int(in_tok),
                    output_tokens=None,
                    tokens_total=None,
                    usage_source="partial_unknown",
                    had_tool_calls=False,
                    tool_call_count=0,
                    tool_call_parse_source=None,
                    text_recovered_tool_call_count=0,
                    native_tool_call_count=0,
                    tool_names=[],
                    error_text=str(normalized_exc),
                    input_token_source="estimated",
                    output_token_source="unknown",
                    network_attempts=network_attempts,
                )
                self._emit_llm_metric(metric)

            raise normalized_exc

    def _emit_llm_metric(self, metric: LLMCallMetric) -> None:
        """Emit llm metric."""
        # Keep events flowing.
        if self.runtime_config.print_events:
            status = "error" if metric.error_text else "ok"
            tool_names = ",".join(metric.tool_names) if metric.tool_names else "none"
            print(
                "[agent-llm] agent={agent} call={call_index} kind={kind} status={status} tokens={tokens} tools={tools} latency_ms={latency_ms}".format(
                    agent=metric.agent_name,
                    call_index=metric.call_index,
                    kind=metric.call_kind,
                    status=status,
                    tokens=metric.tokens_total,
                    tools=tool_names,
                    latency_ms=metric.latency_ms,
                ),
                flush=True,
            )

        if not self.runtime_config.persist_events:
            return

        self._telemetry.emit_llm(metric)

    def _emit_event(
        self,
        *,
        event_type: str,
        node_name: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        status: str | None = None,
        payload_json: dict[str, Any] | None = None,
        payload_text: str | None = None,
    ) -> None:
        """Emit event."""
        # Keep events flowing.
        if self.runtime_config.print_events:
            details = []
            if node_name:
                details.append("node={node}".format(node=node_name))
            if tool_name:
                details.append("tool={tool}".format(tool=tool_name))
            if status:
                details.append("status={status}".format(status=status))
            suffix = " " + " ".join(details) if details else ""
            print("[agent-event] type={event_type}{suffix}".format(event_type=event_type, suffix=suffix), flush=True)

        if not self.runtime_config.persist_events:
            return

        self._events.emit(
            event_type=event_type,
            node_name=node_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status=status,
            payload_json=payload_json,
            payload_text=payload_text,
        )

    async def astream(
        self,
        payload: Any,
        stream_mode: Sequence[str] | str | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Handle the value."""
        # Keep the main step clear.
        async for item in self._agent_runner.astream(payload, stream_mode=stream_mode):
            yield item

    async def ainvoke(self, payload: Any) -> Any:
        """Handle the value."""
        # Keep the main step clear.
        final_output: Any = None
        final_handoff: Any = None
        async for mode, data in self.astream(payload, stream_mode=("values",)):
            if mode == "values" and isinstance(data, Mapping):
                final_output = data.get("output")
                final_handoff = data.get("handoff")
        if final_handoff is not None:
            return {"handoff": final_handoff, "output": final_output}
        return final_output

    def invoke(self, payload: Any) -> Any:
        """Handle the value."""
        # Keep the main step clear.
        return self._run_async_in_thread(self.ainvoke(payload))

    def stream(
        self,
        payload: Any,
        stream_mode: Sequence[str] | str | None = None,
    ):
        """Stream the value."""
        # Keep events flowing.
        async_gen = self.astream(payload, stream_mode=stream_mode)
        yield from self._stream_async_in_thread(async_gen)

    @staticmethod
    def _run_async_in_thread(coro: Any) -> Any:
        """Run async in thread."""
        # Kick off the main step.
        q: Queue[Any] = Queue()

        def _runner() -> None:
            """Handle the value."""
            # Keep the main step clear.
            try:
                result = asyncio.run(coro)
                q.put(("result", result))
            except Exception as exc:
                q.put(("error", exc))

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        kind, value = q.get()
        if kind == "error":
            raise value
        return value

    @staticmethod
    def _stream_async_in_thread(async_gen: AsyncGenerator[tuple[str, Any], None]):
        """Stream async in thread."""
        # Keep events flowing.
        q: Queue[Any] = Queue()
        sentinel = object()

        async def _producer() -> None:
            """Handle the value."""
            # Keep the main step clear.
            try:
                async for item in async_gen:
                    q.put(("item", item))
            except Exception as exc:
                q.put(("error", exc))
            finally:
                q.put(("done", sentinel))

        def _runner() -> None:
            """Handle the value."""
            # Keep the main step clear.
            asyncio.run(_producer())

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

        while True:
            kind, value = q.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                break


# Backward-compatible alias while the codebase migrates to the paper-facing name.
SSEHandrolledAgent = AgentKernel
