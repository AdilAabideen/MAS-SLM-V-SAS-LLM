"""Agent Runner module helpers."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Mapping

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .finalization_policy import FinalizationPolicy
from .handoff_policy import HandoffPolicy
from .runtime_config import RuntimeConfig
from .short_term_memory import ShortTermMemory, ShortTermMemoryConfig
from .runtime_types import (
    BoundModel,
    BuildAIMessageWithToolCalls,
    CurrentTelemetryContext,
    EmitEvent,
    InvokeWithTelemetry,
    JSONFromText,
    LimitToolCalls,
    ParsedToPayloadJSON,
    PayloadToHumanContent,
    RenderSystemPrompt,
    StreamModesInput,
    ValuesStateBuilder,
)
from .tool_executor import ToolExecutor


class AgentRunner:
    """Async runtime loop orchestrator for the hand-rolled SSE agent."""

    def __init__(
        self,
        *,
        bound_model: BoundModel,
        runtime_config: RuntimeConfig,
        agent_node_name: str,
        tools_node_name: str,
        finalization_policy: FinalizationPolicy,
        handoff_policy: HandoffPolicy | None,
        tool_executor: ToolExecutor,
        current_telemetry_context: CurrentTelemetryContext,
        render_system_prompt: RenderSystemPrompt,
        payload_to_human_content: PayloadToHumanContent,
        ainvoke_with_telemetry: InvokeWithTelemetry,
        ai_message_with_tool_calls: BuildAIMessageWithToolCalls,
        limit_tool_calls: LimitToolCalls,
        json_from_text: JSONFromText,
        parsed_to_payload_json: ParsedToPayloadJSON,
        emit_event: EmitEvent,
        values_state: ValuesStateBuilder,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.bound_model = bound_model
        self.runtime_config = runtime_config
        self.agent_node_name = agent_node_name
        self.tools_node_name = tools_node_name
        self.finalization_policy = finalization_policy
        self.handoff_policy = handoff_policy or HandoffPolicy()
        self.tool_executor = tool_executor
        self.current_telemetry_context = current_telemetry_context
        self.render_system_prompt = render_system_prompt
        self.payload_to_human_content = payload_to_human_content
        self.ainvoke_with_telemetry = ainvoke_with_telemetry
        self.ai_message_with_tool_calls = ai_message_with_tool_calls
        self.limit_tool_calls = limit_tool_calls
        self.json_from_text = json_from_text
        self.parsed_to_payload_json = parsed_to_payload_json
        self.emit_event = emit_event
        self.values_state = values_state

    @staticmethod
    def _resolve_modes(stream_mode: StreamModesInput) -> tuple[str, ...]:
        """Resolve modes."""
        # Pick the needed value.
        return (stream_mode,) if isinstance(stream_mode, str) else tuple(stream_mode or ("updates", "values"))

    def _build_call_messages(
        self,
        *,
        human_msg: HumanMessage,
        short_term_memory: ShortTermMemory,
        retry_feedback: HumanMessage | None = None,
    ) -> list[BaseMessage]:
        """Build call messages."""
        # Build the next value.
        call_messages: list[BaseMessage] = [SystemMessage(content=self.render_system_prompt()), human_msg]
        call_messages.extend(short_term_memory.messages())
        if retry_feedback is not None:
            call_messages.append(retry_feedback)
        return call_messages

    @staticmethod
    def _short_excerpt(text: str, *, max_chars: int = 400) -> str:
        """Handle excerpt."""
        # Keep the main step clear.
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "...(truncated)"

    def _build_malformed_tool_retry_feedback(self, ai_msg: AIMessage) -> HumanMessage:
        """Build malformed tool retry feedback."""
        # Build the next value.
        excerpt = self._short_excerpt(str(getattr(ai_msg, "content", "") or ""))
        content = (
            "MALFORMED TOOL CALL DETECTED.\n"
            "Your previous response was not valid for the tool-calling contract.\n"
            "Return EXACTLY one JSON object and no extra text.\n"
            'Required format: {"tool_calls":[{"id":"call_<unique_id>","name":"<tool_name>","arguments":{...}}]}\n'
            "Rules:\n"
            "- Do not use markdown or code fences.\n"
            "- Do not add extra top-level keys.\n"
            "- Put all tool calls inside the single tool_calls array.\n"
            "- Use valid JSON only.\n"
            f"Previous malformed output excerpt:\n{excerpt}"
        )
        return HumanMessage(content=content)

    def _expected_tool_for_retry(self, ai_msg: AIMessage) -> str | None:
        """Handle tool for retry."""
        # Keep the main step clear.
        content = str(getattr(ai_msg, "content", "") or "")
        lowered = content.lower()

        final_answer_tool_name = str(self.finalization_policy.final_answer_tool_name or "").strip()
        if final_answer_tool_name and final_answer_tool_name.lower() in lowered:
            return final_answer_tool_name

        available_tool_names = list(self.tool_executor.tools_by_name.keys())
        matches = [name for name in available_tool_names if name.lower() in lowered]
        if len(matches) == 1:
            return matches[0]

        if len(available_tool_names) == 1:
            return available_tool_names[0]

        return None

    def _retry_key_for_tool(self, tool_name: str | None) -> str:
        """Handle key for tool."""
        # Keep the main step clear.
        return str(tool_name or "__unknown_tool__")

    def _build_tool_specific_malformed_tool_retry_feedback(
        self,
        ai_msg: AIMessage,
        *,
        tool_name: str | None,
    ) -> HumanMessage:
        """Build tool specific malformed tool retry feedback."""
        # Build the next value.
        if not tool_name:
            return self._build_malformed_tool_retry_feedback(ai_msg)

        excerpt = self._short_excerpt(str(getattr(ai_msg, "content", "") or ""))
        content = (
            "MALFORMED TOOL CALL DETECTED.\n"
            f"Your previous response attempted to call `{tool_name}` but the JSON/tool-call format was malformed.\n"
            f"Call EXACTLY one tool: `{tool_name}`.\n"
            'Required format: {"tool_calls":[{"id":"call_<unique_id>","name":"'
            f'{tool_name}'
            '","arguments":{{...}}}]}\n'
            "Rules:\n"
            "- Do not use markdown or code fences.\n"
            "- Do not add prose or explanations.\n"
            "- Do not add extra top-level keys.\n"
            "- Put the single tool call inside the tool_calls array.\n"
            "- Use valid JSON only.\n"
            f"Previous malformed output excerpt:\n{excerpt}"
        )
        return HumanMessage(content=content)

    def _extract_normalized_tool_calls(self, ai_msg: AIMessage) -> list[dict[str, Any]]:
        """Extract normalized tool calls."""
        # Pull out the needed value.
        raw_tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
        normalized_calls: list[dict[str, Any]] = []
        for item in raw_tool_calls:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            call_id = item.get("id")
            args = item.get("args", {})
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(call_id, str) or not call_id.strip():
                continue
            if not isinstance(args, Mapping):
                args = {}
            normalized = {"id": call_id, "name": name.strip(), "args": dict(args)}
            if normalized:
                normalized_calls.append(normalized)
        return normalized_calls

    def _apply_tool_call_limit(
        self,
        *,
        ai_msg: AIMessage,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[AIMessage, list[dict[str, Any]]]:
        """Handle tool call limit."""
        # Keep the main step clear.
        limited_tool_calls, dropped_tool_calls = self.limit_tool_calls(tool_calls)
        if not (dropped_tool_calls and self.runtime_config.drop_extra_tool_calls):
            return ai_msg, tool_calls

        limited_ai_msg = self.ai_message_with_tool_calls(
            ai_msg,
            limited_tool_calls,
            extra_additional_kwargs={
                "dropped_tool_calls": [{"id": tc.get("id"), "name": tc.get("name")} for tc in dropped_tool_calls],
                "dropped_tool_call_count": len(dropped_tool_calls),
            },
        )
        return limited_ai_msg, limited_tool_calls

    def _emit_assistant_event_from_text(self, content: str) -> None:
        """Emit assistant event from text."""
        # Keep events flowing.
        stripped = str(content or "").strip()
        if not stripped:
            return
        parsed, raw = self.json_from_text(stripped)
        self.emit_event(
            event_type="assistant",
            node_name=self.agent_node_name,
            payload_json=self.parsed_to_payload_json(parsed),
            payload_text=raw if parsed is None else None,
        )

    def _emit_tool_call_events(self, tool_calls: list[dict[str, Any]]) -> None:
        """Emit tool call events."""
        # Keep events flowing.
        for tool_call in tool_calls:
            self.emit_event(
                event_type="tool_call",
                node_name=self.agent_node_name,
                tool_name=str(tool_call.get("name") or ""),
                tool_call_id=(str(tool_call.get("id")) if tool_call.get("id") is not None else None),
                payload_json={"args": tool_call.get("args")},
            )

    def _emit_tool_result_event(self, tool_message: ToolMessage) -> None:
        """Emit tool result event."""
        # Keep events flowing.
        content = str(getattr(tool_message, "content", "") or "").strip()
        parsed, raw = self.json_from_text(content)
        self.emit_event(
            event_type="tool_result",
            node_name=self.tools_node_name,
            tool_name=getattr(tool_message, "name", None),
            tool_call_id=getattr(tool_message, "tool_call_id", None),
            status=getattr(tool_message, "status", None),
            payload_json={"result": parsed} if parsed is not None else None,
            payload_text=raw if parsed is None else None,
        )

    @staticmethod
    def _should_emit(modes: tuple[str, ...], mode: str) -> bool:
        """Handle emit."""
        # Keep the main step clear.
        return mode in modes

    async def astream(
        self,
        payload: Any,
        stream_mode: StreamModesInput = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Handle the value."""
        # Keep the main step clear.
        modes = self._resolve_modes(stream_mode)

        human_msg = HumanMessage(content=self.payload_to_human_content(payload))
        short_term_memory = ShortTermMemory(
            config=ShortTermMemoryConfig(
                include_final_assistant_output=self.runtime_config.short_term_memory_include_final_assistant_output,
                include_raw_provider_debug=self.runtime_config.short_term_memory_include_raw_provider_debug,
                verbose=self.runtime_config.short_term_memory_verbose,
                log_token_estimates=self.runtime_config.short_term_memory_log_token_estimates,
            )
        )
        streamed_messages: list[BaseMessage] = []

        final_output: Any = None
        handoff_output: Any = None
        done = False
        iteration = 0
        malformed_tool_retry_counts: dict[str, int] = {}
        pending_retry_feedback: HumanMessage | None = None

        while not done:
            iteration += 1

            call_messages = self._build_call_messages(
                human_msg=human_msg,
                short_term_memory=short_term_memory,
                retry_feedback=pending_retry_feedback,
            )
            pending_retry_feedback = None

            ai_msg = await self.ainvoke_with_telemetry(
                call_kind="main_loop",
                iteration=iteration,
                messages=call_messages,
                invoke_fn=lambda: self.bound_model.ainvoke(call_messages),
            )

            tool_calls = self._extract_normalized_tool_calls(ai_msg)
            if tool_calls and str(ai_msg.content or "").strip():
                ai_msg = self.ai_message_with_tool_calls(
                    ai_msg,
                    tool_calls,
                    extra_additional_kwargs={"raw_tool_text": str(ai_msg.content or "")},
                )
            ai_msg, tool_calls = self._apply_tool_call_limit(ai_msg=ai_msg, tool_calls=tool_calls)

            streamed_messages.append(ai_msg)
            if tool_calls:
                short_term_memory.append_assistant_tool_call(ai_msg)
            self._emit_tool_call_events(tool_calls)

            if self._should_emit(modes, "updates"):
                yield "updates", {self.agent_node_name: {"messages": [ai_msg]}}
            if self._should_emit(modes, "values"):
                yield "values", self.values_state(streamed_messages, iteration, False, None)

            if not tool_calls:
                malformed_detected = bool(
                    (getattr(ai_msg, "additional_kwargs", {}) or {}).get("malformed_tool_call_detected")
                )
                if (
                    malformed_detected
                    and self.runtime_config.malformed_tool_retry_enabled
                ):
                    retry_tool_name = self._expected_tool_for_retry(ai_msg)
                    retry_key = self._retry_key_for_tool(retry_tool_name)
                    retry_count = malformed_tool_retry_counts.get(retry_key, 0)
                    if retry_count < self.runtime_config.max_malformed_tool_retries_per_tool:
                        malformed_tool_retry_counts[retry_key] = retry_count + 1
                        pending_retry_feedback = self._build_tool_specific_malformed_tool_retry_feedback(
                            ai_msg,
                            tool_name=retry_tool_name,
                        )
                        self.emit_event(
                            event_type="runtime_decision",
                            node_name=self.agent_node_name,
                            payload_json={
                                "decision": "retry_after_malformed_tool_call",
                                "tool_name": retry_tool_name,
                                "retry_index": malformed_tool_retry_counts[retry_key],
                                "max_retries_per_tool": self.runtime_config.max_malformed_tool_retries_per_tool,
                            },
                        )
                        if self._should_emit(modes, "values"):
                            yield "values", self.values_state(streamed_messages, iteration, False, None)
                        continue

                self._emit_assistant_event_from_text(str(ai_msg.content or ""))
                short_term_memory.append_final_assistant(ai_msg)

                decision = self.finalization_policy.maybe_finalize_from_assistant_no_tools(ai_msg)
                if decision.finalized:
                    final_output = decision.output
                else:
                    invalid = self.finalization_policy.finalize_invalid_output(
                        reason=decision.reason,
                        raw_output=str(ai_msg.content or ""),
                    )
                    final_output = invalid.output

                done = True
                break

            tool_msgs_by_index: dict[int, ToolMessage] = {}
            run_id, agent_name = self.current_telemetry_context()
            async for idx, tm in self.tool_executor.execute_tool_calls_batched(
                tool_calls,
                iteration=iteration,
                run_id=run_id,
                agent_name=agent_name,
                timeout_s=None,
            ):
                tool_msgs_by_index[idx] = tm
                streamed_messages.append(tm)
                self._emit_tool_result_event(tm)

                if self._should_emit(modes, "updates"):
                    yield "updates", {self.tools_node_name: {"messages": [tm]}}
                if self._should_emit(modes, "values"):
                    yield "values", self.values_state(streamed_messages, iteration, False, None)
            tool_msgs = [tool_msgs_by_index[idx] for idx in range(len(tool_calls)) if idx in tool_msgs_by_index]
            for tm in tool_msgs:
                short_term_memory.append_tool_result(tm)

            for tc, tm in zip(tool_calls, tool_msgs):
                handoff_decision = self.handoff_policy.maybe_handoff_from_tool_result(tc, tm)
                if handoff_decision.should_handoff and handoff_decision.envelope is not None:
                    handoff_output = handoff_decision.envelope.model_dump()
                    self.emit_event(
                        event_type="handoff",
                        node_name=self.tools_node_name,
                        tool_name=str(tc.get("name") or ""),
                        tool_call_id=(str(tc.get("id")) if tc.get("id") is not None else None),
                        status="success",
                        payload_json={"handoff": handoff_output},
                    )
                    done = True
                    break
                if handoff_decision.error:
                    final_output = {
                        "ok": False,
                        "error": "handoff_invalid",
                        "reason": handoff_decision.reason,
                        "details": handoff_decision.error,
                    }
                    self.emit_event(
                        event_type="error",
                        node_name=self.tools_node_name,
                        tool_name=str(tc.get("name") or ""),
                        tool_call_id=(str(tc.get("id")) if tc.get("id") is not None else None),
                        status="error",
                        payload_json={"handoff_error": final_output},
                    )
                    done = True
                    break

                decision = self.finalization_policy.maybe_finalize_from_tool_result(tc, tm)
                if not decision.finalized:
                    if decision.reason == "schema_validation_error":
                        invalid = self.finalization_policy.finalize_invalid_output(
                            reason=decision.reason,
                            raw_output=str(getattr(tm, "content", "") or ""),
                        )
                        final_output = invalid.output
                        final_msg = AIMessage(
                            content=invalid.output_text or json.dumps(final_output, ensure_ascii=False)
                        )
                        streamed_messages.append(final_msg)
                        short_term_memory.append_final_assistant(final_msg)
                        self._emit_assistant_event_from_text(str(final_msg.content or ""))
                        if self._should_emit(modes, "updates"):
                            yield "updates", {self.agent_node_name: {"messages": [final_msg]}}
                        done = True
                        break
                    continue

                final_output = decision.output
                final_text = decision.output_text or json.dumps(final_output, ensure_ascii=False)
                final_msg = AIMessage(content=final_text)
                streamed_messages.append(final_msg)
                short_term_memory.append_final_assistant(final_msg)
                self._emit_assistant_event_from_text(final_text)
                if self._should_emit(modes, "updates"):
                    yield "updates", {self.agent_node_name: {"messages": [final_msg]}}
                done = True
                break

        if final_output is None and handoff_output is None:
            fallback = self.finalization_policy.finalize_no_output()
            final_output = fallback.output
            final_msg = AIMessage(content=fallback.output_text or json.dumps(final_output, ensure_ascii=False))
            streamed_messages.append(final_msg)
            short_term_memory.append_final_assistant(final_msg)
            self._emit_assistant_event_from_text(str(final_msg.content or ""))
            if self._should_emit(modes, "updates"):
                yield "updates", {self.agent_node_name: {"messages": [final_msg]}}

        if self._should_emit(modes, "values"):
            yield "values", self.values_state(
                streamed_messages,
                iteration,
                True,
                final_output,
                handoff_output,
            )
