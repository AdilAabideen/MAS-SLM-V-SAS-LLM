"""Tool Executor module helpers."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Mapping, Sequence, Union

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agentic.protocols import NormalizedToolCall


ToolCallInput = Union[Mapping[str, Any], NormalizedToolCall]


@dataclass(frozen=True)
class ToolExecutionTrace:
    run_id: str | None
    agent_name: str | None
    iteration: int
    tool_call_id: str
    tool_name: str
    started_at: datetime
    ended_at: datetime
    latency_ms: int
    status: str
    error_kind: str | None
    result_char_count: int
    result_estimated_tokens: int
    error_text: str | None = None


class ToolExecutor:
    """Centralized async tool execution with consistent wrapping and metrics."""

    def __init__(
        self,
        tools_by_name: Mapping[str, BaseTool],
        *,
        estimate_tool_result_tokens: Callable[[str], int],
        emit_trace: Callable[[ToolExecutionTrace], None] | None = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.tools_by_name = dict(tools_by_name)
        self._estimate_tool_result_tokens = estimate_tool_result_tokens
        self._emit_trace = emit_trace

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Handle default."""
        # Keep the main step clear.
        if isinstance(value, BaseModel):
            return value.model_dump()
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return value.model_dump()
            except Exception:
                pass
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, (list, tuple)):
            return list(value)
        return str(value)

    @classmethod
    def _tool_output_to_text(cls, value: Any) -> str:
        """Handle output to text."""
        # Keep the main step clear.
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
            return json.dumps(value, ensure_ascii=False, default=cls._json_default)
        return str(value)

    @staticmethod
    def _extract_call_fields(tool_call: ToolCallInput) -> tuple[str | None, str, dict[str, Any]]:
        """Extract call fields."""
        # Pull out the needed value.
        if isinstance(tool_call, NormalizedToolCall):
            name = tool_call.name
            call_id = tool_call.id
            args = tool_call.args if isinstance(tool_call.args, dict) else {}
            return name, call_id, args

        name = tool_call.get("name")
        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        return str(name) if name is not None else None, str(call_id), args

    async def execute_tool_call(
        self,
        tool_call: ToolCallInput,
        *,
        iteration: int,
        run_id: str | None = None,
        agent_name: str | None = None,
    ) -> ToolMessage:
        """Handle tool call."""
        # Keep the main step clear.
        name, call_id, args = self._extract_call_fields(tool_call)

        started_at = datetime.utcnow()
        t0 = time.perf_counter()
        status = "error"
        error_kind: str | None = None
        error_text: str | None = None
        content = ""
        metric_tool_name = str(name or "tool")

        if name not in self.tools_by_name:
            error_kind = "unknown_tool"
            content = f"Unknown tool: {name}"
            error_text = content
            tool_message = ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=name,
                status="error",
            )
        else:
            tool_obj = self.tools_by_name[name]
            metric_tool_name = str(getattr(tool_obj, "name", name) or "tool")
            try:
                result = await tool_obj.ainvoke(args)
                content = self._tool_output_to_text(result)
                status = "success"
                error_kind = None
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=call_id,
                    name=name,
                    artifact=result,
                    status="success",
                )
            except Exception as exc:
                error_kind = "tool_execution_error"
                error_text = str(exc)
                content = error_text
                tool_message = ToolMessage(
                    content=error_text,
                    tool_call_id=call_id,
                    name=name,
                    status="error",
                )

        ended_at = datetime.utcnow()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if status != "success":
            status = "error"

        if self._emit_trace is not None:
            self._emit_trace(
                ToolExecutionTrace(
                    run_id=run_id,
                    agent_name=agent_name,
                    iteration=iteration,
                    tool_call_id=str(call_id),
                    tool_name=metric_tool_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    status=status,
                    error_kind=error_kind,
                    result_char_count=len(content),
                    result_estimated_tokens=self._estimate_tool_result_tokens(content),
                    error_text=error_text,
                )
            )

        return tool_message

    async def execute_tool_calls_batched(
        self,
        tool_calls: Sequence[ToolCallInput],
        *,
        iteration: int,
        run_id: str | None = None,
        agent_name: str | None = None,
        timeout_s: float | None = None,
    ) -> AsyncGenerator[tuple[int, ToolMessage], None]:
        """Handle tool calls batched."""
        # Keep the main step clear.
        async def _run_indexed(idx: int, call: ToolCallInput) -> tuple[int, ToolMessage]:
            """Run indexed."""
            # Kick off the main step.
            msg = await self.execute_tool_call(
                call,
                iteration=iteration,
                run_id=run_id,
                agent_name=agent_name,
            )
            return idx, msg

        tasks = [
            asyncio.create_task(_run_indexed(idx, call))
            for idx, call in enumerate(tool_calls)
            if isinstance(call, (Mapping, NormalizedToolCall))
        ]
        if not tasks:
            return

        try:
            iterator = (
                asyncio.as_completed(tasks)
                if timeout_s is None
                else asyncio.as_completed(tasks, timeout=timeout_s)
            )
            for task in iterator:
                idx, tm = await task
                yield idx, tm
        except TimeoutError:
            for task in tasks:
                task.cancel()
            raise TimeoutError("run_timeout_exceeded")
