"""Handoff module helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _coerce_string_bool(value: Any) -> Any:
    """Handle string bool."""
    # Keep the main step clear.
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def _coerce_string_list(value: Any) -> list[str]:
    """Handle string list."""
    # Keep the main step clear.
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


class CoerciveHandoffPayload(BaseModel):
    """
    Shared coercion layer for handoff payloads.

    Keep meaning-bearing scalar fields strict, but normalize the repeated model
    drift we see in production traces:
    - "true"/"false" strings for boolean flags
    - omitted concern arrays -> []
    - scalar/string concern values -> [value]
    """

    model_config = ConfigDict(extra="ignore")

    _bool_fields: ClassVar[frozenset[str]] = frozenset()
    _list_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _coerce_common_handoff_shapes(cls, value: Any) -> Any:
        """Handle common handoff shapes."""
        # Keep the main step clear.
        if not isinstance(value, dict):
            return value

        data = dict(value)
        for field_name in cls._bool_fields:
            if field_name in data:
                data[field_name] = _coerce_string_bool(data[field_name])

        for field_name in cls._list_fields:
            data[field_name] = _coerce_string_list(data.get(field_name))

        return data


class HandoffResult(BaseModel):
    handoff_name: str = Field(..., description="Stable handoff identifier.")
    from_agent: str = Field(..., description="Agent that initiated the handoff.")
    target_agent: str = Field(..., description="Agent that should receive control next.")
    payload_schema: str = Field(..., description="Payload schema class name used for validation.")
    payload: Dict[str, Any] = Field(..., description="Validated structured handoff payload.")


@dataclass(frozen=True)
class HandoffDefinition:
    source_agent: str
    target_agent: str
    payload_model: Type[BaseModel]
    description: str
    tool_name: Optional[str] = None

    @property
    def name(self) -> str:
        """Handle the value."""
        # Keep the main step clear.
        if self.tool_name:
            return self.tool_name
        return "handoff_to_{target}".format(target=self.target_agent)


@dataclass(frozen=True)
class HandoffCommandConfig:
    messages_state_key: str = "messages"
    active_agent_state_key: str = "active_agent"
    pending_handoff_state_key: str = "pending_handoff"
    handoff_history_state_key: str = "handoff_history"


def define_handoff(
    *,
    source_agent: str,
    target_agent: str,
    payload_model: Type[BaseModel],
    description: str,
    tool_name: Optional[str] = None,
) -> HandoffDefinition:
    """Handle handoff."""
    # Keep the main step clear.
    if not source_agent.strip():
        raise ValueError("source_agent must be non-empty.")
    if not target_agent.strip():
        raise ValueError("target_agent must be non-empty.")
    if not isinstance(payload_model, type) or not issubclass(payload_model, BaseModel):
        raise TypeError("payload_model must be a Pydantic BaseModel subclass.")
    if not description.strip():
        raise ValueError("description must be non-empty.")

    return HandoffDefinition(
        source_agent=source_agent.strip(),
        target_agent=target_agent.strip(),
        payload_model=payload_model,
        description=description.strip(),
        tool_name=(tool_name.strip() if isinstance(tool_name, str) and tool_name.strip() else None),
    )


def create_handoff_tool(definition: HandoffDefinition) -> BaseTool:
    """Create handoff tool."""
    # Build the new value.
    payload_model = definition.payload_model
    tool_name = definition.name
    tool_description = "{description}".format(
        description=definition.description,
        source=definition.source_agent,
        target=definition.target_agent,
    )


    @tool(tool_name, args_schema=payload_model, description=tool_description)
    def _handoff(**kwargs: Any) -> Dict[str, Any]:
        """Handle the value."""
        # Keep the main step clear.
        validated = payload_model.model_validate(kwargs)
        result = HandoffResult(
            handoff_name=tool_name,
            from_agent=definition.source_agent,
            target_agent=definition.target_agent,
            payload_schema=payload_model.__name__,
            payload=validated.model_dump(),
        )
        return result.model_dump()

    return _handoff


def create_handoff_tools(
    source_agent: str,
    handoffs: Sequence[HandoffDefinition],
) -> List[BaseTool]:
    """Create handoff tools."""
    # Build the new value.
    normalized = validate_handoffs(source_agent=source_agent, handoffs=handoffs)
    return [create_handoff_tool(handoff) for handoff in normalized]


def validate_handoffs(
    *,
    source_agent: str,
    handoffs: Sequence[HandoffDefinition],
) -> List[HandoffDefinition]:
    """Validate handoffs."""
    # Fail fast on bad input.
    normalized: List[HandoffDefinition] = []
    seen_tool_names: Dict[str, str] = {}

    for handoff in handoffs:
        if handoff.source_agent != source_agent:
            raise ValueError(
                "Handoff source mismatch for tool '{tool_name}': expected '{expected}', got '{actual}'.".format(
                    tool_name=handoff.name,
                    expected=source_agent,
                    actual=handoff.source_agent,
                )
            )
        if handoff.name in seen_tool_names:
            raise ValueError(
                "Duplicate handoff tool name '{tool_name}' for targets '{first}' and '{second}'.".format(
                    tool_name=handoff.name,
                    first=seen_tool_names[handoff.name],
                    second=handoff.target_agent,
                )
            )
        seen_tool_names[handoff.name] = handoff.target_agent
        normalized.append(handoff)

    return normalized


def handoff_result_to_command(
    handoff: HandoffResult,
    *,
    state: Dict[str, Any],
    tool_call_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    config: Optional[HandoffCommandConfig] = None,
) -> Command:
    """Handle result to command."""
    # Keep the main step clear.
    runtime_config = config or HandoffCommandConfig()

    messages_key = runtime_config.messages_state_key
    active_agent_key = runtime_config.active_agent_state_key
    pending_handoff_key = runtime_config.pending_handoff_state_key
    handoff_history_key = runtime_config.handoff_history_state_key

    update: Dict[str, Any] = {
        active_agent_key: handoff.target_agent,
        pending_handoff_key: handoff.model_dump(),
        handoff_history_key: [
            {
                "handoff_name": handoff.handoff_name,
                "from_agent": handoff.from_agent,
                "target_agent": handoff.target_agent,
                "payload_schema": handoff.payload_schema,
                "payload": handoff.payload,
            }
        ],
    }

    if messages_key in state:
        tool_message = ToolMessage(
            content="Successfully transferred to {agent}".format(agent=handoff.target_agent),
            name=tool_name or handoff.handoff_name,
            tool_call_id=tool_call_id or handoff.handoff_name,
        )
        update[messages_key] = list(state.get(messages_key) or []) + [tool_message]

    return Command(
        goto=handoff.target_agent,
        graph=Command.PARENT,
        update=update,
    )


def handoff_targets(handoffs: Sequence[HandoffDefinition]) -> List[str]:
    """Handle targets."""
    # Keep the main step clear.
    targets: List[str] = []
    for handoff in handoffs:
        if handoff.target_agent not in targets:
            targets.append(handoff.target_agent)
    return targets
