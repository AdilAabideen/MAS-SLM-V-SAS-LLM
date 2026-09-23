"""Payload Builder module helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from mas_slm_research.workflows.esi.payloads.doctor import build_payload as build_doctor_payload
from mas_slm_research.workflows.esi.payloads.esi1 import build_payload as build_esi1_payload
from mas_slm_research.workflows.esi.payloads.esi2 import build_payload as build_esi2_payload
from mas_slm_research.workflows.esi.payloads.esi345 import build_payload as build_esi345_payload
from mas_slm_research.workflows.esi.payloads.vitals import build_payload as build_vitals_payload
from mas_slm_research.mas_contract import AgentName, MASState


PayloadBuilder = Callable[[MASState], Dict[str, Any]]


payload_builders: Dict[AgentName, PayloadBuilder] = {
    "esi1_agent": build_esi1_payload,
    "esi2_agent": build_esi2_payload,
    "esi345_agent": build_esi345_payload,
    "vitals_agent": build_vitals_payload,
    "doctor_agent": build_doctor_payload,
}


def _latest_handoff_for_agent(agent_name: AgentName, state: MASState) -> Optional[Dict[str, Any]]:
    """Find the newest handoff for the target agent."""
    # Prefer the still-active handoff first.
    pending_handoff = state.get("pending_handoff")
    if isinstance(pending_handoff, dict) and pending_handoff.get("target_agent") == agent_name:
        return dict(pending_handoff)

    # Fall back to the most recent matching history item.
    for item in reversed(list(state.get("handoff_history") or [])):
        if isinstance(item, dict) and item.get("target_agent") == agent_name:
            return dict(item)

    return None


def _state_for_agent(agent_name: AgentName, state: MASState) -> MASState:
    """Scope shared state to the next agent handoff."""
    # Keep the main step clear.
    scoped_state = dict(state)
    # Attach only the handoff meant for this agent.
    scoped_state["pending_handoff"] = _latest_handoff_for_agent(agent_name, state)
    return scoped_state  # type: ignore[return-value]


def build_pending_agent_payload(agent_name: AgentName, state: MASState) -> Dict[str, Any]:
    """Build the next agent payload and validate its shape."""
    # Build the next value.
    try:
        builder = payload_builders[agent_name]
    except KeyError as exc:
        raise ValueError("No payload builder registered for agent '{agent}'.".format(agent=agent_name)) from exc

    # Build from the agent-scoped state view.
    payload = builder(_state_for_agent(agent_name, state))
    if not isinstance(payload, dict) or not isinstance(payload.get("llm_payload"), dict):
        raise ValueError(
            "Payload builder for agent '{agent}' must return a dict containing llm_payload.".format(
                agent=agent_name
            )
        )

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(
            "Payload builder for agent '{agent}' returned non-dict metadata.".format(agent=agent_name)
        )

    return payload
