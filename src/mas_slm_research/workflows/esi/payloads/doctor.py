"""Payload Builder module helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mas_slm_research.mas_contract import MASState
from mas_slm_research.workflows.esi.payloads.common import (
    unified_payload_builder,
)


ACUITY_AGENTS = {"esi1_agent", "esi2_agent", "esi345_agent"}


def _doctor_handoffs(state: MASState) -> List[Dict[str, Any]]:
    """Handle handoffs."""
    # Keep the main step clear.
    return [
        dict(item)
        for item in list(state.get("handoff_history") or [])
        if isinstance(item, dict) and item.get("target_agent") == "doctor_agent"
    ]


def _latest_handoff_from(
    handoffs: List[Dict[str, Any]],
    source_agents: set[str],
) -> Optional[Dict[str, Any]]:
    """Handle handoff from."""
    # Keep the main step clear.
    for item in reversed(handoffs):
        if item.get("from_agent") in source_agents:
            return item
    return None


def build_payload(state: MASState) -> Dict[str, Any]:
    """Build a compact doctor mas payload."""
    # Build the next value.
    doctor_handoffs = _doctor_handoffs(state)
    acuity_handoff = _latest_handoff_from(doctor_handoffs, ACUITY_AGENTS)
    vitals_handoff = _latest_handoff_from(doctor_handoffs, {"vitals_agent"})
    acuity_payload = dict((acuity_handoff or {}).get("payload") or {})
    vitals_payload = dict((vitals_handoff or {}).get("payload") or {})
    case_info = unified_payload_builder("doctor_agent", dict(state.get("case_info") or {}))

    source_agent = (acuity_handoff or {}).get("from_agent")

    upstream_esi_level = (
        acuity_payload.get("esi_level")
        or acuity_payload.get("final_esi_level")
        or acuity_payload.get("upstream_esi_level")
        or acuity_payload.get("esi_level_345")
    )

    llm_payload = {
        "case_info": case_info,
        "source_agent": source_agent,
        "upstream_esi_level": upstream_esi_level,
        "vitals_consider_uptriage": (
            vitals_payload.get("consider_uptriage")
            or vitals_payload.get("vitals_consider_uptriage")
            or False

        ),
        "abnormal_vitals": vitals_payload.get("abnormal_vitals") or []
    }
    return {
        "llm_payload": llm_payload,
        "metadata": {
            "agent_role": "doctor_agent",
            "uses_handoff": True,
            "doctor_handoff_count": len(doctor_handoffs),
            "acuity_from_agent": (acuity_handoff or {}).get("from_agent"),
            "vitals_from_agent": (vitals_handoff or {}).get("from_agent"),
        },
    }
