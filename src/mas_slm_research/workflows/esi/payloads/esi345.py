"""Payload Builder module helpers."""

from __future__ import annotations

from typing import Any, Dict

from mas_slm_research.mas_contract import MASState
from mas_slm_research.workflows.esi.payloads.common import (
    unified_payload_builder,
)

def clean_payload(payload):
    """Handle payload."""
    # Keep the main step clear.
    return {
        "carry_forward_concerns" : payload.get("carry_forward_concerns"),
        "brief_reason" : payload.get("reason")
    }


def build_payload(state: MASState) -> Dict[str, Any]:
    """Build a compact ESI345 mas payload."""
    # Build the next value.
    pending_handoff = dict(state.get("pending_handoff") or {})
    handoff_payload = dict(pending_handoff.get("payload") or {})
    case_info = unified_payload_builder("esi345_agent", dict(state.get("case_info") or {}))
    print(case_info)
    llm_payload = {
        "case_info": case_info,
        "handoff_context": (
            "You are receiving a handoff from the ESI2 Agent. "
            "This means the patient was judged to be NOT ESI-2."
        ),
        "prior_agent": pending_handoff.get("from_agent"),
        "prior_result": clean_payload(handoff_payload),
    }
    return {
        "llm_payload": llm_payload,
        "metadata": {
            "agent_role": "esi345_agent",
            "uses_handoff": True,
            "handoff_name": pending_handoff.get("handoff_name"),
            "from_agent": pending_handoff.get("from_agent"),
        },
    }
