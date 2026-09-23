"""Payload Builder module helpers."""

from __future__ import annotations

from typing import Any, Dict

from mas_slm_research.mas_contract import MASState
from mas_slm_research.workflows.esi.payloads.common import (
    unified_payload_builder,
)


def build_payload(state: MASState) -> Dict[str, Any]:
    """Build a compact ESI1 mas payload."""
    # Build the next value.
    case_info = unified_payload_builder("esi1_agent", dict(state.get("case_info") or {}))
    llm_payload = {
        "task": "Evaluate this patient for ESI Decision Point A only.",
        "agent_role": "esi1_agent",
        "case_info": case_info,
        "handoff_context": None,
    }
    return {
        "llm_payload": llm_payload,
        "metadata": {
            "agent_role": "esi1_agent",
            "uses_handoff": False,
        },
    }
