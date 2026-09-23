"""Payload Builders package exports."""

from __future__ import annotations

from typing import Any, Dict

from mas_slm_research.mas_contract import AgentName


def left_side_payload(case_info: Dict[str, Any]) -> Dict[str, Any]:
    """Handle side payload."""
    # Keep the main step clear.
    chief_complaint = case_info.get("chiefcomplaint")
    triage_case = case_info.get("tiragecase")
    age = case_info.get("age")

    return {
        "gender": case_info.get("gender"),
        "race": case_info.get("race"),
        "arrival_transport": case_info.get("arrival_transport"),
        "age": age,
        "age_years": age,
        "pain": case_info.get("pain"),
        "chief_complaint": chief_complaint,
        "chiefcomplaint": chief_complaint,
        "agent_triage_case": triage_case,
        "tiragecase": triage_case,
    }


def right_side_payload(case_info: Dict[str, Any]) -> Dict[str, Any]:
    """Handle side payload."""
    # Keep the main step clear.
    heart_rate = case_info.get("heartrate")
    respiratory_rate = case_info.get("resprate")
    o2_sat = case_info.get("o2sat")

    return {
        "temperature": case_info.get("temperature"),
        "heart_rate": heart_rate,
        "heartrate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "resprate": respiratory_rate,
        "o2_sat": o2_sat,
        "o2sat": o2_sat,
        "sbp": case_info.get("sbp"),
        "dbp": case_info.get("dbp"),
    }


def unified_payload_builder(agent_name: AgentName, case_info: Dict[str, Any]) -> Dict[str, Any]:
    """Handle payload builder."""
    # Keep the main step clear.
    projected = left_side_payload(case_info)
    if agent_name in {"vitals_agent", "doctor_agent"}:
        projected.update(right_side_payload(case_info))
    return projected
