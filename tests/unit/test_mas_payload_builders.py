"""Test Mas Payload Builders test coverage."""

from __future__ import annotations

import pytest

from mas_slm_research.workflows.esi.payload_builder import build_pending_agent_payload
from mas_slm_research.workflows.esi.payloads.common import (
    left_side_payload,
    right_side_payload,
    unified_payload_builder,
)


def _sample_case_info() -> dict:
    """Handle case info."""
    # Keep the main step clear.
    return {
        "gender": "M",
        "race": "WHITE",
        "arrival_transport": "AMBULANCE",
        "pain": "8",
        "chiefcomplaint": "Chest pain",
        "age": 63.0,
        "tiragecase": "63-year-old male with chest pain and dizziness.",
        "temperature": 98.6,
        "heartrate": 122.0,
        "resprate": 24.0,
        "o2sat": 92.0,
        "sbp": 88.0,
        "dbp": 54.0,
    }


@pytest.mark.unit
def test_ut_plb_001_left_side_payload_projects_shared_triage_fields():
    """Handle ut plb 001 left side payload projects shared triage fields."""
    # Keep the main step clear.
    payload = left_side_payload(_sample_case_info())

    assert payload["gender"] == "M"
    assert payload["race"] == "WHITE"
    assert payload["arrival_transport"] == "AMBULANCE"
    assert payload["age"] == 63.0
    assert payload["age_years"] == 63.0
    assert payload["pain"] == "8"
    assert payload["chief_complaint"] == "Chest pain"
    assert payload["chiefcomplaint"] == "Chest pain"
    assert payload["agent_triage_case"] == "63-year-old male with chest pain and dizziness."
    assert payload["tiragecase"] == "63-year-old male with chest pain and dizziness."
    assert "temperature" not in payload


@pytest.mark.unit
def test_ut_plb_002_right_side_payload_projects_structured_vitals_fields():
    """Handle ut plb 002 right side payload projects structured vitals fields."""
    # Keep the main step clear.
    payload = right_side_payload(_sample_case_info())

    assert payload["temperature"] == 98.6
    assert payload["heart_rate"] == 122.0
    assert payload["heartrate"] == 122.0
    assert payload["respiratory_rate"] == 24.0
    assert payload["resprate"] == 24.0
    assert payload["o2_sat"] == 92.0
    assert payload["o2sat"] == 92.0
    assert payload["sbp"] == 88.0
    assert payload["dbp"] == 54.0
    assert "gender" not in payload


@pytest.mark.unit
def test_ut_plb_003_unified_payload_builder_hides_vitals_from_esi_agents():
    """Handle ut plb 003 unified payload builder hides vitals from ESI agents."""
    # Keep the main step clear.
    payload = unified_payload_builder("esi2_agent", _sample_case_info())

    assert payload["chief_complaint"] == "Chest pain"
    assert payload["agent_triage_case"] == "63-year-old male with chest pain and dizziness."
    assert "temperature" not in payload
    assert "heartrate" not in payload
    assert "heart_rate" not in payload
    assert "o2sat" not in payload


@pytest.mark.unit
def test_ut_plb_004_unified_payload_builder_exposes_vitals_to_vitals_and_doctor():
    """Handle ut plb 004 unified payload builder exposes vitals to vitals and doctor."""
    # Keep the main step clear.
    vitals_payload = unified_payload_builder("vitals_agent", _sample_case_info())
    doctor_payload = unified_payload_builder("doctor_agent", _sample_case_info())

    for payload in (vitals_payload, doctor_payload):
        assert payload["chief_complaint"] == "Chest pain"
        assert payload["temperature"] == 98.6
        assert payload["heart_rate"] == 122.0
        assert payload["heartrate"] == 122.0
        assert payload["age_years"] == 63.0


@pytest.mark.unit
def test_ut_plb_005_build_pending_agent_payload_applies_visibility_policy():
    """Handle ut plb 005 build pending agent payload applies visibility policy."""
    # Keep the main step clear.
    state = {
        "case_info": _sample_case_info(),
        "pending_handoff": None,
        "handoff_history": [],
    }

    esi1_case_info = build_pending_agent_payload("esi1_agent", state)["llm_payload"]["case_info"]
    doctor_case_info = build_pending_agent_payload("doctor_agent", state)["llm_payload"]["case_info"]

    assert "temperature" not in esi1_case_info
    assert "heart_rate" not in esi1_case_info
    assert esi1_case_info["chief_complaint"] == "Chest pain"

    assert doctor_case_info["temperature"] == 98.6
    assert doctor_case_info["heart_rate"] == 122.0
    assert doctor_case_info["chief_complaint"] == "Chest pain"
