"""Test Agent Output Schemas schema types."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from mas_slm_research.agents.esi.esi1.schema import ES1AgentInput, ES1AgentOutput
from mas_slm_research.agents.esi.esi2.schema import ES2AgentInput, ES2AgentOutput
from mas_slm_research.agents.esi.esi345.schema import ES345AgentInput, ES345AgentOutput
from mas_slm_research.agents.esi.vitals.schema import VitalsAgentOutput


@pytest.mark.unit
def test_ut_sch_001_vitals_output_schema_accepts_valid_recommendation_payload():
    """Handle ut sch 001 vitals output schema accepts valid recommendation payload."""
    # Keep the main step clear.
    payload = VitalsAgentOutput.model_validate(
        {
            "ok": True,
            "recommendation": {
                "consider_uptriage": True,
                "reasoning_consider_uptriage": "because",
                "confidence": "high",
            },
        }
    )
    assert payload.ok is True


@pytest.mark.unit
def test_ut_sch_002_esi1_output_schema_rejects_confidence_outside_zero_one():
    """Handle ut sch 002 ESI1 output schema rejects confidence outside zero one."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ES1AgentOutput.model_validate(
            {"is_esi1": True, "confidence": 2, "case_summary": "x", "key_risks": [], "missing_information": [], "justification": "x"}
        )


@pytest.mark.unit
def test_ut_sch_003_esi2_output_schema_rejects_confidence_outside_zero_one():
    """Handle ut sch 003 ESI2 output schema rejects confidence outside zero one."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ES2AgentOutput.model_validate(
            {"is_esi2": True, "confidence": -1, "case_summary": "x", "key_risks": [], "missing_information": [], "justification": "x"}
        )


@pytest.mark.unit
def test_ut_sch_004_esi345_output_schema_rejects_esi_level_outside_three_four_five():
    """Handle ut sch 004 ESI345 output schema rejects ESI level outside three four five."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ES345AgentOutput.model_validate(
            {"esi_level": 2, "num_resources": 1, "predicted_resources": [], "confidence": 0.5, "case_summary": "x", "key_risks": [], "missing_information": [], "justification": "x"}
        )


@pytest.mark.unit
def test_ut_sch_005_esi345_output_schema_rejects_negative_num_resources():
    """Handle ut sch 005 ESI345 output schema rejects negative num resources."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ES345AgentOutput.model_validate(
            {"esi_level": 3, "num_resources": -1, "predicted_resources": [], "confidence": 0.5, "case_summary": "x", "key_risks": [], "missing_information": [], "justification": "x"}
        )


@pytest.mark.unit
def test_ut_sch_006_esi1_input_accepts_alias_fields():
    """Handle ut sch 006 ESI1 input accepts alias fields."""
    # Keep the main step clear.
    payload = ES1AgentInput.model_validate(
        {"gender": "m", "race": "x", "arrival": "walk", "pain": "1", "chief complaint": "pain", "age": 20, "triage case": "case"}
    )
    assert payload.arrival_transport == "walk"

