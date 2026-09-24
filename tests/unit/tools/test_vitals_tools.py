"""Test Vitals Tools test coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mas_slm_research.tools.esi.compute_esi_danger_zone import (
    ESI_Danger_Zone_Vitals,
    compute_esi_danger_zone,
)
from mas_slm_research.tools.esi.compute_shock_index import ShockIndexInput, compute_shock_index


@pytest.mark.unit
def test_ut_tol_001_shock_index_returns_missing_vitals_payload_when_hr_missing():
    """Handle ut tol 001 shock index returns missing vitals payload when hr missing."""
    # Keep the main step clear.
    result = compute_shock_index.invoke({"hr": None, "sbp": 120})
    assert result["ok"] is False
    assert result["missing_vitals"] == ["HR"]


@pytest.mark.unit
def test_ut_tol_003_shock_index_normal_band():
    """Handle ut tol 003 shock index normal band."""
    # Keep the main step clear.
    result = compute_shock_index.invoke({"hr": 80, "sbp": 120})
    assert result["band"] == "normal"


@pytest.mark.unit
def test_ut_tol_005_shock_index_hard_band():
    """Handle ut tol 005 shock index hard band."""
    # Keep the main step clear.
    result = compute_shock_index.invoke({"hr": 120, "sbp": 100})
    assert result["band"] == "hard"


@pytest.mark.unit
def test_ut_tol_006_shock_index_schema_rejects_negative_hr():
    """Handle ut tol 006 shock index schema rejects negative hr."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ShockIndexInput(hr=-1, sbp=100)


@pytest.mark.unit
def test_ut_tol_007_shock_index_schema_rejects_non_positive_sbp():
    """Handle ut tol 007 shock index schema rejects non positive sbp."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ShockIndexInput(hr=80, sbp=0)


@pytest.mark.unit
def test_ut_tol_008_danger_zone_adult_no_violations():
    """Handle ut tol 008 danger zone adult no violations."""
    # Keep the main step clear.
    result = compute_esi_danger_zone.invoke(
        {"age_years": 25, "hr": 90, "rr": 18, "spo2": 98, "has_respiratory_compromise": False}
    )
    assert result["status"] == "none"


@pytest.mark.unit
def test_ut_tol_010_danger_zone_red_hr_violation():
    """Handle ut tol 010 danger zone red hr violation."""
    # Keep the main step clear.
    result = compute_esi_danger_zone.invoke(
        {"age_years": 25, "hr": 106, "rr": 18, "spo2": 98, "has_respiratory_compromise": False}
    )
    assert result["status"] == "red"


@pytest.mark.unit
def test_ut_tol_012_danger_zone_spo2_ignored_when_no_respiratory_compromise():
    """Handle ut tol 012 danger zone spo2 ignored when no respiratory compromise."""
    # Keep the main step clear.
    result = compute_esi_danger_zone.invoke(
        {"age_years": 25, "hr": 90, "rr": 18, "spo2": 80, "has_respiratory_compromise": False}
    )
    assert result["status"] == "none"


@pytest.mark.unit
def test_ut_tol_014_danger_zone_spo2_red_when_respiratory_compromise_present():
    """Handle ut tol 014 danger zone spo2 red when respiratory compromise present."""
    # Keep the main step clear.
    result = compute_esi_danger_zone.invoke(
        {"age_years": 25, "hr": 90, "rr": 18, "spo2": 90, "has_respiratory_compromise": True}
    )
    assert result["status"] == "red"


@pytest.mark.unit
def test_ut_tol_015_danger_zone_returns_missing_vital_names_when_missing():
    """Handle ut tol 015 danger zone returns missing vital names when missing."""
    # Keep the main step clear.
    result = compute_esi_danger_zone.invoke(
        {"age_years": 25, "hr": None, "rr": None, "spo2": None, "has_respiratory_compromise": True}
    )
    assert result["missing_vitals"] == ["HR", "RR", "SpO2"]


@pytest.mark.unit
def test_ut_tol_017_danger_zone_schema_rejects_spo2_outside_range():
    """Handle ut tol 017 danger zone schema rejects spo2 outside range."""
    # Keep the main step clear.
    with pytest.raises(ValidationError):
        ESI_Danger_Zone_Vitals(age_years=20, hr=80, rr=20, spo2=101, has_respiratory_compromise=False)
