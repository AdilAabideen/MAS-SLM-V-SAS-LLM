"""Compute Shock Index module helpers."""

from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool


class ShockIndexInput(BaseModel):
    hr: Optional[float] = Field(default=None, description="Heart rate in beats per minute")
    sbp: Optional[float] = Field(default=None, description="Systolic blood pressure in mmHg")

    @field_validator("hr")
    @classmethod
    def validate_hr(cls, v: Optional[float]) -> Optional[float]:
        """Validate hr."""
        # Fail fast on bad input.
        if v is not None and v < 0:
            raise ValueError("hr must be >= 0")
        return v

    @field_validator("sbp")
    @classmethod
    def validate_sbp(cls, v: Optional[float]) -> Optional[float]:
        """Validate sbp."""
        # Fail fast on bad input.
        if v is not None and v <= 0:
            raise ValueError("sbp must be > 0")
        return v


@tool("compute_shock_index", args_schema=ShockIndexInput)
def compute_shock_index(
    hr: Optional[float],
    sbp: Optional[float]
) -> Dict[str, Any]:
    """
    Compute shock index (SI = HR / SBP) and return a compact structured result.

    Intended use:
    - supports reassessment or uptriage concern
    - does not determine final clinical disposition alone

    Banding:
    - normal: SI < 0.9
    - soft:   0.9 <= SI < 1.0
    - hard:   SI >= 1.0
    """

    # Derive the needed value.
    missing = []
    if hr is None:
        missing.append("HR")
    if sbp is None:
        missing.append("SBP")

    if missing:
        return {
            "ok": False,
            "missing_vitals": missing,
            "si": None,
            "band": None,
            "summary": f"Shock index could not be computed; missing: {', '.join(missing)}."
        }

    si = hr / sbp

    if si >= 1.0:
        band = "hard"
        interpretation = "Supports concern for hemodynamic instability."
    elif si >= 0.9:
        band = "soft"
        interpretation = "Borderline elevation; interpret with other findings."
    else:
        band = "normal"
        interpretation = "No shock-index elevation detected."

    return {
        "ok": True,
        "si": round(si, 3),
        "band": band,
        "thresholds": {
            "soft_ge": 0.9,
            "hard_ge": 1.0
        },
        "interpretation": interpretation,
        "summary": f"Shock index {round(si, 3)} in {band} band."
    }
