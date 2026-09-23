"""Compute Esi Danger Zone module helpers."""

from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator


class ESI_Danger_Zone_Vitals(BaseModel):
    age_years: float = Field(description="Patient age in years")
    hr: Optional[float] = Field(default=None, description="Heart rate in beats per minute")
    rr: Optional[float] = Field(default=None, description="Respiratory rate in breaths per minute")
    spo2: Optional[float] = Field(default=None, description="Oxygen saturation percentage")
    has_respiratory_compromise: bool = Field(
        description="Whether the case suggests respiratory compromise from the presentation"
    )

    @field_validator("age_years")
    @classmethod
    def validate_age(cls, v: float) -> float:
        """Validate age."""
        # Fail fast on bad input.
        if v < 0:
            raise ValueError("age_years must be >= 0")
        return v

    @field_validator("hr")
    @classmethod
    def validate_hr(cls, v: Optional[float]) -> Optional[float]:
        """Validate hr."""
        # Fail fast on bad input.
        if v is not None and v < 0:
            raise ValueError("hr must be >= 0")
        return v

    @field_validator("rr")
    @classmethod
    def validate_rr(cls, v: Optional[float]) -> Optional[float]:
        """Validate rr."""
        # Fail fast on bad input.
        if v is not None and v < 0:
            raise ValueError("rr must be >= 0")
        return v

    @field_validator("spo2")
    @classmethod
    def validate_spo2(cls, v: Optional[float]) -> Optional[float]:
        """Validate spo2."""
        # Fail fast on bad input.
        if v is not None and not (0 <= v <= 100):
            raise ValueError("spo2 must be between 0 and 100")
        return v


@tool("compute_esi_danger_zone", args_schema=ESI_Danger_Zone_Vitals)
def compute_esi_danger_zone(
    age_years: float,
    hr: Optional[float],
    rr: Optional[float],
    spo2: Optional[float],
    has_respiratory_compromise: bool,
) -> Dict[str, Any]:
    """
    Compute compact ESI danger-zone signals from age-adjusted HR/RR thresholds and
    conditional SpO2 thresholds.

    This tool is intended to provide structured escalation signals, not a final triage decision.

    Returns:
        {
          "ok": bool,
          "age_band": str,
          "thresholds": {
            "hr_gt": float,
            "rr_gt": float,
            "spo2_lt_when_respiratory_compromise": float
          },
          "status": "none" | "orange" | "red",
          "red_count": int,
          "orange_count": int,
          "missing_vitals": list[str],
          "violations": list[{
              "vital": str,
              "level": "orange" | "red",
              "direction": "high" | "low",
              "value": float,
              "threshold": float,
              "age_band": str
          }],
          "summary": str
        }
    """
    # Age band thresholds based on the tool's intended ESI danger-zone logic
    if age_years < (1 / 12):  # <1 month
        hr_thr, rr_thr = 190.0, 60.0
        age_band = "<1_month"
    elif age_years < 1:  # 1-12 months
        hr_thr, rr_thr = 180.0, 55.0
        age_band = "1_12_months"
    elif age_years < 3:  # 1-3 years
        hr_thr, rr_thr = 140.0, 40.0
        age_band = "1_3_years"
    elif age_years < 5:  # 3-5 years
        hr_thr, rr_thr = 120.0, 35.0
        age_band = "3_5_years"
    elif age_years < 12:  # 5-12 years
        hr_thr, rr_thr = 120.0, 30.0
        age_band = "5_12_years"
    elif age_years < 18:  # 12-18 years
        hr_thr, rr_thr = 100.0, 20.0
        age_band = "12_18_years"
    else:
        hr_thr, rr_thr = 100.0, 20.0
        age_band = "adult"

    # Orange = over threshold but within overband.
    # Red = beyond overband.
    OVERBAND_PCT = 0.05
    SPO2_THR = 92.0
    SPO2_OVERBAND_PCT = 0.01  # 1% below threshold becomes red cutpoint

    missing_vitals: List[str] = []
    violations: List[Dict[str, Any]] = []

    def add_upper_violation(vital: str, value: Optional[float], threshold: float) -> None:
        """Handle upper violation."""
        # Keep the main step clear.
        if value is None:
            missing_vitals.append(vital)
            return

        red_cutoff = threshold * (1 + OVERBAND_PCT)
        if value > red_cutoff:
            violations.append(
                {
                    "vital": vital,
                    "level": "red",
                    "direction": "high",
                    "value": float(value),
                    "threshold": round(red_cutoff, 2),
                    "age_band": age_band,
                }
            )
        elif value > threshold:
            violations.append(
                {
                    "vital": vital,
                    "level": "orange",
                    "direction": "high",
                    "value": float(value),
                    "threshold": float(threshold),
                    "age_band": age_band,
                }
            )

    add_upper_violation("HR", hr, hr_thr)
    add_upper_violation("RR", rr, rr_thr)

    # Only check SpO2 if respiratory compromise is present
    if has_respiratory_compromise:
        if spo2 is None:
            missing_vitals.append("SpO2")
        else:
            red_cutoff_spo2 = SPO2_THR * (1 - SPO2_OVERBAND_PCT)
            if spo2 < red_cutoff_spo2:
                violations.append(
                    {
                        "vital": "SpO2",
                        "level": "red",
                        "direction": "low",
                        "value": float(spo2),
                        "threshold": round(red_cutoff_spo2, 2),
                        "age_band": age_band,
                    }
                )
            elif spo2 < SPO2_THR:
                violations.append(
                    {
                        "vital": "SpO2",
                        "level": "orange",
                        "direction": "low",
                        "value": float(spo2),
                        "threshold": float(SPO2_THR),
                        "age_band": age_band,
                    }
                )

    red_count = sum(v["level"] == "red" for v in violations)
    orange_count = sum(v["level"] == "orange" for v in violations)

    if red_count > 0:
        status = "red"
    elif orange_count > 0:
        status = "orange"
    else:
        status = "none"

    if status == "red":
        summary = f"Red danger-zone status with {red_count} red and {orange_count} orange violations."
    elif status == "orange":
        summary = f"Orange danger-zone status with {orange_count} orange violations."
    else:
        summary = "No danger-zone violations detected."

    return {
        "ok": True,
        "age_band": age_band,
        "thresholds": {
            "hr_gt": hr_thr,
            "rr_gt": rr_thr,
            "spo2_lt_when_respiratory_compromise": SPO2_THR,
        },
        "status": status,
        "red_count": red_count,
        "orange_count": orange_count,
        "missing_vitals": missing_vitals,
        "violations": violations,
        "summary": summary,
    }
