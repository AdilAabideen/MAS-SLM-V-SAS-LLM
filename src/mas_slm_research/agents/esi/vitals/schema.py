"""Schema module helpers."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator


class VitalsAgentInput(BaseModel):
    temperature: float
    heartrate: float
    resprate: float
    o2sat: float
    sbp: float
    dbp: float
    pain: float
    age_years: float
    chiefcomplaint: str


class VitalsRecommendation(BaseModel):
    consider_uptriage: bool = Field(description="Whether the agent recommends uptriage")
    reasoning_consider_uptriage: str = Field(
        description="The reasoning behind the recommendation to consider uptriage"
    )
    abnormal_vitals: list[str] = Field(
        default_factory=list,
        description="A list of vitals that are abnormal or physiologically concerning",
    )
    confidence: Union[float, Literal["low", "medium", "high"]] = Field(
        description="The confidence in the recommendation of the agent"
    )


class VitalsAgentOutput(BaseModel):
    ok: bool = Field(default=True)
    recommendation: VitalsRecommendation

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_or_nested_payload(cls, value):
        """Handle flat or nested payload."""
        # Keep the main step clear.
        if not isinstance(value, dict):
            return value

        if isinstance(value.get("recommendation"), dict):
            normalized = dict(value)
            normalized.setdefault("ok", True)
            return normalized

        recommendation_keys = {
            "consider_uptriage",
            "reasoning_consider_uptriage",
            "abnormal_vitals",
            "confidence",
        }
        if recommendation_keys.intersection(value.keys()):
            recommendation = {
                "consider_uptriage": value.get("consider_uptriage"),
                "reasoning_consider_uptriage": value.get("reasoning_consider_uptriage"),
                "abnormal_vitals": value.get("abnormal_vitals", []),
                "confidence": value.get("confidence"),
            }
            return {
                "ok": value.get("ok", True),
                "recommendation": recommendation,
            }

        return value
