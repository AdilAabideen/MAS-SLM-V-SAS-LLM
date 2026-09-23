"""Schema module helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, List, Union

from pydantic import AliasChoices, BaseModel, Field


class ES345AgentInput(BaseModel):
    gender: str
    race: str
    arrival_transport: str = Field(
        validation_alias=AliasChoices("arrival_transport", "arrivaltransport", "arrival", "transfer")
    )
    pain: str
    chiefcomplaint: str = Field(
        validation_alias=AliasChoices("chiefcomplaint", "chief complaint", "chief_complaint")
    )
    age: Union[float, int]
    tiragecase: str = Field(
        validation_alias=AliasChoices("tiragecase", "triage case", "triage_case")
    )



class ES345AgentOutput(BaseModel):
    esi_level: Literal[3, 4, 5] = Field(
        ...,
        description="Which ESI level the case is in"
    )
    num_resources: int = Field(
        ...,
        ge=0,
        description="The number of resources required for the case."
    )
    predicted_resources: List[str] = Field(
        default_factory=list,
        description="Specific ESI resources likely needed, if any."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 and 1."
    )
    justification: str = Field(
        ...,
        description="Brief rationale for the provisional ESI decision."
    )



