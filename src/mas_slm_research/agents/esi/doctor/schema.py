"""Schema module helpers."""

from pydantic import BaseModel, Field, AliasChoices, model_validator
from typing import List, Literal, Optional, Any, Union


class DoctorAgentInput(BaseModel):
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

    source_agent: Literal["esi1_agent", "esi2_agent", "esi345_agent"] = Field(
        ...,
        description="Which upstream triage pathway is sending the case to the doctor agent."
    )

    esi1_summary: Optional[str] = Field(
        default=None,
        description="Summary from the ESI-1 agent if the case came from the ESI-1 pathway."
    )
    esi1_reason: Optional[str] = Field(
        default=None,
        description="Reason from the ESI-1 agent if available."
    )
    esi1_critical_concerns: List[str] = Field(
        default_factory=list,
        description="Immediate threats identified by the ESI-1 agent."
    )

    esi2_summary: Optional[str] = Field(
        default=None,
        description="Summary from the ESI-2 agent if the case came from the ESI-2 pathway."
    )
    esi2_reason: Optional[str] = Field(
        default=None,
        description="Reason from the ESI-2 agent if available."
    )
    esi2_critical_concerns: List[str] = Field(
        default_factory=list,
        description="High-risk concerns identified by the ESI-2 agent."
    )

    esi_level_345: Optional[Literal[3, 4, 5]] = Field(
        default=None,
        description="Predicted ESI level from the ESI-345 agent."
    )
    num_resources: Optional[int] = Field(
        default=None,
        description="Predicted number of ESI-counted resources from the ESI-345 agent."
    )
    predicted_resources: List[str] = Field(
        default_factory=list,
        description="Predicted ESI-counted resources from the ESI-345 agent."
    )
    esi345_reason: Optional[str] = Field(
        default=None,
        description="Reasoning from the ESI-345 agent."
    )
    esi345_concerns: List[str] = Field(
        default_factory=list,
        description="Concerns or caveats raised by the ESI-345 agent."
    )

    vitals_consider_uptriage: Optional[bool] = Field(
        default=None,
        description="Whether the vitals agent recommends considering up-triage."
    )
    vitals_urgency: Optional[Literal["low", "moderate", "high"]] = Field(
        default=None,
        description="Urgency label from the vitals agent."
    )
    vitals_reason: Optional[str] = Field(
        default=None,
        description="Reasoning from the vitals agent."
    )
    abnormal_vitals: List[str] = Field(
        default_factory=list,
        description="Abnormal vital signs identified by the vitals agent."
    )
    vitals_confidence: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Confidence from the vitals agent."
    )

class DoctorAgentOutput(BaseModel):
    final_esi_level: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Final ESI level after doctor-agent routing review."
    )
    uptriaged: bool = Field(
        ...,
        description="True only if an ESI-345 result was escalated to ESI-2 because of vitals."
    )

    decision_source: Literal[
        "esi1",
        "esi2",
        "esi345",
        "vitals"
    ] = Field(
        ...,
        description="Routing rule used to produce the final decision."
    )

    audit_summary: str = Field(
        ...,
        description="One concise sentence explaining the routing decision."
    )

    abnormal_vitals_considered: List[str] = Field(
        default_factory=list,
        description="Vitals considered only for ESI-345 up-triage review."
    )
