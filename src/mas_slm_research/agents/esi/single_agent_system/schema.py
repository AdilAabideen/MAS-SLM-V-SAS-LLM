"""Schema module helpers."""

from pydantic import BaseModel, Field, AliasChoices
from typing import List, Literal, Optional, Any, Union


class SingleAgentInput(BaseModel):
    gender: str = Field(description="Recorded patient gender at the time of triage.")
    race: str = Field(description="Recorded patient race or ethnicity category used in the source dataset.")
    arrival_transport: str = Field(
        validation_alias=AliasChoices("arrival_transport", "arrivaltransport", "arrival", "transfer"),
        description="How the patient arrived to the emergency department, such as walk-in or ambulance.",
    )
    pain: Optional[str] = Field(description="Reported pain level or pain assessment captured during intake.")
    chiefcomplaint: str = Field(
        validation_alias=AliasChoices("chiefcomplaint", "chief complaint", "chief_complaint"),
        description="Primary presenting complaint recorded at triage.",
    )
    age: Optional[float] = Field(description="Patient age at the time of the encounter.")
    tiragecase: str = Field(
        validation_alias=AliasChoices("tiragecase", "triage case", "triage_case"),
        description="Free-text triage case summary or scenario narrative.",
    )
    temperature: Optional[float] = Field(description="Measured body temperature during triage, typically in Celsius.")
    heartrate: Optional[float] = Field(description="Measured heart rate in beats per minute.")
    resprate: Optional[float] = Field(description="Measured respiratory rate in breaths per minute.")
    o2sat: Optional[float] = Field(description="Measured peripheral oxygen saturation percentage.")
    sbp: Optional[float] = Field(description="Measured systolic blood pressure.")
    dbp: Optional[float] = Field(description="Measured diastolic blood pressure.")

class SingleAgentOutput(BaseModel):
    final_esi_level: Literal[1, 2, 3, 4, 5, "1", "2", "3", "4", "5"] = Field(
        ...,
        description="Final ESI level assigned by the single-agent ESI workflow."
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in the final ESI decision, from 0 to 1."
    )

    decision_source: Literal[
        "esi1_decision_point_a",
        "esi2_decision_point_b",
        "esi345_resource_prediction",
        "esi345_uptriaged_to_esi2_by_vitals",
    ] = Field(
        ...,
        description="Which ESI decision stage produced the final decision."
    )

    uptriaged: bool = Field(
        ...,
        description="True only if an initial ESI-3/4/5 resource-based result was changed to ESI-2 because of vital signs."
    )

    initial_resource_based_esi_level: Optional[Literal[3, 4, 5, "3", "4", "5"]] = Field(
        default=None,
        description="The initial ESI-3/4/5 level before vitals up-triage, if applicable."
    )

    num_resources: Optional[int] = Field(
        default=None,
        ge=0,
        description="Predicted number of distinct ESI-counted resources, if resource prediction was used."
    )

    predicted_resources: List[str] = Field(
        default_factory=list,
        description="Specific ESI-counted resources likely needed, such as labs, ECG, radiograph, CT, IV fluids, IV medications, or consultation."
    )

    abnormal_vitals_considered: bool = Field(
        ...,
        description="Whether vital signs affected the risk assessment or up-triage consideration."
    )

    vitals_summary: str = Field(
        ...,
        description="Concise summary of available vital-sign concerns, reassuring vitals, or missing vital signs."
    )

    case_summary: str = Field(
        ...,
        description="Brief clinician-facing summary of the patient case."
    )

    key_risks: List[str] = Field(
        default_factory=list,
        description="Important acute risks or red flags identified during ESI assessment."
    )

    missing_information: List[str] = Field(
        default_factory=list,
        description="Only genuinely decision-relevant missing information."
    )

    rationale: str = Field(
        ...,
        description="Concise explanation of why the final ESI level was selected."
    )

    next_actions: List[str] = Field(
        default_factory=list,
        description="Immediate triage-facing next actions or workflow suggestions."
    )