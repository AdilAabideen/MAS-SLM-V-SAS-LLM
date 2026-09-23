"""Handoffs module helpers."""

from typing import ClassVar, List

from pydantic import Field

from mas_slm_research.handoff import CoerciveHandoffPayload, define_handoff


class VitalsToDoctorPayload(CoerciveHandoffPayload):
    _bool_fields: ClassVar[frozenset[str]] = frozenset({"consider_uptriage"})
    _list_fields: ClassVar[frozenset[str]] = frozenset({"abnormal_vitals"})

    consider_uptriage: bool = Field(
        ...,
        description="Whether the vitals agent recommends that the case should be considered for up-triage."
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why the vitals pattern is concerning."
    )
    abnormal_vitals: List[str] = Field(
        default_factory=list,
        description="Specific abnormal vital signs or physiological concerns identified."
    )
    confidence: float = Field(
        ...,
        description="Confidence of the vitals agent in the recommendation."
    )

HANDOFFS = [
    define_handoff(
        source_agent="vitals_agent",
        target_agent="doctor_agent",
        payload_model=VitalsToDoctorPayload,
        description="Transfer to the Doctor the Information About the Vitals",
        tool_name="finalise_output"
    ),
]
