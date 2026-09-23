"""Handoffs module helpers."""

from typing import ClassVar, List, Literal

from pydantic import AliasChoices, Field

from mas_slm_research.handoff import CoerciveHandoffPayload, define_handoff


class ESI345ToDoctorPayload(CoerciveHandoffPayload):
    _list_fields: ClassVar[frozenset[str]] = frozenset({"predicted_resources"})

    esi_level: Literal[3, 4, 5] = Field(
        ...,
        description="Which ESI level the case is currently judged to be in before doctor review."
    )
    num_resources: int = Field(
        ...,
        ge=0,
        description="The predicted number of ESI-counted resources required for the case."
    )
    predicted_resources: List[str] = Field(
        default_factory=list,
        description="Specific ESI resources likely needed, if any."
    )
    reason: str = Field(
        ...,
        validation_alias=AliasChoices("reason", "justification"),
        description="Brief rationale for the provisional ESI decision."
    )


HANDOFFS = [
    define_handoff(
        source_agent="esi345_agent",
        target_agent="doctor_agent",
        payload_model=ESI345ToDoctorPayload,
        description="Transfer to Doctor when the ESI-345 assessment identifies concerns requiring escalation or review.",
        tool_name="final_esi345_result_handoff_to_doctor_agent"
    ),
]
