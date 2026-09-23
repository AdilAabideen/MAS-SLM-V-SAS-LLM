"""Handoffs module helpers."""

from typing import ClassVar, List, Literal

from pydantic import AliasChoices, Field

from mas_slm_research.handoff import CoerciveHandoffPayload, define_handoff


class ESI1ToESI2Payload(CoerciveHandoffPayload):
    _bool_fields: ClassVar[frozenset[str]] = frozenset({"is_esi1"})
    _list_fields: ClassVar[frozenset[str]] = frozenset({"carry_forward_concerns"})

    is_esi1: Literal[False] = Field(
        ...,
        description="Confirmed ESI-1 decision"
    )
    brief_reason: str = Field(
        ...,
        validation_alias=AliasChoices("brief_reason", "reason"),
        description="Short explanation of why the case was not judged to require immediate life-saving intervention from the available information."
    )
    carry_forward_concerns: List[str] = Field(
        default_factory=list,
        description="Key concerns or unresolved issues that the ESI-2 agent should keep in mind during high-risk assessment."
    )

class ESI1ToDoctorPayload(CoerciveHandoffPayload):
    _bool_fields: ClassVar[frozenset[str]] = frozenset({"is_esi1"})
    _list_fields: ClassVar[frozenset[str]] = frozenset({"critical_concerns"})

    is_esi1: Literal[True] = Field(
        ...,
        description="Confirmed ESI-1 decision"
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why the patient appears to meet ESI-1 criteria."
    )
    critical_concerns: List[str] = Field(
        default_factory=list,
        description="Key immediate threats or red flags identified from the case, such as airway compromise, severe respiratory distress, shock, or unresponsiveness."
    )

HANDOFFS = [
    define_handoff(
        source_agent="esi1_agent",
        target_agent="esi2_agent",
        payload_model=ESI1ToESI2Payload,
        description="Transfer to ESI-2 when ESI-1 criteria are not met.",
        tool_name="final_esi1_false_handoff_to_esi2_agent"
    ),
    define_handoff(
        source_agent="esi1_agent",
        target_agent="doctor_agent",
        payload_model=ESI1ToDoctorPayload,
        description="Transfer to Doctor when the Case is Confirmed to be ESI1.",
        tool_name="final_esi1_true_handoff_to_doctor_agent"
    )
]
