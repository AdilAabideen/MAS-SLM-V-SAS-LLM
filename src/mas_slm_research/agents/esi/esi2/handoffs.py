"""Handoffs module helpers."""

from typing import Any, ClassVar, List, Literal

from pydantic import Field, model_validator

from mas_slm_research.handoff import CoerciveHandoffPayload, define_handoff

class ESI2ToESI345Payload(CoerciveHandoffPayload):
    _bool_fields: ClassVar[frozenset[str]] = frozenset({"is_esi2"})
    _list_fields: ClassVar[frozenset[str]] = frozenset({"carry_forward_concerns"})

    @model_validator(mode="before")
    @classmethod
    def _coerce_stale_false_branch_fields(cls, value: Any) -> Any:
        """Handle stale false branch fields."""
        # Keep the main step clear.
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if "is_esi2" not in data:
            stale_result = str(data.get("esi1_result", "")).strip().lower()
            stale_decision = str(data.get("decision", "")).strip().lower()
            if stale_result == "not_esi2" or stale_decision == "not_esi2":
                data["is_esi2"] = False
        return data

    is_esi2: Literal[False] = Field(
        ...,
        description="Confirmed ESI-2 decision."
    )
    reason: str = Field(
        ...,
        description="Short explanation of why the case was not judged to require immediate life-saving intervention from the available information."
    )
    carry_forward_concerns: List[str] = Field(
        default_factory=list,
        description="Key concerns or unresolved issues that the ESI-3/4/5 agent should keep in mind during resource prediction."
    )

class ESI2ToDoctorPayload(CoerciveHandoffPayload):
    _bool_fields: ClassVar[frozenset[str]] = frozenset({"is_esi2"})
    _list_fields: ClassVar[frozenset[str]] = frozenset({"critical_concerns"})

    @model_validator(mode="before")
    @classmethod
    def _coerce_stale_true_branch_fields(cls, value: Any) -> Any:
        """Handle stale true branch fields."""
        # Keep the main step clear.
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if "is_esi2" not in data:
            stale_decision = str(data.get("decision", "")).strip().lower()
            if stale_decision == "esi2" or "reason" in data or "critical_concerns" in data:
                data["is_esi2"] = True
        return data

    is_esi2: Literal[True] = Field(
        ...,
        description="Confirmed ESI-2 decision."
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why the patient appears to meet ESI-2 criteria."
    )
    critical_concerns: List[str] = Field(
        default_factory=list,
        description="Key immediate threats or red flags identified from the case, such as airway compromise, severe respiratory distress, shock, or unresponsiveness."
    )

HANDOFFS = [
    define_handoff(
        source_agent="esi2_agent",
        target_agent="esi345_agent",
        payload_model=ESI2ToESI345Payload,
        description="Transfer to ESI-3/4/5 when ESI-2 criteria are not met.",
        tool_name="final_esi2_false_handoff_to_esi345_agent"
    ),
    define_handoff(
        source_agent="esi2_agent",
        target_agent="doctor_agent",
        payload_model=ESI2ToDoctorPayload,
        description="Transfer to Doctor when the Case is Confirmed to be ESI2.",
        tool_name="final_esi2_true_handoff_to_doctor_agent"
    )
]
