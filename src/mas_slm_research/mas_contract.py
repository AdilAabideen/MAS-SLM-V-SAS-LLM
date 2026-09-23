"""Workflow-owned multi-agent identity, handoff, and graph state contracts."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator

from .workflows.definition import WorkflowDefinition


AgentName = str
ExecutionStatus = Literal["handoff", "final", "error"]


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def merge_unique_lists(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    values: List[str] = []
    for item in list(left or []) + list(right or []):
        if item not in values:
            values.append(item)
    return values


def take_latest_str(left: Optional[str], right: Optional[str]) -> Optional[str]:
    return right if right is not None else left


class HandoffResult(BaseModel):
    """Tool result before workflow route validation."""

    handoff_name: str = Field(..., description="Stable handoff identifier.")
    from_agent: str = Field(..., description="Agent that initiated the handoff.")
    target_agent: str = Field(..., description="Agent that should receive control next.")
    payload_schema: str = Field(..., description="Payload schema class name used for validation.")
    payload: Dict[str, Any] = Field(..., description="Validated structured handoff payload.")


class HandoffEnvelope(HandoffResult):
    """A handoff whose route is checked against a supplied workflow."""

    def validate_for(self, workflow: WorkflowDefinition) -> "HandoffEnvelope":
        workflow.validate_handoff(self.from_agent, self.target_agent)
        return self


class AgentExecutionPayload(BaseModel):
    agent_name: AgentName
    case_info: Dict[str, Any] = Field(default_factory=dict)
    active_handoff: Optional[HandoffEnvelope] = None
    handoff_history: List[HandoffEnvelope] = Field(default_factory=list)

    def validate_for(self, workflow: WorkflowDefinition) -> "AgentExecutionPayload":
        workflow.validate_agent(self.agent_name)
        if self.active_handoff is not None:
            self.active_handoff.validate_for(workflow)
            if self.active_handoff.target_agent != self.agent_name:
                raise ValueError("active handoff target must match payload agent")
        for handoff in self.handoff_history:
            handoff.validate_for(workflow)
        return self


class AgentExecutionResult(BaseModel):
    agent_name: AgentName
    status: ExecutionStatus
    output: Dict[str, Any] = Field(default_factory=dict)
    handoff: Optional[HandoffEnvelope] = None
    final_output: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentExecutionResult":
        if self.status == "handoff" and self.handoff is None:
            raise ValueError("status='handoff' requires a handoff envelope")
        if self.status == "final" and self.final_output is None:
            raise ValueError("status='final' requires final_output")
        if self.status != "final" and self.final_output is not None:
            raise ValueError("final_output is only valid when status='final'")
        return self

    def validate_for(self, workflow: WorkflowDefinition) -> "AgentExecutionResult":
        workflow.validate_agent(self.agent_name)
        if self.status == "handoff":
            assert self.handoff is not None
            if self.handoff.from_agent != self.agent_name:
                raise ValueError("handoff source must match result agent")
            self.handoff.validate_for(workflow)
        if self.status == "final" and not workflow.is_finalizing_agent(self.agent_name):
            raise ValueError(f"agent '{self.agent_name}' cannot finalize workflow '{workflow.metadata.workflow_id}'")
        return self


class MASState(TypedDict):
    case_info: Dict[str, Any]
    execution_context: Annotated[Optional[Dict[str, Any]], merge_dicts]
    active_agent: Annotated[Optional[str], take_latest_str]
    pending_handoff: Annotated[Optional[Dict[str, Any]], merge_dicts]
    pending_agent_payload: Annotated[Optional[Dict[str, Any]], merge_dicts]
    handoff_history: Annotated[List[Dict[str, Any]], operator.add]
    completed_agents: Annotated[List[str], merge_unique_lists]
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]
    final_output: Annotated[Optional[Dict[str, Any]], merge_dicts]


def make_initial_mas_state(
    case_info: Dict[str, Any],
    execution_context: Optional[Dict[str, Any]] = None,
) -> MASState:
    return {
        "case_info": dict(case_info),
        "execution_context": dict(execution_context or {}),
        "active_agent": None,
        "pending_handoff": None,
        "pending_agent_payload": None,
        "handoff_history": [],
        "completed_agents": [],
        "execution_trace": [],
        "final_output": None,
    }
