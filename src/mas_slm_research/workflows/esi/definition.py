"""Workflow Definition module helpers."""

from __future__ import annotations

from mas_slm_research.workflows.definition import (
    GateNodeDefinition,
    SourceDefinition,
    WorkflowDefinition,
    WorkflowMetadata,
)


ESI_MAS = WorkflowDefinition(
    metadata=WorkflowMetadata(
        workflow_id="esi_mas",
        name="ESI MAS",
        version="1.0.0",
        description=(
            "Constrained multi-agent ESI workflow with parallel ESI1 and vitals starts, "
            "acuity handoffs through ESI2/ESI345, and doctor finalization."
        ),
    ),
    participating_agents=(
        "esi1_agent",
        "esi2_agent",
        "esi345_agent",
        "vitals_agent",
        "doctor_agent",
    ),
    sources={
        "acuity": SourceDefinition(
            source_id="acuity",
            name="Acuity Branch",
            agent_names=("esi1_agent", "esi2_agent", "esi345_agent"),
            description="The main ESI acuity decision pathway.",
            metadata={"branch_type": "clinical_acuity"},
        ),
        "vitals": SourceDefinition(
            source_id="vitals",
            name="Vitals Branch",
            agent_names=("vitals_agent",),
            description="The vitals-only parallel support branch.",
            metadata={"branch_type": "physiologic_support"},
        ),
    },
    start_agents=(
        "esi1_agent",
        "vitals_agent",
    ),
    finalizing_agents=(
        "doctor_agent",
    ),
    allowed_handoffs={
        "esi1_agent": ("esi2_agent", "doctor_agent"),
        "esi2_agent": ("esi345_agent", "doctor_agent"),
        "esi345_agent": ("doctor_agent",),
        "vitals_agent": ("doctor_agent",),
        "doctor_agent": (),
    },
    gates={
        "doctor_gate": GateNodeDefinition(
            gate_id="doctor_gate",
            name="Doctor Gate",
            description="Waits until both the acuity branch and vitals branch have handed off before doctor finalization.",
            required_sources=("acuity", "vitals"),
            target_node="doctor_agent",
            metadata={
                "gate_type": "readiness_gate",
                "ready_rule": "requires_required_sources_to_handoff_to_target",
                "terminal_when_not_ready": True,
            },
        ),
    },
    agent_metadata={
        "esi1_agent": {
            "role": "acuity",
            "stage": "decision_point_a",
            "can_handoff": True,
            "can_finalize": False,
        },
        "esi2_agent": {
            "role": "acuity",
            "stage": "decision_point_b",
            "can_handoff": True,
            "can_finalize": False,
        },
        "esi345_agent": {
            "role": "acuity",
            "stage": "decision_point_c",
            "can_handoff": True,
            "can_finalize": False,
        },
        "vitals_agent": {
            "role": "vitals",
            "stage": "parallel_support",
            "can_handoff": True,
            "can_finalize": False,
        },
        "doctor_agent": {
            "role": "supervisor",
            "stage": "final_review",
            "can_handoff": False,
            "can_finalize": True,
        },
    },
    workflow_metadata={
        "workflow_family": "esi_mas",
        "execution_model": "constrained_mas",
        "parallel_start": True,
        "doctor_gate_id": "doctor_gate",
    },
)
