"""Gate Evaluator module helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mas_slm_research.mas_contract import MASState
from mas_slm_research.workflows.definition import WorkflowDefinition

if TYPE_CHECKING:
    from mas_slm_research.mas_tracker import InMemoryMASTracker


@dataclass(frozen=True)
class GateEvaluationOutcome:
    gate_id: str
    ready: bool
    satisfied_sources: List[str]
    missing_sources: List[str]
    handoffs_to_target: List[Dict[str, Any]]
    next_target: Optional[str]
    state_updates: Dict[str, Any]
    terminal: bool = False


class GateEvaluator:
    """Reusable runtime adapter for workflow gate nodes."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        execution_tracker: "InMemoryMASTracker | None" = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.workflow = workflow
        self.execution_tracker = execution_tracker

    def evaluate(
        self,
        *,
        gate_id: str,
        state: MASState,
        persist: bool = True,
    ) -> GateEvaluationOutcome:
        """Handle the value."""
        # Keep the main step clear.
        gate = self.workflow.gates[gate_id]
        handoff_history = list(state.get("handoff_history", []))
        ready = self.workflow.is_gate_ready(gate_id, handoff_history)
        satisfied_sources = list(self.workflow.gate_satisfied_sources(gate_id, handoff_history))
        missing_sources = list(self.workflow.gate_missing_sources(gate_id, handoff_history))
        handoffs_to_target = [
            item
            for item in handoff_history
            if isinstance(item, dict) and item.get("target_agent") == gate.target_node
        ]

        state_updates = {
            "execution_trace": [
                {
                    "event": gate_id,
                    "ready": ready,
                    "satisfied_sources": satisfied_sources,
                    "missing_sources": missing_sources,
                    "handoffs_to_target": handoffs_to_target,
                }
            ]
        }

        next_target = gate.target_node if ready else None
        terminal = not ready and bool(gate.metadata.get("terminal_when_not_ready", False))
        outcome = GateEvaluationOutcome(
            gate_id=gate_id,
            ready=ready,
            satisfied_sources=satisfied_sources,
            missing_sources=missing_sources,
            handoffs_to_target=handoffs_to_target,
            next_target=next_target,
            state_updates=state_updates,
            terminal=terminal,
        )
        if persist and self.execution_tracker is not None:
            evaluation_id = self.execution_tracker.record_gate_evaluation(
                gate_id=gate_id,
                state=state,
                outcome=outcome,
            )
            if evaluation_id is not None:
                outcome.state_updates.setdefault("execution_context", {})
                outcome.state_updates["execution_context"]["last_gate_evaluation_id"] = evaluation_id
                outcome.state_updates["execution_context"]["current_gate_id"] = gate_id
        return outcome
