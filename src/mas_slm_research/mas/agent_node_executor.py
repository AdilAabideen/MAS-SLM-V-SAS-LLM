"""Agent Node Executor module helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from mas_slm_research.mas.execution_strategy import ExecutionRequest, ExecutionStrategy
from mas_slm_research.mas_tracker import (
    InMemoryMASTracker,
    TrackedAgentExecution,
    TrackedExecutionOutcome,
)
from mas_slm_research.mas_contract import AgentExecutionResult, AgentName, HandoffEnvelope, MASState
from mas_slm_research.workflows.definition import WorkflowDefinition


PayloadBuilder = Callable[[AgentName, MASState], Dict[str, Any]]


@dataclass(frozen=True)
class AgentNodeExecutionOutcome:
    agent_name: AgentName
    pending_agent_payload: Dict[str, Any]
    result: AgentExecutionResult
    state_updates: Dict[str, Any]
    goto: Optional[str]
    terminal: bool = False


class AgentNodeExecutor:
    """Reusable runtime adapter for one workflow agent-node execution."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        strategy: ExecutionStrategy,
        payload_builder: PayloadBuilder,
        execution_tracker: InMemoryMASTracker | None = None,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.workflow = workflow
        self.strategy = strategy
        self.payload_builder = payload_builder
        self.execution_tracker = execution_tracker

    async def execute(self, *, agent_name: AgentName, state: MASState) -> AgentNodeExecutionOutcome:
        """Handle the value."""
        # Keep the main step clear.
        pending_agent_payload = self.payload_builder(agent_name, state)
        tracked_execution = self._begin_tracking(
            agent_name=agent_name,
            state=state,
            pending_agent_payload=pending_agent_payload,
        )
        state_snapshot = self._state_snapshot_for_request(
            state=state,
            tracked_execution=tracked_execution,
        )
        request = ExecutionRequest(
            workflow=self.workflow,
            agent_name=agent_name,
            pending_agent_payload=dict(pending_agent_payload),
            state_snapshot=state_snapshot,
        )
        try:
            result = await self.strategy.execute(request)
            result.validate_for(self.workflow)
        except Exception as exc:
            # A child failure must finish its tracked run and never become a prediction.
            result = AgentExecutionResult(
                agent_name=agent_name,
                status="error",
                output={"error": "agent_execution_failed", "detail": str(exc) or type(exc).__name__},
            )
        tracking_outcome = self._complete_tracking(tracked=tracked_execution, result=result)
        updates = self._base_updates(
            agent_name=agent_name,
            pending_agent_payload=pending_agent_payload,
            result=result,
            tracked_execution=tracked_execution,
            tracking_outcome=tracking_outcome,
        )
        return self._finalize_outcome(
            agent_name=agent_name,
            pending_agent_payload=pending_agent_payload,
            result=result,
            updates=updates,
            tracking_outcome=tracking_outcome,
        )

    def _base_updates(
        self,
        *,
        agent_name: AgentName,
        pending_agent_payload: Dict[str, Any],
        result: AgentExecutionResult,
        tracked_execution: TrackedAgentExecution | None,
        tracking_outcome: TrackedExecutionOutcome | None,
    ) -> Dict[str, Any]:
        """Handle updates."""
        # Keep the main step clear.
        updates = {
            "active_agent": agent_name,
            "pending_agent_payload": pending_agent_payload,
            "completed_agents": [agent_name],
            "execution_trace": [
                {
                    "event": "agent_executed",
                    "agent": agent_name,
                    "status": result.status,
                    "output": result.output,
                    "payload_metadata": pending_agent_payload.get("metadata", {}),
                }
            ],
        }
        if self.execution_tracker is not None:
            updates.update(
                self.execution_tracker.state_updates_for_completion(
                    tracked=tracked_execution,
                    persisted=tracking_outcome,
                )
            )
        if tracking_outcome is not None:
            updates.setdefault("execution_trace", [])
            updates["execution_trace"].append(
                {
                    "event": "agent_run_recorded",
                    "agent": agent_name,
                    "agent_run_id": tracking_outcome.agent_run_id,
                    "sequence_index": tracking_outcome.sequence_index,
                }
            )
        return updates

    def _finalize_outcome(
        self,
        *,
        agent_name: AgentName,
        pending_agent_payload: Dict[str, Any],
        result: AgentExecutionResult,
        updates: Dict[str, Any],
        tracking_outcome: TrackedExecutionOutcome | None,
    ) -> AgentNodeExecutionOutcome:
        """Handle outcome."""
        # Keep the main step clear.
        if result.status == "final":
            updates["final_output"] = result.final_output or {}
            updates["execution_trace"].append(
                {
                    "event": "final_output_created",
                    "agent": result.agent_name,
                    "final_output": result.final_output,
                }
            )
            return AgentNodeExecutionOutcome(
                agent_name=agent_name,
                pending_agent_payload=pending_agent_payload,
                result=result,
                state_updates=updates,
                goto=None,
                terminal=True,
            )

        if result.status == "error":
            updates["execution_trace"].append(
                {
                    "event": "agent_error",
                    "agent": result.agent_name,
                    "output": result.output,
                }
            )
            return AgentNodeExecutionOutcome(
                agent_name=agent_name,
                pending_agent_payload=pending_agent_payload,
                result=result,
                state_updates=updates,
                goto=None,
                terminal=True,
            )

        handoff = result.handoff
        if handoff is None:
            updates["execution_trace"].append(
                {
                    "event": "missing_handoff",
                    "agent": result.agent_name,
                }
            )
            return AgentNodeExecutionOutcome(
                agent_name=agent_name,
                pending_agent_payload=pending_agent_payload,
                result=result,
                state_updates=updates,
                goto=None,
                terminal=True,
            )

        handoff_dict = self._decorate_handoff_dict(
            handoff=handoff,
            tracking_outcome=tracking_outcome,
        )
        updates["pending_handoff"] = handoff_dict
        updates["handoff_history"] = [handoff_dict]
        updates["execution_trace"].append(
            {
                "event": "handoff_created",
                "from_agent": handoff.from_agent,
                "target_agent": handoff.target_agent,
                "handoff_name": handoff.handoff_name,
                "handoff_id": handoff_dict.get("handoff_id"),
            }
        )
        return AgentNodeExecutionOutcome(
            agent_name=agent_name,
            pending_agent_payload=pending_agent_payload,
            result=result,
            state_updates=updates,
            goto=self._route_handoff(handoff),
            terminal=False,
        )

    def _route_handoff(self, handoff: HandoffEnvelope) -> str:
        """Handle handoff."""
        # Keep the main step clear.
        gate_id = self._matching_gate_for_handoff(handoff)
        if gate_id is not None:
            return gate_id
        return handoff.target_agent

    def _matching_gate_for_handoff(self, handoff: HandoffEnvelope) -> Optional[str]:
        """Handle gate for handoff."""
        # Keep the main step clear.
        matches = []
        source_ids = set(self.workflow.sources_for_agent(handoff.from_agent))
        for gate_id, gate in self.workflow.gates.items():
            if gate.target_node != handoff.target_agent:
                continue
            if gate.incoming_from and handoff.from_agent not in gate.incoming_from:
                continue
            if gate.required_sources and not source_ids.intersection(set(gate.required_sources)):
                continue
            matches.append(gate_id)

        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                "Multiple gates match handoff '{source}' -> '{target}': {gates}".format(
                    source=handoff.from_agent,
                    target=handoff.target_agent,
                    gates=", ".join(sorted(matches)),
                )
            )
        return matches[0]

    def _begin_tracking(
        self,
        *,
        agent_name: AgentName,
        state: MASState,
        pending_agent_payload: Dict[str, Any],
    ) -> TrackedAgentExecution | None:
        """Handle tracking."""
        # Keep the main step clear.
        if self.execution_tracker is None:
            return None
        return self.execution_tracker.begin_agent_execution(
            agent_name=agent_name,
            state=state,
            pending_agent_payload=pending_agent_payload,
        )

    def _complete_tracking(
        self,
        *,
        tracked: TrackedAgentExecution | None,
        result: AgentExecutionResult,
    ) -> TrackedExecutionOutcome | None:
        """Handle tracking."""
        # Keep the main step clear.
        if self.execution_tracker is None:
            return None
        return self.execution_tracker.complete_agent_execution(
            tracked=tracked,
            result=result,
        )

    def _decorate_handoff_dict(
        self,
        *,
        handoff: HandoffEnvelope,
        tracking_outcome: TrackedExecutionOutcome | None,
    ) -> Dict[str, Any]:
        """Handle handoff dict."""
        # Keep the main step clear.
        handoff_dict = handoff.model_dump()
        if self.execution_tracker is None:
            return handoff_dict
        return self.execution_tracker.decorate_handoff_dict(
            handoff_dict=handoff_dict,
            persisted=tracking_outcome,
        )

    @staticmethod
    def _state_snapshot_for_request(
        *,
        state: MASState,
        tracked_execution: TrackedAgentExecution | None,
    ) -> Dict[str, Any]:
        """Handle snapshot for request."""
        # Keep the main step clear.
        snapshot = dict(state)
        if tracked_execution is None:
            return snapshot

        execution_context = dict(snapshot.get("execution_context") or {})
        execution_context["current_agent_run_id"] = tracked_execution.agent_run_id
        execution_context["next_sequence_index"] = int(tracked_execution.sequence_index) + 1
        snapshot["execution_context"] = execution_context
        return snapshot
