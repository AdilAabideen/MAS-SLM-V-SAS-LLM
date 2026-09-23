"""Graph Builder module helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from mas_slm_research.mas.agent_node_executor import AgentNodeExecutionOutcome, AgentNodeExecutor
from mas_slm_research.mas.gate_evaluator import GateEvaluationOutcome, GateEvaluator
from mas_slm_research.mas_contract import MASState, make_initial_mas_state
from mas_slm_research.workflows.definition import WorkflowDefinition


BootstrapUpdateFactory = Callable[[WorkflowDefinition], Dict[str, Any]]
BranchStateFactory = Callable[[MASState], MASState]


def default_bootstrap_update_factory(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Handle bootstrap update factory."""
    # Keep the main step clear.
    return {
        "execution_trace": [
            {
                "event": "bootstrap",
                "parallel_start_agents": list(workflow.start_agents),
            }
        ]
    }


def default_branch_state_factory(state: MASState) -> MASState:
    """Handle branch state factory."""
    # Keep the main step clear.
    return make_initial_mas_state(
        dict(state.get("case_info") or {}),
        execution_context=dict(state.get("execution_context") or {}),
    )


class MASGraphBuilder:
    """Compile a workflow definition into a LangGraph mas graph."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        agent_executor: AgentNodeExecutor,
        gate_evaluator: GateEvaluator,
        bootstrap_update_factory: BootstrapUpdateFactory = default_bootstrap_update_factory,
        branch_state_factory: BranchStateFactory = default_branch_state_factory,
    ) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.workflow = workflow
        self.agent_executor = agent_executor
        self.gate_evaluator = gate_evaluator
        self.bootstrap_update_factory = bootstrap_update_factory
        self.branch_state_factory = branch_state_factory

    def build(self):
        """Build the value."""
        # Build the next value.
        graph = StateGraph(MASState)
        graph.add_node("bootstrap", self._bootstrap_node)

        for agent_name in self.workflow.participating_agents:
            graph.add_node(agent_name, self._build_agent_node(agent_name))

        for gate_id in self.workflow.gates:
            graph.add_node(gate_id, self._build_gate_node(gate_id))

        graph.add_edge(START, "bootstrap")
        graph.add_conditional_edges("bootstrap", self._route_bootstrap)

        for gate_id in self.workflow.gates:
            graph.add_conditional_edges(gate_id, self._build_gate_router(gate_id))

        return graph.compile()

    def _bootstrap_node(self, state: MASState) -> Dict[str, Any]:
        """Handle node."""
        # Keep the main step clear.
        return self.bootstrap_update_factory(self.workflow)

    def _route_bootstrap(self, state: MASState) -> List[Send]:
        """Handle bootstrap."""
        # Keep the main step clear.
        branch_state = self.branch_state_factory(state)
        return [Send(agent_name, dict(branch_state)) for agent_name in self.workflow.start_agents]

    def _build_agent_node(self, agent_name: str):
        """Build agent node."""
        # Build the next value.
        async def node(state: MASState) -> Command:
            """Handle the value."""
            # Keep the main step clear.
            outcome = await self.agent_executor.execute(agent_name=agent_name, state=state)  # type: ignore[arg-type]
            return self._agent_outcome_to_command(outcome)

        return node

    def _build_gate_node(self, gate_id: str):
        """Build gate node."""
        # Build the next value.
        def node(state: MASState) -> Dict[str, Any]:
            """Handle the value."""
            # Keep the main step clear.
            outcome = self.gate_evaluator.evaluate(gate_id=gate_id, state=state, persist=True)
            return outcome.state_updates

        return node

    def _build_gate_router(self, gate_id: str):
        """Build gate router."""
        # Build the next value.
        def route(state: MASState) -> str:
            """Handle the value."""
            # Keep the main step clear.
            if state.get("final_output") is not None:
                return END
            outcome = self.gate_evaluator.evaluate(gate_id=gate_id, state=state, persist=False)
            return self._gate_outcome_to_route(outcome)

        return route

    @staticmethod
    def _agent_outcome_to_command(outcome: AgentNodeExecutionOutcome) -> Command:
        """Handle outcome to command."""
        # Keep the main step clear.
        if outcome.terminal or outcome.goto is None:
            return Command(goto=END, update=outcome.state_updates)
        return Command(goto=outcome.goto, update=outcome.state_updates)

    @staticmethod
    def _gate_outcome_to_route(outcome: GateEvaluationOutcome) -> str:
        """Handle outcome to route."""
        # Keep the main step clear.
        if outcome.ready and outcome.next_target is not None:
            return outcome.next_target
        return END
