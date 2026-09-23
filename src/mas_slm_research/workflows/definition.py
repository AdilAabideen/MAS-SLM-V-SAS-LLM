"""Workflow Definition module helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class WorkflowMetadata(BaseModel):
    workflow_id: str = Field(..., description="Stable identifier for the workflow definition.")
    name: str = Field(..., description="Human-readable workflow name.")
    version: str = Field(..., description="Workflow version string.")
    description: Optional[str] = Field(
        default=None,
        description="Optional human-readable description of the workflow.",
    )

    @model_validator(mode="after")
    def validate_metadata(self) -> "WorkflowMetadata":
        """Validate metadata."""
        # Fail fast on bad input.
        if not self.workflow_id.strip():
            raise ValueError("workflow_id must be non-empty.")
        if not self.name.strip():
            raise ValueError("name must be non-empty.")
        if not self.version.strip():
            raise ValueError("version must be non-empty.")
        return self


class GateNodeDefinition(BaseModel):
    gate_id: str = Field(..., description="Stable node identifier for the gate.")
    name: str = Field(..., description="Human-readable gate name.")
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the gate's role in the workflow.",
    )
    required_sources: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Logical source groups or upstream sources required before the gate may route forward.",
    )
    incoming_from: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Workflow nodes that may route into this gate.",
    )
    target_node: Optional[str] = Field(
        default=None,
        description="Primary node the gate routes to when ready.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-specific gate metadata for rendering or later policy binding.",
    )

    @model_validator(mode="after")
    def validate_gate(self) -> "GateNodeDefinition":
        """Validate gate."""
        # Fail fast on bad input.
        if not self.gate_id.strip():
            raise ValueError("gate_id must be non-empty.")
        if not self.name.strip():
            raise ValueError("gate name must be non-empty.")
        return self


class SourceDefinition(BaseModel):
    source_id: str = Field(..., description="Stable identifier for a logical workflow source or branch.")
    name: str = Field(..., description="Human-readable source name.")
    agent_names: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Agents whose handoffs satisfy this source.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional human-readable description of the source.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional rendering or workflow metadata for this source.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "SourceDefinition":
        """Validate source."""
        # Fail fast on bad input.
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty.")
        if not self.name.strip():
            raise ValueError("source name must be non-empty.")
        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("source agent_names must be unique.")
        return self


class WorkflowDefinition(BaseModel):
    metadata: WorkflowMetadata = Field(..., description="Workflow metadata.")
    participating_agents: Tuple[str, ...] = Field(
        ...,
        description="All agent nodes that participate in this workflow.",
    )
    sources: Dict[str, SourceDefinition] = Field(
        default_factory=dict,
        description="Logical workflow sources or branches used by gates and routing metadata.",
    )
    start_agents: Tuple[str, ...] = Field(
        ...,
        description="Agent nodes that may start the workflow.",
    )
    finalizing_agents: Tuple[str, ...] = Field(
        ...,
        description="Agent nodes allowed to produce final output.",
    )
    allowed_handoffs: Dict[str, Tuple[str, ...]] = Field(
        default_factory=dict,
        description="Directed handoff edges keyed by source agent.",
    )
    gates: Dict[str, GateNodeDefinition] = Field(
        default_factory=dict,
        description="Gate nodes and their metadata.",
    )
    agent_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional rendering or classification metadata for participating agents.",
    )
    workflow_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional workflow-level metadata for rendering or later kernel configuration.",
    )

    @model_validator(mode="after")
    def validate_definition(self) -> "WorkflowDefinition":
        """Validate definition."""
        # Fail fast on bad input.
        participating_agents = list(self.participating_agents)
        if not participating_agents:
            raise ValueError("participating_agents must contain at least one agent.")
        if len(set(participating_agents)) != len(participating_agents):
            raise ValueError("participating_agents must be unique.")

        gate_ids = list(self.gates.keys())
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("gate ids must be unique.")

        overlap = set(participating_agents).intersection(gate_ids)
        if overlap:
            raise ValueError(
                "agent names and gate ids must not overlap: {values}".format(
                    values=", ".join(sorted(overlap))
                )
            )

        for source_id, source in self.sources.items():
            if source.source_id != source_id:
                raise ValueError(
                    "source map key '{key}' must match source_id '{source_id}'.".format(
                        key=source_id,
                        source_id=source.source_id,
                    )
                )
            for agent_name in source.agent_names:
                if agent_name not in participating_agents:
                    raise ValueError(
                        "source '{source_id}' references unknown agent '{agent}'.".format(
                            source_id=source_id,
                            agent=agent_name,
                        )
                    )

        for agent_name in self.start_agents:
            if agent_name not in participating_agents:
                raise ValueError(
                    "start agent '{agent}' is not declared in participating_agents.".format(agent=agent_name)
                )

        for agent_name in self.finalizing_agents:
            if agent_name not in participating_agents:
                raise ValueError(
                    "finalizing agent '{agent}' is not declared in participating_agents.".format(agent=agent_name)
                )

        for source_agent, target_agents in self.allowed_handoffs.items():
            if source_agent not in participating_agents:
                raise ValueError(
                    "allowed_handoffs source '{source}' is not a participating agent.".format(source=source_agent)
                )
            if len(set(target_agents)) != len(target_agents):
                raise ValueError(
                    "allowed_handoffs for source '{source}' must not contain duplicates.".format(
                        source=source_agent
                    )
                )
            for target_agent in target_agents:
                if target_agent not in participating_agents:
                    raise ValueError(
                        "allowed_handoffs target '{target}' from source '{source}' is not a participating agent.".format(
                            source=source_agent,
                            target=target_agent,
                        )
                    )

        all_nodes = set(participating_agents).union(gate_ids)
        for gate_id, gate in self.gates.items():
            if gate.gate_id != gate_id:
                raise ValueError(
                    "gate map key '{key}' must match gate_id '{gate_id}'.".format(
                        key=gate_id,
                        gate_id=gate.gate_id,
                    )
                )
            for source_id in gate.required_sources:
                if source_id not in self.sources:
                    raise ValueError(
                        "gate '{gate_id}' requires unknown source '{source_id}'.".format(
                            gate_id=gate_id,
                            source_id=source_id,
                        )
                    )
            for source_node in gate.incoming_from:
                if source_node not in all_nodes:
                    raise ValueError(
                        "gate '{gate_id}' incoming_from node '{source}' is not a known workflow node.".format(
                            gate_id=gate_id,
                            source=source_node,
                        )
                    )
            if gate.target_node is not None and gate.target_node not in all_nodes:
                raise ValueError(
                    "gate '{gate_id}' target_node '{target}' is not a known workflow node.".format(
                        gate_id=gate_id,
                        target=gate.target_node,
                    )
                )

        return self

    @property
    def gate_ids(self) -> Tuple[str, ...]:
        """Handle ids."""
        # Keep the main step clear.
        return tuple(self.gates.keys())

    @property
    def source_ids(self) -> Tuple[str, ...]:
        """Handle ids."""
        # Keep the main step clear.
        return tuple(self.sources.keys())

    @property
    def all_nodes(self) -> Tuple[str, ...]:
        """Handle nodes."""
        # Keep the main step clear.
        return tuple(list(self.participating_agents) + list(self.gate_ids))

    def allowed_targets_for(self, source_agent: str) -> Tuple[str, ...]:
        """Handle targets for."""
        # Keep the main step clear.
        return tuple(self.allowed_handoffs.get(source_agent, ()))

    def validate_agent(self, agent_name: str) -> str:
        """Require an agent declared by this resolved workflow."""
        if agent_name not in self.participating_agents:
            raise ValueError(f"Unknown agent '{agent_name}' for workflow '{self.metadata.workflow_id}'.")
        return agent_name

    def validate_handoff(self, source_agent: str, target_agent: str) -> tuple[str, str]:
        """Require a declared directed agent-to-agent communication path."""
        self.validate_agent(source_agent)
        self.validate_agent(target_agent)
        if target_agent not in self.allowed_targets_for(source_agent):
            raise ValueError(f"Invalid handoff route '{source_agent}' -> '{target_agent}'.")
        return source_agent, target_agent

    def is_start_agent(self, agent_name: str) -> bool:
        """Handle start agent."""
        # Keep the main step clear.
        return agent_name in self.start_agents

    def is_finalizing_agent(self, agent_name: str) -> bool:
        """Handle finalizing agent."""
        # Keep the main step clear.
        return agent_name in self.finalizing_agents

    def is_gate_node(self, node_name: str) -> bool:
        """Handle gate node."""
        # Keep the main step clear.
        return node_name in self.gates

    def source_agents(self, source_id: str) -> Tuple[str, ...]:
        """Handle agents."""
        # Keep the main step clear.
        source = self.sources.get(source_id)
        if source is None:
            return ()
        return source.agent_names

    def sources_for_agent(self, agent_name: str) -> Tuple[str, ...]:
        """Handle for agent."""
        # Keep the main step clear.
        source_ids: List[str] = []
        for source_id, source in self.sources.items():
            if agent_name in source.agent_names:
                source_ids.append(source_id)
        return tuple(source_ids)

    def gate_incoming_nodes(self, gate_id: str) -> Tuple[str, ...]:
        """Handle incoming nodes."""
        # Keep the main step clear.
        gate = self.gates[gate_id]
        if gate.incoming_from:
            return gate.incoming_from

        incoming: List[str] = []
        for source_id in gate.required_sources:
            for agent_name in self.source_agents(source_id):
                if agent_name not in incoming:
                    incoming.append(agent_name)
        return tuple(incoming)

    def gate_satisfied_sources(
        self,
        gate_id: str,
        handoff_history: List[Dict[str, Any]],
    ) -> Tuple[str, ...]:
        """Handle satisfied sources."""
        # Keep the main step clear.
        gate = self.gates[gate_id]
        target_node = gate.target_node
        if target_node is None:
            return ()

        satisfied: List[str] = []
        for source_id in gate.required_sources:
            source_agents = set(self.source_agents(source_id))
            for item in handoff_history:
                if not isinstance(item, dict):
                    continue
                if item.get("target_agent") != target_node:
                    continue
                if item.get("from_agent") in source_agents:
                    satisfied.append(source_id)
                    break
        return tuple(satisfied)

    def gate_missing_sources(
        self,
        gate_id: str,
        handoff_history: List[Dict[str, Any]],
    ) -> Tuple[str, ...]:
        """Handle missing sources."""
        # Keep the main step clear.
        gate = self.gates[gate_id]
        satisfied = set(self.gate_satisfied_sources(gate_id, handoff_history))
        return tuple(source_id for source_id in gate.required_sources if source_id not in satisfied)

    def is_gate_ready(
        self,
        gate_id: str,
        handoff_history: List[Dict[str, Any]],
    ) -> bool:
        """Handle gate ready."""
        # Keep the main step clear.
        return len(self.gate_missing_sources(gate_id, handoff_history)) == 0

    def graph_edges(self) -> List[Tuple[str, str]]:
        """Handle edges."""
        # Keep the main step clear.
        edges: List[Tuple[str, str]] = []
        for source_agent, target_agents in self.allowed_handoffs.items():
            for target_agent in target_agents:
                edges.append((source_agent, target_agent))
        for gate_id, gate in self.gates.items():
            for source_node in self.gate_incoming_nodes(gate_id):
                edges.append((source_node, gate_id))
            if gate.target_node is not None:
                edges.append((gate_id, gate.target_node))
        return edges
