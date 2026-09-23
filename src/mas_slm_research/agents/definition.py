"""Subclassable agent-construction contract for registered Python definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from mas_slm_research.kernel import AgentKernel
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.workflows.definition import WorkflowDefinition


class AgentDefinition(ABC):
    """Build a fresh kernel from a chosen model and resolved workflow.

    Researchers may subclass this and register an instance under ``agents``.
    The construction factory calls this method once per case/role attempt.
    """

    @abstractmethod
    def build_kernel(
        self,
        *,
        model: Any,
        runtime_config: RuntimeConfig,
        workflow: WorkflowDefinition | None,
        handoff_schemas: Mapping[str, str],
        registry: Any,
    ) -> AgentKernel:
        """Return a new extracted ``AgentKernel``."""
