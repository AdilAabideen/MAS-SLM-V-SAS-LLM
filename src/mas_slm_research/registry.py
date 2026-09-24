"""Explicit, per-experiment component registration.

An extension module exposes ``register_components(registry)``. Loading a module
is an explicit caller action; configuration files never contain Python code.
The registry stores the Python implementation, not a string to evaluate later.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel

from .grading import BaseGrader
from .model_registry import ModelSpec
from .workflows.definition import WorkflowDefinition


ComponentKind = Literal[
    "providers", "models", "agents", "tools", "schemas",
    "payload_builders", "workflows", "dataset_loaders", "graders",
]
COMPONENT_KINDS: tuple[ComponentKind, ...] = (
    "providers", "models", "agents", "tools", "schemas",
    "payload_builders", "workflows", "dataset_loaders", "graders",
)


class ComponentRegistry:
    """Own explicit component IDs for one experiment or preview.

    Providers, agent definitions, payload builders and dataset loaders are
    callable factories. Graders subclass BaseGrader.
    """

    def __init__(self) -> None:
        self._items: dict[ComponentKind, dict[str, Any]] = {
            kind: {} for kind in COMPONENT_KINDS
        }
        self._loaded_modules: set[str] = set()

    def register(self, kind: ComponentKind, identifier: str, component: Any) -> None:
        if kind not in self._items:
            raise ValueError(f"Unknown component kind {kind!r}; choose from {COMPONENT_KINDS}")
        if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
            raise ValueError(f"{kind} component ID must be a nonempty, trimmed string")
        if identifier in self._items[kind]:
            raise ValueError(f"Duplicate {kind} component ID {identifier!r}")
        self._validate_component(kind, identifier, component)
        if kind == "graders" and isinstance(component, type):
            component = component()
        self._items[kind][identifier] = component

    @staticmethod
    def _validate_component(kind: ComponentKind, identifier: str, component: Any) -> None:
        if kind == "schemas":
            valid = isinstance(component, type) and issubclass(component, BaseModel)
        elif kind == "workflows":
            valid = isinstance(component, WorkflowDefinition) or callable(component)
        elif kind == "models":
            valid = isinstance(component, ModelSpec) or callable(component)
        elif kind == "graders":
            valid = (isinstance(component, type) and issubclass(component, BaseGrader)
                     and not getattr(component, "__abstractmethods__", None)) or isinstance(component, BaseGrader)
        elif kind == "agents":
            valid = callable(component) or callable(getattr(component, "build_kernel", None))
        elif kind == "tools":
            valid = callable(component) or callable(getattr(component, "invoke", None))
        else:
            valid = callable(component)
        if not valid:
            raise TypeError(f"{kind} component {identifier!r} has an invalid implementation")

    def resolve(self, kind: ComponentKind, identifier: str) -> Any:
        if kind not in self._items:
            raise ValueError(f"Unknown component kind {kind!r}; choose from {COMPONENT_KINDS}")
        try:
            return self._items[kind][identifier]
        except KeyError as exc:
            available = ", ".join(sorted(self._items[kind])) or "(none)"
            raise KeyError(f"Unknown {kind} component ID {identifier!r}; available: {available}") from exc

    def ids(self, kind: ComponentKind) -> tuple[str, ...]:
        if kind not in self._items:
            raise ValueError(f"Unknown component kind {kind!r}; choose from {COMPONENT_KINDS}")
        return tuple(sorted(self._items[kind]))

    def inventory(self) -> Mapping[ComponentKind, tuple[str, ...]]:
        return {kind: self.ids(kind) for kind in COMPONENT_KINDS}

    def load_module(self, module_name: str) -> None:
        """Run one explicitly named module's registration function once.

        A failed registration leaves this registry unchanged, so the caller can
        correct the extension and retry without a half-populated inventory.
        """
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("extension module name must be nonempty")
        if module_name in self._loaded_modules:
            raise ValueError(f"Extension module {module_name!r} was already loaded")
        module = importlib.import_module(module_name)
        register = getattr(module, "register_components", None)
        if not callable(register):
            raise TypeError(f"Extension module {module_name!r} must define register_components(registry)")
        staged = ComponentRegistry()
        staged._items = {kind: dict(items) for kind, items in self._items.items()}
        staged._loaded_modules = set(self._loaded_modules)
        register(staged)
        self._items = staged._items
        self._loaded_modules = staged._loaded_modules | {module_name}

    def load_modules(self, module_names: Iterable[str]) -> None:
        for module_name in module_names:
            self.load_module(module_name)


def register_builtin_components(registry: ComponentRegistry) -> None:
    """Register preserved, backend-independent ESI workflow and model assets."""
    from .model_factory import builtin_provider_factory
    from .model_registry import list_registered_models
    from .agents.esi.definitions import ESI_AGENTS, ESI_SCHEMAS, ESI_TOOLS
    from .evaluation.esi_final_acuity import ESIFinalAcuityGrader
    from .dataset import load_esi_jsonl, load_jsonl
    from .workflows.esi.definition import ESI_MAS
    from .workflows.esi.payload_builder import build_pending_agent_payload, payload_builders

    registry.register("workflows", "esi.legacy_v1", ESI_MAS)
    registry.register("payload_builders", "esi.legacy_v1", build_pending_agent_payload)
    for provider_id in ("openai", "azure_openai", "openai_api", "dr7", "vllm"):
        registry.register("providers", provider_id, builtin_provider_factory(provider_id))
    for role, builder in payload_builders.items():
        registry.register("payload_builders", f"esi.{role}_v1", builder)
    for model in list_registered_models():
        registry.register("models", model.id, model)
    for identifier, definition in ESI_AGENTS.items():
        registry.register("agents", identifier, definition)
    for identifier, schema in ESI_SCHEMAS.items():
        registry.register("schemas", identifier, schema)
    for identifier, tool in ESI_TOOLS.items():
        registry.register("tools", identifier, tool)
    registry.register("graders", "esi.final_acuity_v1", ESIFinalAcuityGrader)
    registry.register("dataset_loaders", "jsonl", load_jsonl)
    registry.register("dataset_loaders", "esi.jsonl_v1", load_esi_jsonl)
