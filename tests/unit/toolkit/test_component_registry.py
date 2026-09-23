"""Explicit registration and extension loading without backend imports."""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel

from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.workflows.esi.definition import ESI_MAS


def test_builtin_esi_assets_resolve_without_backend() -> None:
    registry = ComponentRegistry()
    register_builtin_components(registry)

    assert registry.resolve("workflows", "esi.legacy_v1") is ESI_MAS
    assert registry.resolve("payload_builders", "esi.legacy_v1")
    assert "esi.doctor_agent_v1" in registry.ids("payload_builders")
    assert registry.resolve("models", "medgemma-4b-it").provider == "dr7"
    assert registry.ids("providers") == ("dr7", "openai", "vllm")


def test_explicit_extension_registers_every_component_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("research_test_extension")

    class ExampleSchema(BaseModel):
        value: str

    class ExampleGrader:
        def evaluate(self, expected, actual):
            return expected == actual

        def aggregate(self, results):
            return sum(results)

    def register_components(registry: ComponentRegistry) -> None:
        registry.register("providers", "test.provider", lambda spec, settings: object())
        registry.register("models", "test.model", lambda settings: object())
        registry.register("agents", "test.agent", lambda model, workflow=None: object())
        registry.register("tools", "test.tool", lambda value: value)
        registry.register("schemas", "test.schema", ExampleSchema)
        registry.register("payload_builders", "test.payload", lambda role, state: {})
        registry.register("workflows", "test.workflow", lambda: ESI_MAS)
        registry.register("dataset_loaders", "test.dataset", lambda path: [])
        registry.register("graders", "test.grader", ExampleGrader)

    module.register_components = register_components
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = ComponentRegistry()
    registry.load_module(module.__name__)

    assert all(registry.ids(kind) for kind in registry.inventory())
    assert registry.resolve("schemas", "test.schema") is ExampleSchema
    assert registry.resolve("graders", "test.grader") is ExampleGrader
    with pytest.raises(ValueError, match="already loaded"):
        registry.load_module(module.__name__)


def test_duplicate_unknown_and_invalid_components_have_context() -> None:
    registry = ComponentRegistry()
    registry.register("agents", "my.agent", lambda: None)
    with pytest.raises(ValueError, match="Duplicate agents component ID 'my.agent'"):
        registry.register("agents", "my.agent", lambda: None)
    with pytest.raises(KeyError, match="Unknown agents component ID 'missing'; available: my.agent"):
        registry.resolve("agents", "missing")
    with pytest.raises(TypeError, match="schemas component 'bad'"):
        registry.register("schemas", "bad", object())
    with pytest.raises(ValueError, match="Unknown component kind"):
        registry.register("other", "x", lambda: None)  # type: ignore[arg-type]


def test_failed_extension_does_not_partially_register(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("research_broken_extension")

    def register_components(registry: ComponentRegistry) -> None:
        registry.register("agents", "test.partial", lambda: None)
        registry.register("agents", "test.partial", lambda: None)

    module.register_components = register_components
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="Duplicate agents"):
        registry.load_module(module.__name__)
    assert registry.ids("agents") == ()
