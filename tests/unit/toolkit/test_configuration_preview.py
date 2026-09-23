"""Preview reflects real constructed kernels without contacting providers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from mas_slm_research.agents.definition import AgentDefinition
from mas_slm_research.configuration import load_configuration
from mas_slm_research.configured_systems import build_configured_systems
from mas_slm_research.handoff import HandoffDefinition, create_handoff_tools
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.preview import inspect_configuration
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from tests.doubles.fake_provider import FakeChatModel


ESI_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "esi" / "experiment.yaml"
ENV = {
    "BASELINE_MODEL_ID": "gpt-4o", "BASELINE_API_KEY": "private-baseline-value",
    "BASELINE_AZURE_ENDPOINT": "https://azure.invalid", "BASELINE_AZURE_API_VERSION": "2024-02-01",
    "SPECIALIST_MODEL_ID": "medgemma-4b-it-Finetuned", "SPECIALIST_API_KEY": "private-specialist-value",
    "SPECIALIST_BASE_URL": "http://localhost:9999/v1",
}


def _esi_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    registry.register("dataset_loaders", "jsonl", lambda path: [])
    return registry


def test_esi_preview_matches_constructed_prompt_model_route_and_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_configuration(ESI_EXAMPLE, registry=_esi_registry(), environment=ENV)
    original_resolve = loaded.registry.resolve

    def refuse_provider(kind, identifier):
        if kind == "providers":
            raise AssertionError("preview attempted provider construction")
        return original_resolve(kind, identifier)

    monkeypatch.setattr(loaded.registry, "resolve", refuse_provider)
    preview = inspect_configuration(loaded)
    systems = build_configured_systems(
        loaded, model_factory=lambda model, role: FakeChatModel([]),
    )
    real_kernel = systems.mas_runner.role_factories["esi1_agent"]()
    shown = preview.details["mas"]["roles"]["esi1_agent"]

    assert shown["assembled_prompt"] == real_kernel._render_system_prompt()
    assert shown["ordered_tool_names"] == [tool.name for tool in real_kernel.tools]
    assert shown["model"]["provider_model_id"] == "esi1-agent-075"
    assert shown["model"]["effective_model_settings"]["max_tokens"] == 4096
    assert shown["model"]["actual_request_decoding"]["max_tokens"] == 250
    assert shown["model"]["actual_request_decoding"]["temperature"] == 0
    assert preview.concise["mas"]["allowed_handoffs"]["esi1_agent"] == ["esi2_agent", "doctor_agent"]
    assert preview.concise["mas"]["gates"]["doctor_gate"]["required_sources"] == ["acuity", "vitals"]
    assert "assembled_prompt" not in preview.concise["sas"]
    assert "assembled_prompt" in preview.details["sas"]
    assert preview.concise["mas"]["roles"]["doctor_agent"]["model_source"] == "agents.doctor.model"
    assert preview.concise["model_precedence"]["mas"][0] == "mas.model_overrides.<role>"
    assert "private-baseline-value" not in preview.render(details=True)
    assert "private-specialist-value" not in preview.render(details=True)
    json.loads(preview.render())
    json.loads(preview.render(details=True))


def test_preview_role_override_matches_constructed_model(tmp_path: Path) -> None:
    import shutil

    experiment_path = tmp_path / "experiment.yaml"
    shutil.copyfile(ESI_EXAMPLE, experiment_path)
    shutil.copyfile(ESI_EXAMPLE.parent / "workflow.yaml", tmp_path / "workflow.yaml")
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    experiment["mas"]["model_overrides"]["doctor_agent"] = "baseline"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    loaded = load_configuration(experiment_path, registry=_esi_registry(), environment=ENV)
    preview = inspect_configuration(loaded)
    systems = build_configured_systems(loaded, model_factory=lambda model, role: FakeChatModel([]))

    shown = preview.concise["mas"]["roles"]["doctor_agent"]
    assert shown["model_source"] == "mas.model_overrides"
    assert shown["model"]["model_id"] == systems.mas_models["doctor_agent"].model_id == "gpt-4o"
    assert shown["model"]["provider_model_id"] == "gpt-4o"


class ToyAnswer(BaseModel):
    value: str


class ToyHandoff(BaseModel):
    note: str


class ToyDefinition(AgentDefinition):
    def __init__(self, name: str, prompt: str) -> None:
        self.name = name
        self.prompt = prompt
        self.output_schema = ToyAnswer

    def build_kernel(self, *, model, runtime_config, workflow, handoff_schemas, registry):
        tools = []
        names = []
        if self.name == "worker" and runtime_config.multi_agent:
            route = HandoffDefinition(
                source_agent="worker", target_agent="reviewer",
                payload_model=registry.resolve("schemas", handoff_schemas["reviewer"]),
                description="Send work for review.", tool_name="send_for_review",
            )
            tools = create_handoff_tools("worker", [route])
            names = ["send_for_review"]
        return AgentKernel(
            model=model, tools=tools, system_prompt=self.prompt,
            response_format=ToyAnswer if self.name == "reviewer" else None,
            agent_node_name=self.name,
            handoff_tool_names=names, handoff_workflow=workflow if names else None,
            runtime_config=runtime_config,
        )


def test_small_external_workflow_previews_its_own_agents_and_gate(tmp_path: Path) -> None:
    experiment = {
        "version": 1, "name": "toy-comparison",
        "models": {"toy": {"provider": "toy.provider", "model_id": "toy-model"}},
        "agents": {
            "baseline": {"definition": "toy.reviewer", "model": "toy"},
            "worker": {"definition": "toy.worker"},
            "reviewer": {"definition": "toy.reviewer"},
        },
        "sas": {"agent": "baseline", "model": "toy"},
        "mas": {"default_model": "toy", "agents": {"worker": "worker", "reviewer": "reviewer"},
                "workflow": "workflow.yaml"},
        "dataset": {"loader": "toy.dataset", "path": "cases.jsonl"},
        "grader": "toy.grade",
    }
    workflow = {
        "version": 1,
        "metadata": {"workflow_id": "toy", "name": "Toy", "version": "1"},
        "participating_agents": ["worker", "reviewer"],
        "sources": {"draft": {"source_id": "draft", "name": "Draft", "agent_names": ["worker"]}},
        "start_agents": ["worker"], "finalizing_agents": ["reviewer"],
        "allowed_handoffs": {"worker": ["reviewer"], "reviewer": []},
        "gates": {"review_gate": {"gate_id": "review_gate", "name": "Review Gate",
                                  "required_sources": ["draft"], "target_node": "reviewer"}},
        "payloads": {
            "worker": {"builder": "toy.payload"},
            "reviewer": {"builder": "toy.payload"},
        },
        "handoff_schemas": {"worker": {"reviewer": "toy.handoff"}},
    }
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    (tmp_path / "workflow.yaml").write_text(yaml.safe_dump(workflow), encoding="utf-8")
    registry = ComponentRegistry()
    registry.register("providers", "toy.provider", lambda model, environment: FakeChatModel([]))
    registry.register("agents", "toy.worker", ToyDefinition("worker", "Worker prompt."))
    registry.register("agents", "toy.reviewer", ToyDefinition("reviewer", "Reviewer prompt."))
    registry.register("schemas", "toy.handoff", ToyHandoff)
    registry.register("payload_builders", "toy.payload", lambda state: {"llm_payload": {"case_info": state.get("case_info")}})
    registry.register("dataset_loaders", "toy.dataset", lambda path: [])
    registry.register("graders", "toy.grade", lambda: None)

    loaded = load_configuration(experiment_path, registry=registry, environment={})
    preview = inspect_configuration(loaded)
    assert preview.concise["mas"]["allowed_handoffs"] == {"worker": ["reviewer"], "reviewer": []}
    assert preview.concise["mas"]["gates"]["review_gate"]["required_sources"] == ["draft"]
    assert preview.details["mas"]["roles"]["worker"]["assembled_prompt"] == "Worker prompt."
    assert preview.details["mas"]["roles"]["reviewer"]["assembled_prompt"] == "Reviewer prompt."
    assert preview.concise["mas"]["roles"]["worker"]["model"]["provider_model_id"] == "toy-model"
    assert preview.concise["mas"]["roles"]["worker"]["handoff_tool_names"] == ["send_for_review"]
