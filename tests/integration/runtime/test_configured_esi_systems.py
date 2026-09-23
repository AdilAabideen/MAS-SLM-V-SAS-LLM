"""The split ESI config builds both extracted systems with preserved contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from mas_slm_research.configuration import load_configuration
from mas_slm_research.comparison import compare_experiment, configured_prices
from mas_slm_research.configured_systems import build_configured_systems
from mas_slm_research.contracts import RunIdentity, RunStatus
from mas_slm_research.experiment import ExperimentStatus, run_configured_experiment
from mas_slm_research.multi_agent import MultiAgentRunner
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.telemetry import token_estimator
from tests.doubles.fake_provider import FakeChatModel
from tests.regression.test_research_baseline_contracts import PROMPT_DIGESTS, TOOL_CONTRACT_DIGESTS


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "esi" / "experiment.yaml"
ENV = {
    "BASELINE_MODEL_ID": "gpt-4o",
    "BASELINE_API_KEY": "test-only-key",
    "BASELINE_AZURE_ENDPOINT": "https://azure.invalid",
    "BASELINE_AZURE_API_VERSION": "2024-02-01",
    "SPECIALIST_MODEL_ID": "medgemma-4b-it-Finetuned",
    "SPECIALIST_API_KEY": "test-only-key",
    "SPECIALIST_BASE_URL": "http://localhost:9999/v1",
}
CASE = {
    "gender": "female", "race": "unknown", "arrival_transport": "walk-in",
    "pain": "0", "chiefcomplaint": "minor symptom", "age": 30,
    "tiragecase": "Synthetic triage case for offline tests.",
    "temperature": 36.8, "heartrate": 80, "resprate": 16,
    "o2sat": 99, "sbp": 120, "dbp": 80,
}


def _registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    return registry


def _tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": name, "name": name, "args": args}])


def _baseline_output() -> dict:
    return {
        "final_esi_level": 3, "confidence": 0.8,
        "decision_source": "esi345_resource_prediction", "uptriaged": False,
        "abnormal_vitals_considered": True, "vitals_summary": "Normal supplied vitals.",
        "case_summary": "Synthetic case.", "rationale": "Offline test response.",
    }


def _doctor_output(level: int, source: str) -> dict:
    return {
        "final_esi_level": level, "uptriaged": False,
        "decision_source": source, "audit_summary": "Synthetic routing decision.",
    }


def _responses(path: tuple[str, ...]) -> dict[str, AIMessage]:
    esi1_target = "doctor_agent" if len(path) == 1 else "esi2_agent"
    esi1_args = (
        {"is_esi1": True, "reason": "Synthetic immediate need", "critical_concerns": []}
        if esi1_target == "doctor_agent"
        else {"is_esi1": False, "brief_reason": "No immediate need", "carry_forward_concerns": []}
    )
    result = {
        "esi1_agent": _tool_call(
            "final_esi1_true_handoff_to_doctor_agent" if len(path) == 1
            else "final_esi1_false_handoff_to_esi2_agent", esi1_args,
        ),
        "vitals_agent": _tool_call("finalise_output", {
            "consider_uptriage": False, "reason": "Normal supplied vitals",
            "abnormal_vitals": [], "confidence": 0.8,
        }),
        "doctor_agent": _tool_call(
            "final_answer", _doctor_output(1 if len(path) == 1 else 3, "esi1" if len(path) == 1 else "esi345"),
        ),
    }
    if len(path) > 1:
        result["esi2_agent"] = _tool_call("final_esi2_false_handoff_to_esi345_agent", {
            "is_esi2": False, "reason": "No high-risk finding", "carry_forward_concerns": [],
        })
        result["esi345_agent"] = _tool_call("final_esi345_result_handoff_to_doctor_agent", {
            "esi_level": 3, "num_resources": 2,
            "predicted_resources": ["labs", "imaging"], "reason": "Two resources",
        })
    return result


@pytest.fixture(autouse=True)
def _offline_token_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(token_estimator, "tiktoken", None)


def test_constructed_prompt_tools_schemas_and_model_mapping_match_esi() -> None:
    from app.agentic.agents.esi1.prompt import HANDOFF_REQUIREMENTS, SYSTEM_PROMPT
    from app.agentic.agents.esi1.schema import ES1AgentOutput
    from app.agentic.agents.single_agent_system.prompt import SYSTEM_PROMPT as BASELINE_PROMPT

    loaded = load_configuration(EXAMPLE, registry=_registry(), environment=ENV)
    observed: list[tuple[str, str]] = []

    def fake_factory(model, role):
        observed.append((role, model.model_id))
        return FakeChatModel([])

    systems = build_configured_systems(loaded, model_factory=fake_factory)
    sas = systems.sas_runner.kernel_factory()
    esi1 = systems.mas_runner.role_factories["esi1_agent"]()
    doctor = systems.mas_runner.role_factories["doctor_agent"]()

    assert sas.system_prompt == BASELINE_PROMPT
    assert [tool.name for tool in sas.tools] == [
        "compute_esi_danger_zone", "compute_shock_index", "create_plan", "log_thought", "final_answer",
    ]
    assert esi1.system_prompt == SYSTEM_PROMPT
    assert HANDOFF_REQUIREMENTS.strip() in esi1._render_system_prompt()
    assert esi1.response_format.model_json_schema() == ES1AgentOutput.model_json_schema()
    assert [tool.name for tool in esi1.tools] == [
        "create_plan", "log_thought", "final_esi1_false_handoff_to_esi2_agent",
        "final_esi1_true_handoff_to_doctor_agent",
    ]
    assert [tool.name for tool in doctor.tools] == ["create_plan", "log_thought", "final_answer"]
    assert observed == [
        ("baseline", "gpt-4o"), ("esi1_agent", "medgemma-4b-it-Finetuned"),
        ("doctor_agent", "medgemma-4b-it-Finetuned"),
    ]
    assert systems.workflow is loaded.registry.resolve("workflows", "esi.legacy_v1")


def test_every_preserved_esi_prompt_and_ordered_tool_contract_matches_baseline() -> None:
    from mas_slm_research.agents.esi.definitions import ESI_AGENTS
    from mas_slm_research.runtime.runtime_config import RuntimeConfig
    from mas_slm_research.workflows.esi.definition import ESI_MAS

    registry = _registry()
    for definition in ESI_AGENTS.values():
        modes = ("single",) if definition.name == "single_agent" else ("single", "multi")
        for mode in modes:
            schemas = {
                handoff.target_agent: (
                    f"esi.{handoff.source_agent.removesuffix('_agent')}_to_"
                    f"{handoff.target_agent.removesuffix('_agent')}_v1"
                )
                for handoff in definition.handoffs
            } if mode == "multi" else {}
            kernel = definition.build_kernel(
                model=FakeChatModel([]), runtime_config=RuntimeConfig(multi_agent=mode == "multi"),
                workflow=ESI_MAS if mode == "multi" else None,
                handoff_schemas=schemas, registry=registry,
            )
            prompt_digest = hashlib.sha256(kernel._render_system_prompt().encode()).hexdigest()
            tool_contract = [{
                "name": tool.name, "description": tool.description,
                "schema": tool.args_schema.model_json_schema() if tool.args_schema else None,
            } for tool in kernel.tools]
            tool_digest = hashlib.sha256(
                json.dumps(tool_contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            assert prompt_digest == PROMPT_DIGESTS[(definition.name, mode)]
            assert tool_digest == TOOL_CONTRACT_DIGESTS[(definition.name, mode)]


def test_configured_role_override_and_finetuned_checkpoints(tmp_path: Path) -> None:
    import shutil
    import yaml

    experiment = tmp_path / "experiment.yaml"
    shutil.copyfile(EXAMPLE, experiment)
    shutil.copyfile(EXAMPLE.parent / "workflow.yaml", tmp_path / "workflow.yaml")
    data = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    data["mas"]["model_overrides"]["doctor_agent"] = "baseline"
    data["agents"]["esi1"]["runtime"] = {"max_tool_calls_per_turn": 3}
    experiment.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    loaded = load_configuration(experiment, registry=_registry(), environment=ENV)
    systems = build_configured_systems(loaded, model_factory=lambda model, role: FakeChatModel([]))
    assert systems.mas_models["doctor_agent"].model_id == "gpt-4o"
    assert systems.mas_models["esi1_agent"].model_id == "medgemma-4b-it-Finetuned"
    assert systems.mas_runner.role_factories["esi1_agent"]().runtime_config.max_tool_calls_per_turn == 3

    preserved_model = loaded.registry.resolve("providers", "vllm")(
        loaded.models["specialist"], ENV,
    )
    assert preserved_model.agent_model_id_overrides["esi1_agent"] == "esi1-agent-075"
    assert preserved_model.agent_model_id_overrides["doctor_agent"] == "medgemma-4b-it"


def test_declarative_workflow_and_selected_payload_builder_execute(tmp_path: Path) -> None:
    import shutil
    import yaml
    from mas_slm_research.workflows.esi.payloads.esi1 import build_payload as original_esi1_builder

    experiment = tmp_path / "experiment.yaml"
    workflow_path = tmp_path / "workflow.yaml"
    shutil.copyfile(EXAMPLE, experiment)
    shutil.copyfile(EXAMPLE.parent / "workflow.yaml", workflow_path)
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    data["payloads"]["esi1_agent"]["builder"] = "test.esi1_payload_v1"
    workflow_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    observed: list[str] = []

    def custom_builder(state):
        observed.append("esi1_agent")
        return original_esi1_builder(state)

    registry = _registry()
    registry.register("payload_builders", "test.esi1_payload_v1", custom_builder)
    loaded = load_configuration(experiment, registry=registry, environment=ENV)
    responses = _responses(("esi1_agent",))
    systems = build_configured_systems(
        loaded, model_factory=lambda model, role: FakeChatModel([responses.get(role, AIMessage(content=""))]),
    )
    declarative_runner = MultiAgentRunner(
        workflow=loaded.workflow,
        role_factories=systems.mas_runner.role_factories,
        payload_builder=systems.mas_runner.payload_builder,
        output_validator=systems.mas_runner.output_validator,
    )
    run = asyncio.run(declarative_runner.run_case(
        identity=RunIdentity(experiment_id="esi", system_id="declarative", case_id="c1", repetition=1, run_id="graph-1"),
        case_info=CASE,
    ))
    assert run.result.status == RunStatus.COMPLETED
    assert run.result.output.to_dict()["final_esi_level"] == 1
    assert observed == ["esi1_agent"]


def test_partial_case_keeps_preserved_payload_tolerance() -> None:
    loaded = load_configuration(EXAMPLE, registry=_registry(), environment=ENV)
    responses = _responses(("esi1_agent",))
    systems = build_configured_systems(
        loaded, model_factory=lambda model, role: FakeChatModel([responses.get(role, AIMessage(content=""))]),
    )
    run = asyncio.run(systems.mas_runner.run_case(
        identity=RunIdentity(experiment_id="esi", system_id="multi", case_id="partial", repetition=1, run_id="mas-partial"),
        case_info={"chiefcomplaint": "synthetic incomplete observation"},
    ))
    assert run.result.status == RunStatus.COMPLETED
    assert run.result.output.to_dict()["final_esi_level"] == 1


def test_registered_custom_provider_is_used_without_core_edits(tmp_path: Path) -> None:
    import shutil
    import yaml

    experiment = tmp_path / "experiment.yaml"
    shutil.copyfile(EXAMPLE, experiment)
    shutil.copyfile(EXAMPLE.parent / "workflow.yaml", tmp_path / "workflow.yaml")
    data = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    data["models"]["baseline"] = {"provider": "test.provider", "model_id": "test-model"}
    experiment.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    registry = _registry()
    observed: list[str] = []

    def provider(model, environment):
        observed.append(model.model_id)
        return FakeChatModel([])

    registry.register("providers", "test.provider", provider)
    loaded = load_configuration(experiment, registry=registry, environment=ENV)
    systems = build_configured_systems(loaded, environment=ENV)
    systems.sas_runner.kernel_factory()
    assert observed == ["test-model"]


@pytest.mark.integration
@pytest.mark.parametrize("path", [("esi1_agent",), ("esi1_agent", "esi2_agent", "esi345_agent")])
def test_configured_sas_and_mas_execute_fake_provider_routes(path: tuple[str, ...]) -> None:
    loaded = load_configuration(EXAMPLE, registry=_registry(), environment=ENV)
    responses = _responses(path)

    def fake_factory(model, role):
        return FakeChatModel([
            _tool_call("final_answer", _baseline_output()) if role == "baseline" else responses.get(role, AIMessage(content=""))
        ])

    systems = build_configured_systems(loaded, model_factory=fake_factory)
    sas = asyncio.run(systems.sas_runner.run_case(
        identity=RunIdentity(experiment_id="esi", system_id="single", case_id="c1", repetition=1, run_id="sas-1"),
        payload=CASE,
    ))
    mas = asyncio.run(systems.mas_runner.run_case(
        identity=RunIdentity(experiment_id="esi", system_id="multi", case_id="c1", repetition=1, run_id="mas-1"),
        case_info=CASE,
    ))
    assert sas.result.status == RunStatus.COMPLETED
    assert sas.result.output.to_dict()["final_esi_level"] == 3
    assert mas.result.status == RunStatus.COMPLETED
    assert mas.result.output.to_dict()["final_esi_level"] == (1 if len(path) == 1 else 3)
    assert {item.agent_name for item in mas.agent_records} == set(path) | {"vitals_agent", "doctor_agent"}
    assert len(mas.handoff_records) == len(path) + 1
    assert mas.gate_records


def test_configured_construction_imports_no_backend_in_fresh_process() -> None:
    code = '''
import sys
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.configuration import load_configuration
from mas_slm_research.configured_systems import build_configured_systems
from tests.doubles.fake_provider import FakeChatModel
r = ComponentRegistry(); register_builtin_components(r)
loaded = load_configuration(sys.argv[1], registry=r, environment={
    "BASELINE_MODEL_ID": "gpt-4o", "BASELINE_API_KEY": "x",
    "BASELINE_AZURE_ENDPOINT": "https://azure.invalid", "BASELINE_AZURE_API_VERSION": "2024-02-01",
    "SPECIALIST_MODEL_ID": "medgemma-4b-it-Finetuned", "SPECIALIST_API_KEY": "x",
    "SPECIALIST_BASE_URL": "http://localhost:9999/v1",
})
systems = build_configured_systems(loaded, model_factory=lambda model, role: FakeChatModel([]))
systems.sas_runner.kernel_factory()
systems.mas_runner.role_factories["esi1_agent"]()
assert "app" not in sys.modules
assert "sqlalchemy" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLE)],
        env={"PYTHONPATH": f"src{':' + __import__('os').environ['PYTHONPATH'] if __import__('os').environ.get('PYTHONPATH') else ''}"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_registered_esi_dataset_runs_both_arms_in_system_major_order() -> None:
    loaded = load_configuration(EXAMPLE, registry=_registry(), environment=ENV)
    responses = _responses(("esi1_agent",))

    def fake_factory(model, role):
        response = (_tool_call("final_answer", _baseline_output()) if role == "baseline"
                    else responses.get(role, AIMessage(content="")))
        return FakeChatModel([response])

    run = asyncio.run(run_configured_experiment(
        loaded, model_factory=fake_factory, experiment_id="offline-esi",
    ))
    assert run.status == ExperimentStatus.COMPLETED
    assert len(run.attempts) == 6
    assert [attempt.system_id for attempt in run.attempts] == ["single"] * 3 + ["multi"] * 3
    assert all(pair.single and pair.multi for pair in run.pairs)
    assert all(attempt.result.status == RunStatus.COMPLETED for attempt in run.attempts)
    assert all(attempt.grade.status.value == "graded" for attempt in run.attempts)
    grader = loaded.registry.resolve("graders", loaded.experiment.grader)
    report = compare_experiment(run, grader=grader, prices_by_role=configured_prices(loaded))
    assert report.systems["single"].attempted == 3
    assert report.systems["multi"].attempted == 3
    assert report.systems["single"].grader_summary["attempted"] == 3
    assert report.systems["multi"].grader_summary["attempted"] == 3
    assert report.systems["single"].cost_usd_estimate is None
    assert len(report.pairs) == 3
