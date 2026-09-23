"""Split YAML loading fails safely before provider or runner construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mas_slm_research.configuration import ConfigurationError, load_configuration
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.workflows.esi.definition import ESI_MAS


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "esi"


def _registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    registry.register("dataset_loaders", "jsonl", lambda path: [])
    return registry


def _env() -> dict[str, str]:
    return {
        "BASELINE_MODEL_ID": "gpt-test-baseline",
        "BASELINE_API_KEY": "secret-baseline-123",
        "BASELINE_AZURE_ENDPOINT": "https://azure.invalid",
        "BASELINE_AZURE_API_VERSION": "2024-02-01",
        "SPECIALIST_MODEL_ID": "medgemma-test",
        "SPECIALIST_API_KEY": "secret-specialist-456",
        "SPECIALIST_BASE_URL": "https://provider.invalid/api",
    }


def _copy_example(tmp_path: Path) -> tuple[Path, Path]:
    experiment = tmp_path / "experiment.yaml"
    workflow = tmp_path / "workflow.yaml"
    experiment.write_text((EXAMPLE / "experiment.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    workflow.write_text((EXAMPLE / "workflow.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return experiment, workflow


def _edit_yaml(path: Path, edit) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_valid_esi_split_config_resolves_without_inference_or_secrets(tmp_path: Path) -> None:
    experiment, workflow = _copy_example(tmp_path)
    loaded = load_configuration(experiment, registry=_registry(), environment=_env())

    assert loaded.workflow == ESI_MAS
    assert loaded.workflow_path == workflow
    assert loaded.dataset_path == (tmp_path / "../../data/cases.jsonl").resolve()
    assert loaded.models["baseline"].model_id == "gpt-test-baseline"
    assert loaded.experiment.sas.agent == "baseline"
    assert loaded.experiment.mas.agents["doctor_agent"] == "doctor"
    assert loaded.workflow_file.payloads["doctor_agent"].builder == "esi.doctor_agent_v1"
    snapshot = json.dumps(loaded.safe_snapshot())
    assert "secret-baseline-123" not in snapshot
    assert "secret-specialist-456" not in snapshot
    assert "BASELINE_API_KEY" in snapshot
    with pytest.raises(Exception):
        loaded.experiment.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        loaded.experiment.mas.agents["doctor_agent"] = "changed"
    with pytest.raises(TypeError, match="immutable"):
        loaded.workflow_file.handoff_schemas["esi1_agent"]["doctor_agent"] = "changed"


@pytest.mark.parametrize(
    ("field", "bad_value", "expected"),
    [
        ("sas.agent", "missing", "sas.agent: unknown agent"),
        ("mas.default_model", "missing", "mas.default_model: unknown model"),
        ("agents.esi1.definition", "missing", "agents.esi1.definition"),
        ("models.specialist.provider", "missing", "models.specialist.provider"),
        ("dataset.loader", "missing", "dataset.loader"),
    ],
)
def test_unknown_references_report_experiment_field(
    tmp_path: Path, field: str, bad_value: str, expected: str
) -> None:
    experiment, _ = _copy_example(tmp_path)

    def change(data):
        current = data
        parts = field.split(".")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = bad_value

    _edit_yaml(experiment, change)
    with pytest.raises(ConfigurationError, match=expected) as exc:
        load_configuration(experiment, registry=_registry(), environment=_env())
    assert str(experiment) in str(exc.value)


def test_missing_env_and_unknown_key_are_early_errors(tmp_path: Path) -> None:
    experiment, _ = _copy_example(tmp_path)
    environment = _env()
    environment.pop("BASELINE_API_KEY")
    with pytest.raises(ConfigurationError, match="models.baseline.api_key_env.*missing or blank"):
        load_configuration(experiment, registry=_registry(), environment=environment)

    _edit_yaml(experiment, lambda data: data["sas"].update({"surprise": True}))
    with pytest.raises(ConfigurationError, match="sas.surprise") as exc:
        load_configuration(experiment, registry=_registry(), environment=_env())
    assert str(experiment) in str(exc.value)


def test_duplicate_yaml_key_reports_filename_and_line(tmp_path: Path) -> None:
    experiment, _ = _copy_example(tmp_path)
    experiment.write_text("version: 1\nname: first\nname: second\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate YAML key 'name' at line 3") as exc:
        load_configuration(experiment, registry=_registry(), environment=_env())
    assert str(experiment) in str(exc.value)


def test_invalid_route_and_unreachable_finalizer_fail(tmp_path: Path) -> None:
    experiment, workflow = _copy_example(tmp_path)
    _edit_yaml(workflow, lambda data: data["allowed_handoffs"]["esi1_agent"].append("missing"))
    with pytest.raises(ConfigurationError, match="allowed_handoffs.*missing") as exc:
        load_configuration(experiment, registry=_registry(), environment=_env())
    assert str(workflow) in str(exc.value)

    _copy_example(tmp_path)
    def disconnect(data):
        data["allowed_handoffs"]["esi1_agent"] = []
        data["allowed_handoffs"]["esi2_agent"] = []
        data["allowed_handoffs"]["esi345_agent"] = []
    _edit_yaml(workflow, disconnect)
    with pytest.raises(ConfigurationError, match="unreachable finalizers|unreachable agents|no finalizer reachable"):
        load_configuration(experiment, registry=_registry(), environment=_env())


def test_cycle_and_payload_contract_gap_fail(tmp_path: Path) -> None:
    experiment, workflow = _copy_example(tmp_path)
    _edit_yaml(workflow, lambda data: data["allowed_handoffs"]["doctor_agent"].append("esi1_agent"))
    with pytest.raises(ConfigurationError, match="unbounded cycle"):
        load_configuration(experiment, registry=_registry(), environment=_env())

    _copy_example(tmp_path)
    _edit_yaml(workflow, lambda data: data["handoff_schemas"]["esi1_agent"].pop("doctor_agent"))
    with pytest.raises(ConfigurationError, match="handoff_schemas: missing"):
        load_configuration(experiment, registry=_registry(), environment=_env())


def test_declared_workflow_cannot_silently_diverge_from_selected_definition(tmp_path: Path) -> None:
    experiment, workflow = _copy_example(tmp_path)
    _edit_yaml(workflow, lambda data: data["metadata"].update({"name": "A changed workflow"}))
    with pytest.raises(ConfigurationError, match="definition: declared workflow differs"):
        load_configuration(experiment, registry=_registry(), environment=_env())
