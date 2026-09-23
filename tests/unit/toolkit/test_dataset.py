"""Dataset IDs, inputs, and expected labels are validated before inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mas_slm_research.configuration import load_configuration
from mas_slm_research.dataset import (
    DatasetCase, DatasetError, load_configured_dataset, load_dataset, load_jsonl,
)
from mas_slm_research.evaluation.esi_final_acuity import ESIFinalAcuityGrader
from mas_slm_research.registry import ComponentRegistry, register_builtin_components


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "esi" / "experiment.yaml"
ENV = {
    "BASELINE_MODEL_ID": "gpt-4o", "BASELINE_API_KEY": "secret",
    "BASELINE_AZURE_ENDPOINT": "https://azure.invalid", "BASELINE_AZURE_API_VERSION": "2024-02-01",
    "SPECIALIST_MODEL_ID": "medgemma-4b-it-Finetuned", "SPECIALIST_API_KEY": "secret",
    "SPECIALIST_BASE_URL": "http://localhost:9999/v1",
}


def _registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    register_builtin_components(registry)
    return registry


def _row(case_id="c1", *, expected=None, input=None):
    return {"case_id": case_id, "input": input or {"question": "A?"},
            "expected": expected or {"acuity": 2}}


def _write(path: Path, *rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_versioned_synthetic_esi_dataset_loads_with_hidden_labels() -> None:
    loaded = load_configuration(EXAMPLE, registry=_registry(), environment=ENV)
    dataset = load_configured_dataset(loaded, split="demo")

    assert dataset.loader_id == "esi.jsonl_v1"
    assert dataset.case_ids == (
        "synthetic-esi1-001", "synthetic-esi2-001", "synthetic-esi4-001",
    )
    assert all(case.metadata["provenance"] == "fabricated synthetic example" for case in dataset.cases)
    assert all(case.metadata["clinical_validation"] is False for case in dataset.cases)
    assert all("acuity" not in case.agent_input() for case in dataset.cases)
    assert [case.expected_label()["acuity"] for case in dataset.cases] == [1, 2, 4]
    facts = dataset.cases[0].agent_input()
    facts["chiefcomplaint"] = "mutated"
    assert dataset.cases[0].agent_input()["chiefcomplaint"] == "unresponsive on arrival"


def test_generic_jsonl_rejects_duplicate_ids_before_any_model_call(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write(path, _row("same"), _row("same"))
    with pytest.raises(DatasetError, match="duplicate case_id 'same'"):
        load_dataset(path=path, loader_id="jsonl", loader=load_jsonl, grader=ESIFinalAcuityGrader())


def test_malformed_row_duplicate_key_and_invalid_label_report_location(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"c1","input":{},"expected":{"acuity":2}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="cases.jsonl:1"):
        load_jsonl(path)

    path.write_text('{"case_id":"c1","case_id":"c2","input":{},"expected":{"acuity":2}}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate JSON key 'case_id'"):
        load_jsonl(path)

    _write(path, _row(expected={"acuity": 9}))
    with pytest.raises(DatasetError, match=r"case 1 \(c1\): invalid expected label"):
        load_dataset(path=path, loader_id="jsonl", loader=load_jsonl, grader=ESIFinalAcuityGrader())

    _write(path, {**_row(), "unexpected": 5})
    with pytest.raises(DatasetError, match="unknown case fields"):
        load_dataset(path=path, loader_id="jsonl", loader=load_jsonl, grader=ESIFinalAcuityGrader())


def test_esi_alias_projection_and_target_leak_rejection(tmp_path: Path) -> None:
    from mas_slm_research.dataset import load_esi_jsonl

    path = tmp_path / "cases.jsonl"
    _write(path, _row(input={
        "gender": "female", "race": "not recorded", "arrivaltransport": "walk-in",
        "chief complaint": "synthetic symptom", "triage_case": "entirely fictional",
    }))
    dataset = load_dataset(path=path, loader_id="esi.jsonl_v1", loader=load_esi_jsonl,
                           grader=ESIFinalAcuityGrader())
    assert dataset.cases[0].agent_input()["arrival_transport"] == "walk-in"
    assert dataset.cases[0].agent_input()["chiefcomplaint"] == "synthetic symptom"
    assert dataset.cases[0].agent_input()["tiragecase"] == "entirely fictional"
    assert dataset.cases[0].agent_input()["temperature"] is None

    _write(path, _row(input={**dataset.cases[0].agent_input(), "acuity": 2}))
    with pytest.raises(DatasetError, match="contains target labels"):
        load_esi_jsonl(path)


def test_selection_validates_ids_and_keeps_expected_separate(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write(path,
           {**_row("c1"), "split": "train", "tags": ["toy"]},
           {**_row("c2"), "split": "test", "tags": ["toy"]})
    dataset = load_dataset(path=path, loader_id="jsonl", loader=load_jsonl,
                           grader=ESIFinalAcuityGrader(), split="test", case_ids=["c2"])
    assert dataset.case_ids == ("c2",)
    assert dataset.cases[0].agent_input() == {"question": "A?"}
    assert dataset.cases[0].expected_label() == {"acuity": 2}
    with pytest.raises(DatasetError, match="selected case IDs not found"):
        load_dataset(path=path, loader_id="jsonl", loader=load_jsonl,
                     grader=ESIFinalAcuityGrader(), case_ids=["missing"])


def test_backend_free_dataset_import_in_fresh_process() -> None:
    code = '''
import sys
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.dataset import load_dataset
r = ComponentRegistry(); register_builtin_components(r)
assert "app" not in sys.modules
assert "sqlalchemy" not in sys.modules
assert "esi.jsonl_v1" in r.ids("dataset_loaders")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": f"src{':' + __import__('os').environ['PYTHONPATH'] if __import__('os').environ.get('PYTHONPATH') else ''}"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
