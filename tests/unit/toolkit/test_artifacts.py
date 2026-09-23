"""Incremental artifact integrity and offline partial-run interpretation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mas_slm_research.artifacts import ArtifactError, ArtifactWriter, summarize_artifacts
from mas_slm_research.cli import _load_fixture
from mas_slm_research.comparison import compare_experiment, configured_prices
from mas_slm_research.configuration import load_configuration
from mas_slm_research.dataset import load_configured_dataset
from mas_slm_research.experiment import run_configured_experiment
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.telemetry import token_estimator


ROOT = Path(__file__).resolve().parents[3]


def _fixture_run(monkeypatch):
    monkeypatch.setattr(token_estimator, "tiktoken", None)
    env, factory = _load_fixture(ROOT / "examples" / "esi" / "offline_fixture.json")
    registry = ComponentRegistry()
    register_builtin_components(registry)
    loaded = load_configuration(ROOT / "examples" / "esi" / "experiment.yaml",
                                registry=registry, environment=env)
    dataset = load_configured_dataset(loaded)
    run = asyncio.run(run_configured_experiment(loaded, model_factory=factory, experiment_id="artifact-test"))
    grader = loaded.registry.resolve("graders", loaded.experiment.grader)
    report = compare_experiment(run, grader=grader, prices_by_role=configured_prices(loaded))
    return loaded, dataset, run, report, env


def test_partial_interruption_retains_one_readable_attempt_and_empty_pair_slots(tmp_path: Path, monkeypatch) -> None:
    loaded, dataset, run, _, env = _fixture_run(monkeypatch)
    writer = ArtifactWriter(
        directory=tmp_path / "partial", loaded=loaded, dataset=dataset,
        experiment_id=run.experiment_id, prices_by_role={}, environment=env,
    )
    writer.record_attempt(run.attempts[0])
    writer.mark_interrupted()
    manifest = json.loads((writer.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "interrupted" and manifest["attempts_written"] == 1
    report = summarize_artifacts(writer.directory)
    assert report.experiment_status == "interrupted"
    assert report.systems["single"].attempted == 1
    assert report.systems["multi"].attempted == 0
    assert len(report.pairs) == 3
    assert report.pairs[0].multi_run_id is None


def test_round_trip_hashes_and_secret_scrubbing(tmp_path: Path, monkeypatch) -> None:
    loaded, dataset, run, report, env = _fixture_run(monkeypatch)
    writer = ArtifactWriter(
        directory=tmp_path / "complete", loaded=loaded, dataset=dataset,
        experiment_id=run.experiment_id, prices_by_role={}, environment=env,
        include_events=True,
    )
    for attempt in run.attempts:
        writer.record_attempt(attempt)
    writer.finalize(run, report)
    assert summarize_artifacts(writer.directory).to_dict() == report.to_dict()
    for path in writer.directory.iterdir():
        assert "fixture-only" not in path.read_text(encoding="utf-8")
    manifest = json.loads((writer.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "completed"
    assert manifest["hashes"]["dataset"]
    assert manifest["source_revision"] is not None
    assert manifest["package_version"] == "0.0.0"
    (writer.directory / "summary.json").unlink()
    assert summarize_artifacts(writer.directory).to_dict() == report.to_dict()
    with pytest.raises(ArtifactError, match="summary.json"):
        summarize_artifacts(writer.directory, verify_derived=True)
    with (writer.directory / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        summarize_artifacts(writer.directory)
