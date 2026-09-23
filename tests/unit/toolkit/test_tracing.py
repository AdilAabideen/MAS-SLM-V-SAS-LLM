"""Optional span parentage, duplicate prevention, and exporter isolation."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml

from mas_slm_research.cli import _load_fixture
from mas_slm_research.configuration import ConfigurationError, TelemetryConfig, load_configuration
from mas_slm_research.experiment import run_configured_experiment
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.telemetry import token_estimator
from mas_slm_research.tracing import trace_experiment


ROOT = Path(__file__).resolve().parents[3]


def _run(monkeypatch):
    monkeypatch.setattr(token_estimator, "tiktoken", None)
    env, factory = _load_fixture(ROOT / "examples" / "esi" / "offline_fixture.json")
    registry = ComponentRegistry()
    register_builtin_components(registry)
    loaded = load_configuration(ROOT / "examples" / "esi" / "experiment.yaml",
                                registry=registry, environment=env)
    run = asyncio.run(run_configured_experiment(loaded, model_factory=factory, experiment_id="trace-test"))
    return run


def test_in_memory_tree_has_one_owner_and_no_duplicate_spans(monkeypatch) -> None:
    run = _run(monkeypatch)
    class MemorySink:
        def __init__(self): self.batches = []
        def export(self, spans): self.batches.append(tuple(spans))
    sink = MemorySink()
    off = trace_experiment(run, config=TelemetryConfig(), sink=sink)
    assert off.spans == () and sink.batches == []

    on = trace_experiment(run, config=TelemetryConfig(enabled=True), sink=sink)
    assert len(sink.batches) == 1
    assert on.spans == sink.batches[0]
    by_id = {span.span_id: span for span in on.spans}
    assert len(by_id) == len(on.spans)
    assert sum(span.name == "experiment" for span in on.spans) == 1
    assert sum(span.name == "case" for span in on.spans) == 3
    assert sum(span.name.startswith("system.") for span in on.spans) == 6
    assert sum(span.name == "model.call" for span in on.spans) == 12
    assert sum(span.name == "tool.call" for span in on.spans) == 12
    for span in on.spans:
        if span.parent_id is not None:
            assert span.parent_id in by_id
    assert all(by_id[span.parent_id].name.startswith("agent.") for span in on.spans
               if span.name in {"model.call", "tool.call"})
    assert any(event.name == "gate_evaluated" for span in on.spans for event in span.events)
    assert any(event.name.startswith("handoff") for span in on.spans for event in span.events)
    assert all("fixture-only" not in str(span) for span in on.spans)


def test_failing_exporter_changes_no_prediction_or_accounting(monkeypatch) -> None:
    run = _run(monkeypatch)
    before = [(attempt.result.to_dict(), attempt.grade.to_dict()) for attempt in run.attempts]
    class FailingSink:
        def export(self, spans): raise RuntimeError("collector unavailable")
    outcome = trace_experiment(run, config=TelemetryConfig(enabled=True), sink=FailingSink())
    assert len(outcome.spans) > 0
    assert outcome.warnings == ("trace export failed: RuntimeError",)
    assert before == [(attempt.result.to_dict(), attempt.grade.to_dict()) for attempt in run.attempts]


def test_missing_optional_sdk_is_visible_but_no_collector_is_contacted(monkeypatch) -> None:
    run = _run(monkeypatch)
    outcome = trace_experiment(
        run, config=TelemetryConfig(enabled=True, exporter="otlp_http", endpoint_env="TEST_OTLP"),
        environment={"TEST_OTLP": "http://localhost:9999/v1/traces"},
    )
    assert outcome.spans
    assert outcome.warnings == ("trace export failed: optional OpenTelemetry dependency unavailable",)


def test_enabled_otlp_references_are_validated_and_secret_values_stay_out_of_snapshot(tmp_path: Path) -> None:
    source = ROOT / "examples" / "esi"
    for name in ("experiment.yaml", "workflow.yaml", "cases.jsonl"):
        shutil.copyfile(source / name, tmp_path / name)
    experiment = tmp_path / "experiment.yaml"
    data = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    data["telemetry"] = {"enabled": True, "exporter": "otlp_http",
                         "endpoint_env": "TRACE_ENDPOINT", "header_env": {"api_key": "TRACE_TOKEN"}}
    experiment.write_text(yaml.safe_dump(data), encoding="utf-8")
    env, _ = _load_fixture(source / "offline_fixture.json")
    registry = ComponentRegistry()
    register_builtin_components(registry)
    with pytest.raises(ConfigurationError, match="TRACE_ENDPOINT"):
        load_configuration(experiment, registry=registry, environment=env)
    loaded = load_configuration(
        experiment, registry=registry,
        environment={**env, "TRACE_ENDPOINT": "http://localhost:9999/v1/traces",
                     "TRACE_TOKEN": "secret-value"},
    )
    assert loaded.experiment.telemetry.enabled
    assert "secret-value" not in str(loaded.safe_snapshot())
