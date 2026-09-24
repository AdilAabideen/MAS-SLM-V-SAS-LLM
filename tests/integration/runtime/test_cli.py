"""The public CLI works from a clone using scripted offline responses."""

from __future__ import annotations

import json
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from mas_slm_research.cli import _execute, _parser


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "examples" / "esi" / "experiment.yaml"
FIXTURE = ROOT / "examples" / "esi" / "offline_fixture.json"
DR7_CONFIG = ROOT / "examples" / "esi" / "experiment-dr7.yaml"
DR7_FIXTURE = ROOT / "examples" / "esi" / "offline_fixture_openai_dr7.json"


def _dotenv_experiment(tmp_path: Path) -> Path:
    example_dir = tmp_path / "study" / "examples" / "esi"
    example_dir.mkdir(parents=True)
    for name in ("experiment-dr7.yaml", "workflow.yaml", "cases.jsonl"):
        (example_dir / name).write_bytes((DR7_CONFIG.parent / name).read_bytes())
    (tmp_path / ".env").write_text(
        "BASELINE_MODEL_ID=dotenv-baseline\n"
        "OPENAI_API_KEY=dotenv-openai-key\n"
        "DR7_API_KEY=dotenv-dr7-key\n"
        "DR7_BASE_URL=https://dr7.invalid/api/v1/medical\n",
        encoding="utf-8",
    )
    return example_dir / "experiment-dr7.yaml"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mas_slm_research.cli", *args],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )


def test_help_and_invalid_config_have_documented_exit_codes() -> None:
    help_result = _cli("--help")
    assert help_result.returncode == 0
    assert all(command in help_result.stdout for command in ("validate", "inspect", "run", "compare", "summarize"))
    invalid = _cli("validate", "missing-experiment.yaml")
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert "invalid input" in invalid.stderr


def test_validate_and_inspect_are_offline_and_reveal_no_fixture_credentials() -> None:
    options = (str(CONFIG), "--fixture", str(FIXTURE))
    validated = _cli("validate", *options)
    inspected = _cli("inspect", *options)
    assert validated.returncode == inspected.returncode == 0
    assert json.loads(validated.stdout)["cases"] == 3
    assert json.loads(inspected.stdout)["sas"]["model"]["model_id"] == "offline-baseline"
    assert "fixture-only" not in validated.stdout + inspected.stdout


def test_cli_loads_ancestor_dotenv_and_exported_values_win(tmp_path: Path) -> None:
    config = _dotenv_experiment(tmp_path)
    environment = {key: value for key, value in os.environ.items() if key not in {
        "BASELINE_MODEL_ID", "OPENAI_API_KEY", "DR7_API_KEY", "DR7_BASE_URL",
    }}
    environment["PYTHONPATH"] = str(ROOT / "src")

    def validate() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "mas_slm_research.cli", "validate", str(config)],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
        )

    loaded = validate()
    assert loaded.returncode == 0, loaded.stderr
    assert json.loads(loaded.stdout)["configuration"]["resolved_models"]["baseline"]["model_id"] == "dotenv-baseline"
    assert "dotenv-openai-key" not in loaded.stdout + loaded.stderr
    assert "dotenv-dr7-key" not in loaded.stdout + loaded.stderr

    environment["BASELINE_MODEL_ID"] = "exported-baseline"
    overridden = validate()
    assert overridden.returncode == 0, overridden.stderr
    assert json.loads(overridden.stdout)["configuration"]["resolved_models"]["baseline"]["model_id"] == "exported-baseline"

    fixture = subprocess.run(
        [sys.executable, "-m", "mas_slm_research.cli", "validate", str(config),
         "--fixture", str(DR7_FIXTURE)],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    assert fixture.returncode == 0, fixture.stderr
    assert json.loads(fixture.stdout)["configuration"]["resolved_models"]["baseline"]["model_id"] == "offline-baseline"


def test_cli_passes_dotenv_values_to_live_provider_factory_without_network(
    tmp_path: Path, monkeypatch,
) -> None:
    config = _dotenv_experiment(tmp_path)
    for name in ("BASELINE_MODEL_ID", "OPENAI_API_KEY", "DR7_API_KEY", "DR7_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    constructed: list[dict] = []

    def fake_chat_openai(**options):
        constructed.append(options)
        from mas_slm_research.cli import _load_fixture
        _, factory = _load_fixture(DR7_FIXTURE)
        return factory(None, "baseline")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_chat_openai)
    result = asyncio.run(_execute(_parser().parse_args([
        "run", str(config), "--system", "single", "--case", "synthetic-esi1-001",
        "--console", "none",
    ])))
    assert result["result"]["status"] == "completed"
    assert len(constructed) == 1
    assert constructed[0]["model"] == "dotenv-baseline"
    assert str(constructed[0]["api_key"]) == "dotenv-openai-key"


def test_run_compare_and_summarize_scripted_clone_fixture(tmp_path: Path) -> None:
    options = (str(CONFIG), "--fixture", str(FIXTURE))
    single = _cli("run", *options, "--system", "single", "--case", "synthetic-esi1-001")
    assert single.returncode == 0
    single_data = json.loads(single.stdout)
    assert single_data["result"]["status"] == "completed"
    assert single_data["grade"]["passed"] is False

    multi = _cli("run", *options, "--system", "multi", "--case", "synthetic-esi1-001")
    assert multi.returncode == 0
    assert json.loads(multi.stdout)["grade"]["passed"] is True

    compared = _cli("compare", *options, "--no-artifacts")
    assert compared.returncode == 0
    report = json.loads(compared.stdout)
    assert report["systems"]["single"]["attempted"] == 3
    assert report["systems"]["multi"]["attempted"] == 3
    assert len(report["pairs"]) == 3
    path = tmp_path / "report.json"
    path.write_text(compared.stdout, encoding="utf-8")
    summary = _cli("summarize", str(path))
    assert summary.returncode == 0
    assert json.loads(summary.stdout)["systems"] == report["systems"]
    assert "fixture-only" not in compared.stdout + summary.stdout


def test_unknown_case_and_bad_summary_fail_before_inference(tmp_path: Path) -> None:
    result = _cli("run", str(CONFIG), "--fixture", str(FIXTURE),
                  "--system", "single", "--case", "missing")
    assert result.returncode == 2
    assert result.stdout == ""
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert _cli("summarize", str(path)).returncode == 2


def test_console_trace_is_ordered_on_stderr_and_can_be_disabled() -> None:
    options = (str(CONFIG), "--fixture", str(FIXTURE),
               "--system", "multi", "--case", "synthetic-esi1-001")
    traced = _cli("run", *options, "--console", "events", "--color", "never")
    quiet = _cli("run", *options, "--console", "none")
    assert traced.returncode == quiet.returncode == 0
    assert json.loads(traced.stdout)["result"]["status"] == json.loads(quiet.stdout)["result"]["status"]
    assert quiet.stderr == ""
    assert "[esi1_agent] tool_call" in traced.stderr
    assert "[vitals_agent] tool_call" in traced.stderr
    assert "handoff_created -> doctor_agent" in traced.stderr
    assert "gate_evaluated doctor_gate ready=True" in traced.stderr
    assert traced.stderr.index("[esi1_agent] agent_started") < traced.stderr.index("[doctor_agent] agent_started")
    assert "\x1b[" not in traced.stderr
    assert "fixture-only" not in traced.stderr

    full = _cli("run", *options, "--console", "full", "--color", "never")
    assert full.returncode == 0
    assert '"is_esi1": true' in full.stderr
    assert '"result"' in full.stderr


def test_cli_trace_on_off_preserves_case_result() -> None:
    options = (str(CONFIG), "--fixture", str(FIXTURE), "--system", "multi",
               "--case", "synthetic-esi1-001", "--console", "none")
    off = _cli("run", *options, "--no-trace")
    on = _cli("run", *options, "--trace")
    assert off.returncode == on.returncode == 0
    assert json.loads(off.stdout)["result"]["output"] == json.loads(on.stdout)["result"]["output"]
    assert json.loads(off.stdout)["grade"]["score"] == json.loads(on.stdout)["grade"]["score"]
    assert off.stderr == ""
    assert "spans captured" in on.stderr


def test_artifact_directory_recomputes_summary_and_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "research-run"
    options = (str(CONFIG), "--fixture", str(FIXTURE), "--console", "none",
               "--output-dir", str(output_dir), "--events-file")
    first = _cli("compare", *options)
    assert first.returncode == 0, first.stderr
    terminal = json.loads(first.stdout)
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "results.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "events.jsonl").is_file()
    offline = _cli("summarize", str(output_dir))
    assert offline.returncode == 0, offline.stderr
    assert json.loads(offline.stdout)["systems"] == terminal["systems"]
    assert "fixture-only" not in (output_dir / "manifest.json").read_text(encoding="utf-8")
    again = _cli("compare", *options)
    assert again.returncode == 2
    assert "already exists" in again.stderr


def test_interrupted_comparison_keeps_readable_partial_artifacts(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for rows in fixture["roles"].values():
        for row in rows:
            row["delay_ms"] = 600
    delayed = tmp_path / "delayed.json"
    delayed.write_text(json.dumps(fixture), encoding="utf-8")
    output_dir = tmp_path / "interrupted-run"
    command = [sys.executable, "-m", "mas_slm_research.cli", "compare", str(CONFIG),
               "--fixture", str(delayed), "--output-dir", str(output_dir), "--console", "none"]
    process = subprocess.Popen(command, cwd=ROOT,
                               env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 12
        manifest_path = output_dir / "manifest.json"
        while time.monotonic() < deadline:
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["attempts_written"] >= 1:
                    break
            time.sleep(0.05)
        else:
            raise AssertionError("first attempt was not saved before timeout")
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stderr
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] in {"cancelled", "interrupted"}
        offline = _cli("summarize", str(output_dir))
        assert offline.returncode == 0, offline.stderr
        assert json.loads(offline.stdout)["systems"]["single"]["attempted"] >= 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
