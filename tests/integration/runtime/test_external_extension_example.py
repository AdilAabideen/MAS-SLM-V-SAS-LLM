"""Copying the extension out of the checkout requires no toolkit source edits."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "examples" / "external_extension"


def _cli(extension: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mas_slm_research.cli", *args],
        cwd=extension.parent,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(extension)))},
        capture_output=True, text=True, check=False,
    )


def test_external_extension_copies_inspects_and_runs(tmp_path: Path) -> None:
    extension = tmp_path / "my-study"
    shutil.copytree(SOURCE, extension)
    config = str(extension / "experiment.yaml")
    fixture = str(extension / "offline_fixture.json")

    inspected = _cli(extension, "inspect", config, "--fixture", fixture)
    assert inspected.returncode == 0, inspected.stderr
    preview = json.loads(inspected.stdout)
    assert preview["mas"]["selected_definition"] == "research.word_count_v1"
    assert preview["mas"]["allowed_handoffs"] == {"counter_agent": ["reviewer_agent"], "reviewer_agent": []}
    reviewer = preview["mas"]["roles"]["reviewer_agent"]
    assert reviewer["model_source"] == "mas.model_overrides"
    assert reviewer["model"]["model_id"] == "offline-word-count-reviewer"
    assert reviewer["payload_builder"] == "research.reviewer_payload_v1"
    assert preview["mas"]["roles"]["counter_agent"]["outgoing_handoff_schemas"] == {
        "reviewer_agent": "research.count_handoff_v1"
    }

    compared = _cli(extension, "compare", config, "--fixture", fixture,
                    "--no-artifacts", "--console", "events", "--color", "never")
    assert compared.returncode == 0, compared.stderr
    report = json.loads(compared.stdout)
    assert report["systems"]["single"]["passed"] == 1
    assert report["systems"]["multi"]["passed"] == 1
    assert report["ties"] == 1
    assert "handoff_created -> reviewer_agent" in compared.stderr
    assert "gate_evaluated reviewer_gate ready=True" in compared.stderr


def test_invalid_extension_registration_fails_before_inference(tmp_path: Path) -> None:
    extension = tmp_path / "bad-study"
    shutil.copytree(SOURCE, extension)
    (extension / "bad_extension.py").write_text(
        "def register_components(registry):\n"
        "    registry.register('schemas', 'bad.schema_v1', object())\n",
        encoding="utf-8",
    )
    config = extension / "experiment.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "extensions: [research_extension]", "extensions: [bad_extension]"
    ), encoding="utf-8")
    result = _cli(extension, "validate", str(config), "--fixture", str(extension / "offline_fixture.json"))
    assert result.returncode == 2
    assert result.stdout == ""
    assert "bad.schema_v1" in result.stderr
    assert "invalid implementation" in result.stderr
