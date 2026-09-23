"""A non-ESI extension exercises the public SAS/MAS comparison path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "arithmetic"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mas_slm_research.cli", *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(EXAMPLE)))},
        capture_output=True, text=True, check=False,
    )


def test_core_import_does_not_eagerly_import_esi_benchmark() -> None:
    process = subprocess.run(
        [sys.executable, "-c", "import sys; import mas_slm_research.configuration; "
         "assert not any('.esi' in name for name in sys.modules if name.startswith('mas_slm_research.'))"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr


def test_arithmetic_cli_routes_tools_handoffs_gate_and_artifacts(tmp_path: Path) -> None:
    config = str(EXAMPLE / "experiment.yaml")
    fixture = str(EXAMPLE / "offline_fixture.json")
    output = tmp_path / "arithmetic-results"
    validated = _cli("validate", config, "--fixture", fixture)
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["case_ids"] == ["arithmetic-001"]

    compared = _cli("compare", config, "--fixture", fixture, "--output-dir", str(output),
                    "--console", "events", "--color", "never", "--events-file")
    assert compared.returncode == 0, compared.stderr
    report = json.loads(compared.stdout)
    assert report["systems"]["single"]["passed"] == 1
    assert report["systems"]["multi"]["passed"] == 1
    assert report["ties"] == 1
    assert report["uncomparable_pairs"] == 0
    for event in ("tool_call sum_numbers", "handoff_created -> final_agent",
                  "gate_evaluated final_gate ready=True", "[final_agent] agent_completed"):
        assert event in compared.stderr
    assert "[adder_agent]" in compared.stderr and "[checker_agent]" in compared.stderr
    assert (output / "manifest.json").is_file()
    assert (output / "results.jsonl").is_file()
    assert (output / "events.jsonl").is_file()
    summary = _cli("summarize", str(output))
    assert summary.returncode == 0, summary.stderr
    assert json.loads(summary.stdout)["systems"] == report["systems"]
