"""The published ESI fixture is the same benchmark as the clone example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from mas_slm_research.benchmarks.esi import ASSET_NAMES, copy_esi_benchmark


ROOT = Path(__file__).resolve().parents[3]


def test_packaged_assets_match_the_preserved_clone_example(tmp_path: Path) -> None:
    copied = copy_esi_benchmark(tmp_path / "esi")
    assert tuple(sorted(path.name for path in copied.iterdir())) == tuple(sorted(ASSET_NAMES))
    resources = files("mas_slm_research.benchmarks.esi")
    for name in ASSET_NAMES:
        original = (ROOT / "examples" / "esi" / name).read_bytes()
        assert resources.joinpath(name).read_bytes() == original
        assert (copied / name).read_bytes() == original

    with pytest.raises(ValueError, match="already exists"):
        copy_esi_benchmark(copied)


def test_copied_esi_benchmark_runs_both_systems_offline(tmp_path: Path) -> None:
    copied = copy_esi_benchmark(tmp_path / "esi")
    result = subprocess.run(
        [sys.executable, "-m", "mas_slm_research.cli", "compare", str(copied / "experiment.yaml"),
         "--fixture", str(copied / "offline_fixture.json"), "--no-artifacts", "--console", "none"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["systems"]["single"]["attempted"] == 3
    assert report["systems"]["multi"]["attempted"] == 3
    assert len(report["pairs"]) == 3
