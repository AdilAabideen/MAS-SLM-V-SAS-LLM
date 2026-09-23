"""Preserved ESI benchmark assets plus explicitly fabricated case fixtures."""

from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path


ASSET_NAMES = ("experiment.yaml", "workflow.yaml", "cases.jsonl", "offline_fixture.json")


def copy_esi_benchmark(destination: Path) -> Path:
    """Create a new editable benchmark directory from the packaged assets."""
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"benchmark destination already exists: {destination}") from exc
    resources = files(__package__)
    for name in ASSET_NAMES:
        with resources.joinpath(name).open("rb") as source, (destination / name).open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination


__all__ = ["ASSET_NAMES", "copy_esi_benchmark"]
