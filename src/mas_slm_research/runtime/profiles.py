"""Named, versioned runtime interventions for controlled comparisons."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, Mapping

from .runtime_config import RuntimeConfig


RuntimeProfile = Literal["legacy_v1", "strict_v1", "slm_assisted_v1"]

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "legacy_v1": {},
    "strict_v1": {
        "allow_text_tool_recovery": False,
        "malformed_tool_retry_enabled": False,
        "allow_plain_json_final_output": False,
        "max_model_calls": 8,
        "max_tool_calls_total": 16,
        "max_elapsed_seconds": 120.0,
    },
    "slm_assisted_v1": {
        "allow_text_tool_recovery": True,
        "malformed_tool_retry_enabled": True,
        "max_malformed_tool_retries_per_tool": 1,
        "allow_plain_json_final_output": True,
        "max_model_calls": 8,
        "max_tool_calls_total": 16,
        "max_elapsed_seconds": 120.0,
    },
}


def runtime_config_for_profile(
    profile: RuntimeProfile, *, multi_agent: bool,
    overrides: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Apply explicit agent overrides after the named policy defaults."""
    if profile not in _PROFILE_DEFAULTS:
        raise ValueError(f"unknown runtime profile {profile!r}")
    values = asdict(RuntimeConfig(multi_agent=multi_agent))
    values.update(_PROFILE_DEFAULTS[profile])
    values.update(dict(overrides or {}))
    values.update(policy_id=profile, multi_agent=multi_agent, persist_events=True, print_events=False)
    return RuntimeConfig(**values)


def mas_budget_for_profile(profile: RuntimeProfile) -> tuple[int | None, float | None]:
    if profile == "legacy_v1":
        return None, None
    if profile in {"strict_v1", "slm_assisted_v1"}:
        return 8, 240.0
    raise ValueError(f"unknown runtime profile {profile!r}")
