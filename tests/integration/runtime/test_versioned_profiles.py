"""Identical scripted responses reveal named runtime intervention differences."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from mas_slm_research.comparison import _measure
from mas_slm_research.configuration import load_configuration
from mas_slm_research.contracts import RunStatus, RunTiming
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.preview import inspect_configuration
from mas_slm_research.registry import ComponentRegistry, register_builtin_components
from mas_slm_research.runtime.profiles import mas_budget_for_profile, runtime_config_for_profile
from mas_slm_research.single_agent import SingleAgentRunner
from tests.doubles.fake_provider import FakeChatModel
from tests.integration.runtime.test_single_agent_runner import Answer, final_answer, identity
from tests.integration.runtime.test_configured_esi_systems import ENV, EXAMPLE


def _attempt(profile, *, overrides=None):
    responses = [
        AIMessage(content='{"tool_calls":[{"name":"final_answer","arguments":'),
        AIMessage(content="", tool_calls=[
            {"id": "final", "name": "final_answer", "args": {"recommendation": {"value": "ok"}}},
        ]),
    ]
    config = runtime_config_for_profile(profile, multi_agent=False, overrides=overrides)
    run = asyncio.run(SingleAgentRunner(
        kernel_factory=lambda: AgentKernel(
            model=FakeChatModel(list(responses)), tools=[final_answer],
            response_format=Answer, runtime_config=config,
        ), agent_name="baseline",
    ).run_case(identity=identity(profile + str(overrides)), payload="case"))
    measure = _measure(SimpleNamespace(
        execution=SimpleNamespace(llm_calls=run.llm_calls, tool_calls=run.tool_calls, events=run.events),
        result=SimpleNamespace(identity=run.result.identity, timing=RunTiming(wall_seconds=0.1), status=run.result.status),
    ), {})
    return run, measure


@pytest.mark.integration
def test_strict_assisted_and_disabled_repair_have_distinct_results():
    strict, strict_metrics = _attempt("strict_v1")
    assisted, assisted_metrics = _attempt("slm_assisted_v1")
    disabled, disabled_metrics = _attempt("slm_assisted_v1", overrides={"malformed_tool_retry_enabled": False})

    assert strict.result.status == RunStatus.FAILED
    assert disabled.result.status == RunStatus.FAILED
    assert assisted.result.status == RunStatus.COMPLETED
    assert assisted_metrics.first_pass_valid is False
    assert assisted_metrics.repaired_valid is True
    assert assisted_metrics.repair_extra_calls == 1
    assert assisted_metrics.repair_extra_tokens is not None
    assert len(assisted.llm_calls) == 2
    assert assisted.llm_calls[1]["call_kind"] == "malformed_repair"
    assert strict_metrics.repaired_valid is False
    assert disabled_metrics.repair_extra_calls == 0


@pytest.mark.integration
def test_legacy_policy_is_labeled_and_new_profiles_have_finite_limits():
    legacy = runtime_config_for_profile("legacy_v1", multi_agent=False)
    strict = runtime_config_for_profile("strict_v1", multi_agent=True)
    assisted = runtime_config_for_profile("slm_assisted_v1", multi_agent=True)
    assert legacy.policy_id == "legacy_v1" and legacy.max_model_calls is None
    assert strict.policy_id == "strict_v1" and strict.max_model_calls == 8
    assert strict.allow_text_tool_recovery is False
    assert strict.malformed_tool_retry_enabled is False
    assert assisted.policy_id == "slm_assisted_v1"
    assert assisted.malformed_tool_retry_enabled is True
    assert mas_budget_for_profile("strict_v1") == (8, 240.0)
    assert mas_budget_for_profile("legacy_v1") == (None, None)

    registry = ComponentRegistry()
    register_builtin_components(registry)
    loaded = load_configuration(EXAMPLE, registry=registry, environment=ENV)
    configured = replace(loaded, experiment=loaded.experiment.model_copy(update={"runtime_profile": "strict_v1"}))
    view = inspect_configuration(configured).concise
    assert view["runtime_profile"] == "strict_v1"
    assert view["sas"]["runtime_policy"]["max_model_calls"] == 8
    assert view["mas"]["max_handoffs"] == 8
    assert view["mas"]["roles"]["esi1_agent"]["runtime_policy"]["policy_id"] == "strict_v1"
