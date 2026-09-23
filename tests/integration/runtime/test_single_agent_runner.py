"""A database-free single-agent case returns results and measurements in memory."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from mas_slm_research.contracts import FailureKind, RunIdentity, RunStatus, UsageSource
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.single_agent import SingleAgentRunner
from mas_slm_research.telemetry import token_estimator
from tests.doubles.fake_provider import FakeChatModel


class Answer(BaseModel):
    recommendation: dict


def lookup_value(value: str) -> dict:
    """Return a deterministic tool payload."""
    return {"value": value}


def final_answer(recommendation: dict) -> dict:
    """Return the final payload."""
    return {"recommendation": recommendation}


def identity(run_id: str = "attempt-1") -> RunIdentity:
    return RunIdentity(experiment_id="demo", system_id="sas", case_id="case-1", repetition=1, run_id=run_id)


@pytest.fixture(autouse=True)
def offline_token_estimation(monkeypatch):
    monkeypatch.setattr(token_estimator, "tiktoken", None)


@pytest.mark.integration
def test_single_case_returns_validated_output_and_in_memory_trace_without_database(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def build_kernel():
        return AgentKernel(
            model=FakeChatModel([
                AIMessage(content="", tool_calls=[{"id": "lookup", "name": "lookup_value", "args": {"value": "abc"}}]),
                AIMessage(content="", tool_calls=[{"id": "final", "name": "final_answer", "args": {"recommendation": {"value": "abc"}}}]),
            ]),
            tools=[lookup_value, final_answer],
            response_format=Answer,
        )

    run = asyncio.run(SingleAgentRunner(kernel_factory=build_kernel, agent_name="baseline").run_case(
        identity=identity(), payload={"case": "example"},
    ))
    assert run.result.status == RunStatus.COMPLETED
    assert run.result.output.to_dict() == {"recommendation": {"value": "abc"}, "ok": True}
    assert run.result.failure is None
    assert run.result.usage.source == UsageSource.ESTIMATED
    assert run.result.usage.total_tokens > 0
    assert run.result.timing.wall_seconds >= 0
    assert run.result.timing.child_seconds_sum is not None
    assert len(run.llm_calls) == 2
    assert len(run.tool_calls) == 2
    assert run.events and all(event["run_id"] == "attempt-1" for event in run.events)
    assert capsys.readouterr().out == ""
    assert not list(tmp_path.iterdir())


@pytest.mark.integration
@pytest.mark.parametrize("failure_mode, expected_kind", [
    ("missing_final", FailureKind.VALIDATION),
    ("provider_error", FailureKind.PROVIDER),
])
def test_single_case_retains_classified_failures(failure_mode, expected_kind):
    def build_kernel():
        if failure_mode == "provider_error":
            model = FakeChatModel([RuntimeError("provider unavailable")])
        else:
            model = FakeChatModel([AIMessage(content='{"recommendation":{"value":"plain"}}')])
        return AgentKernel(
            model=model, tools=[], response_format=Answer,
            runtime_config=RuntimeConfig(require_final_answer_tool=True, allow_plain_json_final_output=False),
        )

    run = asyncio.run(SingleAgentRunner(kernel_factory=build_kernel, agent_name="baseline").run_case(
        identity=identity(failure_mode), payload="case",
    ))
    assert run.result.status == RunStatus.FAILED
    assert run.result.output is None
    assert run.result.failure.kind == expected_kind
    assert run.result.to_dict()["failure"]["message"]
    assert len(run.llm_calls) == 1


@pytest.mark.integration
def test_fresh_process_single_agent_run_uses_no_backend_modules_or_database(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = """
import asyncio
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from mas_slm_research.contracts import RunIdentity, RunStatus
from mas_slm_research.kernel import AgentKernel
from mas_slm_research.single_agent import SingleAgentRunner
from mas_slm_research.telemetry import token_estimator
from tests.doubles.fake_provider import FakeChatModel
token_estimator.tiktoken = None
class Answer(BaseModel):
    recommendation: dict
def final_answer(recommendation: dict) -> dict:
    '''Return a final answer.'''
    return {'recommendation': recommendation}
runner = SingleAgentRunner(
    kernel_factory=lambda: AgentKernel(
        model=FakeChatModel([AIMessage(content='', tool_calls=[
            {'id': 'final', 'name': 'final_answer', 'args': {'recommendation': {'value': 'ok'}}}
        ])]), tools=[final_answer], response_format=Answer,
    ), agent_name='baseline',
)
run = asyncio.run(runner.run_case(
    identity=RunIdentity(experiment_id='demo', system_id='sas', case_id='case', repetition=1, run_id='run'),
    payload='case',
))
assert run.result.status == RunStatus.COMPLETED, run.result.to_dict()
assert len(run.llm_calls) == 1
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
assert not any(name == 'fastapi' or name.startswith('fastapi.') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(root), str(root / "src")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert not list(tmp_path.rglob("*.db"))
