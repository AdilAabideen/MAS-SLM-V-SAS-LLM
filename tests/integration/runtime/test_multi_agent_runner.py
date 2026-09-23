"""A fake-provider ESI MAS case returns correlated records and a common result."""

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
from mas_slm_research.mas_contract import HandoffResult
from mas_slm_research.multi_agent import MultiAgentRunner
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.telemetry import token_estimator
from mas_slm_research.workflows.esi.definition import ESI_MAS
from mas_slm_research.workflows.esi.payload_builder import build_pending_agent_payload
from tests.doubles.fake_provider import FakeChatModel


class DoctorAnswer(BaseModel):
    recommendation: dict


def final_answer(recommendation: dict) -> dict:
    """Return the final ESI recommendation."""
    return {"recommendation": recommendation}


def make_handoff_tool(source: str, target: str):
    name = f"handoff_{source}_to_{target}"

    def transfer(payload: dict) -> dict:
        """Transfer a structured payload to the next role."""
        return HandoffResult(
            handoff_name=name, from_agent=source, target_agent=target,
            payload_schema="TestPayload", payload=payload,
        ).model_dump()

    transfer.__name__ = name
    return transfer


def role_factories(acuity_path, *, failing_role=None, constructed=None):
    constructed = constructed if constructed is not None else []
    factories = {}
    for role in ESI_MAS.participating_agents:
        if role == "doctor_agent":
            def build_doctor(role=role):
                constructed.append(role)
                model = FakeChatModel([AIMessage(content="", tool_calls=[
                    {"id": "final", "name": "final_answer", "args": {"recommendation": {"esi": 2}}},
                ])])
                return AgentKernel(model=model, tools=[final_answer], response_format=DoctorAnswer)

            factories[role] = build_doctor
            continue

        target = "doctor_agent" if role == "vitals_agent" or role == acuity_path[-1] else (
            acuity_path[acuity_path.index(role) + 1] if role in acuity_path else "doctor_agent"
        )

        def build_handoff(role=role, target=target):
            constructed.append(role)
            tool = make_handoff_tool(role, target)
            script = [RuntimeError("provider unavailable")] if role == failing_role else [
                AIMessage(content="", tool_calls=[{
                    "id": f"call_{role}", "name": tool.__name__, "args": {"payload": {"source": role}},
                }]),
            ]
            return AgentKernel(
                model=FakeChatModel(script), tools=[tool],
                handoff_tool_names=[tool.__name__], handoff_workflow=ESI_MAS,
                runtime_config=RuntimeConfig(multi_agent=True),
            )

        factories[role] = build_handoff
    return factories


def output_validator(value):
    if value.get("ok") is not True or "recommendation" not in value:
        raise ValueError("doctor output is invalid")
    return value


def identity(run_id):
    return RunIdentity(experiment_id="comparison", system_id="mas", case_id="case-1", repetition=1, run_id=run_id)


@pytest.fixture(autouse=True)
def offline_token_estimation(monkeypatch):
    monkeypatch.setattr(token_estimator, "tiktoken", None)


@pytest.mark.integration
@pytest.mark.parametrize("acuity_path", [
    ("esi1_agent",),
    ("esi1_agent", "esi2_agent", "esi345_agent"),
])
def test_fake_provider_esi_mas_returns_common_result_and_correlated_children(acuity_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    constructed = []
    runner = MultiAgentRunner(
        workflow=ESI_MAS, role_factories=role_factories(acuity_path, constructed=constructed),
        payload_builder=build_pending_agent_payload, output_validator=output_validator,
    )
    run = asyncio.run(runner.run_case(
        identity=identity("run-1"),
        case_info={"chiefcomplaint": "chest pain", "heartrate": 110},
    ))
    assert run.result.status == RunStatus.COMPLETED
    assert run.result.output.to_dict()["recommendation"] == {"esi": 2}
    assert run.result.failure is None
    assert run.result.usage.source == UsageSource.ESTIMATED
    assert run.result.usage.total_tokens > 0
    assert run.result.timing.wall_seconds >= 0
    assert run.result.timing.child_seconds_sum is not None
    assert len(constructed) == len(ESI_MAS.participating_agents)
    assert {record.agent_name for record in run.agent_records} == set(acuity_path) | {"vitals_agent", "doctor_agent"}
    doctor = next(record for record in run.agent_records if record.agent_name == "doctor_agent")
    assert len(doctor.incoming_handoff_ids) == 2
    assert all(handoff.to_agent_run_id == doctor.agent_run_id for handoff in run.handoff_records if handoff.to_agent_name == "doctor_agent")
    assert run.gate_records[-1].ready is True
    assert [gate.ready for gate in run.gate_records] == ([True] if len(acuity_path) == 1 else [False, True])
    assert all(record.measurements is not None for record in run.agent_records)
    assert run.llm_calls and run.tool_calls and run.events
    assert run.sink_errors == ()
    assert not list(tmp_path.rglob("*.db"))

    second = asyncio.run(runner.run_case(
        identity=identity("run-2"), case_info={"chiefcomplaint": "headache", "heartrate": 90},
    ))
    assert second.result.status == RunStatus.COMPLETED
    assert len(constructed) == 2 * len(ESI_MAS.participating_agents)
    assert {record.mas_run_id for record in second.agent_records} == {"run-2"}
    assert not {record.agent_run_id for record in run.agent_records} & {record.agent_run_id for record in second.agent_records}


@pytest.mark.integration
def test_fake_provider_error_stays_failed_with_child_metric_and_no_prediction():
    runner = MultiAgentRunner(
        workflow=ESI_MAS,
        role_factories=role_factories(("esi1_agent", "esi2_agent", "esi345_agent"), failing_role="esi2_agent"),
        payload_builder=build_pending_agent_payload, output_validator=output_validator,
    )
    run = asyncio.run(runner.run_case(identity=identity("run-error"), case_info={"chiefcomplaint": "pain"}))
    assert run.result.status == RunStatus.FAILED
    assert run.result.output is None
    assert run.result.failure.kind == FailureKind.PROVIDER
    assert all(record.finished_at is not None for record in run.agent_records)
    assert any(record.agent_name == "esi2_agent" and record.status == "failed" for record in run.agent_records)
    assert all(record.agent_name != "doctor_agent" for record in run.agent_records)
    assert any(call["error_text"] == "provider unavailable" for call in run.llm_calls)


@pytest.mark.integration
def test_fresh_process_esi_mas_case_needs_no_sql_or_legacy_app(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = """
import asyncio
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from mas_slm_research.contracts import RunStatus
from mas_slm_research.multi_agent import MultiAgentRunner
from mas_slm_research.telemetry import token_estimator
from mas_slm_research.workflows.esi.definition import ESI_MAS
from mas_slm_research.workflows.esi.payload_builder import build_pending_agent_payload
from tests.integration.runtime.test_multi_agent_runner import identity, output_validator, role_factories
token_estimator.tiktoken = None
runner = MultiAgentRunner(
    workflow=ESI_MAS, role_factories=role_factories(('esi1_agent',)),
    payload_builder=build_pending_agent_payload, output_validator=output_validator,
)
result = asyncio.run(runner.run_case(identity=identity('isolated-run'), case_info={'chiefcomplaint': 'pain'}))
assert result.result.status == RunStatus.COMPLETED, result.result.to_dict()
assert len(result.gate_records) == 1
assert not any(name == 'sqlalchemy' or name.startswith('sqlalchemy.') for name in sys.modules)
assert not any(name == 'app' or name.startswith('app.') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(root), str(root / "src")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert not list(tmp_path.iterdir())
