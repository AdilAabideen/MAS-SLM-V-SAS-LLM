"""Test Runtime Config And Helpers test coverage."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from mas_slm_research.kernel import AgentKernel
from mas_slm_research.contracts import FailureKind
from mas_slm_research.runtime.runtime_config import RuntimeConfig
from mas_slm_research.workflows.definition import WorkflowDefinition, WorkflowMetadata


def _noop_tool() -> dict:
    """Return an empty payload."""
    # Keep the main step clear.
    return {}


class _PermissiveGenericModel:
    def __init__(self) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self.bound_kwargs = None

    def bind_tools(self, tools, tool_choice="any", **kwargs):
        """Handle tools."""
        # Keep the main step clear.
        self.bound_kwargs = dict(kwargs)
        return self

    async def ainvoke(self, _messages):
        """Handle the value."""
        # Keep the main step clear.
        return AIMessage(content='{"value":"ok"}')


class _RuntimeHintAwareModel(_PermissiveGenericModel):
    def _llm_type(self) -> str:
        """Handle type."""
        # Keep the main step clear.
        return "vllm-chat"


@pytest.mark.unit
def test_ut_run_028_runtime_config_rejects_max_tool_calls_per_turn_below_one():
    """Handle ut run 028 runtime config rejects max tool calls per turn below one."""
    # Keep the main step clear.
    with pytest.raises(ValueError):
        RuntimeConfig(max_tool_calls_per_turn=0)


@pytest.mark.unit
def test_ut_run_029_runtime_config_rejects_negative_malformed_retry_count():
    """Handle ut run 029 runtime config rejects negative malformed retry count."""
    # Keep the main step clear.
    with pytest.raises(ValueError):
        RuntimeConfig(max_malformed_tool_retries_per_tool=-1)


@pytest.mark.unit
def test_ut_run_030_common_failure_kinds_remain_stable():
    """Handle ut run 030 failure taxonomy values remain stable."""
    # Keep the main step clear.
    assert FailureKind.TOOL.value == "tool"
    assert FailureKind.VALIDATION.value == "validation"


@pytest.mark.unit
def test_ut_run_031_payload_to_human_content_extracts_plain_string_input():
    """Handle ut run 031 payload to human content extracts plain string input."""
    # Keep the main step clear.
    assert AgentKernel._payload_to_human_content("hello") == "hello"


@pytest.mark.unit
def test_ut_run_033_payload_to_human_content_extracts_latest_user_message_from_dict_messages():
    """Handle ut run 033 payload to human content extracts latest user message from dict messages."""
    # Keep the main step clear.
    payload = {"messages": [{"role": "user", "content": "first"}, {"role": "user", "content": "latest"}]}
    assert AgentKernel._payload_to_human_content(payload) == "latest"


@pytest.mark.unit
def test_ut_run_034_limit_tool_calls_splits_kept_and_dropped_calls_correctly():
    """Handle ut run 034 limit tool calls splits kept and dropped calls correctly."""
    # Keep the main step clear.
    agent = object.__new__(AgentKernel)
    agent.max_tool_calls = 2
    kept, dropped = AgentKernel._limit_tool_calls(
        agent,
        [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"},
        ],
    )
    assert [item["id"] for item in kept] == ["1", "2"]
    assert [item["id"] for item in dropped] == ["3"]


@pytest.mark.unit
def test_ut_run_035_parsed_to_payload_json_wraps_scalar_values_under_value():
    """Handle ut run 035 parsed to payload json wraps scalar values under value."""
    # Keep the main step clear.
    assert AgentKernel._parsed_to_payload_json("x") == {"value": "x"}


@pytest.mark.unit
def test_ut_run_036_bind_model_tools_omits_runtime_hints_for_generic_models():
    """Handle ut run 036 bind model tools omits runtime hints for generic models."""
    # Keep the main step clear.
    model = _PermissiveGenericModel()
    AgentKernel(model=model, tools=[_noop_tool])
    assert model.bound_kwargs == {}


@pytest.mark.unit
def test_ut_run_037_bind_model_tools_includes_runtime_hints_for_supported_wrappers():
    """Handle ut run 037 bind model tools includes runtime hints for supported wrappers."""
    # Keep the main step clear.
    model = _RuntimeHintAwareModel()
    workflow = WorkflowDefinition(
        metadata=WorkflowMetadata(workflow_id="demo", name="Demo", version="1"),
        participating_agents=("demo_agent", "other"),
        start_agents=("demo_agent",), finalizing_agents=("other",),
        allowed_handoffs={"demo_agent": ("other",), "other": ()},
    )
    agent = AgentKernel(
        model=model,
        tools=[_noop_tool],
        agent_node_name="demo_agent",
        handoff_tool_names=["handoff_to_other"],
        handoff_workflow=workflow,
        runtime_config=RuntimeConfig(multi_agent=True),
    )
    assert model.bound_kwargs == {
        "agent_name": agent.agent_node_name,
        "multi_agent": True,
        "handoff_names": ["handoff_to_other"],
    }
