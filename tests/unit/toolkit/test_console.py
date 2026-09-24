"""Terminal rendering is optional, redacted, and width-aware."""

from __future__ import annotations

import io

from mas_slm_research.console import ConsoleRenderer


def test_tool_call_json_is_multiline_and_secret_fields_are_redacted(monkeypatch) -> None:
    stream = io.StringIO()
    renderer = ConsoleRenderer(mode="full", color="never", stream=stream, width=48)
    renderer._event("agent_event", {
        "agent_name": "specialist", "event_type": "tool_call", "tool_name": "example",
        "payload_json": {"args": {"api_key": "do-not-print", "nested": {"password": "hidden"},
                                  "details": "many words in a long synthetic argument"}},
    })
    value = stream.getvalue()
    assert "[Tool call] example" in value
    assert "\n  {\n" in value
    assert value.count("\n") >= 3
    assert "do-not-print" not in value and "hidden" not in value
    assert "[redacted]" in value
    assert "tool_result" not in value


def test_mas_tool_calls_name_the_agent_without_startup_agent_lines() -> None:
    stream = io.StringIO()
    renderer = ConsoleRenderer(mode="events", color="never", stream=stream)
    renderer.start_case(system_id="multi", case_id="case-1", repetition=1)
    renderer.render_event("graph_event", {"event_type": "agent_started", "agent_name": "vitals_agent"})
    renderer.render_event("agent_event", {
        "event_type": "tool_call", "agent_name": "vitals_agent", "tool_name": "create_plan",
    })
    lines = stream.getvalue().splitlines()
    assert "Agent: vitals_agent | [Tool call] create_plan" in lines
    assert "Agent: vitals_agent" not in lines


def test_sas_tool_call_keeps_existing_label() -> None:
    stream = io.StringIO()
    renderer = ConsoleRenderer(mode="events", color="never", stream=stream)
    renderer.start_case(system_id="single", case_id="case-1", repetition=1, agent_name="baseline")
    renderer.render_event("agent_event", {
        "event_type": "tool_call", "agent_name": "baseline", "tool_name": "create_plan",
    })
    lines = stream.getvalue().splitlines()
    assert "Agent: baseline" in lines
    assert "[Tool call] create_plan" in lines
    assert "Agent: baseline | [Tool call] create_plan" not in lines


def test_no_color_environment_wins_over_forced_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    stream = io.StringIO()
    renderer = ConsoleRenderer(mode="summary", color="always", stream=stream)
    renderer._line("test", color="31")
    assert stream.getvalue() == "test\n"


def test_console_off_writes_nothing() -> None:
    stream = io.StringIO()
    renderer = ConsoleRenderer(mode="none", stream=stream)
    renderer._line("must not appear")
    assert stream.getvalue() == ""
