"""Terminal rendering is optional, redacted, and width-aware."""

from __future__ import annotations

import io

from mas_slm_research.console import ConsoleRenderer


def test_secret_fields_are_redacted_and_lines_wrap(monkeypatch) -> None:
    stream = io.StringIO()
    renderer = ConsoleRenderer(mode="full", color="never", stream=stream, width=48)
    renderer._event("agent_event", {
        "agent_name": "specialist", "event_type": "tool_call", "tool_name": "example",
        "payload_json": {"args": {"api_key": "do-not-print", "nested": {"password": "hidden"},
                                  "details": "many words in a long synthetic argument"}},
    })
    value = stream.getvalue()
    assert "[specialist] tool_call example" in value
    assert value.count("\n") >= 3
    assert "do-not-print" not in value and "hidden" not in value
    assert "[redacted]" in value


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
