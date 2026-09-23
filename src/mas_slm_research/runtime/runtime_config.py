"""Runtime Config module helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Central runtime policy switches for hand-rolled agent execution."""

    multi_agent: bool = False
    print_events: bool = False
    persist_events: bool = True
    disable_final_answer_tool_when_handoff_tools_present: bool = True
    max_tool_calls_per_turn: int = 2
    require_final_answer_tool: bool = True
    allow_text_tool_recovery: bool = True
    malformed_tool_retry_enabled: bool = True
    max_malformed_tool_retries_per_tool: int = 1
    allow_plain_json_final_output: bool = True
    drop_extra_tool_calls: bool = True
    short_term_memory_include_final_assistant_output: bool = False
    short_term_memory_include_raw_provider_debug: bool = False
    short_term_memory_verbose: bool = False
    short_term_memory_log_token_estimates: bool = True

    def __post_init__(self) -> None:
        """Handle init."""
        # Keep the main step clear.
        if self.max_tool_calls_per_turn < 1:
            raise ValueError("max_tool_calls_per_turn must be >= 1.")
        if self.max_malformed_tool_retries_per_tool < 0:
            raise ValueError("max_malformed_tool_retries_per_tool must be >= 0.")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dict for telemetry and experiment logging."""
        # Keep the main step clear.
        return asdict(self)
