"""Prompt Protocol module helpers."""

from __future__ import annotations

from typing import Optional, Sequence


def normalize_prompt_block(block: Optional[str]) -> str:
    """Trim a prompt block while preserving its internal formatting."""
    # Keep the output consistent.
    if not isinstance(block, str):
        return ""
    return block.strip()


def join_prompt_sections(*sections: Optional[str]) -> str:
    """Join non-empty prompt sections with stable double-newline separators."""
    # Keep the main step clear.
    normalized_sections = [
        normalize_prompt_block(section)
        for section in sections
    ]
    filtered_sections = [section for section in normalized_sections if section]
    return "\n\n".join(filtered_sections)


def build_system_prompt(
    base_prompt: str,
    multi_agent_addon: Optional[str] = None,
    single_agent_addon: Optional[str] = None,
    *,
    multi_agent: bool = False,
    extra_sections: Optional[Sequence[str]] = None,
) -> str:
    """Build the final system prompt from a neutral base prompt plus mode-specific addons.

    Assembly order is:
    1. base prompt
    2. mode-specific addon
    3. optional extra sections
    """
    # Build the next value.
    mode_addon = multi_agent_addon if multi_agent else single_agent_addon
    extras = [normalize_prompt_block(section) for section in list(extra_sections or [])]
    return join_prompt_sections(base_prompt, mode_addon, *extras)
