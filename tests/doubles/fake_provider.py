"""Fake Provider test coverage."""

from __future__ import annotations

from collections import deque
from typing import Any


class FakeBoundModel:
    def __init__(self, scripted: list[Any]) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self._scripted = deque(scripted)
        self.messages_seen: list[list[Any]] = []

    async def ainvoke(self, messages):
        """Handle the value."""
        # Keep the main step clear.
        self.messages_seen.append(list(messages))
        if not self._scripted:
            raise AssertionError("FakeBoundModel exhausted")
        item = self._scripted.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class FakeChatModel:
    def __init__(self, scripted: list[Any]) -> None:
        """Handle the value."""
        # Keep the main step clear.
        self._scripted = deque(scripted)
        self.bound_tools = None
        self.bound_tool_choice = None
        self.messages_seen: list[list[Any]] = []

    def bind_tools(self, tools, tool_choice="any"):
        """Handle tools."""
        # Keep the main step clear.
        self.bound_tools = list(tools)
        self.bound_tool_choice = tool_choice
        return self

    async def ainvoke(self, messages):
        """Handle the value."""
        # Keep the main step clear.
        self.messages_seen.append(list(messages))
        if not self._scripted:
            raise AssertionError("FakeChatModel exhausted")
        item = self._scripted.popleft()
        if isinstance(item, Exception):
            raise item
        return item
