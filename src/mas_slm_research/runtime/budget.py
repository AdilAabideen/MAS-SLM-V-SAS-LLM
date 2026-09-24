"""Explicit limits for the corrected, bounded execution policy."""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """A configured cumulative model, tool, or handoff limit was reached."""

    def __init__(self, *, counter: str, used: int, limit: int) -> None:
        self.counter = counter
        self.used = used
        self.limit = limit
        super().__init__(f"{counter}_budget_exceeded: used={used} limit={limit}")
