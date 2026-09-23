"""Log Thought module helpers."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class LogThoughtInput(BaseModel):
    step: str = Field(
        ...,
        description="Exact plan step ID, e.g. S1, S2, S3."
    )
    thought: str = Field(
        ...,
        description="Short reasoning trace line for observability. No final diagnosis, no treatment recommendation, <= 25 words."
    )


@tool("log_thought", args_schema=LogThoughtInput)
def log_thought(step: str, thought: str) -> dict:
    """Emit a short step-linked reasoning trace line."""
    # Keep the main step clear.
    return {
        "step": step,
        "thought": thought,
    }
