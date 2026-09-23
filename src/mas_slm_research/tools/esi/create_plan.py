"""Create Plan module helpers."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Any, List, Optional

class Step(BaseModel):
    step_id: str = Field(
        ...,
        description="Stable step identifier such as S1, S2, S3."
    )
    description: str = Field(
        ...,
        description="Short description of the step."
    )


class CreatePlanInput(BaseModel):
    objective: str = Field(
        ...,
        description="The main task the agent is trying to complete."
    )
    steps: List[Step] = Field(
        ...,
        description="Ordered plan steps."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional brief note for the plan."
    )


@tool("create_plan", args_schema=CreatePlanInput)
def create_plan(
    objective: str,
    steps: List[Step],
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a lightweight structured plan for the current case.
    """
    # Build the new value.
    return {
        "objective": objective,
        "steps": [step.model_dump() for step in steps],
        "notes": notes,
    }
