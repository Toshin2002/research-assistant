from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ── Request ───────────────────────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    goal:           str
    max_iterations: Optional[int] = None


# ── Response ──────────────────────────────────────────────────────────────────

class StepOut(BaseModel):
    id:          str
    node:        str
    tool_called: Optional[str]
    input:       Optional[str]
    output:      Optional[str]
    timestamp:   datetime

    model_config = {"from_attributes": True}


class RunOut(BaseModel):
    id:              str
    goal:            str
    status:          str
    report:          Optional[str]
    error:           Optional[str]
    iteration_count: int
    created_at:      datetime
    updated_at:      datetime

    model_config = {"from_attributes": True}


class RunDetailOut(RunOut):
    steps: list[StepOut] = []
