from typing import TypedDict, Optional


class StepRecord(TypedDict):
    node:   str
    tool:   Optional[str]
    input:  str
    output: str


class AgentState(TypedDict):

    # ── Set once, never changed ───────────────────────────────────────────────
    goal:           str
    run_id:         str
    max_iterations: int

    # ── Grows with every act → observe cycle ──────────────────────────────────
    plan:                list[str]
    current_step_index:  int
    steps_taken:         list[StepRecord]
    notes:               list[str]
    iteration:           int

    # ── Written by the reflect node ───────────────────────────────────────────
    satisfied:  bool
    reflection: str
    next_focus: str

    # ── Handoff keys between act and observe ──────────────────────────────────
    pending_tool_name:          Optional[str]
    pending_tool_input:         dict
    pending_tool_text_fallback: Optional[str]

    # ── Written at the very end ───────────────────────────────────────────────
    report: Optional[str]
    error:  Optional[str]
