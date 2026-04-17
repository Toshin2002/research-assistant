import logging

from app.agent.graph import compiled_graph
from app.agent.state import AgentState
from app.db.repository import RunRepository
from app.db.session import AsyncSessionLocal
from app.config import settings

logger = logging.getLogger(__name__)


async def run_agent(run_id: str, goal: str, max_iterations: int | None = None) -> None:
    effective_max = max_iterations if max_iterations is not None else settings.max_iterations

    # ── 1. Mark run as running ─────────────────────────────────────────────────
    async with AsyncSessionLocal() as session:
        await RunRepository(session).set_running(run_id)

    # ── 2. Build initial state ─────────────────────────────────────────────────
    initial_state: AgentState = {
        "goal":                       goal,
        "run_id":                     run_id,
        "max_iterations":             effective_max,
        "plan":                       [],
        "current_step_index":         0,
        "steps_taken":                [],
        "notes":                      [],
        "iteration":                  0,
        "satisfied":                  False,
        "reflection":                 "",
        "next_focus":                 "",
        "pending_tool_name":          None,
        "pending_tool_input":         {},
        "pending_tool_text_fallback": None,
        "report":                     None,
        "error":                      None,
    }

    # ── 3. Run the graph ───────────────────────────────────────────────────────
    try:
        final_state: AgentState = await compiled_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("[runner] agent crashed  run_id=%s", run_id)
        async with AsyncSessionLocal() as session:
            await RunRepository(session).set_failed(run_id, str(exc))
        return

    # ── 4. Persist steps and final status ─────────────────────────────────────
    async with AsyncSessionLocal() as session:
        repo = RunRepository(session)

        # Tool call steps (act → observe cycles)
        for step in final_state.get("steps_taken", []):
            await repo.add_step(
                run_id=run_id,
                node=step["node"],
                input_data={"input": step["input"]},
                output_data={"output": step["output"]},
                tool_called=step.get("tool"),
            )

        # Plan step — what research steps were generated
        await repo.add_step(
            run_id=run_id,
            node="plan",
            input_data={"goal": goal},
            output_data={"plan": final_state.get("plan", [])},
        )

        # Reflect step — final self-evaluation verdict
        await repo.add_step(
            run_id=run_id,
            node="reflect",
            input_data={"iteration": final_state.get("iteration", 0)},
            output_data={
                "satisfied": final_state.get("satisfied"),
                "reasoning": final_state.get("reflection"),
            },
        )

        # Mark completed
        await repo.set_completed(
            run_id=run_id,
            report=final_state.get("report") or "[no report generated]",
            iteration_count=final_state.get("iteration", 0),
        )

    logger.info(
        "[runner] completed  run_id=%s  iterations=%d",
        run_id,
        final_state.get("iteration", 0),
    )
