import logging

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent import nodes

logger = logging.getLogger(__name__)


# ── Conditional edge ──────────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    if state["satisfied"]:
        logger.info("[router] satisfied=True → report")
        return "report"

    if state["iteration"] >= state["max_iterations"]:
        logger.info("[router] max iterations reached → report")
        return "report"

    logger.info("[router] iteration=%d → act", state["iteration"])
    return "act"


# ── Graph definition ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("plan",    nodes.plan)
    graph.add_node("act",     nodes.act)
    graph.add_node("observe", nodes.observe)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("report",  nodes.report)

    # Fixed edges
    graph.add_edge(START,     "plan")
    graph.add_edge("plan",    "act")
    graph.add_edge("act",     "observe")
    graph.add_edge("observe", "reflect")
    graph.add_edge("report",  END)

    # Conditional edge — the loop
    graph.add_conditional_edges(
        "reflect",
        _should_continue,
        {"act": "act", "report": "report"},
    )

    return graph


compiled_graph = build_graph().compile()
