import json
import logging

from groq import Groq

from app.agent.state import AgentState, StepRecord
from app.agent.tools import TOOL_REGISTRY
from app.config import settings

logger = logging.getLogger(__name__)
client = Groq(api_key=settings.groq_api_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chat(messages: list[dict], max_tokens: int = 1024, tools: list = None) -> any:
    """Single place where every Groq API call is made."""
    kwargs = {
        "model":      settings.model_name,
        "messages":   messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"]        = tools
        kwargs["tool_choice"]  = "auto"

    return client.chat.completions.create(**kwargs)


def _text(response) -> str:
    """Extract plain text from a Groq response."""
    return response.choices[0].message.content or ""


# ── plan node ─────────────────────────────────────────────────────────────────

def plan(state: AgentState) -> dict:
    logger.info("[plan] goal=%s", state["goal"])

    messages = [
        {
            "role": "system",
            "content": "You are a research planning expert. Return ONLY a JSON array of strings. No explanation. No markdown fences.",
        },
        {
            "role": "user",
            "content": (
                f"Goal: {state['goal']}\n\n"
                "Break this into 3 to 6 specific, ordered research steps.\n"
                'Example: ["Step one", "Step two", "Step three"]'
            ),
        },
    ]

    raw = _text(_chat(messages, max_tokens=512)).strip()

    try:
        steps = json.loads(raw)
        if not isinstance(steps, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError):
        steps = [raw]

    logger.info("[plan] %d steps generated", len(steps))

    return {
        "plan":                list(steps),
        "current_step_index":  0,
        "iteration":           0,
        "steps_taken":         [],
        "notes":               [],
        "satisfied":           False,
        "reflection":          "",
        "next_focus":          "",
        "report":              None,
        "error":               None,
    }


# ── act node ──────────────────────────────────────────────────────────────────

def act(state: AgentState) -> dict:
    idx   = state["current_step_index"]
    steps = state["plan"]

    current_step = (
        steps[idx] if idx < len(steps)
        else state["next_focus"] or "Continue researching the goal"
    )

    logger.info("[act] iteration=%d  step=%s", state["iteration"], current_step)

    notes_block = (
        "\n".join(f"- {n}" for n in state["notes"][-8:])
        or "None yet."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous research agent. "
                "You have two tools available:\n\n"
                "1. web_search(query: str, max_results: int = 5)\n"
                "   Use this to search the web for information.\n\n"
                "2. fetch_page(url: str)\n"
                "   Use this to read the full text of a specific web page.\n\n"
                "Respond with ONLY a JSON object — no explanation, no markdown fences.\n"
                'Example: {"tool": "web_search", "arguments": {"query": "your query here"}}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research goal: {state['goal']}\n\n"
                f"Current step: {current_step}\n\n"
                f"Notes so far:\n{notes_block}\n\n"
                "Which tool should you call and with what arguments? "
                "Respond with ONLY the JSON object."
            ),
        },
    ]

    raw = _text(_chat(messages, max_tokens=256)).strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    raw = raw.strip()

    try:
        parsed     = json.loads(raw)
        tool_name  = parsed.get("tool")
        tool_input = parsed.get("arguments", {})

        if not tool_name:
            raise ValueError("missing 'tool' key")

        logger.info("[act] tool=%s  input=%s", tool_name, tool_input)
        return {
            "pending_tool_name":          tool_name,
            "pending_tool_input":         tool_input,
            "pending_tool_text_fallback": None,
        }
    except (json.JSONDecodeError, ValueError):
        logger.warning("[act] could not parse tool JSON — using text fallback")
        return {
            "pending_tool_name":          None,
            "pending_tool_input":         {},
            "pending_tool_text_fallback": raw,
        }


# ── observe node ──────────────────────────────────────────────────────────────

def observe(state: AgentState) -> dict:
    tool_name  = state.get("pending_tool_name")
    tool_input = state.get("pending_tool_input") or {}
    fallback   = state.get("pending_tool_text_fallback")

    if tool_name is None:
        observation = fallback or "[no output]"
    else:
        logger.info("[observe] tool=%s  input=%s", tool_name, tool_input)
        tool_fn = TOOL_REGISTRY.get(tool_name)

        if tool_fn is None:
            observation = f"[error: unknown tool '{tool_name}']"
        else:
            try:
                result      = tool_fn(**tool_input)
                observation = json.dumps(result) if not isinstance(result, str) else result
            except Exception as exc:
                logger.exception("[observe] tool error")
                observation = f"[tool error: {exc}]"

    note = _distill(state["goal"], observation, tool_name or "text")

    step: StepRecord = {
        "node":   "act->observe",
        "tool":   tool_name,
        "input":  json.dumps(tool_input),
        "output": observation[:2000],
    }

    return {
        "notes":                      state["notes"] + [note],
        "steps_taken":                state["steps_taken"] + [step],
        "iteration":                  state["iteration"] + 1,
        "current_step_index":         min(
                                          state["current_step_index"] + 1,
                                          len(state["plan"]) - 1,
                                      ),
        "pending_tool_name":          None,
        "pending_tool_input":         {},
        "pending_tool_text_fallback": None,
    }


def _distill(goal: str, raw: str, tool_name: str) -> str:
    """Summarise raw tool output into a concise research note."""
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Write concise, factual notes.",
        },
        {
            "role": "user",
            "content": (
                f"Research goal: {goal}\n\n"
                f"Raw tool output ({tool_name}):\n{raw[:5000]}\n\n"
                "Write a concise note (2–4 sentences) capturing the key facts "
                "relevant to the goal. State findings only — no meta-commentary."
            ),
        },
    ]
    return _text(_chat(messages, max_tokens=256)).strip()


# ── reflect node ──────────────────────────────────────────────────────────────

def reflect(state: AgentState) -> dict:
    logger.info("[reflect] iteration=%d", state["iteration"])

    notes_block = "\n".join(
        f"{i + 1}. {n}" for i, n in enumerate(state["notes"])
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a critical research evaluator. "
                "Return ONLY valid JSON. No markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research goal: {state['goal']}\n\n"
                f"Notes gathered so far:\n{notes_block}\n\n"
                f"Iterations used: {state['iteration']} / {state['max_iterations']}\n\n"
                "Do you have enough information to write a comprehensive report?\n\n"
                "Return exactly this JSON shape:\n"
                '{"satisfied": true or false, '
                '"reasoning": "one or two sentences", '
                '"next_focus": "specific topic to research next, or empty string if satisfied"}'
            ),
        },
    ]

    raw = _text(_chat(messages, max_tokens=256)).strip()

    try:
        result     = json.loads(raw)
        satisfied  = bool(result.get("satisfied", False))
        reasoning  = str(result.get("reasoning", ""))
        next_focus = str(result.get("next_focus", ""))
    except (json.JSONDecodeError, KeyError):
        logger.warning("[reflect] could not parse JSON — defaulting to not satisfied")
        satisfied  = False
        reasoning  = raw
        next_focus = ""

    logger.info("[reflect] satisfied=%s", satisfied)
    return {
        "satisfied":  satisfied,
        "reflection": reasoning,
        "next_focus": next_focus,
    }


# ── report node ───────────────────────────────────────────────────────────────

def report(state: AgentState) -> dict:
    logger.info("[report] writing final report")

    notes_block = "\n".join(
        f"{i + 1}. {n}" for i, n in enumerate(state["notes"])
    )

    messages = [
        {
            "role": "system",
            "content": "You are a professional research analyst. Write clear, structured markdown reports.",
        },
        {
            "role": "user",
            "content": (
                f"Research goal: {state['goal']}\n\n"
                f"Research notes:\n{notes_block}\n\n"
                "Write a comprehensive research report in markdown with these sections:\n\n"
                "## Executive Summary\n"
                "## Key Findings\n"
                "## Detailed Analysis\n"
                "## Sources & Confidence\n"
                "## Conclusion\n\n"
                "Be specific. Cite numbers, names, and facts wherever available."
            ),
        },
    ]

    return {"report": _text(_chat(messages, max_tokens=2048)).strip()}
