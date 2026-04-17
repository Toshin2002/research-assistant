"""
Unit tests for agent nodes and the graph router.

All Groq API calls are mocked — these tests run with no API keys and no network.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.agent import nodes
from app.agent.graph import _should_continue
from app.agent.state import AgentState


# ── Test helpers ──────────────────────────────────────────────────────────────

def _state(**overrides) -> AgentState:
    """Build a fully-populated AgentState with sensible defaults."""
    base: AgentState = {
        "goal":                       "Analyze AI coding assistants",
        "run_id":                     "test-run-id",
        "max_iterations":             5,
        "plan":                       ["Search for tools", "Compare features"],
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
    base.update(overrides)
    return base


def _mock_groq_response(text: str) -> MagicMock:
    """Build a fake Groq response object that returns the given text."""
    message          = MagicMock()
    message.content  = text
    message.tool_calls = None

    choice           = MagicMock()
    choice.message   = message

    response         = MagicMock()
    response.choices = [choice]
    return response


def _mock_tool_call_response(tool_name: str, tool_args: dict) -> MagicMock:
    """Build a fake Groq response that contains a tool call."""
    fn            = MagicMock()
    fn.name       = tool_name
    fn.arguments  = json.dumps(tool_args)

    call          = MagicMock()
    call.function = fn

    message            = MagicMock()
    message.content    = None
    message.tool_calls = [call]

    choice           = MagicMock()
    choice.message   = message

    response         = MagicMock()
    response.choices = [choice]
    return response


# ── Router tests (no Groq calls needed) ───────────────────────────────────────

class TestRouter:

    def test_satisfied_goes_to_report(self):
        state = _state(satisfied=True, iteration=1)
        assert _should_continue(state) == "report"

    def test_max_iterations_goes_to_report(self):
        state = _state(satisfied=False, iteration=5, max_iterations=5)
        assert _should_continue(state) == "report"

    def test_over_max_iterations_goes_to_report(self):
        # Defensive: iteration somehow exceeds max
        state = _state(satisfied=False, iteration=7, max_iterations=5)
        assert _should_continue(state) == "report"

    def test_not_satisfied_continues_to_act(self):
        state = _state(satisfied=False, iteration=2, max_iterations=5)
        assert _should_continue(state) == "act"

    def test_iteration_zero_continues_to_act(self):
        state = _state(satisfied=False, iteration=0, max_iterations=5)
        assert _should_continue(state) == "act"


# ── plan node ─────────────────────────────────────────────────────────────────

class TestPlanNode:

    @patch("app.agent.nodes.client")
    def test_returns_list_of_steps(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            '["Step 1", "Step 2", "Step 3"]'
        )
        result = nodes.plan(_state())

        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 3
        assert result["plan"][0] == "Step 1"

    @patch("app.agent.nodes.client")
    def test_resets_mutable_state(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            '["Step 1"]'
        )
        # Start with dirty state to verify plan resets it
        result = nodes.plan(_state(iteration=3, notes=["old note"], satisfied=True))

        assert result["iteration"]   == 0
        assert result["notes"]       == []
        assert result["satisfied"]   is False
        assert result["steps_taken"] == []

    @patch("app.agent.nodes.client")
    def test_handles_malformed_json_gracefully(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "This is not JSON at all"
        )
        result = nodes.plan(_state())

        # Falls back to a single-element list
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 1

    @patch("app.agent.nodes.client")
    def test_handles_non_list_json_gracefully(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            '{"step": "search"}'
        )
        result = nodes.plan(_state())

        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 1


# ── reflect node ──────────────────────────────────────────────────────────────

class TestReflectNode:

    @patch("app.agent.nodes.client")
    def test_satisfied_true(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({
                "satisfied":  True,
                "reasoning":  "All key areas are covered.",
                "next_focus": "",
            })
        )
        result = nodes.reflect(_state(notes=["Note 1", "Note 2"]))

        assert result["satisfied"]  is True
        assert result["reflection"] == "All key areas are covered."
        assert result["next_focus"] == ""

    @patch("app.agent.nodes.client")
    def test_satisfied_false_provides_next_focus(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({
                "satisfied":  False,
                "reasoning":  "Missing pricing data.",
                "next_focus": "Search for pricing of each tool",
            })
        )
        result = nodes.reflect(_state(notes=["Note 1"]))

        assert result["satisfied"]  is False
        assert "pricing" in result["next_focus"]

    @patch("app.agent.nodes.client")
    def test_defaults_to_not_satisfied_on_bad_json(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "I think we have enough information."
        )
        result = nodes.reflect(_state(notes=["Note 1"]))

        assert result["satisfied"] is False


# ── act node ──────────────────────────────────────────────────────────────────

class TestActNode:

    @patch("app.agent.nodes.client")
    def test_extracts_tool_name_and_input(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({"tool": "web_search", "arguments": {"query": "AI coding assistants 2024"}})
        )
        result = nodes.act(_state())

        assert result["pending_tool_name"]  == "web_search"
        assert result["pending_tool_input"] == {"query": "AI coding assistants 2024"}

    @patch("app.agent.nodes.client")
    def test_text_fallback_when_bad_json(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "I will search for AI coding tools."
        )
        result = nodes.act(_state())

        assert result["pending_tool_name"]          is None
        assert result["pending_tool_text_fallback"] == "I will search for AI coding tools."

    @patch("app.agent.nodes.client")
    def test_strips_markdown_fences(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            '```json\n{"tool": "web_search", "arguments": {"query": "test"}}\n```'
        )
        result = nodes.act(_state())

        assert result["pending_tool_name"] == "web_search"

    @patch("app.agent.nodes.client")
    def test_uses_next_focus_when_plan_exhausted(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({"tool": "web_search", "arguments": {"query": "pricing comparison"}})
        )
        result = nodes.act(_state(
            current_step_index=99,
            next_focus="Search for pricing comparison",
        ))

        assert result["pending_tool_name"] == "web_search"


# ── observe node ──────────────────────────────────────────────────────────────

class TestObserveNode:

    @patch("app.agent.nodes.client")
    @patch("app.agent.tools.web_search")
    def test_calls_tool_and_appends_note(self, mock_search, mock_client):
        mock_search.return_value = [
            {"title": "Tool A", "url": "http://a.com", "content": "details about Tool A"}
        ]
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Tool A is a leading AI coding assistant with strong IDE integration."
        )

        result = nodes.observe(_state(
            pending_tool_name="web_search",
            pending_tool_input={"query": "AI coding assistants"},
        ))

        assert len(result["notes"])      == 1
        assert "Tool A" in result["notes"][0]
        assert result["iteration"]       == 1
        assert len(result["steps_taken"]) == 1

    @patch("app.agent.nodes.client")
    def test_unknown_tool_records_error_note(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Error noted."
        )
        result = nodes.observe(_state(
            pending_tool_name="nonexistent_tool",
            pending_tool_input={},
        ))

        # Should not raise — should record an error note and continue
        assert result["iteration"] == 1
        assert len(result["notes"]) == 1

    @patch("app.agent.nodes.client")
    def test_clears_pending_tool_keys(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response("Note.")
        result = nodes.observe(_state(
            pending_tool_name="nonexistent_tool",
            pending_tool_input={"x": 1},
        ))

        assert result["pending_tool_name"]  is None
        assert result["pending_tool_input"] == {}

    @patch("app.agent.nodes.client")
    def test_text_fallback_becomes_note(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Summarized text note."
        )
        result = nodes.observe(_state(
            pending_tool_name=None,
            pending_tool_text_fallback="Some raw text from act node",
        ))

        assert result["iteration"] == 1
        assert len(result["notes"]) == 1

    @patch("app.agent.nodes.client")
    def test_advances_step_index(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response("Note.")
        result = nodes.observe(_state(
            current_step_index=0,
            plan=["Step 1", "Step 2"],
            pending_tool_name=None,
            pending_tool_text_fallback="text",
        ))

        assert result["current_step_index"] == 1

    @patch("app.agent.nodes.client")
    def test_step_index_does_not_exceed_plan_length(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response("Note.")
        result = nodes.observe(_state(
            current_step_index=1,       # already at last step (plan has 2 items)
            plan=["Step 1", "Step 2"],
            pending_tool_name=None,
            pending_tool_text_fallback="text",
        ))

        assert result["current_step_index"] == 1  # stays at 1, not 2


# ── report node ───────────────────────────────────────────────────────────────

class TestReportNode:

    @patch("app.agent.nodes.client")
    def test_returns_report_string(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "## Executive Summary\nAI coding tools are growing rapidly."
        )
        result = nodes.report(_state(notes=["Note 1", "Note 2"]))

        assert isinstance(result["report"], str)
        assert len(result["report"]) > 0

    @patch("app.agent.nodes.client")
    def test_report_contains_model_output(self, mock_client):
        expected = "## Executive Summary\nThis is the report."
        mock_client.chat.completions.create.return_value = _mock_groq_response(expected)

        result = nodes.report(_state(notes=["Note 1"]))

        assert result["report"] == expected
