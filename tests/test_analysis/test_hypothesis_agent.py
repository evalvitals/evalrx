from __future__ import annotations

from evalvitals.analysis.hypothesis_agent import Hypothesis, HypothesisAgent, _parse_hypotheses


class ScriptedJudge:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self._response


class QueuedJudge:
    """Returns each response in order, repeating the last once exhausted —
    for tests that need to script a follow-up (repair) call."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


_REPORT = {
    "question": "What predicts yield?",
    "takeaways": [{
        "title": "Higher temperature batches yield more (70.1% vs 88.4%).",
        "analysis": "Mean yield rises with temperature across bins.",
    }],
    "observations": ["30 batches, no missing values."],
    "candidate_signals": [{"name": "temperature", "display_name": "Temperature", "rationale": "r=0.86"}],
}

_GOOD_RESPONSE = """\
HYPOTHESIS: Higher temperature accelerates the reaction, raising yield.
PLAIN: Warmer batches speed up the reaction, so more product comes out.
BASIS: Higher temperature batches yield more (70.1% vs 88.4%).
TEST: Run a controlled temperature-ramp experiment holding pressure fixed.

HYPOTHESIS: Catalyst B underperforms due to a side reaction at high pressure.
PLAIN: Catalyst B does worse because it triggers an unwanted extra reaction under high pressure.
BASIS: Temperature signal only
TEST: Compare catalyst B vs C yield holding temperature fixed."""

_JARGON_RESPONSE = """\
HYPOTHESIS: Higher temperature accelerates the reaction, raising yield.
PLAIN: This is collinear with the AUC of the reaction rate.
BASIS: Higher temperature batches yield more (70.1% vs 88.4%).
TEST: Run a controlled temperature-ramp experiment holding pressure fixed."""


def test_parse_hypotheses_splits_multiple_entries():
    out = _parse_hypotheses(_GOOD_RESPONSE)
    assert len(out) == 2
    assert out[0].statement == "Higher temperature accelerates the reaction, raising yield."
    assert out[0].plain_statement == "Warmer batches speed up the reaction, so more product comes out."
    assert out[0].basis == "Higher temperature batches yield more (70.1% vs 88.4%)."
    assert out[0].test_design == "Run a controlled temperature-ramp experiment holding pressure fixed."
    assert out[1].statement.startswith("Catalyst B underperforms")


def test_parse_hypotheses_plain_line_is_optional():
    """Older/repaired responses may omit PLAIN entirely — parsing must not break."""
    raw = (
        "HYPOTHESIS: Higher temperature accelerates the reaction, raising yield.\n"
        "BASIS: Higher temperature batches yield more.\n"
        "TEST: Run a controlled experiment."
    )
    out = _parse_hypotheses(raw)
    assert len(out) == 1
    assert out[0].plain_statement == ""


def test_parse_hypotheses_no_hypothesis_marker_yields_empty():
    assert _parse_hypotheses("NO_HYPOTHESIS") == []
    assert _parse_hypotheses("") == []


def test_parse_hypotheses_dedupes_restated_hypothesis_in_a_cli_trajectory():
    """CliAgentResult.raw_output for the CLI-agent backend is the full
    rendered tool-call trajectory, not just a final answer — an agent that
    narrates a plan before its final answer can restate the same hypothesis
    twice. This must not double-count it (observed on a real claude_code run)."""
    trajectory = (
        "[assistant] Let me think through this...\n"
        + _GOOD_RESPONSE
        + "\n\n[assistant] Here is my final answer.\n"
        + _GOOD_RESPONSE
    )

    out = _parse_hypotheses(trajectory)

    assert len(out) == 2
    statements = [h.statement for h in out]
    assert len(statements) == len(set(statements))
    assert out[0].statement == "Higher temperature accelerates the reaction, raising yield."


def test_propose_uses_judge_and_returns_parsed_hypotheses():
    judge = ScriptedJudge(_GOOD_RESPONSE)
    agent = HypothesisAgent(judge=judge)

    out = agent.propose(_REPORT)

    assert len(out) == 2
    assert all(isinstance(h, Hypothesis) for h in out)
    assert out[0].plain_statement == "Warmer batches speed up the reaction, so more product comes out."
    # the prompt actually carries the takeaway/observation/signal content
    assert "Higher temperature batches yield more" in judge.prompts[0]
    assert "temperature" in judge.prompts[0].lower()
    # PLAIN line already passed the check — no repair round needed
    assert len(judge.prompts) == 1


def test_propose_repairs_a_jargon_plain_line():
    fixed = (
        "HYPOTHESIS: Higher temperature accelerates the reaction, raising yield.\n"
        "PLAIN: Warmer batches produce more output.\n"
        "BASIS: Higher temperature batches yield more (70.1% vs 88.4%).\n"
        "TEST: Run a controlled temperature-ramp experiment holding pressure fixed."
    )
    judge = QueuedJudge(_JARGON_RESPONSE, fixed)
    agent = HypothesisAgent(judge=judge)

    out = agent.propose(_REPORT)

    assert len(judge.prompts) == 2
    assert "collinear" in judge.prompts[0]
    # the repair prompt names the specific violation and echoes the original answer
    assert "jargon" in judge.prompts[1].lower()
    assert "AUC" in judge.prompts[1] or "collinear" in judge.prompts[1]
    assert len(out) == 1
    assert out[0].plain_statement == "Warmer batches produce more output."


def test_propose_keeps_original_when_repair_still_fails():
    """If the repair round itself doesn't fix the jargon, fall back to
    whatever was parsed rather than dropping the hypothesis entirely."""
    judge = ScriptedJudge(_JARGON_RESPONSE)  # same jargon-y answer every call
    agent = HypothesisAgent(judge=judge)

    out = agent.propose(_REPORT)

    assert len(judge.prompts) == 2
    assert len(out) == 1
    assert out[0].statement == "Higher temperature accelerates the reaction, raising yield."


def test_propose_returns_empty_without_a_configured_backend():
    agent = HypothesisAgent()  # no judge, no cli_config
    assert agent.available is False
    assert agent.propose(_REPORT) == []


def test_propose_returns_empty_when_report_has_nothing_to_reason_over():
    judge = ScriptedJudge(_GOOD_RESPONSE)
    agent = HypothesisAgent(judge=judge)

    out = agent.propose({"question": "q"})

    assert out == []
    assert judge.prompts == []  # never even called the backend


def test_propose_never_raises_on_backend_failure():
    class BrokenJudge:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("boom")

    agent = HypothesisAgent(judge=BrokenJudge())
    assert agent.propose(_REPORT) == []


# ── agent-trajectory hint ─────────────────────────────────────────────────────
_AGENT_REPORT = {
    "question": "What predicts failures in this agent run?",
    "takeaways": [{
        "title": "Failures repeat tool calls",
        "analysis": "FAIL cases have max_consecutive_repeat >= 2 far more often than PASS.",
    }],
    "observations": ["n_tool_calls is right-skewed"],
    "candidate_signals": [{"name": "shap_outcome_image_zoom_in", "rationale": "zoom drives passing"}],
}


def test_agent_report_appends_intervenable_cause_hint():
    from evalvitals.analysis.hypothesis_agent import _is_agent_report

    judge = ScriptedJudge("NO_HYPOTHESIS")
    HypothesisAgent(judge=judge).propose(_AGENT_REPORT)
    assert _is_agent_report(_AGENT_REPORT) is True
    assert "AGENT TRAJECTORIES" in judge.prompts[0]
    assert "INTERVENABLE cause" in judge.prompts[0]
    assert "failure_mode is a judge-assigned label" in judge.prompts[0]


def test_non_agent_report_gets_no_hint():
    from evalvitals.analysis.hypothesis_agent import _is_agent_report

    judge = ScriptedJudge("NO_HYPOTHESIS")
    HypothesisAgent(judge=judge).propose(_REPORT)
    assert _is_agent_report(_REPORT) is False
    assert "AGENT TRAJECTORIES" not in judge.prompts[0]
