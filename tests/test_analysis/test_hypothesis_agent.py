from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalrx.agent_runtime.cli_types import CliAgentConfig, CliAgentResult
from evalrx.analysis.hypothesis_agent import Hypothesis, HypothesisAgent


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

def _answer(*items: tuple[str, str]) -> str:
    return json.dumps({"hypotheses": [
        {"hypothesis": h, "plain_statement": plain, "basis": "Higher temperature batches yield more.",
         "test": "Run a controlled temperature-ramp experiment holding pressure fixed."}
        for h, plain in items
    ]})


_GOOD_RESPONSE = _answer(
    ("Higher temperature accelerates the reaction, raising yield.",
     "Warmer batches speed up the reaction, so more product comes out."),
    ("Catalyst B underperforms due to a side reaction at high pressure.",
     "Catalyst B does worse because it triggers an unwanted extra reaction under high pressure."),
)

_JARGON_RESPONSE = _answer(
    ("Higher temperature accelerates the reaction, raising yield.",
     "This is collinear with the AUC of the reaction rate."),
)

_NONE = json.dumps({"hypotheses": []})


def test_propose_uses_judge_and_returns_typed_hypotheses():
    judge = ScriptedJudge(f"Here you go:\n```json\n{_GOOD_RESPONSE}\n```")
    agent = HypothesisAgent(judge=judge)

    out = agent.propose(_REPORT)

    assert len(out) == 2
    assert all(isinstance(h, Hypothesis) for h in out)
    assert out[0].statement == "Higher temperature accelerates the reaction, raising yield."
    assert out[0].plain_statement == "Warmer batches speed up the reaction, so more product comes out."
    assert out[0].basis == "Higher temperature batches yield more."
    assert out[0].test_design.startswith("Run a controlled")
    # the prompt actually carries the takeaway/observation/signal content
    assert "Higher temperature batches yield more" in judge.prompts[0]
    assert "temperature" in judge.prompts[0].lower()
    # the plain statements already pass the check: one call
    assert len(judge.prompts) == 1


def test_an_empty_list_means_no_hypotheses():
    assert HypothesisAgent(judge=ScriptedJudge(_NONE)).propose(_REPORT) == []


def test_propose_sends_a_jargon_plain_statement_back_once():
    fixed = _answer(("Higher temperature accelerates the reaction, raising yield.",
                     "Warmer batches produce more output."))
    judge = QueuedJudge(_JARGON_RESPONSE, fixed)

    out = HypothesisAgent(judge=judge).propose(_REPORT)

    assert len(judge.prompts) == 2
    assert "plain-language check" in judge.prompts[1]
    assert "collinear" in judge.prompts[1]
    assert [h.plain_statement for h in out] == ["Warmer batches produce more output."]


def test_propose_keeps_the_answer_when_the_revision_still_has_jargon():
    judge = ScriptedJudge(_JARGON_RESPONSE)  # same answer every call

    out = HypothesisAgent(judge=judge).propose(_REPORT)

    assert len(judge.prompts) == 2
    assert [h.statement for h in out] == ["Higher temperature accelerates the reaction, raising yield."]


def test_an_answer_without_the_hypotheses_field_is_not_read_as_none():
    judge = QueuedJudge('{"note": "thinking"}', _GOOD_RESPONSE)
    out = HypothesisAgent(judge=judge).propose(_REPORT)
    assert len(judge.prompts) == 2 and len(out) == 2


def test_propose_returns_empty_when_no_valid_answer_arrives():
    judge = ScriptedJudge("HYPOTHESIS: an old-style answer")
    assert HypothesisAgent(judge=judge).propose(_REPORT) == []
    assert len(judge.prompts) == 2


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
    from evalrx.analysis.hypothesis_agent import _is_agent_report

    judge = ScriptedJudge(_NONE)
    HypothesisAgent(judge=judge).propose(_AGENT_REPORT)
    assert _is_agent_report(_AGENT_REPORT) is True
    assert "AGENT TRAJECTORIES" in judge.prompts[0]
    assert "INTERVENABLE cause" in judge.prompts[0]
    assert "failure_mode is a judge-assigned label" in judge.prompts[0]


def test_non_agent_report_gets_no_hint():
    from evalrx.analysis.hypothesis_agent import _is_agent_report

    judge = ScriptedJudge(_NONE)
    HypothesisAgent(judge=judge).propose(_REPORT)
    assert _is_agent_report(_REPORT) is False
    assert "AGENT TRAJECTORIES" not in judge.prompts[0]


# ── CLI coding-agent backend ──────────────────────────────────────────────────
class _FakeRunner:
    """Stands in for CodegenRunner: 'the agent' writes answer.json (or not)."""

    answers: list = []
    workdirs: list = []

    def __init__(self, cli_config):
        pass

    def run(self, prompt, *, workdir: Path, timeout_sec=None):
        _FakeRunner.workdirs.append(workdir)
        assert "answer.json" in prompt
        answer = _FakeRunner.answers.pop(0)
        if isinstance(answer, str):
            (workdir / "answer.json").write_text(answer)
            return CliAgentResult(files={}, provider_name="claude_code", elapsed_sec=1.0,
                                  raw_output="[assistant] draft " + _JARGON_RESPONSE)
        return CliAgentResult(files={}, provider_name="claude_code", elapsed_sec=1.0, **answer)


@pytest.fixture
def fake_runner(monkeypatch):
    import evalrx.agent_runtime.codegen as codegen

    monkeypatch.setattr(codegen, "CodegenRunner", _FakeRunner)
    _FakeRunner.answers, _FakeRunner.workdirs = [], []
    return _FakeRunner


def test_cli_backend_reads_the_answer_file_not_the_trajectory(fake_runner):
    fake_runner.answers = [_GOOD_RESPONSE]
    agent = HypothesisAgent(cli_config=CliAgentConfig(provider="claude_code"))
    out = agent.propose(_REPORT)
    assert [h.plain_statement for h in out][0] == "Warmer batches speed up the reaction, so more product comes out."


def test_cli_backend_retries_in_a_fresh_directory_when_no_file_is_written(fake_runner):
    fake_runner.answers = [{"raw_output": "I wrote nothing"}, _GOOD_RESPONSE]
    out = HypothesisAgent(cli_config=CliAgentConfig(provider="claude_code")).propose(_REPORT)
    assert len(out) == 2
    assert len(set(fake_runner.workdirs)) == 2


def test_cli_backend_failure_yields_no_hypotheses(fake_runner):
    fake_runner.answers = [{"error": "[TIMEOUT] agent killed after 90s"}]
    assert HypothesisAgent(cli_config=CliAgentConfig(provider="claude_code")).propose(_REPORT) == []
