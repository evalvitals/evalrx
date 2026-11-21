"""M2→M3 handoff: DiagnosisAgent must consume M2's LLM conclusion + evidence
chain + statistical verdicts, not just the threshold narrative.

Regression for the gap where an analyzer surfaced a real failure mode (rich M2
conclusion) but M3 saw only "no anomalies" (severity=none) and returned 0
hypotheses.
"""

from __future__ import annotations

from evalrx.analysis.stats_agent import StatsAnalysisReport
from evalrx.analysis.stats_tools import StatsToolResult
from evalrx.core.capability import Capability
from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel
from tests.judge_replies import hypothesis, keep_all, proposal


def _stats_report_with_conclusion() -> StatsAnalysisReport:
    # severity=none and no threshold findings, but a rich LLM conclusion +
    # evidence + a statistical verdict — exactly the relative_attention case.
    return StatsAnalysisReport(
        model_name="vlm",
        findings=[],
        severity="none",
        narrative="No anomalies detected — all metrics within normal ranges.",
        raw_results={},
        conclusion="The model ignores the image and answers from language priors.",
        evidence_chain=[
            "relative_attention max weight only 1.69x (near-uniform)",
            "diffuse attention connects to the counting/colour errors",
        ],
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True,
                            summary="signal vs FAIL: effect=+0.80 -> REJECT H0",
                            effect=0.8, reject=True),
        ],
    )


class CapturingJudge(FakeModel):
    def __init__(self) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.prompts: list[str] = []

    def generate(self, inputs, **kw) -> str:
        # diagnose() makes a second adversarial-validation call; record all and
        # let tests inspect the first (the diagnosis prompt). The proposal
        # carries a clean plain_statement because the prompt demands one — a
        # missing or jargon-y one costs a third call (the plain-language
        # revision), which is what test_plain_language_revision covers.
        prompt = str(inputs)
        self.prompts.append(prompt)
        if "adversarial ML reviewer" in prompt:
            return keep_all(1)
        return _proposal()


def _proposal(plain: str = "The model answers from the wording of the question "
                           "instead of looking at the picture.") -> str:
    return proposal(hypothesis("model fails to ground answers in the image",
                               "weak_visual_grounding", plain=plain))


def test_prompt_includes_conclusion_evidence_and_stats():
    judge = CapturingJudge()
    DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    p = judge.prompts[0]
    assert "ignores the image and answers from language priors" in p   # conclusion
    assert "near-uniform" in p                                          # evidence chain
    assert "REJECT H0" in p                                             # stats verdict
    # The prompt must steer M3 away from trusting threshold severity alone.
    assert "threshold severity" in p.lower()


def test_prompt_and_critic_include_complete_protocol_contract():
    report = _stats_report_with_conclusion()
    report.protocol = ExperimentProtocol(
        description="Answer an audio multiple-choice question.",
        success_criteria="Reply with only one option letter.",
        output_contract={"kind": "multiple_choice_letter", "choices": ["A", "B", "C", "D"]},
    )
    judge = CapturingJudge()
    DiagnosisAgent(judge=judge).diagnose(report)
    assert all("Reply with only one option letter" in p for p in judge.prompts[:2])
    assert all('"kind": "multiple_choice_letter"' in p for p in judge.prompts[:2])


def test_hypothesis_generated_despite_severity_none():
    # The whole point: a real hypothesis comes out even though severity=none.
    judge = CapturingJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    assert len(diag.hypotheses) == 1
    assert diag.hypotheses[0].predicted_failure_mode == "weak_visual_grounding"


def test_failure_modes_none_by_default_adds_nothing_to_the_prompt():
    judge = CapturingJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    assert "FAILURE MODES" not in judge.prompts[0]
    assert diag.failure_modes_used is False


def test_failure_modes_report_enters_the_prompt_when_supplied():
    from evalrx.analysis.failure_modes import FailureMode, FailureModeReport

    fm_report = FailureModeReport(
        clusters=[FailureMode(name="small_object_miss", description="objects too small to detect", size=7)],
        method="cosine_greedy",
    )
    judge = CapturingJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(
        _stats_report_with_conclusion(), failure_modes=fm_report,
    )
    p = judge.prompts[0]
    assert "FAILURE MODES" in p
    assert "small_object_miss" in p
    assert "objects too small to detect" in p
    assert diag.failure_modes_used is True


def test_failure_modes_with_zero_clusters_adds_nothing():
    from evalrx.analysis.failure_modes import FailureModeReport

    judge = CapturingJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(
        _stats_report_with_conclusion(), failure_modes=FailureModeReport(),
    )
    assert "FAILURE MODES" not in judge.prompts[0]
    assert diag.failure_modes_used is False


def test_critic_reviews_against_the_proposers_context_and_a_label_summary():
    """audiocaps 2026-08-20: the critic rejected a correct 'answers Yes regardless
    of the audio' lead for 'no ground-truth present/absent field' -- it saw only
    the findings JSON while the proposer had M2's conclusion and the explore
    breakdown. Both calls now read the same context, plus a label summary."""
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label

    judge = CapturingJudge()
    cases = CaseBatch(
        [FailureCase(id=f"f{i}", inputs=Inputs(prompt="Is there a dog?"),
                     observed="Yes", expected="No", label=Label.FAIL) for i in range(5)]
        + [FailureCase(id=f"p{i}", inputs=Inputs(prompt="Is there a cat?"),
                       observed="Yes", expected="Yes", label=Label.PASS) for i in range(5)]
    )
    DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion(), cases=cases)
    assert len(judge.prompts) == 2, "proposer call, then the critic call"
    critic = judge.prompts[1]
    assert "Context the proposer worked from" in critic
    assert "ignores the image and answers from language priors" in critic   # conclusion
    assert "near-uniform" in critic                                          # evidence chain
    assert "REJECT H0" in critic                                             # stats verdict
    assert "LABEL SUMMARY" in critic and "gold=no  answered=yes      n=5    FAIL=5" in critic
    # the proposer's own prompt is unchanged by the cases kwarg
    assert "LABEL SUMMARY" not in judge.prompts[0]
    # and the critic prompt travels on the result so the run logger can persist it
    # (the request itself; the judge also saw the Review schema ask_judge adds)
    diag = DiagnosisAgent(judge=CapturingJudge()).diagnose(_stats_report_with_conclusion(), cases=cases)
    assert "LABEL SUMMARY" in diag.critic_prompt and diag.critic_prompt in judge.prompts[1]


def test_diagnose_without_cases_keeps_the_critic_prompt_label_free():
    judge = CapturingJudge()
    DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    assert "LABEL SUMMARY" not in judge.prompts[1]
    assert "Context the proposer worked from" in judge.prompts[1]  # conclusion/evidence still travel


def test_plain_language_revision_runs_once_when_the_plain_statement_is_jargon():
    """The prompt asking for plain language is not enough on its own — a judge
    will reuse the technical line — so the host checks it and spends ONE extra
    call to fix it. The analysis-side hypothesis agent already worked this way;
    this is the path the benchmark runner actually takes."""
    class JargonThenPlain(FakeModel):
        def __init__(self) -> None:
            super().__init__(capabilities={Capability.GENERATE})
            self.prompts: list[str] = []

        def generate(self, inputs, **kw) -> str:
            prompt = str(inputs)
            self.prompts.append(prompt)
            if "adversarial ML reviewer" in prompt:
                return keep_all(1)
            return _proposal("Spearman rho of 0.42 between the coefficient and the p-value"
                             if len(self.prompts) == 1
                             else "The model answers Yes whatever the picture shows.")

    judge = JargonThenPlain()
    diag = DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    assert len(judge.prompts) == 3, "proposer, its plain-language revision, then the critic"
    assert "fail the plain-language check" in judge.prompts[1]
    assert "Spearman rho of 0.42" in judge.prompts[1]          # the rejected answer goes back
    assert diag.hypotheses[0].plain_statement == (
        "The model answers Yes whatever the picture shows."
    )
    assert [(call["role"], call["metadata"]["attempt"]) for call in diag.model_calls] == [
        ("diagnosis_judge", 1), ("diagnosis_judge", 2), ("hypothesis_critic", 1),
    ]
    assert [call["inputs"] for call in diag.model_calls] == judge.prompts
    assert all(call["duration_sec"] >= 0 for call in diag.model_calls)


def test_a_judge_error_on_the_plain_language_revision_raises():
    """The revision is attempt 2 of the same typed call, so a judge that raises
    there fails the call like any other judge failure — JudgeCallError carrying
    both exchanges — and the loop decides whether to retry M3."""
    import pytest

    from evalrx.agent_runtime.judge_call import JudgeCallError

    class RaisesOnRevision(FakeModel):
        def __init__(self) -> None:
            super().__init__(capabilities={Capability.GENERATE})
            self.calls = 0

        def generate(self, inputs, **kw) -> str:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("judge went away")
            return _proposal("rho of 0.42 against the coefficient")

    with pytest.raises(JudgeCallError, match="judge went away") as err:
        DiagnosisAgent(judge=RaisesOnRevision()).diagnose(_stats_report_with_conclusion())
    first, second = err.value.exchanges
    assert first.output and first.error is None
    assert second.output is None and "judge went away" in second.error


def test_loop_hands_the_case_batch_to_agents_that_accept_it_only():
    from evalrx.eval_agent.loop.stages import _diagnose_with_optional_context

    class Modern:
        def diagnose(self, stats_report, prior_cycles=None, explore_context=None,
                     failure_modes=None, cases=None):
            self.seen = {"cases": cases, "explore_context": explore_context}
            return "ok"

    class Legacy:
        def diagnose(self, stats_report, prior_cycles=None):
            return "ok"

    modern = Modern()
    assert _diagnose_with_optional_context(modern, "report", [], None, cases=["c"]) == "ok"
    assert modern.seen == {"cases": ["c"], "explore_context": None}
    # a legacy agent without the kwarg is called without it (no TypeError)
    assert _diagnose_with_optional_context(Legacy(), "report", [], None, cases=["c"]) == "ok"
