"""M2→M3 handoff: DiagnosisAgent must consume M2's LLM conclusion + evidence
chain + statistical verdicts, not just the threshold narrative.

Regression for the gap where an analyzer surfaced a real failure mode (rich M2
conclusion) but M3 saw only "no anomalies" (severity=none) and returned 0
hypotheses.
"""

from __future__ import annotations

from evalvitals.analysis.stats_agent import StatsAnalysisReport
from evalvitals.analysis.stats_tools import StatsToolResult
from evalvitals.core.capability import Capability
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
from tests.conftest import FakeModel


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
        # let tests inspect the first (the diagnosis prompt).
        self.prompts.append(str(inputs))
        return "HYPOTHESIS: model fails to ground answers in the image\nFAILURE_MODE: weak_visual_grounding"


def test_prompt_includes_conclusion_evidence_and_stats():
    judge = CapturingJudge()
    DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    p = judge.prompts[0]
    assert "ignores the image and answers from language priors" in p   # conclusion
    assert "near-uniform" in p                                          # evidence chain
    assert "REJECT H0" in p                                             # stats verdict
    # The prompt must steer M3 away from trusting threshold severity alone.
    assert "threshold severity" in p.lower()


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
    from evalvitals.analysis.failure_modes import FailureMode, FailureModeReport

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
    from evalvitals.analysis.failure_modes import FailureModeReport

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
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

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


def test_diagnose_without_cases_keeps_the_critic_prompt_label_free():
    judge = CapturingJudge()
    DiagnosisAgent(judge=judge).diagnose(_stats_report_with_conclusion())
    assert "LABEL SUMMARY" not in judge.prompts[1]
    assert "Context the proposer worked from" in judge.prompts[1]  # conclusion/evidence still travel


def test_loop_hands_the_case_batch_to_agents_that_accept_it_only():
    from evalvitals.eval_agent.loop import _diagnose_with_optional_context

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
