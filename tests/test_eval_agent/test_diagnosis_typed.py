"""M3 asks the judge for typed answers: a ``Proposal`` and a critic ``Review``.

The proposer's reply is validated against the ``Proposal`` schema (an invalid
one goes back once with the errors, then raises); the critic's ``Review``
annotates hypotheses by number and never removes one.
"""

from __future__ import annotations

import pytest

from evalrx.agent_runtime.judge_call import JudgeCallError
from evalrx.analysis.analysis_module import AnalysisFinding, AnalysisReport
from evalrx.core.capability import Capability
from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent
from tests.conftest import FakeModel
from tests.judge_replies import hypothesis, keep_all, proposal, review


class _Judge(FakeModel):
    """Answers the proposer from *proposals* in order and the critic with *critic*.

    A reply that is an exception instance is raised instead of returned.
    """

    def __init__(self, proposals: list, critic: object = None) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self._proposals = iter(proposals)
        self._critic = critic
        self.prompts: list[str] = []

    def generate(self, inputs, **kw) -> str:
        prompt = str(inputs)
        self.prompts.append(prompt)
        reply = self._critic if "adversarial ML reviewer" in prompt else next(self._proposals)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _report(*findings: AnalysisFinding, severity: str = "none") -> AnalysisReport:
    return AnalysisReport(
        model_name="test-model", findings=list(findings), severity=severity,
        narrative="narrative", raw_results={},
    )


def _finding(severity: str, analyzer: str = "attention_sink") -> AnalysisFinding:
    return AnalysisFinding(
        analyzer=analyzer, metric="mean_sink_mass", value=0.85, threshold=0.6,
        direction="above", severity=severity, message="over-attends to the first token",
    )


_THREE = proposal(
    hypothesis("the model ignores visual evidence entirely", "visual_blindness"),
    hypothesis("the model truncates long answers", "termination"),
    hypothesis("the model copies the option order it was shown", "position_bias"),
)


def test_a_valid_proposal_maps_every_field_onto_the_hypothesis():
    judge = _Judge([proposal(hypothesis(
        "the model ignores the image on counting questions", "visual_neglect",
        plain="The model counts without looking at the picture.",
        test="relative_attention.max_relative_weight LOWER on failing cases",
        expected="lower_on_failures",
    ))], critic=keep_all(1))
    diag = DiagnosisAgent(judge=judge).diagnose(_report())

    [h] = diag.hypotheses
    assert h.statement == "the model ignores the image on counting questions"
    assert h.target_model == "test-model"
    assert h.predicted_failure_mode == "visual_neglect"
    assert h.plain_statement == "The model counts without looking at the picture."
    assert h.test_design == "relative_attention.max_relative_weight LOWER on failing cases"
    assert h.expected_association == "lower_on_failures"
    assert [c["role"] for c in diag.model_calls] == ["diagnosis_judge", "hypothesis_critic"]


@pytest.mark.parametrize("bad_mode", ["`visual_neglect`", "VisualNeglect"])
def test_a_failure_mode_that_is_not_snake_case_is_sent_back_and_the_second_reply_used(bad_mode):
    judge = _Judge([
        proposal(hypothesis("the model ignores the image on counting questions", bad_mode)),
        proposal(hypothesis("the model ignores the image on counting questions", "visual_neglect")),
    ], critic=keep_all(1))
    diag = DiagnosisAgent(judge=judge).diagnose(_report())

    assert [h.predicted_failure_mode for h in diag.hypotheses] == ["visual_neglect"]
    assert len(judge.prompts) == 3, "proposal, its one retry, then the critic"
    retry = judge.prompts[1]
    assert "Your previous answer" in retry and bad_mode in retry
    # the validation error goes back with it (the schema alone is in every prompt)
    assert "hypotheses.0.failure_mode" in retry
    assert "hypotheses.0.failure_mode" not in judge.prompts[0]
    assert [(c["role"], c["metadata"]["attempt"]) for c in diag.model_calls] == [
        ("diagnosis_judge", 1), ("diagnosis_judge", 2), ("hypothesis_critic", 1),
    ]


def test_critic_decisions_annotate_by_number_and_order_kept_first():
    judge = _Judge([_THREE], critic=review(
        (3, "keep", "the option order explains the failures"),
        (1, "reject", "nothing in the evidence tests vision"),
    ))
    diag = DiagnosisAgent(judge=judge).diagnose(_report())

    # The critic lists hypotheses numbered in the proposed order.
    critic_prompt = judge.prompts[-1]
    assert "1. the model ignores visual evidence entirely  (failure_mode: visual_blindness)" in critic_prompt
    assert "3. the model copies the option order it was shown  (failure_mode: position_bias)" in critic_prompt

    # Nothing is dropped; kept first, then the one with no decision, then rejected.
    assert [h.predicted_failure_mode for h in diag.hypotheses] == [
        "position_bias", "termination", "visual_blindness"]
    assert [h.metadata["critic"] for h in diag.hypotheses] == ["keep", "unparsed", "reject"]
    assert diag.hypotheses[0].metadata["critic_reason"] == "the option order explains the failures"
    assert "critic_reason" not in diag.hypotheses[1].metadata
    # The review record stays in proposed order.
    assert [d["decision"] for d in diag.review_decisions] == ["reject", "unparsed", "keep"]
    assert diag.n_critic_kept == 1 and diag.n_critic_rejected == 1


@pytest.mark.parametrize("critic", [RuntimeError("judge went away"), "no JSON here at all"],
                         ids=["judge_raises", "unreadable_review"])
def test_a_critic_that_gives_no_usable_review_keeps_every_hypothesis(critic):
    judge = _Judge([_THREE], critic=critic)
    diag = DiagnosisAgent(judge=judge).diagnose(_report())

    assert [h.predicted_failure_mode for h in diag.hypotheses] == [
        "visual_blindness", "termination", "position_bias"]
    assert all("critic" not in h.metadata for h in diag.hypotheses)
    assert [d["decision"] for d in diag.review_decisions] == ["review_unavailable"] * 3
    assert diag.n_critic_kept == 0 and diag.n_critic_rejected == 0
    # The failed critic exchanges are still recorded.
    critic_calls = diag.model_calls[1:]
    assert critic_calls and all(c["role"] == "hypothesis_critic" for c in critic_calls)


def test_an_empty_proposal_with_medium_findings_yields_template_hypotheses():
    judge = _Judge([proposal()])
    diag = DiagnosisAgent(judge=judge).diagnose(
        _report(_finding("medium"), _finding("low", analyzer="logit_lens"), severity="medium"),
    )

    assert diag.proposed_hypotheses == []
    assert [h.predicted_failure_mode for h in diag.hypotheses] == ["attention_sink"]   # low is not templated
    assert "over-attends to the first token" in diag.hypotheses[0].statement
    assert len(judge.prompts) == 1, "nothing proposed, so no critic call"


def test_an_empty_proposal_without_findings_is_no_problem():
    diag = DiagnosisAgent(judge=_Judge([proposal()])).diagnose(_report())
    assert diag.hypotheses == [] and diag.review_decisions == []


def test_a_proposer_that_raises_gives_judge_call_error():
    judge = _Judge([RuntimeError("quota exhausted")])
    with pytest.raises(JudgeCallError, match="quota exhausted") as err:
        DiagnosisAgent(judge=judge).diagnose(_report(_finding("high"), severity="high"))
    [exchange] = err.value.exchanges
    assert exchange.role == "diagnosis_judge" and exchange.output is None
