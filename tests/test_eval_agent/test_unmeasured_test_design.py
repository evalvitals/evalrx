"""M3 is asked to rewrite a test that names only evidence M2 did not measure."""

from __future__ import annotations

from evalrx.analysis.stats_tools import StatsToolResult
from evalrx.eval_agent.stages.diagnosis import (
    HypothesisDraft,
    Proposal,
    _unmeasured_test_problems,
)
from evalrx.eval_agent.stages.hypothesis_tester import unmeasured_identifiers

MEASURED = [
    StatsToolResult(tool="signal_label_assoc", effect=0.25,
                    config={"signal": "modality_ablation.grounded_in_image"}),
    StatsToolResult(tool="signal_label_assoc", effect=None,     # ran, no effect: not usable
                    config={"signal": "pope.gold_yes"}),
]


def _draft(test: str) -> HypothesisDraft:
    return HypothesisDraft(hypothesis="The model answers without looking at the image.",
                           failure_mode="language_prior_bias", test=test)


def test_a_design_is_measured_when_it_names_a_signal_or_its_analyzer():
    for design in ("modality_ablation.grounded_in_image is lower on failures",
                   "grounded_in_image should be false on the failing cases",
                   "rerun modality_ablation and compare",
                   "failing cases should change less when the image is removed"):  # prose only
        assert unmeasured_identifiers(design, MEASURED) == set()


def test_a_design_that_names_only_case_ids_or_made_up_signals_is_unmeasured():
    assert unmeasured_identifiers("rad_pres_5 flips between runs", MEASURED) == {"rad_pres_5"}
    assert unmeasured_identifiers("fixed_by_semantic_grading rises", MEASURED) == {
        "fixed_by_semantic_grading"}
    assert unmeasured_identifiers("pope.gold_yes is higher on failures", MEASURED) == {
        "pope.gold_yes"}                                          # measured without an effect


def test_the_feedback_names_the_hypothesis_and_what_it_asked_for():
    proposal = Proposal(hypotheses=[
        _draft("grounded_in_image is lower on failures"),
        _draft("rad_pres_5 and rad_easy_1 are graded wrongly"),
    ])
    feedback = _unmeasured_test_problems(proposal, MEASURED)
    assert feedback is not None
    assert "hypothesis 2's test names rad_easy_1, rad_pres_5" in feedback
    assert "hypothesis 1" not in feedback
    assert feedback.endswith("Signals with a result: modality_ablation.grounded_in_image")
    assert _unmeasured_test_problems(Proposal(hypotheses=proposal.hypotheses[:1]), MEASURED) is None
