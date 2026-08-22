"""P3+P4: hypothesis test designs, evidence routing, and depth-tiered stopping.

P3 — M3 attaches a ``test_design`` to each hypothesis; M5 routes evidence by it
(deterministic) before falling back to keywords; M1 folds the designs into
cycle-2 analyzer selection.
P4 — verdicts carry an ``evidence_grade`` (intervention > observational) and
``stopping_criteria_met`` can require intervention-grade evidence.
"""

from __future__ import annotations

import pytest

from evalvitals.analysis.stats_agent import StatsAnalysisReport
from evalvitals.analysis.stats_tools import StatsToolResult
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent import HypothesisTester, ProbeAgent
from evalvitals.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    hypothesis_from_dict,
    hypothesis_to_dict,
)
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
from evalvitals.eval_agent.stages.hypothesis_tester import (
    HypothesisTestResult,
    _evidence_grade,
)
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel

# ── fixtures ────────────────────────────────────────────────────────────────


def _labeled_batch() -> CaseBatch:
    cases = [FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}"),
                         label=Label.FAIL if i < 2 else Label.PASS) for i in range(4)]
    return CaseBatch(cases)


def _report() -> StatsAnalysisReport:
    """Three signal tools: decisive pope FN, non-decisive attention, decisive
    intervention-derived prompt_contrast repair flag."""
    return StatsAnalysisReport(
        model_name="m",
        findings=[],
        severity="none",
        narrative="",
        raw_results={},
        conclusion="c",
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.8,
                            ci=(0.5, 1.0), reject=True,
                            config={"signal": "pope.false_negative"},
                            summary="pope.false_negative vs FAIL"),
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=-0.3,
                            ci=(-0.8, 0.2), reject=False,
                            config={"signal": "relative_attention.max_relative_weight"},
                            summary="relative_attention.max_relative_weight vs FAIL"),
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.6,
                            ci=(0.4, 0.9), reject=True,
                            config={"signal": "prompt_contrast.fixed_by_describe_first"},
                            summary="prompt_contrast.fixed_by_describe_first vs FAIL"),
        ],
    )


def _hyp(statement: str, mode: str = "mode", design: str = "") -> Hypothesis:
    return Hypothesis(statement=statement, target_model="m",
                      predicted_failure_mode=mode, test_design=design)


# ── evidence grading (P4) ───────────────────────────────────────────────────


def test_evidence_grade_unit():
    assert _evidence_grade("mcnemar_evalue", "") == "intervention"
    assert _evidence_grade("friedman_nemenyi", "") == "intervention"
    assert _evidence_grade("signal_label_assoc", "prompt_contrast.fixed_by_x") == "intervention"
    assert _evidence_grade("signal_label_assoc", "pope.false_negative") == "observational"


def test_invalid_min_grade_raises():
    with pytest.raises(ValueError):
        HypothesisTester(min_evidence_grade="causal")


# ── P3 routing ──────────────────────────────────────────────────────────────


def test_test_design_routes_to_designated_signal():
    # Statement keywords would match pope ("negative"), but the explicit test
    # design names the attention signal — design must win.
    h = _hyp("Failures stem from negative answer bias.",
             design="relative_attention.max_relative_weight")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "test_design"
    assert tr.evidence["chosen_tool"] == "signal_label_assoc"
    assert tr.status == HypothesisStatus.INCONCLUSIVE  # attention CI crosses 0
    assert tr.effect_size == -0.3


def test_design_routing_to_intervention_signal_grades_intervention():
    h = _hyp("Describing first interferes with perception.",
             design="prompt_contrast describe_first repair")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "test_design"
    assert tr.status == HypothesisStatus.SUPPORTED
    assert tr.evidence_grade == "intervention"


def test_keyword_fallback_when_no_design():
    h = _hyp("The model produces false negative answers on presence questions.")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "keywords"
    assert tr.status == HypothesisStatus.SUPPORTED
    assert tr.evidence_grade == "observational"


def test_unavailable_preregistered_design_does_not_use_keyword_substitute():
    h = _hyp(
        "Zero correct samples indicate a visual attention lock-on.",
        design="new_attention_probe.target_region_mass",
    )
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.INCONCLUSIVE
    assert tr.test_name == "stats_results"
    # one canonical marker for "design named evidence M2 did not measure"
    assert tr.evidence["routed_by"] == "test_design_unmet"
    assert tr.evidence_grade == "none"


def test_preregistered_lower_on_failures_direction_supports_negative_effect():
    report = _report()
    report.stats_results[1].reject = True
    report.stats_results[1].ci = (-0.7, -0.1)
    h = Hypothesis(
        statement="Failures have depressed target-region attention.",
        target_model="m",
        predicted_failure_mode="attention_sink",
        test_design="relative_attention.max_relative_weight",
        expected_association="lower_on_failures",
    )
    tr = HypothesisTester().test([h], report, _labeled_batch())[0]
    assert tr.status == HypothesisStatus.SUPPORTED
    assert tr.effect_size == -0.3
    assert tr.evidence["expected_association"] == "lower_on_failures"


def test_descriptive_tool_never_becomes_inconclusive_headline():
    """single_rate_evalue (rate−0.5 = large |effect|) must NOT be surfaced as the
    'best' result when nothing discriminates — that printed a meaningless
    'vs p0=0.50 → reject' as the verdict (defect 6).

    Since "Generalize M2 analysis planning" (commit 5fd27e2), a "no
    discriminating M2 result" verdict (evidence_grade == "none") is not the
    final word: the tester falls through to the rigorous label-free fallback
    (the same per-case ``compare()`` used when M2 supplies no stats_results at
    all) before giving up. That fallback is independent of single_rate_evalue's
    artifact effect, so defect 6 stays fixed either way — here the fallback
    also finds nothing (no per-case signal for this hypothesis in the fixture
    batch), so the final verdict is an honest "insufficient data", never the
    descriptive tool's misleading "reject"."""
    report = StatsAnalysisReport(
        model_name="m", findings=[], severity="none", narrative="",
        raw_results={}, conclusion="c",
        stats_results=[
            # no relevant signal tool matches the hypothesis; only a descriptive
            # global rejection with a huge artifact effect is present.
            StatsToolResult(tool="single_rate_evalue", ok=True, effect=-0.43,
                            e_value=340.0, reject=True, config={"p0": 0.5},
                            summary="FAIL rate 7.0% (105/1500) vs p0=0.50: e=340 -> reject"),
        ],
    )
    h = _hyp("Intermediate layers encode absence before late suppression.",
             mode="late_layer_suppression")
    tr = HypothesisTester().test([h], report, _labeled_batch())[0]
    assert tr.status == HypothesisStatus.INCONCLUSIVE
    assert "reject" not in tr.verdict.lower()
    assert "insufficient" in tr.verdict.lower()
    assert tr.evidence["source"] == "fallback_compare"


# ── P4 stopping tiers ───────────────────────────────────────────────────────


def _result(grade: str, confidence: float = 0.5) -> HypothesisTestResult:
    return HypothesisTestResult(
        hypothesis=_hyp("h"), status=HypothesisStatus.SUPPORTED, test_name="t",
        effect_size=0.5, is_consistent_with_protocol=True,
        confidence=confidence, verdict="v", evidence_grade=grade,
    )


def test_observational_min_grade_keeps_plan_a_behavior():
    tester = HypothesisTester()  # default observational
    assert tester.stopping_criteria_met([_result("observational")]) is True


def test_intervention_min_grade_rejects_observational_support():
    tester = HypothesisTester(min_evidence_grade="intervention")
    assert tester.stopping_criteria_met([_result("observational")]) is False
    assert tester.stopping_criteria_met([_result("intervention")]) is True


def test_best_hypotheses_prefers_intervention_grade():
    tester = HypothesisTester()
    obs = _result("observational", confidence=0.9)
    interv = _result("intervention", confidence=0.6)
    best = tester.best_hypotheses([obs, interv])
    assert best[0].evidence_grade == "intervention"


# ── P3 diagnosis: TEST line + available evidence ────────────────────────────


class ScriptedJudge(FakeModel):
    def __init__(self, answer: str) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.prompts: list[str] = []
        self._answer = answer

    def generate(self, inputs, **kw) -> str:
        self.prompts.append(str(inputs))
        return self._answer


class TwoAnswerJudge(FakeModel):
    def __init__(self, answers: list[str]) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self._answers = iter(answers)

    def generate(self, inputs, **kw) -> str:
        return next(self._answers)


def test_diagnosis_parses_test_line_into_design():
    judge = ScriptedJudge(
        "HYPOTHESIS: the model ignores the image\n"
        "FAILURE_MODE: visual_blindness\n"
        "TEST: relative_attention.max_relative_weight association\n"
        "EXPECTED_ASSOCIATION: lower_on_failures\n"
    )
    diag = DiagnosisAgent(judge=judge).diagnose(_report())
    assert len(diag.hypotheses) == 1
    assert diag.hypotheses[0].test_design == "relative_attention.max_relative_weight association"
    assert diag.hypotheses[0].expected_association == "lower_on_failures"


def test_critic_rejection_annotates_instead_of_deleting():
    """A critic that rejects every proposal used to end the run at
    '0 hypothesis/es' (three llm_benchmark runs on 2026-08-20) with no
    M5/M4/fix. The verdict is provenance now: the hypothesis stays, flagged,
    and the held-out M5 decides."""
    judge = TwoAnswerJudge([
        "HYPOTHESIS: the model ignores visual evidence entirely\n"
        "FAILURE_MODE: visual_blindness\n"
        "TEST: relative_attention.max_relative_weight\n"
        "EXPECTED_ASSOCIATION: lower_on_failures\n",
        "REJECT: the model ignores visual evidence entirely\n"
        "REASON: the cited evidence does not test that mechanism\n",
    ])
    diag = DiagnosisAgent(judge=judge).diagnose(_report())
    assert len(diag.hypotheses) == 1
    h = diag.hypotheses[0]
    assert h.metadata["critic"] == "reject"
    assert h.metadata["critic_reason"] == "the cited evidence does not test that mechanism"
    assert diag.n_critic_rejected == 1 and diag.n_critic_kept == 0
    assert "REJECT:" in diag.critic_raw_output


def test_critic_record_carries_the_proposer_reviewer_shape_too():
    """The same review in the record shape the report/Langfuse side reads:
    the proposer's full list plus a per-hypothesis decision derived from the
    annotations — the reviewer records, it never removes (2026-08-21 merge of
    the two M3-critic designs)."""
    judge = TwoAnswerJudge([
        "HYPOTHESIS: the model ignores visual evidence entirely\n"
        "FAILURE_MODE: visual_blindness\n"
        "HYPOTHESIS: the model truncates long answers\n"
        "FAILURE_MODE: termination\n",
        "KEEP: the model truncates long answers\n"
        "REASON: termination_audit shows it\n"
        "REJECT: the model ignores visual evidence entirely\n"
        "REASON: the cited evidence does not test that mechanism\n",
    ])
    diag = DiagnosisAgent(judge=judge).diagnose(_report())
    assert [h.statement for h in diag.proposed_hypotheses] == [
        "the model ignores visual evidence entirely", "the model truncates long answers"]
    assert [h.statement for h in diag.hypotheses] == [          # kept first, nothing dropped
        "the model truncates long answers", "the model ignores visual evidence entirely"]
    assert diag.review_decisions == [
        {"statement": "the model ignores visual evidence entirely", "decision": "reject",
         "reason": "the cited evidence does not test that mechanism"},
        {"statement": "the model truncates long answers", "decision": "keep",
         "reason": "termination_audit shows it"},
    ]
    assert diag.review_raw == diag.critic_raw_output and diag.review_prompt == diag.critic_prompt
    assert diag.n_critic_kept == 1 and diag.n_critic_rejected == 1


def test_diagnosis_prompt_lists_available_evidence():
    judge = ScriptedJudge("NO_ISSUE")
    DiagnosisAgent(judge=judge).diagnose(_report())
    p = judge.prompts[0]
    assert "AVAILABLE EVIDENCE" in p
    assert "pope.false_negative" in p
    assert "relative_attention.max_relative_weight" in p


def test_hypothesis_test_design_round_trips():
    h = _hyp("s", design="pope.false_negative")
    h.expected_association = "higher_on_failures"
    restored = hypothesis_from_dict(hypothesis_to_dict(h))
    assert restored.test_design == "pope.false_negative"
    assert restored.expected_association == "higher_on_failures"


# ── P3 M1: designs reach cycle-2 selection prompt ───────────────────────────


def test_m1_selection_prompt_includes_test_designs():
    class CapturingJudge(FakeModel):
        def __init__(self) -> None:
            super().__init__(capabilities={Capability.GENERATE})
            self.prompt = ""

        def generate(self, inputs, **kw) -> str:
            self.prompt = str(inputs)
            return '{"analyzers": ["self_consistency"], "rationale": "r"}'

    judge = CapturingJudge()
    model = FakeModel(capabilities={Capability.GENERATE})
    agent = ProbeAgent(judge=judge, max_analyzers=1)
    prior = [_hyp("attention is diffuse", design="run prompt_contrast describe_first")]
    agent.probe(model, _labeled_batch(),
                protocol=ExperimentProtocol(description="d"), prior_hypotheses=prior)
    assert "proposed test: run prompt_contrast describe_first" in judge.prompt

# ── M5 must read the corrected verdict, not the raw one ─────────────────────
def _bh_family_report() -> StatsAnalysisReport:
    """Three signal_label_assoc results as correct_results() leaves them.

    Numbers are qwen3.5-2b/bbh_word_sorting run3's deciding signals. BH over
    the three keeps only changed_answer (p=.0013); cot_sentences (p=.039) and
    max_value_drop (p=.143, n_signal=3) fail. All three carry raw reject=True
    because signal_label_assoc rejects on a bootstrap CI, and a CI over three
    cases cannot straddle zero — that is the arm M5 must not be allowed to read.
    """
    from evalvitals.analysis.stats_tools import fdr_correct

    def _r(signal, effect, p, n, ci):
        return StatsToolResult(
            tool="signal_label_assoc", ok=True, effect=effect, ci=ci, reject=True,
            p_value=p, config={"signal": signal},
            analysis_key=f"signal_label_assoc:{signal}", correction_family="bh",
            raw_reject=True, details={"n_signal": n},
            summary=f"signal '{signal}' vs FAIL: effect=+{effect:.4f} -> REJECT H0",
        )

    results = [
        _r("cot_faithfulness.cot_sentences", 0.5, 0.0391, 12, (0.17, 0.83)),
        _r("self_repair.changed_answer", 0.5431, 0.00133, 9, (0.46, 0.64)),
        _r("step_rollout_value.max_value_drop", 0.8, 0.1429, 3, (0.4, 1.0)),
    ]
    corrected = fdr_correct(results, alpha=0.05)
    assert corrected["rejected_result_keys"] == ["signal_label_assoc:self_repair.changed_answer"]
    return StatsAnalysisReport(
        model_name="m", findings=[], severity="none", narrative="", raw_results={},
        conclusion="c", stats_results=results, corrected_rejections=corrected,
    )


def test_a_signal_bh_killed_cannot_support_a_hypothesis():
    """p=0.143 over three cases printed 'REJECT H0' and M5 said SUPPORTED.

    `reject` stays raw for BH members on purpose (multiplicity.py keeps the
    tool's own verdict visible to the loop); the corrected verdict lives in
    fdr_corrected. M5 read the former.
    """
    h = _hyp("Value drops mid-chain cause the failure.",
             design="step_rollout_value.max_value_drop")
    tr = HypothesisTester().test([h], _bh_family_report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "test_design"
    assert tr.status == HypothesisStatus.INCONCLUSIVE
    assert "[BH: NOT survived, p=0.143, n_signal=3]" in tr.verdict


def test_the_borderline_one_is_not_rescued_by_a_sibling_that_survived():
    """Every signal_label_assoc test shares one tool NAME. A tool-level
    survivor set said 'rejected' for cot_sentences (p=.039, fails BH) because
    changed_answer (p=.0013) survived under the same name."""
    h = _hyp("Longer chains corrupt the set.", design="cot_faithfulness.cot_sentences")
    tr = HypothesisTester().test([h], _bh_family_report(), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.INCONCLUSIVE
    assert "NOT survived" in tr.verdict


def test_the_bh_survivor_still_supports():
    h = _hyp("Self-repair changes the answer on failures.",
             design="self_repair.changed_answer")
    tr = HypothesisTester().test([h], _bh_family_report(), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.SUPPORTED
    assert "[BH: survived, p=0.00133, n_signal=9]" in tr.verdict


def test_an_uncorrected_ci_only_result_keeps_its_own_verdict():
    """Outside every family (no p, no e, correction never ran) the tool's own
    reject flag is the only verdict there is — the pre-existing fixture path."""
    h = _hyp("The model produces false negative answers on presence questions.")
    tr = HypothesisTester().test([h], _report(), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.SUPPORTED
    assert "[BH" not in tr.verdict


# ── routing precision (2026-08-18, qwen3.5-2b/bbh_causal_judgement) ─────────


def _cj_report() -> StatsAnalysisReport:
    """What M2 measured on causal_judgement: a strong PROTECTIVE
    answer_extraction_audit signal plus a cot_faithfulness signal."""
    return StatsAnalysisReport(
        model_name="m", findings=[], severity="none", narrative="", raw_results={},
        conclusion="c",
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=-0.4756,
                            ci=(-0.65, -0.27), reject=True,
                            config={"signal": "answer_extraction_audit.gold_in_answer_region"},
                            summary="gold_in_answer_region vs FAIL"),
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.42,
                            ci=(0.08, 0.75), reject=True,
                            config={"signal": "cot_faithfulness.cot_sentences"},
                            summary="cot_sentences vs FAIL"),
        ],
    )


def test_generic_words_in_a_design_do_not_route_to_an_unrelated_signal():
    """The overthinking hypothesis's design says 'answer_trajectory' and 'answer';
    it must route to cot_faithfulness (its named analyzer), not be REFUTED by
    answer_extraction_audit.gold_in_answer_region via the word 'answer'."""
    h = _hyp("The model reaches the correct answer early and then argues itself out of it.",
             mode="overthinking",
             design="cot_faithfulness paired columns on held-out cases — drift_away vs "
                    "late_rescue and answer_trajectory at the 0.25/0.5/0.75 truncations")
    tr = HypothesisTester().test([h], _cj_report(), _labeled_batch())[0]
    assert tr.evidence["routed_by"] == "test_design"
    assert "cot_sentences" in tr.verdict and "gold_in_answer_region" not in tr.verdict
    assert tr.status == HypothesisStatus.SUPPORTED


def test_design_naming_unmeasured_evidence_stays_inconclusive_not_refuted():
    """A design that names signals M2 did not test ('generated:probe1.answers_yes',
    'self_repair.self_says_incorrect') is not judged on whatever else M2
    measured — it is inconclusive with the designated names in the verdict."""
    h = _hyp("The Yes/No verdict is fixed by the prompt encoding before any deliberation.",
             mode="post_hoc_rationalization",
             design="self_repair.self_says_incorrect x self_repair.changed_answer, plus "
                    "generated:probe1.answers_yes stratified by but-for dependence")
    tr = HypothesisTester().test([h], _cj_report(), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.INCONCLUSIVE
    assert tr.evidence["routed_by"] == "test_design_unmet"
    assert "not measured" in tr.verdict and "self_repair" in tr.verdict


def test_identifier_helpers():
    from evalvitals.eval_agent.stages.hypothesis_tester import (
        _identifiers, _signal_keywords, _tool_ids, _tool_keywords,
    )

    assert _identifiers("Re-run `perturbation_battery` and cot_faithfulness.drift_away") == {
        "perturbation_battery", "cot_faithfulness.drift_away"}
    assert "answer" not in _signal_keywords("the answer region and the gold string")
    assert {"gold", "region", "string"} <= _signal_keywords("the answer region and the gold string")
    r = StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.1, ci=(0, 0.2), reject=False,
                        config={"signal": "answer_extraction_audit.gold_in_answer_region"})
    assert _tool_ids(r, level="analyzer") == {"answer_extraction_audit"}
    assert "gold_in_answer_region" in _tool_ids(r) and "answer_extraction_audit.gold_in_answer_region" in _tool_ids(r)
    assert "answer" not in _tool_keywords(r) and {"gold", "region", "extraction"} <= _tool_keywords(r)


# ── direction-aware verdicts (2026-08-19) ───────────────────────────────────


def _signed_report(signal: str, effect: float) -> StatsAnalysisReport:
    """One decisive (uncorrected, CI-only) signal_label_assoc result."""
    lo, hi = (effect - 0.2, effect + 0.2)
    return StatsAnalysisReport(
        model_name="m", findings=[], severity="none", narrative="", raw_results={},
        conclusion="c",
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=effect,
                            ci=(lo, hi), reject=True, config={"signal": signal},
                            summary=f"{signal} vs FAIL"),
        ],
    )


def test_a_hypothesis_predicting_a_low_signal_is_supported_by_a_negative_effect():
    """minervamath: 'step_rollout_value.initial_value — 5/8 chains sit at 0.0 from
    step 0' predicts the signal LOWER on failures; the observed effect was
    negative (low initial value <-> FAIL). Reading every negative effect as
    'protective -> refuted' rejected the hypothesis the data agreed with."""
    h = _hyp("Set-up failures at step 0: the chain starts wrong.",
             design="corroborate with `step_rollout_value.initial_value` (5/8 chains "
                    "sit at 0.0 from step 0)")
    tr = HypothesisTester().test(
        [h], _signed_report("step_rollout_value.initial_value", -0.67), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.SUPPORTED
    assert tr.evidence["expected_direction"] == "lower_in_fail"
    assert "predicts the signal LOWER on failures" in tr.verdict and "consistent" in tr.verdict


def test_a_hypothesis_predicting_a_low_signal_is_refuted_by_a_positive_effect():
    h = _hyp("x", design="self_consistency.consistency is LOWER on failing cases")
    tr = HypothesisTester().test(
        [h], _signed_report("self_consistency.consistency", +0.5), _labeled_batch())[0]
    assert tr.status == HypothesisStatus.REFUTED
    assert "opposite" in tr.verdict


def test_without_a_direction_cue_the_classic_reading_stands():
    """No cue: effect > 0 (signal group fails more) supports, < 0 refutes."""
    h = _hyp("Repetition marks the failures.", design="termination_audit.repetition_score")
    up = HypothesisTester().test(
        [h], _signed_report("termination_audit.repetition_score", +0.4), _labeled_batch())[0]
    down = HypothesisTester().test(
        [h], _signed_report("termination_audit.repetition_score", -0.4), _labeled_batch())[0]
    assert up.status == HypothesisStatus.SUPPORTED and up.evidence["expected_direction"] is None
    assert down.status == HypothesisStatus.REFUTED


def test_expected_sign_parser_cases():
    from evalvitals.eval_agent.stages.hypothesis_tester import _expected_sign

    def sign(design: str, signal: str, statement: str = "x"):
        return _expected_sign(_hyp(statement, design=design), signal)

    assert sign("`gold_in_output` = 1 with `gold_in_answer_region` = 0",
                "answer_extraction_audit.gold_in_output") == 1
    assert sign("`gold_in_output` = 1 with `gold_in_answer_region` = 0",
                "answer_extraction_audit.gold_in_answer_region") == -1
    assert sign("termination_audit.repetition_score should be HIGHER on failing cases",
                "termination_audit.repetition_score") == 1
    assert sign("signal 'self_repair.self_says_incorrect' absent on FAIL",
                "self_repair.self_says_incorrect") == -1
    assert sign("`foo.bar` = 0.5 vs 1", "foo.bar") is None            # a threshold, not a level
    assert sign("check perturbation_battery.noop_clause_flipped",
                "perturbation_battery.noop_clause_flipped") is None
    assert sign("coverage_verification_gap.majority_share (>=0.6 vs <=0.4)",
                "coverage_verification_gap.majority_share") is None
    assert sign("", "foo.bar") is None


def test_diagnosis_prompt_asks_for_the_direction():
    from evalvitals.eval_agent.prompts.diagnosis import _DIAGNOSE_PROMPT
    assert "HIGHER or LOWER on failing cases" in _DIAGNOSE_PROMPT
    # binary tasks: TEST lines go to the direction marginals, not to text/label fields
    assert "answer_extraction_audit.answered_yes" in _DIAGNOSE_PROMPT
    assert "never on\nextracted_answer, labelled_fail" in _DIAGNOSE_PROMPT


def test_an_outcome_regrade_cannot_be_m5_evidence_even_from_an_old_m2():
    """h3 on minervamath named `gold_in_answer_region` = 0; the old M2 results
    still carry that column (84% label copy). It must not decide the verdict
    in either direction — the design is unmet, not refuted / supported."""
    h = _hyp("A minority of FAILs are last-step target errors",
             design="subgroup on `answer_extraction_audit.gold_in_output == 1 & "
                    "gold_in_answer_region == 0`, then a targeted re-ask")
    tr = HypothesisTester().test(
        [h], _signed_report("answer_extraction_audit.gold_in_answer_region", -0.63),
        _labeled_batch())[0]
    assert tr.status == HypothesisStatus.INCONCLUSIVE
    assert tr.evidence["routed_by"] == "test_design_unmet"
