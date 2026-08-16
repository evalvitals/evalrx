"""Behavioral tests for the 2026-08 reasoning probes.

Each test scripts a model with a KNOWN pathology and asserts the probe's
columns expose it — the structural half lives in test_analyzer_contract.py.
The probes exist to keep pathologies APART, so most tests here pin a
discrimination (slip vs chain break, repair vs damage, coverage vs selection)
rather than a single number.
"""

from __future__ import annotations

import pytest

from evalvitals.analyzers.perturbation.perturbation_battery import (
    PerturbationBattery,
    append_noop_clause,
    perturb_numbers,
    rename_entities,
)
from evalvitals.analyzers.reasoning._text import (
    answer_equal,
    extract_answer,
    find_equations,
    looks_like_give_up,
    looks_truncated,
    normalize_answer,
    repetition_score,
    safe_eval_arithmetic,
)
from evalvitals.analyzers.reasoning.answer_extraction_audit import AnswerExtractionAudit
from evalvitals.analyzers.reasoning.arith_audit import ArithmeticAudit
from evalvitals.analyzers.reasoning.contamination import ContaminationProbe, overlap_score
from evalvitals.analyzers.reasoning.knowledge_split import KnowledgeReasoningSplit
from evalvitals.analyzers.reasoning.self_repair import SelfRepairAnalyzer, _parse_verdict
from evalvitals.analyzers.reasoning.step_rollout_value import (
    StepRolloutValueAnalyzer,
    split_steps,
)
from evalvitals.analyzers.reasoning.termination_audit import TerminationAudit
from evalvitals.analyzers.uncertainty.coverage_gap import CoverageVerificationGap, pass_at_k
from evalvitals.analyzers.uncertainty.self_consistency import (
    SelfConsistencyAnalyzer,
    cluster_by_equivalence,
    lexical_equivalent,
)
from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model


class ScriptModel(Model):
    """generate() answers by cycling a script and records every prompt."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, answers):
        self._answers = list(answers)
        self._i = 0
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs):
        self.prompts.append(str(inputs))
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer

    def logprobs(self, inputs, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def forward(self, inputs, capture, spec=None):  # pragma: no cover
        raise NotImplementedError


class NoCapModel(ScriptModel):
    capabilities = frozenset()


def _case(prompt="q", observed=None, expected=None, label=Label.UNKNOWN, **kw):
    return FailureCase(
        inputs=Inputs(prompt=prompt),
        observed=observed,
        expected=expected,
        label=label,
        **kw,
    )


# ── shared helpers ────────────────────────────────────────────────────────────
def test_extract_answer_takes_the_last_marker_by_position_then_last_line():
    """Whichever of ``\\boxed{}`` / ``Answer:`` comes LAST wins — not boxed-always.

    This used to be a fixed ladder (drain every box, only then look at tags), so
    a chain that boxed its intermediate working outranked its own final answer
    line.  Measured on minervamath / Qwen3.5-9B: 28 of 272 were scored FAIL that
    way (gold ``2.45e6``, the model closed with ``Answer: 2.45e6``, extraction
    returned a mid-chain ``7.353e14``), and regrading flipped 28 FAIL->PASS with
    zero PASS->FAIL.
    """
    assert extract_answer("work \\boxed{18} more work\nAnswer: 3") == "3"
    assert extract_answer("Answer: 3\nwork \\boxed{18}") == "18"
    assert extract_answer("blah\nAnswer: 42") == "42"
    assert extract_answer("only a line") == "only a line"


def test_extract_answer_keeps_a_bare_option_label():
    """``(A)`` is a multiple-choice ANSWER, not an echo of the format hint.

    The placeholder guard that skips ``Answer: <answer>`` matched ``Answer: (A)``
    just as well, so extraction walked back into the chain-of-thought and picked
    up the prose there.  Measured on BBH tracking_shuffled_objects_seven_objects
    / Qwen3.5-9B: 244 of 250 final claims are a bare ``(X)``, 99 correct answers
    scored FAIL, and the slice read 0.592 instead of 0.988.
    """
    assert extract_answer("Claire is dancing with **Lola**.\n\nAnswer: (A)") == "(A)"
    assert extract_answer("Answer: (D)") == "(D)"
    assert extract_answer("Answer: B.") == "B."
    # a genuine format echo is still skipped
    assert extract_answer("Answer: 42\nremember to write Answer: <answer>") == "42"
    assert extract_answer("Answer: 42\nremember to write Answer: {answer}") == "42"


def test_answer_equal_is_numeric_aware():
    # substring matching would accept 180 for a gold of 18 — this must not
    assert not answer_equal("the answer is 180", "18")
    assert answer_equal("the answer is 18", "18")
    assert answer_equal("1,234", "1234")
    # short golds need a standalone token, not a substring of a word
    assert not answer_equal("probably", "b")
    assert answer_equal("Answer: B", "B")


def test_safe_eval_rejects_non_arithmetic():
    assert safe_eval_arithmetic("2 + 3 * 4") == 14.0
    assert safe_eval_arithmetic("__import__('os').system('ls')") is None
    assert safe_eval_arithmetic("x + 1") is None


def test_find_equations_skips_definitions_and_prose_numbers():
    found = find_equations("First 3 + 4 = 7 then 7 * 6 = 41. In 2020 he had 5 pears. x = 9")
    assert [expr for expr, _, _ in found] == ["3 + 4 = 7", "7 * 6 = 41"]
    assert found[1][1] == 41.0 and found[1][2] == 42.0  # stated vs computed


def test_repetition_and_termination_helpers():
    assert repetition_score("a b c d e f g h " * 5, 8) > 0.5
    assert repetition_score("the quick brown fox jumps over the lazy dog again", 8) == 0.0
    assert looks_truncated("I think the next step is")
    assert not looks_truncated("Answer: 5")
    assert looks_like_give_up("I cannot determine the answer")
    assert not looks_like_give_up("The answer is 5.")


def test_normalize_answer_strips_decoration():
    assert normalize_answer("$1,234.00**") == "1234.00"
    assert normalize_answer("The Answer") == "answer"


# ── answer_extraction_audit ───────────────────────────────────────────────────
def test_extraction_audit_separates_parse_miss_from_real_failure():
    batch = CaseBatch([
        # boxed answer a strict harness regex would miss -> suspect
        _case("q1", r"So we get \boxed{12}.", "12", Label.FAIL),
        # genuinely wrong -> not suspect
        _case("q2", "Answer: 15", "12", Label.FAIL),
        # gold present only as an intermediate quantity -> loose flags, region does not
        _case("q3", "First 12 boxes counted. " + "filler " * 60 + "\nAnswer: 15", "12", Label.FAIL),
    ])
    f = AnswerExtractionAudit().run(ScriptModel(["x"]), batch).findings
    suspects = [c["extraction_suspect"] for c in f["per_case"]]
    assert suspects == [1, 0, 0]
    assert f["per_case"][2]["gold_in_output"] == 1  # the loose bound over-counts
    assert f["suspect_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_extraction_audit_reask_needs_generate():
    batch = CaseBatch([_case("q", r"\boxed{12}", "12", Label.FAIL)])
    f = AnswerExtractionAudit(reask=True).run(ScriptModel(["12"]), batch).findings
    assert f["n_reasked"] == 1 and f["n_reask_confirmed"] == 1
    with pytest.raises(CapabilityError):
        AnswerExtractionAudit(reask=True).run(NoCapModel(["12"]), batch)


def test_extraction_audit_runs_without_generate_by_default():
    batch = CaseBatch([_case("q", "Answer: 12", "12", Label.PASS)])
    f = AnswerExtractionAudit().run(NoCapModel([]), batch).findings
    assert f["n_gradable"] == 1


# ── termination_audit ─────────────────────────────────────────────────────────
def test_termination_audit_classifies_each_stop_reason():
    batch = CaseBatch([
        _case("q1", "Step one. Step two.\nAnswer: 4", "4", Label.PASS),
        _case("q2", "I start by computing the value of", "4", Label.FAIL),
        _case("q3", "a b c d e f g h " * 6, "4", Label.FAIL),
        _case("q4", "I cannot determine the answer.", "4", Label.FAIL),
    ])
    f = TerminationAudit().run(ScriptModel(["...done\nAnswer: 4"]), batch).findings
    assert [c["termination_class"] for c in f["per_case"]] == [
        "clean", "truncated", "degenerate", "gave_up",
    ]
    assert f["clean_rate"] == 0.25
    # the continuation reaches the gold, so the non-clean cases were budget-bound
    assert f["recovered_rate"] == 1.0


def test_termination_audit_degeneration_wins_over_truncation():
    """A looping generation is also cut off; the loop is the actionable cause."""
    batch = CaseBatch([_case("q", "x y z w " * 20, "4", Label.FAIL)])
    f = TerminationAudit(continue_non_clean=False).run(ScriptModel(["_"]), batch).findings
    assert f["per_case"][0]["termination_class"] == "degenerate"
    assert f["per_case"][0]["looks_truncated"] == 1


# ── arith_audit ───────────────────────────────────────────────────────────────
def test_arith_audit_flags_computation_slip():
    # 3 * 4 stated as 13; repairing it lands exactly on the gold
    batch = CaseBatch([_case("q", "He has 3 * 4 = 13 pens.\nAnswer: 13", "12", Label.FAIL)])
    f = ArithmeticAudit().run(ScriptModel(["x"]), batch).findings
    entry = f["per_case"][0]
    assert entry["error_class"] == "computation_slip"
    assert entry["slip_explains_final"] == 1
    assert entry["first_error_stated"] == 13.0 and entry["first_error_correct"] == 12.0
    assert f["computation_slip_rate"] == 1.0 and f["chain_break_rate"] == 0.0


def test_arith_audit_flags_chain_break_when_arithmetic_is_clean():
    batch = CaseBatch([_case("q", "3 + 4 = 7 so the total is 7.\nAnswer: 7", "12", Label.FAIL)])
    f = ArithmeticAudit().run(ScriptModel(["x"]), batch).findings
    assert f["per_case"][0]["error_class"] == "chain_break"
    assert f["chain_break_rate"] == 1.0


def test_arith_audit_flags_wrong_arithmetic_with_a_right_answer():
    batch = CaseBatch([_case("q", "2 * 2 = 5 anyway.\nAnswer: 12", "12", Label.PASS)])
    f = ArithmeticAudit().run(ScriptModel(["x"]), batch).findings
    assert f["per_case"][0]["error_class"] == "errors_but_correct"
    assert f["n_errors_but_correct"] == 1


def test_arith_audit_reports_unmeasured_when_no_equations_written():
    batch = CaseBatch([_case("q", "I just know it.\nAnswer: 7", "12", Label.FAIL)])
    f = ArithmeticAudit().run(ScriptModel(["x"]), batch).findings
    assert f["per_case"][0]["error_class"] == "unknown"
    assert f["n_with_equations"] == 0


# ── self_repair ───────────────────────────────────────────────────────────────
def test_parse_verdict_handles_negation():
    assert _parse_verdict("INCORRECT") is True
    assert _parse_verdict("CORRECT") is False
    assert _parse_verdict("This is not correct") is True   # naive search reads this backwards
    assert _parse_verdict("no verdict here") is None


def test_self_repair_reports_damage_alongside_repair():
    batch = CaseBatch([
        _case("q1", "Answer: 5", "12", Label.FAIL),
        _case("q2", "Answer: 7", "7", Label.PASS),
    ])
    model = ScriptModel([
        "INCORRECT", "Answer: 12",   # case 1: detected, repaired
        "INCORRECT", "Answer: 9",    # case 2: false alarm, damaged
    ])
    f = SelfRepairAnalyzer().run(model, batch).findings
    assert f["repair_rate"] == 1.0
    assert f["damage_rate"] == 1.0
    assert f["net_revision_gain"] == 0.0      # the headline repair rate is a mirage
    assert f["false_alarm_rate"] == 1.0
    assert f["detection_accuracy"] == 0.5


def test_self_repair_damage_rate_is_none_without_pass_cases():
    batch = CaseBatch([_case("q", "Answer: 5", "12", Label.FAIL)])
    f = SelfRepairAnalyzer().run(ScriptModel(["INCORRECT", "Answer: 12"]), batch).findings
    assert f["damage_rate"] is None and f["repair_rate"] == 1.0


# ── step_rollout_value ────────────────────────────────────────────────────────
def test_split_steps_prefers_lines_and_drops_the_answer_line():
    steps = split_steps("First step.\nSecond step.\nAnswer: 4", max_steps=7)
    assert steps == ["First step.", "Second step."]
    assert len(split_steps("\n".join(f"s{i}" for i in range(20)), max_steps=5)) == 5


def test_step_rollout_locates_the_break_step():
    chain = "Step one.\nStep two.\nStep three.\nAnswer: 4"
    batch = CaseBatch([_case("q", chain, "4", Label.FAIL)])
    # rollouts from step 1 all reach the gold; from steps 2-3 none do
    model = ScriptModel([
        "Answer: 4", "Answer: 4",
        "Answer: 9", "Answer: 9",
        "Answer: 9", "Answer: 9",
    ])
    f = StepRolloutValueAnalyzer(n_rollouts=2, gen_kwargs={"temperature": 0.8}).run(
        model, batch
    ).findings
    entry = f["per_case"][0]
    assert entry["step_values"] == [1.0, 0.0, 0.0]
    assert entry["break_step_idx"] == 1        # the value collapses entering step 2
    assert entry["recoverable"] == 0


def test_step_rollout_skips_ungraded_cases():
    f = StepRolloutValueAnalyzer().run(
        ScriptModel(["x"]), CaseBatch([_case("q", "chain", None, Label.FAIL)])
    ).findings
    assert f["per_case"][0]["skipped"] == "no gold answer"


# ── knowledge_reasoning_split ─────────────────────────────────────────────────
def test_knowledge_split_calls_it_reasoning_when_own_facts_rescue_it():
    batch = CaseBatch([_case("q", None, "Paris", Label.FAIL)])
    model = ScriptModel([
        "Answer: Lyon",     # baseline wrong
        "Answer: Lyon",     # decomposed still wrong
        "- France's capital is Paris",  # fact recall
        "Answer: Paris",    # answering from its OWN facts works
    ])
    f = KnowledgeReasoningSplit(use_context=False).run(model, batch).findings
    entry = f["per_case"][0]
    assert entry["deficit_class"] == "reasoning"
    assert entry["own_facts_correct"] == 1 and entry["baseline_correct"] == 0
    assert f["own_facts_gain"] == 1.0


def test_knowledge_split_calls_it_knowledge_when_only_the_context_helps():
    batch = CaseBatch([
        _case("q", None, "Paris", Label.FAIL, metadata={"context": "The capital is Paris."})
    ])
    model = ScriptModel([
        "Answer: Lyon", "Answer: Lyon",
        "- some unrelated fact", "Answer: Lyon",
        "Answer: Paris",  # open book
    ])
    f = KnowledgeReasoningSplit().run(model, batch).findings
    assert f["per_case"][0]["deficit_class"] == "knowledge"
    assert f["per_case"][0]["open_book_correct"] == 1


# ── contamination_score ───────────────────────────────────────────────────────
def test_overlap_score_is_recall_oriented():
    assert overlap_score("a b c d", "a b c d", 2) == 1.0
    assert overlap_score("prefix a b c d suffix", "a b c d", 2) == 1.0  # verbosity must not hide it
    assert overlap_score("totally different words here", "a b c d", 2) == 0.0


def test_contamination_flags_guided_reconstruction():
    prompt = "the quick brown fox jumps over the lazy dog near the river bank at dawn"
    batch = CaseBatch([_case(prompt, None, None, Label.PASS)])
    model = ScriptModel([
        "something else entirely different",        # general arm: no idea
        "the lazy dog near the river bank at dawn",  # guided arm: verbatim
    ])
    f = ContaminationProbe(dataset_name="FakeBench", ngram=3).run(model, batch).findings
    entry = f["per_case"][0]
    assert entry["guided_overlap"] > entry["general_overlap"]
    assert entry["verbatim_flag"] == 1
    assert f["accuracy_on_flagged"] == 1.0


def test_contamination_skips_short_prompts():
    f = ContaminationProbe().run(ScriptModel(["x"]), CaseBatch([_case("tiny")])).findings
    assert f["per_case"][0]["skipped"].startswith("prompt too short")


# ── perturbation_battery ──────────────────────────────────────────────────────
_WORD_PROBLEM = "Alice has 5 apples and Bob gives her 3 more. How many apples does Alice have?"


def test_perturbation_helpers_apply_the_intended_edit():
    renamed = rename_entities(_WORD_PROBLEM)
    assert "Alice" not in renamed and "apples" in renamed
    assert perturb_numbers(_WORD_PROBLEM) != _WORD_PROBLEM
    assert "5" not in perturb_numbers(_WORD_PROBLEM).split("apples")[0]
    assert append_noop_clause(_WORD_PROBLEM).endswith("archived.")
    assert rename_entities("no names here, only words") is None


def test_perturbation_battery_flags_memorization():
    """Answering 8 even after the numbers change is recall, not computation."""
    batch = CaseBatch([_case(_WORD_PROBLEM, None, "8", Label.PASS)])
    f = PerturbationBattery().run(ScriptModel(["Answer: 8"]), batch).findings
    entry = f["per_case"][0]
    assert entry["invariance_break_rate"] == 0.0
    assert entry["sensitivity_rate"] == 0.0
    assert entry["memorization_suspect"] == 1
    assert f["n_memorization_suspect"] == 1


def test_perturbation_battery_flags_invariance_break():
    # baseline 8, then every perturbed variant answers differently
    model = ScriptModel(["Answer: 8", "Answer: 1", "Answer: 2", "Answer: 3", "Answer: 4"])
    batch = CaseBatch([_case(_WORD_PROBLEM, None, "8", Label.PASS)])
    f = PerturbationBattery().run(model, batch).findings
    entry = f["per_case"][0]
    assert entry["invariance_break_rate"] == 1.0
    assert entry["noop_clause_flipped"] == 1
    assert entry["sensitivity_rate"] == 1.0        # correct behaviour on the altering arm
    assert "memorization_suspect" not in entry


# ── coverage_verification_gap ─────────────────────────────────────────────────
def test_pass_at_k_estimator():
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(5, 0, 5) == 0.0
    assert 0.0 < pass_at_k(5, 1, 2) < 1.0


def test_coverage_gap_separates_selection_from_capability():
    batch = CaseBatch([_case("q", None, "12", Label.FAIL)])
    model = ScriptModel(["Answer: 9", "Answer: 9", "Answer: 12", "Answer: 9", "Answer: 8"])
    f = CoverageVerificationGap(k=5, gen_kwargs={"temperature": 0.8}).run(model, batch).findings
    entry = f["per_case"][0]
    assert entry["pass_at_k"] == 1.0          # the answer WAS produced
    assert entry["majority_correct"] == 0     # and the vote missed it
    assert entry["coverage_gap"] == 1
    assert f["degenerate_sampling"] is False


def test_coverage_gap_reports_degenerate_sampling():
    batch = CaseBatch([_case("q", None, "12", Label.FAIL)])
    f = CoverageVerificationGap(k=3).run(ScriptModel(["Answer: 9"]), batch).findings
    assert f["degenerate_sampling"] is True
    assert f["per_case"][0]["coverage_gap"] == 0   # structurally 0, not measured


# ── self_consistency (semantic-entropy merge) ─────────────────────────────────
def test_lexical_equivalence_is_bidirectional():
    assert lexical_equivalent("18 apples", "18 apples")
    # one-way containment would merge a short answer into any longer one
    assert not lexical_equivalent("18", "18 apples were left over after lunch")


def test_semantic_clustering_merges_paraphrases():
    samples = [
        "The answer is 18 apples",
        "18 apples is the answer",
        "The answer is 18 apples",
        "Twenty two",
        "The answer is 18 apples",
    ]
    clusters = cluster_by_equivalence(samples, lexical_equivalent)
    assert len(clusters) == 2
    f = SelfConsistencyAnalyzer(n=5).run(
        ScriptModel(samples), CaseBatch([_case("q")])
    ).findings
    assert f["consistency"] == 0.6           # surface agreement understates it
    assert f["semantic_consistency"] == 0.8  # meaning agreement
    assert f["n_semantic_clusters"] == 2
    assert 0.0 < f["normalized_semantic_entropy"] < 1.0


def test_semantic_columns_are_opt_out():
    f = SelfConsistencyAnalyzer(n=2, semantic=False).run(
        ScriptModel(["a", "b"]), CaseBatch([_case("q")])
    ).findings
    assert "semantic_entropy" not in f
    assert f["consistency"] == 0.5


def test_self_repair_survives_an_unparseable_critique():
    """A real model answers the critique in prose sometimes; that must not crash
    the rates or be silently counted as 'said correct'."""
    batch = CaseBatch([
        _case("q1", "Answer: 5", "12", Label.FAIL),
        _case("q2", "Answer: 7", "7", Label.PASS),
    ])
    model = ScriptModel([
        "Well, it depends on how you look at it.", "Answer: 12",   # unparsed verdict
        "CORRECT", "Answer: 7",
    ])
    f = SelfRepairAnalyzer().run(model, batch).findings
    assert f["n_critique_unparsed"] == 1
    assert f["detection_accuracy"] == 1.0   # only the parseable verdict counts
    assert f["false_alarm_rate"] == 0.0
    assert f["repair_rate"] == 1.0 and f["damage_rate"] == 0.0


# ── robustness: real models return things fixtures never do ───────────────────
_ADVERSARIAL_OUTPUTS = [
    "",                                   # empty completion (budget or backend hiccup)
    "   \n\n  ",                          # whitespace only
    "I cannot answer that.",              # refusal, no tag
    "x " * 3000,                          # degenerate repetition
    "Answer:",                            # tag with nothing after it
    "The answer is the answer",           # self-referential prose
    "\\boxed{}",                          # empty box
    "答案是 42",                            # non-latin script
    "Answer: 1/0 and 2 / 0 = inf",        # division by zero inside an equation
    "5 + 5 = 10 = 10 = 10",               # chained equalities
]


class AdversarialModel(ScriptModel):
    capabilities = frozenset({Capability.GENERATE})

    def __init__(self):
        super().__init__(_ADVERSARIAL_OUTPUTS)


@pytest.mark.parametrize(
    "analyzer",
    [
        AnswerExtractionAudit(reask=True),
        TerminationAudit(max_cases=6),
        ArithmeticAudit(generate_missing=True),
        SelfRepairAnalyzer(max_cases=4, revise_with_critique=True),
        StepRolloutValueAnalyzer(n_rollouts=2, max_cases=2),
        KnowledgeReasoningSplit(max_cases=3),
        ContaminationProbe(dataset_name="FakeBench", max_cases=4),
        PerturbationBattery(max_cases=3),
        CoverageVerificationGap(k=3, max_cases=3),
    ],
    ids=lambda a: a.name,
)
def test_probes_survive_adversarial_outputs(analyzer):
    """Every probe must produce JSON-serialisable findings on garbage output.

    Empty completions, refusals, repetition loops and division by zero are what
    an endpoint actually returns under load — a probe that raises on them takes
    down the whole M1 stage, not just its own column.
    """
    import json

    batch = CaseBatch([
        _case(f"Alice has {i} apples and Bob gives her {i + 1} more. How many?",
              observed=text, expected="8",
              label=Label.FAIL if i % 2 else Label.PASS)
        for i, text in enumerate(_ADVERSARIAL_OUTPUTS)
    ])
    findings = analyzer.run(AdversarialModel(), batch).findings
    json.dumps(findings)  # must not raise
    assert isinstance(findings.get("per_case"), list)


# ── extraction regressions found by a live model run ──────────────────────────
def test_extract_answer_skips_the_instruction_placeholder():
    """Models restate the requested format; that echo is LAST and would win."""
    text = "...give it as 'Answer: <answer>'.\nSo we compute.\nAnswer: 25"
    assert extract_answer(text) == "25"
    assert extract_answer(r"\boxed{}  and later \boxed{7}") == "7"


def test_answer_equal_checks_both_ends_of_the_span():
    # tagged span that STARTS with the answer and trails commentary
    assert answer_equal("620, since 12 beds are broken", "620")
    # bare sentence ENDING in the answer
    assert answer_equal("the answer is 18", "18")
    # and still no substring match
    assert not answer_equal("the answer is 180", "18")


def test_knowledge_split_reports_unmeasurable_rather_than_zero():
    """Without gold context the open-book arm never runs, so 'knowledge' cannot
    be separated from 'both'. A 0.0 share there would read as a measurement."""
    batch = CaseBatch([_case(f"q{i}", None, "Paris", Label.FAIL) for i in range(3)])
    model = ScriptModel(["Answer: Lyon"])   # every arm wrong, no context available
    f = KnowledgeReasoningSplit().run(model, batch).findings
    assert f["n_baseline_fail"] == 3
    assert f["n_classified"] == 0 and f["unclassified_share"] == 1.0
    # None = "this batch cannot answer the question", not "measured zero"
    assert f["knowledge_deficit_share"] is None
    assert f["both_deficit_share"] is None


def test_knowledge_split_shares_are_over_classified_failures():
    batch = CaseBatch([
        _case("q1", None, "Paris", Label.FAIL, metadata={"context": "The capital is Paris."}),
        _case("q2", None, "Paris", Label.FAIL),   # no context -> unclassifiable
    ])
    model = ScriptModel([
        "Answer: Lyon", "Answer: Lyon", "- a fact", "Answer: Lyon", "Answer: Paris",
        "Answer: Lyon", "Answer: Lyon", "- a fact", "Answer: Lyon",
    ])
    f = KnowledgeReasoningSplit().run(model, batch).findings
    assert f["n_classified"] == 1 and f["n_unclassified"] == 1
    assert f["knowledge_deficit_share"] == 1.0   # 1 of 1 CLASSIFIED, not 1 of 2
