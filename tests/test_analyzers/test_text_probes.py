"""Behavioral tests for the 2026-08 text probes.

Each test scripts a model with a KNOWN pathology and asserts the probe's
columns expose it — the structural half lives in test_analyzer_contract.py.
"""

from __future__ import annotations

from evalrx.analyzers.hallucination.selfcheck import (
    SelfCheckConsistencyAnalyzer,
    containment,
    split_sentences,
)
from evalrx.analyzers.perturbation.context_shap import ContextShapAnalyzer
from evalrx.analyzers.perturbation.cot_faithfulness import (
    CoTFaithfulnessAnalyzer,
    default_answer_fn,
)
from evalrx.analyzers.perturbation.format_sensitivity import (
    FormatSensitivityAnalyzer,
    extract_options,
    parse_choice,
)
from evalrx.analyzers.uncertainty.calibration import (
    CalibrationAnalyzer,
    expected_calibration_error,
)
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model, TokenLogprob


class ScriptModel(Model):
    """generate() answers by cycling a script; logprobs() is constant."""

    capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
    modalities = frozenset({"text"})

    def __init__(self, answers, logprob=-0.25):
        self._answers = list(answers)
        self._logprob = logprob
        self._i = 0
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs):
        self.prompts.append(str(inputs))
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer

    def logprobs(self, inputs, **kwargs):
        return [TokenLogprob(token="t", logprob=self._logprob, top={"t": self._logprob})]

    def forward(self, inputs, capture, spec=None):  # pragma: no cover
        raise NotImplementedError


class PromptAwareModel(ScriptModel):
    """Answers 'red' only when the supporting fact is present in the prompt."""

    def generate(self, inputs, **kwargs):
        prompt = str(inputs)
        self.prompts.append(prompt)
        return "red" if "red car" in prompt else "unknown"


# ── selfcheck_consistency ─────────────────────────────────────────────────────
def test_selfcheck_flags_unsupported_sentence():
    case = FailureCase(
        inputs=Inputs(prompt="Tell me about Paris."),
        observed="Paris is in France. The city was founded on the moon.",
        label=Label.FAIL,
    )
    model = ScriptModel(["Paris is in France. It is a large capital city."])
    f = SelfCheckConsistencyAnalyzer(n_samples=3).run(model, CaseBatch([case])).findings
    entry = f["per_case"][0]
    assert entry["baseline_from"] == "observed"
    assert entry["n_sentences"] == 2
    # the moon sentence has no support in the resamples; the France one does
    assert entry["selfcheck_worst_sentence"] > entry["selfcheck_inconsistency"] - 1e-9
    assert "moon" in entry["worst_sentence_text"]


def test_selfcheck_consistent_answer_scores_low():
    case = FailureCase(inputs=Inputs(prompt="q"), observed="Paris is in France.", label=Label.PASS)
    model = ScriptModel(["Paris is in France."])
    f = SelfCheckConsistencyAnalyzer(n_samples=2).run(model, CaseBatch([case])).findings
    assert f["per_case"][0]["selfcheck_inconsistency"] == 0.0


def test_selfcheck_helpers():
    assert split_sentences("Short. This sentence is long enough.") == [
        "This sentence is long enough."
    ]
    assert containment("red car", "a red car parked") == 1.0
    assert abs(containment("the red car", "a red car parked") - 2 / 3) < 1e-9


# ── format_sensitivity ────────────────────────────────────────────────────────
def _mc_case():
    return FailureCase(
        inputs=Inputs(prompt="Pick one.\nA. apple\nB. banana\nC. cherry\nReply with the letter."),
        observed="A",
        label=Label.FAIL,
    )


def test_format_sensitivity_detects_letter_sticky_model():
    model = ScriptModel(["A"])  # always answers 'A' whatever the rotation
    f = FormatSensitivityAnalyzer(n_variants=2).run(model, CaseBatch([_mc_case()])).findings
    entry = f["per_case"][0]
    assert entry["positional_bias"] == 1.0       # same letter every time
    assert entry["format_flip_rate"] > 0.5       # content changes under it


def test_format_sensitivity_content_tracking_model():
    # rotations shift apple to positions C (shift 1) and B (shift 2)
    model = ScriptModel(["C", "B"])
    f = FormatSensitivityAnalyzer(n_variants=2).run(model, CaseBatch([_mc_case()])).findings
    entry = f["per_case"][0]
    assert entry["format_flip_rate"] == 0.0      # always the same CONTENT (apple)
    assert entry["positional_bias"] < 1.0


def test_format_sensitivity_skips_without_options():
    case = FailureCase(inputs=Inputs(prompt="open question, no options"), label=Label.FAIL)
    f = FormatSensitivityAnalyzer().run(ScriptModel(["x"]), CaseBatch([case])).findings
    assert f["per_case"][0]["skipped"]
    assert f["mean_flip_rate"] is None


def test_format_helpers():
    options, block = extract_options(_mc_case())
    assert options == ["apple", "banana", "cherry"]
    assert block.startswith("A. apple") and block.endswith("C. cherry")
    assert parse_choice("The answer is B", 3) == "B"
    assert parse_choice("Answer: C", 3) == "C"
    assert parse_choice("no letter here", 3) == ""


def test_parse_choice_review_regressions():
    # tagged answer beats a trailing article-'A' (adversarial-review finding)
    assert parse_choice("Answer: B. That is a fact.", 4) == "B"
    # article 'A' followed by a lowercase word is prose, not a choice
    assert parse_choice("blue. A nice colour.", 2) == ""
    # letter ranges beyond 26 options must not crash the regex
    assert parse_choice("The answer is B", 28) == "B"


def test_extract_option_block_ignores_fewshot_blocks():
    from evalrx.analyzers.perturbation.format_sensitivity import extract_option_block
    prompt = (
        "Example:\nA. cat\nB. dog\n\nNow the real question:\n"
        "Which city?\nA. Paris\nB. London\nReply with the letter."
    )
    options, block = extract_option_block(prompt)
    assert options == ["Paris", "London"]
    assert "Paris" in block and "cat" not in block


# ── cot_faithfulness ──────────────────────────────────────────────────────────
def test_cot_faithfulness_posthoc_chain():
    # early answers always equal the full-CoT answer -> unfaithful signature
    model = ScriptModel([
        "4",                                                    # direct
        "First step reasoning. Second step reasoning.\nAnswer: 4",  # full CoT
        "Answer: 4", "Answer: 4",                               # early answers
    ])
    f = CoTFaithfulnessAnalyzer(truncation_fracs=(0.5, 1.0)).run(
        model, CaseBatch([_mc_case()])
    ).findings
    entry = f["per_case"][0]
    assert entry["early_answer_match_rate"] == 1.0
    assert entry["cot_changed_answer"] == 0


def test_cot_faithfulness_loadbearing_chain():
    model = ScriptModel([
        "3",                                                    # direct
        "Long derivation happens here. It flips the result.\nAnswer: 4",
        "Answer: 3", "Answer: 3",                               # early answers differ
    ])
    f = CoTFaithfulnessAnalyzer(truncation_fracs=(0.5, 1.0)).run(
        model, CaseBatch([_mc_case()])
    ).findings
    entry = f["per_case"][0]
    assert entry["early_answer_match_rate"] == 0.0
    assert entry["cot_changed_answer"] == 1


def test_default_answer_fn():
    assert default_answer_fn("blah\nAnswer: 42") == "42"
    assert default_answer_fn("only a line") == "only a line"


def _graded_case(expected):
    return FailureCase(
        inputs=Inputs(prompt="Pick one.\nA. apple\nB. banana\nReply with the letter."),
        observed="A",
        expected=expected,
        label=Label.FAIL,
    )


def test_cot_faithfulness_trajectory_flags_drift_away():
    """Right at the truncation point, wrong at the end — the chain destroyed it."""
    model = ScriptModel([
        "4",                                       # direct
        "Long derivation.\nAnswer: 9",             # full CoT lands on the WRONG answer
        "Answer: 4",                               # the early answer was right
    ])
    f = CoTFaithfulnessAnalyzer(truncation_fracs=(0.5,)).run(
        model, CaseBatch([_graded_case("4")])
    ).findings
    entry = f["per_case"][0]
    assert entry["final_correct"] == 0
    assert f["answer_trajectory_by_case"][entry["sample_id"]] == [1, 0]
    assert entry["drift_away"] == 1 and entry["late_rescue"] == 0
    assert f["drift_away_rate"] == 1.0


def test_cot_faithfulness_trajectory_flags_late_rescue():
    """Wrong early, right at the end — the reasoning is doing real work."""
    model = ScriptModel([
        "9",                                       # direct
        "Long derivation.\nAnswer: 4",             # full CoT lands on the gold
        "Answer: 9",                               # early answer wrong
    ])
    f = CoTFaithfulnessAnalyzer(truncation_fracs=(0.5,)).run(
        model, CaseBatch([_graded_case("4")])
    ).findings
    entry = f["per_case"][0]
    assert entry["late_rescue"] == 1 and entry["drift_away"] == 0
    assert entry["first_correct_frac"] == 1.0
    assert entry["wasted_reasoning_frac"] == 0.0


def test_cot_faithfulness_trajectory_measures_wasted_reasoning():
    model = ScriptModel([
        "4", "Long derivation.\nAnswer: 4", "Answer: 4",
    ])
    f = CoTFaithfulnessAnalyzer(truncation_fracs=(0.25,)).run(
        model, CaseBatch([_graded_case("4")])
    ).findings
    entry = f["per_case"][0]
    # already right at 25% of the chain -> 75% of the reasoning was surplus
    assert entry["first_correct_frac"] == 0.25
    assert entry["wasted_reasoning_frac"] == 0.75


def test_cot_faithfulness_trajectory_columns_absent_without_gold():
    model = ScriptModel(["4", "Step.\nAnswer: 4", "Answer: 4"])
    f = CoTFaithfulnessAnalyzer(truncation_fracs=(0.5,)).run(
        model, CaseBatch([_mc_case()])
    ).findings
    assert "final_correct" not in f["per_case"][0]
    assert f["drift_away_rate"] is None


# ── context_shap ──────────────────────────────────────────────────────────────
def test_context_shap_attributes_to_supporting_chunk():
    ctx = "Alice owns a red car.\n\nBob owns a blue bike."
    case = FailureCase(
        inputs=Inputs(prompt=f"Context:\n{ctx}\n\nWhat colour is Alice's car?"),
        observed="red",
        label=Label.PASS,
        metadata={"context": ctx},
    )
    model = PromptAwareModel([])
    f = ContextShapAnalyzer(n_samples=8, seed=0).run(model, CaseBatch([case])).findings
    entry = f["per_case"][0]
    assert entry["n_chunks"] == 2
    assert entry["top_chunk_index"] == 0          # the Alice chunk carries the answer
    assert entry["context_dependence"] > 0.0      # removing context changes the answer
    assert entry["top_chunk_share"] > 0.9


def test_context_shap_skips_when_context_missing():
    case = FailureCase(inputs=Inputs(prompt="no context"), label=Label.FAIL)
    f = ContextShapAnalyzer().run(ScriptModel(["x"]), CaseBatch([case])).findings
    assert f["per_case"][0]["skipped"]


# ── calibration ───────────────────────────────────────────────────────────────
def test_calibration_overconfident_model():
    # stated 90% confidence, actual accuracy 50% -> positive overconfidence gap
    batch = CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"q{i}"),
                    label=Label.PASS if i % 2 == 0 else Label.FAIL)
        for i in range(8)
    ])
    model = ScriptModel(["answer\nConfidence: 90"], logprob=-0.105)  # ~0.9 seq prob
    f = CalibrationAnalyzer(n_bins=4).run(model, batch).findings
    assert f["verbalized_channel"]["n"] == 8
    assert f["verbalized_channel"]["overconfidence_gap"] > 0.3
    assert f["logprob_channel"]["overconfidence_gap"] > 0.3
    assert f["verbalized_channel"]["ece"] > 0.3


def test_calibration_skips_unlabelled():
    batch = CaseBatch([FailureCase(inputs=Inputs(prompt="q"), label=Label.UNKNOWN)])
    f = CalibrationAnalyzer().run(ScriptModel(["x\nConfidence: 50"]), batch).findings
    assert f["n_unlabelled_skipped"] == 1
    assert f["logprob_channel"]["n"] == 0


def test_calibration_keeps_other_cases_when_one_request_fails():
    class FlakyModel(ScriptModel):
        def generate(self, inputs, **kwargs):
            if len(self.prompts) == 0:
                self.prompts.append(str(inputs))
                raise TimeoutError("temporary backend timeout")
            return super().generate(inputs, **kwargs)

    batch = CaseBatch([
        FailureCase(inputs=Inputs(prompt="q1"), label=Label.PASS),
        FailureCase(inputs=Inputs(prompt="q2"), label=Label.FAIL),
    ])
    f = CalibrationAnalyzer().run(FlakyModel(["Confidence: 80"]), batch).findings
    assert f["verbalized_channel"]["n"] == 1
    assert "verbalized_error" in f["per_case"][0]


def test_ece_helper():
    perfect = [(0.9, True)] * 9 + [(0.9, False)]
    assert expected_calibration_error(perfect, 10) == 0.0
    assert expected_calibration_error([], 10) is None


# ── (A)-style option blocks ──────────────────────────────────────────────────
# BBH / AGIEval / several MMLU redistributions write "(A) foo". The block regex
# required the letter at the start of the line, so every one of those prompts
# parsed to zero options and format_sensitivity reported n_scored=0 -- a silent
# no-op on a benchmark that is nothing but multiple choice.
_BBH_PROMPT = (
    "Alice has a blue ball, Bob has a orange ball, Claire has a black ball.\n"
    "Alice and Bob swap balls. At the end of the game, Bob has the\n"
    "Options:\n(A) blue ball\n(B) orange ball\n(C) black ball\n"
)


def test_extract_option_block_reads_parenthesised_letters():
    from evalrx.analyzers.perturbation.format_sensitivity import extract_option_block

    options, block = extract_option_block(_BBH_PROMPT)
    assert options == ["blue ball", "orange ball", "black ball"]
    assert block.startswith("(A)")


def test_plain_and_parenthesised_styles_both_parse():
    from evalrx.analyzers.perturbation.format_sensitivity import extract_option_block

    for text in ("Q?\nA. one\nB. two\n", "Q?\nA) one\nB) two\n", "Q?\n(A) one\n(B) two\n"):
        assert extract_option_block(text)[0] == ["one", "two"], text


def test_rotation_preserves_the_prompts_own_letter_style():
    """Re-rendering "(A)" as "A." would change the delimiter and the position at
    once, so a measured flip could be either cause."""
    from evalrx.analyzers.perturbation.format_sensitivity import (
        FormatSensitivityAnalyzer,
        extract_option_block,
    )

    options, block = extract_option_block(_BBH_PROMPT)
    rotated_prompt, rotated = FormatSensitivityAnalyzer._variant_prompt(
        _BBH_PROMPT, block, options, 1)
    assert "(A) orange ball" in rotated_prompt
    assert "A. orange ball" not in rotated_prompt
    assert rotated[0] == "orange ball"

    plain = "Q?\nA. one\nB. two\n"
    opts, blk = extract_option_block(plain)
    out, _ = FormatSensitivityAnalyzer._variant_prompt(plain, blk, opts, 1)
    assert "A. two" in out and "(A) two" not in out


def test_parse_choice_accepts_the_answer_in_paren_style():
    from evalrx.analyzers.perturbation.format_sensitivity import parse_choice

    assert parse_choice("Answer: (C)", 7) == "C"
    assert parse_choice("Answer: C", 7) == "C"
    assert parse_choice("the final answer is (G)", 7) == "G"


# ── self_consistency on a reasoning model ────────────────────────────────────
def _chain_model(answers):
    """A model whose chains all differ but whose ANSWERS are given."""
    from evalrx.core.capability import Capability
    from evalrx.core.model import Model

    class _M(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text"})

        def __init__(self):
            self.i = 0

        def generate(self, inputs, **kwargs):
            a = answers[self.i]
            self.i += 1
            return f"Step {self.i}: unique reasoning text {self.i}\n</think>\nAnswer: {a}"

        def forward(self, *a, **k):
            raise NotImplementedError

    return _M()


def _one_case():
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label

    return CaseBatch([FailureCase(inputs=Inputs(prompt="q"), observed="",
                                  expected="(C)", label=Label.FAIL)])


def test_raw_text_consistency_is_a_constant_on_a_reasoning_model():
    """Five identical ANSWERS still read 1/n when whole chains are compared."""
    from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer

    f = SelfConsistencyAnalyzer(n=5, semantic=False).run(
        _chain_model(["(C)"] * 5), _one_case()).findings
    assert f["consistency"] == 0.2 and f["n_unique"] == 5
    assert f["compared_on"] == "raw_text"


def test_answer_fn_makes_consistency_measure_answers():
    from evalrx.analyzers.reasoning._text import extract_answer
    from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer

    f = SelfConsistencyAnalyzer(n=5, semantic=False, answer_fn=extract_answer).run(
        _chain_model(["(C)"] * 5), _one_case()).findings
    assert f["consistency"] == 1.0 and f["n_unique"] == 1
    assert f["compared_on"] == "answer"
    # the misleading number is still reported, so the two cannot be conflated
    assert f["raw_text_consistency"] == 0.2


def test_answer_fn_still_detects_genuine_disagreement():
    from evalrx.analyzers.reasoning._text import extract_answer
    from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer

    f = SelfConsistencyAnalyzer(n=4, semantic=False, answer_fn=extract_answer).run(
        _chain_model(["(A)", "(B)", "(A)", "(C)"]), _one_case()).findings
    assert f["consistency"] == 0.5 and f["n_unique"] == 3
