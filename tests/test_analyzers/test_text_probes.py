"""Behavioral tests for the 2026-08 text probes.

Each test scripts a model with a KNOWN pathology and asserts the probe's
columns expose it — the structural half lives in test_analyzer_contract.py.
"""

from __future__ import annotations

from evalvitals.analyzers.hallucination.selfcheck import (
    SelfCheckConsistencyAnalyzer,
    containment,
    split_sentences,
)
from evalvitals.analyzers.perturbation.context_shap import ContextShapAnalyzer
from evalvitals.analyzers.perturbation.cot_faithfulness import (
    CoTFaithfulnessAnalyzer,
    default_answer_fn,
)
from evalvitals.analyzers.perturbation.format_sensitivity import (
    FormatSensitivityAnalyzer,
    extract_options,
    parse_choice,
)
from evalvitals.analyzers.uncertainty.calibration import (
    CalibrationAnalyzer,
    expected_calibration_error,
)
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model, TokenLogprob


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
    from evalvitals.analyzers.perturbation.format_sensitivity import extract_option_block
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


def test_ece_helper():
    perfect = [(0.9, True)] * 9 + [(0.9, False)]
    assert expected_calibration_error(perfect, 10) == 0.0
    assert expected_calibration_error([], 10) is None
