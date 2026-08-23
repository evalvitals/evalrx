"""M3's line parser must read a judge that answers in Markdown.

Live failure (qwen3.5-2b / bbh_word_sorting smoke, 2026-08-18): the judge wrote
three well-formed hypotheses as ``**HYPOTHESIS:** …`` / ``**FAILURE_MODE:** …``
/ ``**TEST:** …`` and ``_parse_hypotheses`` — a bare ``startswith("HYPOTHESIS:")``
— returned zero, so the loop stopped with "no_hypotheses" after a 30-minute M1.
The label is the contract, not its decoration.
"""

from __future__ import annotations

from evalvitals.eval_agent.stages.diagnosis import (
    _normalise_label_line,
    _parse_hypotheses,
    _validate_hypotheses,
)
from tests.conftest import FakeModel

_LIVE_STYLE = """Three hypotheses, ordered so the instrumentation check comes first.

**HYPOTHESIS:** `generated:probe1`'s metrics use a misaligned reference, so the +0.73 effect rests on a broken probe.
**FAILURE_MODE:** probe_metric_misalignment
**TEST:** Re-run the probe with the gold string as the answer and require misplaced=0.

**HYPOTHESIS:** The long-list failures are multiset errors on an otherwise ordered output.
**FAILURE_MODE:** computation_slip
**TEST:** Recompute `dropped/duplicated` from raw strings.
"""


def test_markdown_bold_labels_parse_like_bare_ones():
    hs = _parse_hypotheses(_LIVE_STYLE, "qwen")
    assert [h.predicted_failure_mode for h in hs] == ["probe_metric_misalignment", "computation_slip"]
    assert hs[0].statement.startswith("`generated:probe1`'s metrics")
    assert hs[0].test_design.startswith("Re-run the probe")
    assert hs[1].test_design.startswith("Recompute")


def test_every_common_decoration_is_normalised():
    cases = {
        "HYPOTHESIS: plain": "HYPOTHESIS: plain",
        "**HYPOTHESIS:** bold-outside": "HYPOTHESIS: bold-outside",
        "**HYPOTHESIS**: bold-inside": "HYPOTHESIS: bold-inside",
        "- FAILURE_MODE: bullet": "FAILURE_MODE: bullet",
        "* TEST: star bullet": "TEST: star bullet",
        "1. HYPOTHESIS: numbered": "HYPOTHESIS: numbered",
        "2) TEST: numbered paren": "TEST: numbered paren",
        "### FAILURE_MODE: heading": "FAILURE_MODE: heading",
        "__hypothesis__: underscored lower": "HYPOTHESIS: underscored lower",
        "`KEEP:` code": "KEEP: code",
        "- **KEEP:** the *first* one **": "KEEP: the *first* one",   # inner emphasis kept, trailing dropped
    }
    for raw, want in cases.items():
        assert _normalise_label_line(raw) == want, raw
    # non-label lines are only stripped
    assert _normalise_label_line("  evidence: HYPOTHESIS-like words inside  ") == "evidence: HYPOTHESIS-like words inside"
    assert _normalise_label_line("The hypothesis: is not a label") == "The hypothesis: is not a label"


def test_json_output_is_untouched_by_the_text_normaliser():
    raw = '[{"hypothesis": "json path statement long enough", "failure_mode": "fm"}]'
    hs = _parse_hypotheses(raw, "m")
    assert [h.statement for h in hs] == ["json path statement long enough"]


class _Critic(FakeModel):
    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply

    def generate(self, inputs, **kw):
        return self.reply


def test_critic_keep_lines_in_markdown_are_honoured():
    hs = _parse_hypotheses(_LIVE_STYLE, "qwen")
    out = _validate_hypotheses(
        hs, "{}", _Critic("Verdicts:\n- **KEEP:** the long-list failures are multiset errors on an otherwise ordered output.\n- **REJECT:** the probe one"),
    )
    # annotate, never filter: every proposal comes back, kept ones first
    assert len(out) == len(hs)
    assert out[0].predicted_failure_mode == "computation_slip"
    assert out[0].metadata["critic"] == "keep"
    assert {h.metadata["critic"] for h in out[1:]} <= {"reject", "unparsed"}


def test_critic_rejecting_everything_keeps_flagged_leads():
    hs = _parse_hypotheses(_LIVE_STYLE, "qwen")
    cap = {}
    out = _validate_hypotheses(
        hs, "{}",
        _Critic("\n".join(
            f"REJECT: {h.statement}\nREASON: n=32 cannot separate this from chance"
            for h in hs)),
        capture=cap,
    )
    assert len(out) == len(hs)
    assert all(h.metadata["critic"] == "reject" for h in out)
    assert all(h.metadata["critic_reason"].startswith("n=32") for h in out)
    assert cap["n_rejected"] == len(hs) and cap["n_kept"] == 0
    assert "REJECT:" in cap["raw"]


def test_unparsable_critic_marks_hypotheses_unreviewed():
    hs = _parse_hypotheses(_LIVE_STYLE, "qwen")
    out = _validate_hypotheses(hs, "{}", _Critic("I cannot review these."))
    assert len(out) == len(hs)
    assert all(h.metadata["critic"] == "unparsed" for h in out)


def test_markdown_expected_association_label_is_normalised():
    raw = (
        "**HYPOTHESIS:** the chain breaks on long lists of words\n"
        "**FAILURE_MODE:** chain_break\n"
        "**TEST:** step_rollout_value.max_value_drop HIGHER on failing cases\n"
        "**EXPECTED_ASSOCIATION:** higher_on_failures\n"
    )
    hs = _parse_hypotheses(raw, "m")
    assert len(hs) == 1
    assert hs[0].expected_association == "higher_on_failures"
    assert hs[0].test_design.startswith("step_rollout_value.max_value_drop")


def test_critic_context_is_optional_and_lands_in_its_prompt():
    """The critic reviews against what the proposer saw: *context* (conclusion,
    evidence, stats, explore notes, label summary) goes into its prompt; the
    old 3-positional call keeps working and adds no context block."""
    hs = _parse_hypotheses(_LIVE_STYLE, "qwen")
    cap = {}
    _validate_hypotheses(hs, "{}", _Critic("KEEP: everything"), capture=cap,
                         context="LABEL SUMMARY: gold=no answered=yes n=27 FAIL=27")
    assert "Context the proposer worked from" in cap["prompt"]
    assert "gold=no answered=yes n=27 FAIL=27" in cap["prompt"]
    assert cap["prompt"].index("Context the proposer") < cap["prompt"].index("Findings summary")

    bare = {}
    _validate_hypotheses(hs, "{}", _Critic("KEEP: everything"), capture=bare)
    assert "Context the proposer worked from" not in bare["prompt"]


def test_label_context_tabulates_gold_by_answer_on_binary_batches():
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
    from evalvitals.eval_agent.stages.diagnosis import _format_label_context

    assert _format_label_context(None) == ""

    def _c(i, observed, expected, label):
        return FailureCase(id=f"c{i}", inputs=Inputs(prompt="Is there a dog?"),
                           observed=observed, expected=expected, label=label)

    binary = CaseBatch(
        [_c(i, "Yes", "No", Label.FAIL) for i in range(3)]
        + [_c(10 + i, "Yes", "Yes", Label.PASS) for i in range(4)]
        + [_c(20, "No", "Yes", Label.FAIL), _c(21, "No", "No", Label.PASS)]
    )
    text = _format_label_context(binary)
    assert "labelled cases: 9 (4 FAIL / 5 PASS)" in text
    assert "gold=no  answered=yes      n=3    FAIL=3" in text
    assert "gold=yes answered=yes      n=4    FAIL=0" in text
    assert "answered yes on 7/9 binary cases" in text

    numeric = CaseBatch([_c(i, "Answer: 12", "12", Label.PASS) for i in range(5)]
                        + [_c(9, "Answer: 7", "12", Label.FAIL)])
    text = _format_label_context(numeric)
    assert "labelled cases: 6 (1 FAIL / 5 PASS)" in text
    assert "gold x answer" not in text


# Live failure #2 (VideoLLaMA2.1-7B-AV / Music-AVQA, 560 cases, 2026-08-23):
# Opus at high effort numbered its three hypotheses the way anyone numbers a
# list — HYPOTHESIS 1: / HYPOTHESIS 2: / HYPOTHESIS 3: — and the parser, which
# by then handled every Markdown decoration and every LEADING list marker,
# returned zero again. The first fix took the ordinal before the label ("1.
# TEST:") and missed the ordinal after it. Two and a half hours of GPU and a
# high-effort judge, discarded over a digit.
#
# Verbatim from that run's prompts/c0_m3_diagnosis.response.txt, trimmed.
_NUMBERED_STYLE = """HYPOTHESIS 1: The M2 item set contains unrendered prompt templates \u2014 31% of questions reach the model with literal `<LR>/<FL>` tokens where a spatial referent should be.
PLAIN_STATEMENT: Roughly a third of the questions were never filled in properly.
FAILURE_MODE: prompt_template_unsubstituted
TEST: `generated:probe1.has_placeholder` on failing vs passing cases (already +0.33, p\u22481e-15).
EXPECTED_ASSOCIATION: higher_on_failures

HYPOTHESIS 2: On the well-formed remainder, the model answers audio-grounded questions from visual priors rather than the audio stream.
PLAIN_STATEMENT: The model is mostly watching, not listening.
FAILURE_MODE: ignored_obs
TEST: `modality_ablation.grounded_in_audio` on failing vs passing cases within the placeholder-free stratum.
EXPECTED_ASSOCIATION: higher_on_failures

HYPOTHESIS 3: The model conditions its answer on its own generated text rather than on the raw AV evidence.
PLAIN_STATEMENT: If you ask it to describe the clip first, it answers its own description.
FAILURE_MODE: self_conditioning_cascade
TEST: `prompt_contrast` describe_first contrast, plus `perturbation_battery.noop_clause_flipped` HIGHER on failing cases.
EXPECTED_ASSOCIATION: higher_on_failures
"""


def test_an_ordinal_after_the_label_parses():
    hs = _parse_hypotheses(_NUMBERED_STYLE, "videollama2.1-7b-av")
    assert [h.predicted_failure_mode for h in hs] == [
        "prompt_template_unsubstituted", "ignored_obs", "self_conditioning_cascade",
    ]
    # The ordinal is a label decoration, not part of the claim.
    assert hs[0].statement.startswith("The M2 item set contains")
    assert hs[1].test_design.startswith("`modality_ablation.grounded_in_audio`")
    assert hs[2].plain_statement.startswith("If you ask it to describe")


def test_ordinals_in_either_position_and_every_marker_shape():
    """Before and after the label, with and without punctuation."""
    for line, want in {
        "HYPOTHESIS 1: a": "HYPOTHESIS: a",
        "HYPOTHESIS 12: a": "HYPOTHESIS: a",
        "HYPOTHESIS #2: a": "HYPOTHESIS: a",
        "TEST 3.: a": "TEST: a",
        "REASON (4): a": "REASON: a",
        "1. TEST: a": "TEST: a",
        "**HYPOTHESIS 1:** a": "HYPOTHESIS: a",
        "- FAILURE_MODE 2: a": "FAILURE_MODE: a",
    }.items():
        assert _normalise_label_line(line) == want, line


def test_a_word_after_the_label_is_not_an_ordinal():
    """Only digits are decoration. `HYPOTHESIS TESTING:` is a different label
    and must not be silently read as `HYPOTHESIS:`."""
    assert _normalise_label_line("HYPOTHESIS TESTING: a") == "HYPOTHESIS TESTING: a"


def test_a_wrapped_failure_mode_unwraps_but_prose_keeps_its_code_spans():
    """FAILURE_MODE is a lookup key; TEST is a sentence. They need opposite rules.

    Regression from the fix that stopped the normaliser eating content
    backticks: a judge writing ``FAILURE_MODE: `ignored_obs` `` then produced
    the key "`ignored_obs", which matches nothing in
    `_FAILURE_MODE_TO_ANALYZERS` -- so the next cycle's focused re-probe
    silently falls back to the generic ranking. Seen live on the Music-AVQA run,
    where all three hypotheses came out with a leading backtick.
    """
    from evalvitals.eval_agent.stages.diagnosis import _unwrap_value

    assert _unwrap_value("`ignored_obs`") == "ignored_obs"
    assert _unwrap_value("**language_prior_bias**") == "language_prior_bias"
    assert _unwrap_value("plain_mode") == "plain_mode"
    # Only an ENTIRELY wrapped value is decoration.
    assert _unwrap_value("some `sig` text") == "some `sig` text"
    assert _unwrap_value("`unclosed") == "`unclosed"

    raw = (
        "HYPOTHESIS 1: The audio branch contributes a clip-level prior.\n"
        "FAILURE_MODE: `ignored_obs`\n"
        "TEST: `modality_ablation.grounded_in_audio` HIGHER on failing cases\n"
        "EXPECTED_ASSOCIATION: `higher_on_failures`\n"
    )
    h = _parse_hypotheses(raw, "videollama2.1-7b-av")[0]
    assert h.predicted_failure_mode == "ignored_obs"          # a key, unwrapped
    assert h.expected_association == "higher_on_failures"
    assert h.test_design.startswith("`modality_ablation")     # a sentence, intact


def test_the_failure_mode_keys_the_next_cycle_actually_routes_on():
    """The unwrapped mode must hit the routing table, or the fix is cosmetic."""
    from evalvitals.eval_agent.stages.probe import _FAILURE_MODE_TO_ANALYZERS
    from evalvitals.eval_agent.stages.diagnosis import _unwrap_value

    for wrapped in ("`ignored_obs`", "`language_prior_bias`"):
        assert _unwrap_value(wrapped) in _FAILURE_MODE_TO_ANALYZERS
