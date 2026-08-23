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


_NUMBERED_STYLE = """## Hypotheses

**HYPOTHESIS 1:** The "No" prior is produced *inside* the chain of thought.
**PLAIN_STATEMENT:** The model grades the stories like a law professor.
**FAILURE_MODE:** task_framing_override
**TEST:** New per-case lexical analyzer over stored outputs: count of strict-doctrine markers HIGHER on failing cases.
**EXPECTED_ASSOCIATION:** higher_on_failures

**HYPOTHESIS 2:** The bias is a length-dependent deliberation drift.
**PLAIN_STATEMENT:** The longer it thinks out loud, the more objections it invents.
**FAILURE_MODE:** overthinking
**TEST:** `termination_audit.output_words` HIGHER on failing cases.
**EXPECTED_ASSOCIATION:** higher_on_failures
"""


def test_numbered_labels_parse_like_bare_ones():
    """gemma-4-e2b / bbh_causal_judgement chain (2026-08-22): ``**HYPOTHESIS 1:**``
    / ``**HYPOTHESIS 2:**`` — three well-formed hypotheses, zero parsed, and the
    run silently diagnosed an analysis-module template instead."""
    hs = _parse_hypotheses(_NUMBERED_STYLE, "gemma")
    assert [h.predicted_failure_mode for h in hs] == ["task_framing_override", "overthinking"]
    assert hs[0].statement == 'The "No" prior is produced *inside* the chain of thought.'
    assert hs[0].plain_statement.startswith("The model grades")
    assert hs[0].test_design.startswith("New per-case lexical analyzer")
    assert hs[1].statement.startswith("The bias is a length-dependent")
    for raw, want in {
        "**HYPOTHESIS 1:** one": "HYPOTHESIS: one",
        "HYPOTHESIS #2: two": "HYPOTHESIS: two",
        "Hypothesis (3): three": "HYPOTHESIS: three",
        "**TEST 1:** t": "TEST: t",
        "3. **HYPOTHESIS 3:** bullet and number": "HYPOTHESIS: bullet and number",
    }.items():
        assert _normalise_label_line(raw) == want, raw
    # a number that is part of the statement, not the label, stays
    assert _normalise_label_line("HYPOTHESIS: 3 of the cases") == "HYPOTHESIS: 3 of the cases"


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
