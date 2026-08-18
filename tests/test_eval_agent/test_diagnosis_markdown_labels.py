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
    kept = _validate_hypotheses(
        hs, "{}", _Critic("Verdicts:\n- **KEEP:** the long-list failures are multiset errors on an otherwise ordered output.\n- **REJECT:** the probe one"),
    )
    assert [h.predicted_failure_mode for h in kept] == ["computation_slip"]
