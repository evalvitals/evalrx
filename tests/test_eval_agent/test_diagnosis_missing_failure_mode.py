"""A hypothesis whose ``FAILURE_MODE:`` line the judge forgot is still a hypothesis.

Live failure (gemma-4-e2b / gsm8k, agy gemini-3.1-pro-preview, 2026-08-27): the
judge answered with three complete HYPOTHESIS / PLAIN_STATEMENT / TEST blocks
and no FAILURE_MODE / EXPECTED_ASSOCIATION lines. ``_parse_hypotheses`` only
appended on ``FAILURE_MODE:``, so three well-formed claims parsed as zero, the
loop stopped with ``no_hypotheses`` and the fix stage never ran — silently: the
run log recorded ``n_hypotheses: 0`` next to a 2.3 KB ``raw_judge_output``.
"""

from __future__ import annotations

import logging

from evalvitals.core.model import Capability
from evalvitals.eval_agent.stages.diagnosis import (
    MISSING_FAILURE_MODE,
    DiagnosisAgent,
    _parse_hypotheses,
)
from tests.test_eval_agent.test_auto_diagnose import ScriptedModel, _make_report

_LIVE_NO_MODE = """HYPOTHESIS: Output truncation (`answer_extraction_audit.output_truncated`) is driven by an excessive number of chain-of-thought sentences.
TEST: Re-run the evaluation with a higher maximum output token limit and check whether truncation drops.
PLAIN_STATEMENT: The model talks too much while reasoning and hits the length limit before printing the answer.

HYPOTHESIS: Mid-thought self-correction (`cot_faithfulness.cot_changed_answer`) inflates the token count.
TEST: Check whether `cot_changed_answer` co-occurs with `output_truncated`.
PLAIN_STATEMENT: The model changes its mind halfway through and runs out of room.

HYPOTHESIS: Problems with many distinct quantities (`coverage_verification_gap.n_unique`) exhaust the output budget.
TEST: Measure generation length as a function of `n_unique`.
PLAIN_STATEMENT: When a problem has many details the model writes too much and gets cut off.
"""

_WELL_FORMED = """HYPOTHESIS: The model over-attends to the first token.
PLAIN_STATEMENT: The model keeps looking at the start of the prompt instead of the question.
FAILURE_MODE: attention_sink
TEST: attention_sink.mean_sink_mass
EXPECTED_ASSOCIATION: higher_on_failures
"""


def test_blocks_without_failure_mode_are_kept_with_a_placeholder_mode():
    hs = _parse_hypotheses(_LIVE_NO_MODE, "gemma")
    assert len(hs) == 3
    assert {h.predicted_failure_mode for h in hs} == {MISSING_FAILURE_MODE}
    assert hs[0].statement.startswith("Output truncation")
    assert hs[0].test_design.startswith("Re-run the evaluation")
    assert hs[0].plain_statement.startswith("The model talks too much")
    assert hs[2].test_design.startswith("Measure generation length")
    assert all(h.target_model == "gemma" for h in hs)


def test_well_formed_blocks_keep_the_old_reading():
    hs = _parse_hypotheses(_WELL_FORMED, "m")
    assert len(hs) == 1
    h = hs[0]
    assert h.predicted_failure_mode == "attention_sink"
    assert h.test_design == "attention_sink.mean_sink_mass"
    assert h.expected_association == "higher_on_failures"
    assert h.plain_statement.startswith("The model keeps looking")


def test_mixed_blocks_parse_in_order():
    raw = _WELL_FORMED + "\n" + _LIVE_NO_MODE.split("\n\n")[0] + "\n"
    hs = _parse_hypotheses(raw, "m")
    assert [h.predicted_failure_mode for h in hs] == ["attention_sink", MISSING_FAILURE_MODE]
    assert hs[1].test_design.startswith("Re-run the evaluation")


def test_lines_after_failure_mode_still_attach_to_the_closed_hypothesis():
    raw = "HYPOTHESIS: a\nFAILURE_MODE: m\nTEST: t1\nEXPECTED_ASSOCIATION: lower_on_failures\n"
    hs = _parse_hypotheses(raw, "m")
    assert len(hs) == 1 and hs[0].test_design == "t1"
    assert hs[0].expected_association == "lower_on_failures"


def test_a_bare_hypothesis_label_with_no_statement_is_not_a_hypothesis():
    assert _parse_hypotheses("HYPOTHESIS:\nTEST: t\n", "m") == []


def test_diagnose_reasks_once_when_a_non_empty_answer_parses_to_zero(caplog):
    """The judge's first answer carries no label lines at all; the agent asks it
    once more for the exact format instead of substituting template hypotheses."""
    judge = ScriptedModel(
        answers=[
            "Three paragraphs of prose about attention sinks, without any label lines.",
            _WELL_FORMED,
        ],
        capabilities={Capability.GENERATE},
    )
    with caplog.at_level(logging.WARNING, logger="evalvitals.eval_agent.stages.diagnosis"):
        diag = DiagnosisAgent(judge=judge).diagnose(_make_report())
    assert [h.predicted_failure_mode for h in diag.hypotheses] == ["attention_sink"]
    assert diag.hypotheses[0].statement == "The model over-attends to the first token."
    assert any("re-asking" in r.message for r in caplog.records)
    assert "over-attends" in diag.raw_judge_output


def test_diagnose_does_not_reask_on_a_no_issue_verdict():
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    diag = DiagnosisAgent(judge=judge).diagnose(_make_report(severity="none"))
    assert diag.hypotheses == [] and judge._i == 1
