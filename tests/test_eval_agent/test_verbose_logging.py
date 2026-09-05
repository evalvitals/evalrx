"""VLDiagnoseLoop(verbose=True) / FixAgent(verbose=True) enable console logging.

Cheap constructor-level checks -- no .run()/.propose_and_validate() (those need
a full M1-M4 or fix-validation pass); this only verifies the wiring reaches
evalrx.logging_utils's idempotent handler.
"""

from __future__ import annotations

import logging

import evalrx
from evalrx.eval_agent.loop import VLDiagnoseLoop
from evalrx.eval_agent.stages.fix_agent import FixAgent
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from evalrx.logging_utils import _MARKER_ATTR, TOP_LEVEL_LOGGER_NAME
from tests.conftest import FakeModel


def _has_console_handler() -> bool:
    top = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
    return any(getattr(h, _MARKER_ATTR, False) for h in top.handlers)


def test_loop_verbose_false_leaves_logging_untouched():
    evalrx.disable_console_logging()
    VLDiagnoseLoop(
        model=FakeModel(), protocol=ExperimentProtocol(description="x"), verbose=False
    )
    assert not _has_console_handler()


def test_loop_verbose_true_enables_console_logging():
    try:
        evalrx.disable_console_logging()
        VLDiagnoseLoop(
            model=FakeModel(), protocol=ExperimentProtocol(description="x"), verbose=True
        )
        assert _has_console_handler()
    finally:
        evalrx.disable_console_logging()


def test_fix_agent_verbose_true_enables_console_logging():
    try:
        evalrx.disable_console_logging()
        FixAgent(verbose=True)
        assert _has_console_handler()
    finally:
        evalrx.disable_console_logging()


def test_loop_stage_narration_is_actually_visible(capsys):
    """logger.info from loop.py itself becomes visible once verbose=True."""
    try:
        evalrx.disable_console_logging()
        VLDiagnoseLoop(
            model=FakeModel(), protocol=ExperimentProtocol(description="x"), verbose=True
        )
        logging.getLogger("evalrx.eval_agent.loop").info("m1 probing started")
        out = capsys.readouterr()
        assert "m1 probing started" in out.out
    finally:
        evalrx.disable_console_logging()


# ── externalised payloads must not crash the human formatter ─────────────────
def _format_payload(payload):
    import logging

    from evalrx.eval_agent.run_logger import _VerboseFormatter

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "run_event", (), None)
    record._payload = payload
    return _VerboseFormatter().format(record)


def test_m2_line_survives_an_externalised_stats_plan():
    """_externalize_if_large swaps a big list for {path, n_items, bytes}.

    Iterating that dict yields its KEYS, so ``s['tool']`` raised
    ``TypeError: string indices must be integers`` inside ``logging.emit``,
    where Python swallows the exception and prints a traceback to stderr.

    Only the CONSOLE line is lost -- the JSON handler is a separate formatter,
    so the JSONL record survives intact.  That is what makes it easy to miss:
    the run completes, the data is on disk, and the only symptom is a missing
    [M2] line next to a traceback that names neither M2 nor the log.  Observed
    live on qwen3.5-2b / minervamath, whose stats_plan externalised at 53 items
    / 9846 bytes.
    """
    out = _format_payload({
        "event": "analysis", "cycle": 0, "severity": "high", "conclusion": "c",
        "stats_plan": {"bytes": 91234, "n_items": 7,
                       "path": "artifacts/c0_m2_stats_plan.json"},
        "stats_tool_results": {"bytes": 5123, "n_items": 4, "path": "artifacts/x.json"},
        "corrected_rejections": {"bytes": 9, "n_items": 1, "path": "p"},
    })
    assert "[M2]" in out
    # it points at the artifact rather than silently dropping the line
    assert "artifacts/c0_m2_stats_plan.json" in out
    assert "7 items externalised" in out


def test_m2_line_still_renders_the_inline_shape():
    out = _format_payload({
        "event": "analysis", "cycle": 0, "severity": "low", "conclusion": "c",
        "stats_plan": [{"tool": "mcnemar"}, {"tool": "bootstrap"}],
        "stats_tool_results": [{"name": "mcnemar", "conclusion": "p=0.01"}],
        "corrected_rejections": {"rejected_tools": ["mcnemar"]},
    })
    assert "['mcnemar', 'bootstrap']" in out
    assert "fdr_survive: 1: ['mcnemar']" in out
    assert "stats_tool : mcnemar - p=0.01" in out


def test_fdr_survive_lists_results_not_tool_names():
    """42 signal_label_assoc tests collapsed to the one word 'signal_label_assoc'."""
    out = _format_payload({
        "event": "analysis", "cycle": 0, "severity": "low", "conclusion": "c",
        "corrected_rejections": {
            "n_tested": 42,
            "rejected_tools": ["signal_label_assoc"],
            "rejected_result_keys": ["signal_label_assoc:probe1.dropped",
                                     "signal_label_assoc:self_repair.changed_answer"],
        },
    })
    assert "fdr_survive: 2 of 42: ['signal_label_assoc:probe1.dropped'" in out
    assert "['signal_label_assoc']" not in out


def test_a_plan_entry_missing_its_tool_key_does_not_raise():
    out = _format_payload({
        "event": "analysis", "cycle": 0, "severity": "low", "conclusion": "c",
        "stats_plan": [{"config": {}}],
    })
    assert "[M2]" in out
