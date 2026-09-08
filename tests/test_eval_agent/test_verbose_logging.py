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


# V1's console narration (_VerboseFormatter) rendered stats_plan/
# corrected_rejections/etc. field-by-field, which is where this suite used
# to guard against a real crash (an externalised {path, n_items, bytes}
# stand-in iterated as if it were still the raw list — see git history for
# the fix). RunLoggerV2's narration (_V2JsonFormatter.line) is a one-line
# `[stage] event cycle=N` summary that never touches those fields, so that
# bug class does not exist to regress here anymore.
