"""evalrx.enable_console_logging() / disable_console_logging()."""

from __future__ import annotations

import logging

import evalrx
from evalrx.logging_utils import _MARKER_ATTR, TOP_LEVEL_LOGGER_NAME


def _console_handlers():
    top = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
    return [h for h in top.handlers if getattr(h, _MARKER_ATTR, False)]


def test_silent_by_default(capsys):
    logging.getLogger("evalrx.some_module").info("quiet please")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_enable_console_logging_surfaces_info(capsys):
    try:
        evalrx.enable_console_logging()
        logging.getLogger("evalrx.some_module").info("hello")
        out = capsys.readouterr()
        assert "hello" in out.out
    finally:
        evalrx.disable_console_logging()


def test_enable_is_idempotent():
    try:
        h1 = evalrx.enable_console_logging()
        h2 = evalrx.enable_console_logging()
        assert h1 is h2
        assert len(_console_handlers()) == 1
    finally:
        evalrx.disable_console_logging()


def test_level_filtering(capsys):
    try:
        evalrx.enable_console_logging(level=logging.INFO)
        logging.getLogger("evalrx.some_module").debug("should not appear")
        out = capsys.readouterr()
        assert "should not appear" not in out.out
    finally:
        evalrx.disable_console_logging()


def test_disable_removes_only_the_console_handler():
    top = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
    n_before = len(top.handlers)  # includes the package's NullHandler
    evalrx.enable_console_logging()
    assert len(top.handlers) == n_before + 1
    evalrx.disable_console_logging()
    assert len(top.handlers) == n_before
    assert _console_handlers() == []


def test_disable_without_enable_is_a_noop():
    evalrx.disable_console_logging()  # must not raise
    evalrx.disable_console_logging()
