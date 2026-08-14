"""Console logging convenience for evalvitals' existing per-module loggers.

Every submodule follows the standard library idiom
``logger = logging.getLogger(__name__)`` and already narrates real events —
M1-M5 stage transitions, capability negotiation, paper-method dispatch
(``vcd``/``icd``/...), fix-candidate generation and validation, audio decode
fallbacks.  None of it is visible by default: Python's logging module only
invokes an implicit "handler of last resort" for ``WARNING``-and-above
records when NO handler exists anywhere in a logger's propagation chain, and
even that prints unformatted straight to stderr.  ``.info()``/``.debug()``
calls — most of the narration — never appear at all without a handler.

A **library** must never call ``logging.basicConfig()`` itself: that
configures the ROOT logger process-wide and would clobber a host
application's own logging setup the moment ``import evalvitals`` runs. The
standard, documented pattern (see the stdlib logging HOWTO's "Configuring
Logging for a Library" section) is instead:

  * attach a :class:`logging.NullHandler` to the package's top-level logger
    (done in :mod:`evalvitals` on import) so the library is silent — no
    "No handlers could be found" noise, no surprise raw stderr output — until
    a caller opts in;
  * expose an explicit, opt-in helper for callers who *do* want to see it.
    That's this module.

Usage::

    import evalvitals
    evalvitals.enable_console_logging()          # INFO and above, readable
    model = evalvitals.load("qwen2.5-7b-instruct")
    ...                                            # now narrated to stdout

    evalvitals.enable_console_logging(level=logging.DEBUG)  # noisier
    evalvitals.disable_console_logging()                     # back to quiet
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Optional

# Every module's `logging.getLogger(__name__)` (e.g. "evalvitals.eval_agent.loop")
# is a child of this in the dot-hierarchy, so one handler here reaches all of
# them via normal propagation -- except RunLogger's own "evalvitals.run.<dir>"
# tree, which sets propagate=False by design to keep its structured JSONL
# event stream a separate channel from this ad-hoc narration.
TOP_LEVEL_LOGGER_NAME = "evalvitals"

# Marks a handler as "ours" so enable_console_logging() is idempotent (calling
# it twice reconfigures the level instead of stacking a second handler) and
# disable_console_logging() only ever removes handlers it installed itself --
# never a caller's own handler on the same logger.
_MARKER_ATTR = "_evalvitals_console_handler"


class _ConsoleFormatter(logging.Formatter):
    """Compact single-line format: time, level, short module name, message."""

    def format(self, record: logging.LogRecord) -> str:
        record.short_name = record.name.rsplit(".", 1)[-1]
        return super().format(record)


def enable_console_logging(
    level: int = logging.INFO, *, stream: Optional[IO[str]] = None
) -> logging.Handler:
    """Attach a readable stdout handler to evalvitals' logger tree.

    Idempotent: a second call reconfigures the level of the existing handler
    rather than attaching a duplicate (no doubled lines).

    Args:
        level:  Minimum level to show. ``logging.INFO`` (default) surfaces the
                stage-by-stage narration most callers want; ``logging.DEBUG``
                is noisier (per-candidate detail, low-level capability checks).
        stream: Defaults to ``sys.stdout``.

    Returns:
        The attached :class:`logging.Handler`, in case a caller wants to
        remove or further customise it directly.
    """
    top_logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
    for handler in top_logger.handlers:
        if getattr(handler, _MARKER_ATTR, False):
            handler.setLevel(level)
            top_logger.setLevel(level)
            return handler

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(
        _ConsoleFormatter(
            fmt="%(asctime)s %(levelname)-7s %(short_name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    setattr(handler, _MARKER_ATTR, True)
    top_logger.addHandler(handler)
    top_logger.setLevel(level)
    return handler


def disable_console_logging() -> None:
    """Remove the handler(s) installed by :func:`enable_console_logging`.

    A no-op if none was ever attached. Never touches handlers a caller added
    themselves directly to ``logging.getLogger("evalvitals")`` or below.
    """
    top_logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
    for handler in list(top_logger.handlers):
        if getattr(handler, _MARKER_ATTR, False):
            top_logger.removeHandler(handler)
