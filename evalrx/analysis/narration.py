"""Live terminal narration for the CLI-facing pipelines (`evalrx explore`,
`evalrx run-codebase`).

The project's own landing page shows a "simulated run" terminal — one
aligned line per M-stage, a dot leader, then the real tally. This module is
the non-simulated version of that: it renders the *same* stage transitions
`--progress-path`'s :class:`~evalrx.analysis.workbench.EventSink` already
receives, as they actually happen, instead of only writing them to a JSONL
file for a separate workbench UI to read later.

:class:`TerminalNarrator` duck-types ``EventSink.emit(stage, status,
message, *, attempt=None, artifact_refs=(), metrics=None)`` so it can be
passed anywhere a ``progress_sink`` is accepted, or fanned out alongside a
real ``EventSink`` via :class:`MultiSink`. Nothing here invents data — every
count and duration comes from the ``metrics``/``message`` the caller passed
at the real event.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO, Any, Iterable

# Stage keys line up with the M1-M5 pipeline naming used throughout the
# docs and the landing page's own terminal walkthrough. `evalrx explore`
# only ever drives m2/m3(/m4-as-holdout-confirm); `run-codebase` adds a
# pre-M2 "run" step (running the user's own codebase, not a numbered stage).
_STAGE_LABELS = {
    "run": "codebase",
    "m1": "probe",
    "m2": "explore",
    "m2_codegen": "explore",
    "m2_execute": "explore",
    "m3": "diagnose",
    "m4": "verify",
    "m5": "intervene",
    "persist": "write output",
}

# Sub-stages of a numbered stage: narrated only on retry or failure, never on
# a clean first-attempt pass, so the common case stays exactly one line per
# stage (matching the mockup) while a real repair loop is still visible.
_SUBSTAGES = {"m2_codegen", "m2_execute"}

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"


def _supports_color(stream: IO[str]) -> bool:
    """Respect NO_COLOR (https://no-color.org) and never color a non-tty."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("EVALRX_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _pluralize(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class TerminalNarrator:
    """Prints one aligned line per pipeline stage, live, as it happens.

    Format matches the landing page's terminal walkthrough:
    ``M2  explore        ············ 23 figures · 4 candidate signals``.
    """

    LABEL_WIDTH = 12

    def __init__(self, *, stream: IO[str] | None = None, color: bool | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.color = _supports_color(self.stream) if color is None else color
        self._started_at: dict[str, float] = {}

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self.color and text else text

    def _row(self, stage_key: str, detail: str) -> str:
        label = _STAGE_LABELS.get(stage_key, stage_key)
        # "m2" -> "M2 "; "run" -> "RUN" (no numbered stage of its own).
        key = self._c(_BOLD + _CYAN, stage_key.upper().ljust(3))
        dots = self._c(_DIM, "·" * max(4, self.LABEL_WIDTH + 2 - len(label)))
        return f"{key} {label.ljust(self.LABEL_WIDTH)}{dots} {detail}"

    def _print(self, stage_key: str, detail: str) -> None:
        print(self._row(stage_key, detail), file=self.stream)

    def emit(
        self,
        stage: str,
        status: str,
        message: str,
        *,
        attempt: int | None = None,
        artifact_refs: Iterable[Any] = (),
        metrics: dict[str, Any] | None = None,
    ) -> None:
        del artifact_refs  # not narrated — the summary printed at the end lists paths
        metrics = metrics or {}
        # "m2_codegen"/"m2_execute" -> "m2"; anything else is its own key.
        stage_key = stage.split("_")[0]

        if stage in _SUBSTAGES:
            # Only worth a line on retry (attempt > 1) or failure — a clean
            # first attempt is already covered by the parent stage's line.
            if status == "failed":
                self._print(stage_key, self._c(_RED, "✗") + f" {message}"
                            + (f" (attempt {attempt})" if attempt else ""))
            elif attempt and attempt > 1 and status == "started":
                self._print(stage_key, self._c(_YELLOW, "↻") + f" retry {attempt} — {message.lower()}")
            return

        if status == "started":
            self._started_at[stage_key] = time.monotonic()
            self._print(stage_key, self._c(_DIM, "running…"))
            return

        elapsed = time.monotonic() - self._started_at.get(stage_key, time.monotonic())
        mark = self._c(_GREEN, "✓") if status == "completed" else self._c(_RED, "✗")
        detail = self._summary(stage_key, status, message, metrics) or message
        timing = self._c(_DIM, f" ({elapsed:.1f}s)") if stage_key in self._started_at else ""
        self._print(stage_key, f"{mark} {detail}{timing}")

    def _summary(self, stage: str, status: str, message: str, metrics: dict[str, Any]) -> str | None:
        if status != "completed":
            return None
        if stage == "run":
            return message  # "Harvested N record(s)" is already the real tally
        if stage == "m2":
            parts = []
            if "n_figures" in metrics:
                parts.append(_pluralize(metrics["n_figures"], "figure"))
            if "n_candidate_signals" in metrics:
                parts.append(_pluralize(metrics["n_candidate_signals"], "candidate signal"))
            return " · ".join(parts) if parts else None
        if stage == "m3":
            n = metrics.get("n_hypotheses")
            if n is not None:
                noun = "falsifiable hypothesis" if n == 1 else "falsifiable hypotheses"
                return f"{n} {noun}"
        if stage == "m4":
            n_rows = metrics.get("n_rows")
            n_reject = metrics.get("n_rejected")
            n_total = metrics.get("n_adjudicated")
            if n_total is not None:
                bits = [f"held-out n={n_rows}" if n_rows is not None else "held-out"]
                bits.append(f"{n_reject}/{n_total} reject")
                return " · ".join(bits)
        return None


class MultiSink:
    """Fan one `.emit(...)` call out to several progress sinks.

    Lets the terminal narrator run alongside a real
    :class:`~evalrx.analysis.workbench.EventSink` (``--progress-path``)
    without either one knowing about the other. `None` entries are dropped
    so callers can pass an optional sink straight through.
    """

    def __init__(self, *sinks: Any) -> None:
        self._sinks = [s for s in sinks if s is not None]

    def __bool__(self) -> bool:
        return bool(self._sinks)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for sink in self._sinks:
            sink.emit(*args, **kwargs)
