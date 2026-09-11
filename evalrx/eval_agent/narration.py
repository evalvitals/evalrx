"""Live per-stage terminal narration for the M1-M5 diagnosis loop
(:class:`~evalrx.eval_agent.run_logger_v2.RunLoggerV2`).

Sibling of :class:`evalrx.analysis.narration.TerminalNarrator` (built for
the `evalrx explore`/`run-codebase` CLI pipeline) — same visual grammar
("M2  explore ············ ...") and the same non-simulated promise: every
count here comes from the real event RunLoggerV2 just logged, nothing is
invented. The two narrators don't share code (small, deliberate style-block
duplication below) because their event shapes differ: EventSink emits
started/completed pairs, RunLoggerV2 emits one already-finished record per
stage occurrence, so this one always prints exactly one line per event
instead of a running/completed pair.

Opt in on any ``RunLoggerV2``::

    RunLoggerV2(run_dir=out / "logs", narrate=True)
"""

from __future__ import annotations

import os
import sys
from typing import IO, Any

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"

_STAGE_LABELS = {
    "M1": "probe",
    "M2": "explore",
    "M3": "diagnose",
    "M4": "verify",
    "M5": "intervene",
}
# Not in _STAGE_LABELS on purpose: that dict also gates on_event()'s "is this
# a real M-stage" check, and "RUN" (the run-level bookend) is never a valid
# `stage` argument there -- only on_run_start/on_run_event use this label.
_RUN_LABEL = "loop"

# Event keys worth a narrated line. Everything else RunLoggerV2 logs
# (model_call, tool_codegen, tool_registry, agent_decision/tool, cases,
# report_published, diagnose_reports) is real detail too, but at a finer
# grain than "what stage is the loop on" — narrating all of it would bury
# the M1-M5 signal this exists to show.
_NARRATED_KEYS = {"probe", "analysis", "explore", "diagnosis", "surgery", "experiment", "fix"}


def _supports_color(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("EVALRX_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _cycle_prefix(record: dict[str, Any]) -> str:
    cycle = record.get("cycle")
    return f"cycle {cycle} · " if cycle not in (None, -1) else ""


def _or_na(value: Any) -> Any:
    return "n/a" if value is None else value


class LoopNarrator:
    """Prints one real, aligned line per M1-M5 event, as RunLoggerV2 logs it."""

    LABEL_WIDTH = 12

    def __init__(self, *, stream: IO[str] | None = None, color: bool | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.color = _supports_color(self.stream) if color is None else color

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self.color and text else text

    def _row(self, key: str, detail: str) -> str:
        label = _STAGE_LABELS.get(key, _RUN_LABEL if key == "RUN" else key.lower())
        badge = self._c(_BOLD + _CYAN, key.ljust(3))
        dots = self._c(_DIM, "·" * max(4, self.LABEL_WIDTH + 2 - len(label)))
        return f"{badge} {label.ljust(self.LABEL_WIDTH)}{dots} {detail}"

    def _print(self, key: str, detail: str) -> None:
        print(self._row(key, detail), file=self.stream)

    # -- stage events (M1-M5), from RunLoggerV2._append_stage -------------

    def on_event(self, stage: str, key: str, record: dict[str, Any]) -> None:
        if key not in _NARRATED_KEYS or stage not in _STAGE_LABELS:
            return
        if key == "fix":
            self._print_fix(stage, record)
            return
        detail = self._summarize(stage, key, record)
        if detail:
            self._print(stage, detail)

    def _summarize(self, stage: str, key: str, record: dict[str, Any]) -> "str | None":
        prefix = _cycle_prefix(record)
        dur = record.get("duration_sec")
        timing = self._c(_DIM, f" ({dur:.1f}s)") if dur is not None else ""

        if key == "probe":
            n = len(record.get("analyzers") or [])
            return f"{prefix}{n} analyzer{'s' if n != 1 else ''} run{timing}"

        if key == "analysis":
            bits = [f"{record.get('n_findings', 0)} finding(s)"]
            if record.get("severity"):
                bits.append(f"severity={record['severity']}")
            if record.get("figures"):
                bits.append(f"{len(record['figures'])} figure(s)")
            return f"{prefix}" + " · ".join(bits) + timing

        if key == "explore":
            return f"{prefix}explore report ready{timing}"

        if key == "diagnosis":
            n = record.get("n_hypotheses", 0)
            noun = "hypothesis" if n == 1 else "hypotheses"
            return f"{prefix}{n} falsifiable {noun} proposed{timing}"

        if key == "surgery":
            status = record.get("status") or "?"
            # M4 (hypothesis verification): the mark is the verdict itself
            # -- "fixed" isn't even a meaningful concept until M5. M5
            # (intervention): the mark is whether it actually fixed the
            # failure, which is exactly what "fixed" means there.
            ok = str(status).lower() == "supported" if stage == "M4" else bool(record.get("fixed"))
            mark = self._c(_GREEN, "✓") if ok else self._c(_RED, "✗")
            hyp = str(record.get("hypothesis") or "")
            hyp = hyp if len(hyp) <= 70 else hyp[:67] + "..."
            return f"{prefix}{mark} {hyp} — {status}{timing}"

        if key == "experiment":
            verdict = record.get("verdict") or record.get("status") or "?"
            mark = self._c(_GREEN, "✓") if record.get("fixed") else self._c(_RED, "✗")
            return f"{prefix}{mark} repair experiment — {verdict}"

        return None

    def _print_fix(self, stage: str, record: dict[str, Any]) -> None:
        """M5's tiered repair ladder — one line per candidate tried, then the pick."""
        attempted = record.get("attempted") or []
        for attempt in attempted:
            self._print(stage, self._tier_line(attempt))
        best = record.get("best") or {}
        if best.get("name"):
            effect = best.get("effect")
            eff_str = f"Δ={effect:+.3f} " if isinstance(effect, (int, float)) else ""
            self._print(stage, f"best: {best['name']} {eff_str}"
                        f"(n_fixed={_or_na(best.get('n_fixed'))}, n_broken={_or_na(best.get('n_broken'))})")
        else:
            self._print(stage, f"{len(attempted)} candidate(s) tried — none selected")

    def _tier_line(self, attempt: dict[str, Any]) -> str:
        # A full row per candidate (same convention as the explore narrator's
        # substage retry lines: every printed line re-states its M-stage tag
        # rather than relying on terminal-fragile manual indentation).
        tier = attempt.get("tier", "?")
        name = attempt.get("name", "?")
        mark = self._c(_GREEN, "✓") if attempt.get("fixed") else self._c(_RED, "✗")
        effect = attempt.get("effect")
        eff_str = f"{effect:+.3f}" if isinstance(effect, (int, float)) else "?"
        return (f"{self._c(_DIM, tier)} {name}  {mark} {eff_str} "
                f"(n_fixed={_or_na(attempt.get('n_fixed'))}, n_broken={_or_na(attempt.get('n_broken'))})")

    # -- run-level events, from RunLoggerV2._append_run / log_run_start ---

    def on_run_start(self, entry: dict[str, Any]) -> None:
        model = entry.get("model") or "target model"
        n_cases = entry.get("n_cases")
        bits = [f"model={model}"]
        if n_cases is not None:
            bits.append(f"n_cases={n_cases}")
        self._print("RUN", "starting · " + " · ".join(bits))

    def on_run_event(self, key: str, record: dict[str, Any]) -> None:
        if key != "loop_end":
            return
        bits = [f"{record.get('cycles', '?')} cycle(s)"]
        if "resolved" in record:
            bits.append(f"resolved={record['resolved']}")
        if "stopped_by" in record:
            bits.append(f"stopped_by={record['stopped_by']}")
        n = record.get("n_hypotheses")
        if n is not None:
            bits.append(f"{n} hypothes{'is' if n == 1 else 'es'}")
        self._print("RUN", "done · " + " · ".join(bits))
