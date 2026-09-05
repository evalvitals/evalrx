"""Compatibility loaders plus a dependency-free HTML report server.

A single-shot pipeline produces one output directory; this loader reads it.
Two product shapes are recognised:

- **explore output**: an ``exploratory_report.json`` (or ``fused_report.json``)
  directly in the directory, with ``figures/`` and ``tables/`` beside it.
- **loop run**: a ``logs_*/run_log.jsonl`` (M2 stats / M3 hypotheses + chart
  references / M4 / Fix) plus an optional ``fused_report.json``.

There is no multi-turn chat session anymore — ``load_run`` replaces the old
``load_session`` that walked ``turn_*`` subdirectories.
"""

from __future__ import annotations

import functools
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from evalrx.reporting.compiler import compile_diagnostic_report


class _QuietHandler(SimpleHTTPRequestHandler):
    """Do not emit one log line per asset request in the report server."""

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _report_path(run_dir: Path, *, no_audio: bool) -> Path:
    """Return a finished report, compiling it if needed."""
    report_path = run_dir / "report.html"
    if report_path.exists():
        return report_path
    from evalrx.reporting.html_report import build_html_report

    return build_html_report(run_dir, out_path=report_path, no_audio=no_audio)


def serve_report(
    run_dir: str | Path,
    *,
    port: int = 8501,
    no_audio: bool = False,
    open_browser: bool = True,
    block: bool = True,
) -> int:
    """Serve a generated report locally without a UI framework.

    The server is deliberately bound to loopback: reports may contain prompts,
    model outputs, and embedded user media.  ``block=False`` is provided for
    callers/tests that manage the returned process lifetime themselves.
    """
    root = Path(run_dir).resolve()
    report = _report_path(root, no_audio=no_audio)
    handler = functools.partial(_QuietHandler, directory=str(report.parent))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_port}/{report.name}"
    print(f"Serving diagnostic report at {url}")
    if open_browser:
        webbrowser.open(url)
    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return 0
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def launch_dashboard(run_dir: str | Path, *, port: int | None = None) -> int:
    """Deprecated compatibility alias for :func:`serve_report`."""
    print("'evalrx dashboard' is deprecated; use 'evalrx serve'.")
    return serve_report(run_dir, port=port or 8501)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a single explore-output or loop-run directory into a view dict.

    Returns ``{root, kind, runs, story, case_studies}`` where:
      - ``kind`` is ``"explore"``, ``"loop"``, ``"casebench"`` or ``"empty"``;
      - ``runs`` is a list of ``{name, dir, report}`` (explore reports found,
        most-specific first); the dashboard renders these as report cards;
      - ``story`` is a parsed loop narrative (``None`` for explore output) —
        see :func:`load_loop_story`;
      - ``case_studies`` is a list of :class:`~evalrx.analysis.case_studio.
        CaseStudy` (non-empty only for ``kind == "casebench"``) — a
        paper-method bench run under ``examples/``, whose report pairs per-case
        outcomes with the benchmark manifest beside it so each case can be
        played/read and answered by a human.
    """
    root = Path(run_dir).resolve()
    runs: list[dict[str, Any]] = []

    for fname in ("exploratory_report.json", "fused_report.json"):
        report = _read_json(root / fname)
        if report is not None:
            runs.append({"name": fname.replace(".json", ""), "dir": str(root), "report": report})

    story = load_loop_story(root)
    if story is not None:
        # A loop run may also carry a fused_report.json (Step 1 explore artifact).
        return {"root": str(root), "kind": "loop", "runs": runs, "story": story,
                "case_studies": []}

    if runs:
        return {"root": str(root), "kind": "explore", "runs": runs, "story": None,
                "case_studies": []}

    # A paper-method bench run (examples/**/run.py): no explore report and no
    # loop log, just outputs/<name>.json + the benchmark manifest it read.
    case_studies = load_case_studies(root)
    if case_studies:
        return {"root": str(root), "kind": "casebench", "runs": [], "story": None,
                "case_studies": case_studies}

    # Example/project roots conventionally keep a loop run under ``outputs/``.
    # Accept the documented project-root invocation as well as the run directory
    # itself, while retaining case-book precedence for legacy bench examples
    # whose report JSON also lives under ``outputs/``.
    outputs_dir = root / "outputs"
    if outputs_dir.is_dir():
        nested = load_run(outputs_dir)
        if nested["kind"] != "empty":
            return nested

    # Legacy / fallback: a directory of turn_* explore reports (pre-retirement).
    for turn_dir in sorted(root.glob("turn_*")):
        report = _read_json(turn_dir / "exploratory_report.json")
        if report is not None:
            runs.append({"name": turn_dir.name, "dir": str(turn_dir), "report": report})
    kind = "explore" if runs else "empty"
    return {"root": str(root), "kind": kind, "runs": runs, "story": None, "case_studies": []}


def load_case_studies(run_dir: str | Path) -> list[Any]:
    """Every paper-method bench report under *run_dir*, joined with its manifest.

    Separate from :func:`load_run` so the upload workbench and tests can ask for
    case books directly. Returns ``[]`` when the directory holds no bench report.
    """
    from evalrx.analysis.case_studio import build_case_study, find_bench_reports

    studies = [build_case_study(p) for p in find_bench_reports(Path(run_dir).resolve())]
    return [s for s in studies if s is not None]


def load_loop_story(run_dir: str | Path) -> dict[str, Any] | None:
    """Parse a loop run's ``run_log.jsonl`` into an ordered diagnostic story.

    Looks for ``run_log.jsonl`` directly or under any ``logs*`` subdirectory.
    Returns ``None`` when no loop log is present (i.e. this is explore output).
    """
    root = Path(run_dir).resolve()
    # A single run can be split across several logs (e.g. logs_m1/ for M1 and a
    # logs_m2_5/ or logs_analysis/ for the M2+ arc). We MERGE the shared M1 probe
    # with the M2+ arc — but a directory may hold SEVERAL M2+ arcs that are
    # different runs (e.g. a descriptive analysis-phase pass AND a stale
    # all-in-one confirm pass). Merging both mixes a descriptive run with
    # surgeries/verdicts from another run. So: keep every M1-only log, but among
    # the M2+ logs keep only the single most-recent arc.
    candidate_paths = [
        p for p in [root / "run_log.jsonl", *sorted(root.glob("logs*/run_log.jsonl"))]
        if p.exists()
    ]
    if not candidate_paths:
        return None

    _M2PLUS = {"analysis", "diagnosis", "surgery", "fix"}
    events_by_path: dict[Path, list[dict[str, Any]]] = {}
    for lp in candidate_paths:
        evs: list[dict[str, Any]] = []
        try:
            for line in lp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
        events_by_path[lp] = evs

    m2plus_logs = [p for p, evs in events_by_path.items()
                   if any(str(e.get("event")) in _M2PLUS for e in evs)]
    if len(m2plus_logs) > 1:
        newest = max(m2plus_logs, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        # Drop the older/stale M2+ arcs; keep M1-only logs and the newest arc.
        log_paths = [p for p in candidate_paths
                     if p not in m2plus_logs or p == newest]
    else:
        log_paths = candidate_paths

    events: list[dict[str, Any]] = [e for lp in log_paths for e in events_by_path.get(lp, [])]
    if not events:
        return None

    by_event: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_event.setdefault(str(ev.get("event", "")), []).append(ev)

    # Display the log that actually carries the diagnostic arc (most events).
    log_path = max(log_paths, key=lambda p: p.stat().st_size if p.exists() else 0)
    explore_report, explore_dir = _find_explore_report(root, log_path)

    # run_start/loop_end bracket the whole run — prefer the carrier log's own
    # (the arc actually being displayed), falling back to any log so an M1-only
    # log still surfaces a run_start when no M2+ arc exists yet.
    carrier_events = events_by_path.get(log_path, [])
    run_start = next((e for e in carrier_events if e.get("event") == "run_start"), None)
    if run_start is None:
        run_start = next((e for e in events if e.get("event") == "run_start"), None)
    _carrier_loop_ends = [e for e in carrier_events if e.get("event") == "loop_end"]
    loop_end = _carrier_loop_ends[-1] if _carrier_loop_ends else None
    if loop_end is None:
        _all_loop_ends = [e for e in events if e.get("event") == "loop_end"]
        loop_end = _all_loop_ends[-1] if _all_loop_ends else None

    agent_steps = _merge_agent_steps(
        by_event.get("agent_decision", []), by_event.get("agent_tool", [])
    )

    story = {
        "log_path": str(log_path),
        "run_start": run_start,
        "loop_end": loop_end,
        "probes": by_event.get("probe", []),
        "agent_steps": agent_steps,
        "mode": "agentic" if agent_steps else "loop",
        "analyses": by_event.get("analysis", []),
        "diagnoses": by_event.get("diagnosis", []),
        "surgeries": by_event.get("surgery", []),
        "fixes": by_event.get("fix", []),
        "n_events": len(events),
        # The Step-1 explorer report (charts/observations M3 was allowed to consult)
        # usually lives in a SIBLING dir (e.g. OUT/fused/) — not next to the log —
        # so resolve it across the run tree rather than requiring co-location.
        "explore_report": explore_report,
        "explore_dir": explore_dir,
        # Convention files a stage may have written next to the log (not events —
        # so no new run_log schema surface is needed for these).
        "m4_results": _read_sibling_json(root, log_path, "report/m4_results.json") or [],
        "failure_modes": _read_sibling_json(root, log_path, "artifacts/failure_modes.json"),
    }
    if not story["diagnoses"]:
        proposed = _read_proposed_hypotheses(root)
        if proposed:
            story["diagnoses"] = [{
                "event": "diagnosis",
                "cycle": 0,
                "n_hypotheses": len(proposed),
                "hypotheses": proposed,
                "source": "analysis/proposed_hypotheses.json",
            }]
    story["diagnostic_report"] = compile_diagnostic_report(story, explore_report).to_dict()
    return story


def _read_proposed_hypotheses(root: Path) -> list[dict[str, Any]]:
    """Fallback for analysis-phase dashboards when the M3 log event is absent."""
    for path in (root / "analysis" / "proposed_hypotheses.json", root / "proposed_hypotheses.json"):
        raw = _read_json(path)
        if isinstance(raw, list):
            return [h for h in raw if isinstance(h, dict)]
        if isinstance(raw, dict) and isinstance(raw.get("hypotheses"), list):
            return [h for h in raw["hypotheses"] if isinstance(h, dict)]
    return []


def _find_explore_report(root: Path, log_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Locate the Step-1 explore/fused report for a loop run.

    The loop writes ``run_log.jsonl`` under ``logs*/`` while the fused pipeline
    writes ``fused_report.json`` under a separate ``fused/`` dir. Search the run
    tree and its parents/siblings so the dashboard can show what M3 consulted."""
    bases = [root, log_path.parent, log_path.parent.parent]
    seen: set[Path] = set()
    for base in bases:
        if base in seen or not base.exists():
            continue
        seen.add(base)
        for pattern in (
            "fused_report.json", "exploratory_report.json",
            "fused/fused_report.json", "*/fused_report.json", "*/exploratory_report.json",
        ):
            for cand in sorted(base.glob(pattern)):
                report = _read_json(cand)
                if report is not None and (report.get("charts") or report.get("observations")):
                    return report, str(cand.parent)
    return None, None


def _read_sibling_json(root: Path, log_path: Path, *rel_paths: str) -> Any | None:
    """Look for any of *rel_paths* (e.g. ``"report/m4_results.json"``) under
    ``root``, then the log's parent and grandparent — mirrors
    :func:`_find_explore_report`'s search so convention files resolve whether
    *run_dir* is a bare RunContext root or a ``logs*/`` subdir was selected as
    the carrier log. Returns the first hit, or ``None``."""
    bases = [root, log_path.parent, log_path.parent.parent]
    seen: set[Path] = set()
    for base in bases:
        if base in seen or not base.exists():
            continue
        seen.add(base)
        for rel in rel_paths:
            data = _read_json(base / rel)
            if data is not None:
                return data
    return None


def _merge_agent_steps(
    decisions: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge ``agent_decision`` + ``agent_tool`` events (AgenticDiagnoseLoop)
    into one ordered per-step list.

    Each entry: ``{step, action, params, rationale, valid, repair_attempts,
    fallback_used, judge_io, outcome}`` where ``outcome`` is
    ``{tool, ok, summary, error, duration_sec} | None`` (``None`` when the
    matching ``agent_tool`` event wasn't logged, e.g. a truncated run)."""
    tools_by_step = {t.get("step"): t for t in tools}
    steps: list[dict[str, Any]] = []
    for d in sorted(decisions, key=lambda e: e.get("step", 0)):
        tool_ev = tools_by_step.get(d.get("step"))
        outcome = None
        if tool_ev is not None:
            outcome = {
                "tool": tool_ev.get("tool"),
                "ok": tool_ev.get("ok"),
                "summary": tool_ev.get("summary", ""),
                "error": tool_ev.get("error"),
                "duration_sec": tool_ev.get("duration_sec"),
            }
        steps.append({
            "step": d.get("step"),
            "action": d.get("action"),
            "params": d.get("params") or {},
            "rationale": d.get("rationale", ""),
            "valid": d.get("valid", True),
            "repair_attempts": d.get("repair_attempts", 0),
            "fallback_used": d.get("fallback_used", False),
            "judge_io": d.get("judge_io"),
            "outcome": outcome,
        })
    return steps
