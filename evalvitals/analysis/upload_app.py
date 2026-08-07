"""EvalVitals web workbench — upload a .zip of results, run M2+M3, read the report.

``evalvitals web [WORKSPACE] [--port N] [--backend B] ...`` serves this
Streamlit app. Unlike ``dashboard_app.py`` (a read-only viewer over one
existing run directory), this app *creates* runs: each uploaded archive
becomes one run directory under the workspace and is analysed by the ordinary
``evalvitals explore`` CLI (M2 exploratory analysis + M3 hypothesis proposal)
running as a detached subprocess — closing the browser tab never kills an
analysis, and a finished run renders with the exact same tabs as
``evalvitals dashboard <out>``.

Run layout::

    <workspace>/<run-name>/
        upload.zip     the archive exactly as received
        data/          extracted payload (what the explorer reads)
        output/        explore artifacts (exploratory_report.json, figures/, ...)
        explore.log    live stdout+stderr of the analysis
        job.sh         the exact command (re-runnable by hand)
        job.json       parameters + pid + start time
        exit_code      written when the subprocess finishes
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from evalvitals.analysis.workbench import (
    EventSink,
    ThreadStore,
    UploadLimits,
    extract_archive,
    ingest_directory,
)

# Backends the explore CLI accepts; the first three are skill-capable and get
# the bundled figure/analysis skills automatically (see agent_runtime.skills).
# The workbench intentionally exposes the three configured local coding-agent
# backends.  Other adapters remain available to the lower-level CLI API.
BACKENDS = ("claude_code", "codex", "antigravity")

DEFAULT_QUESTION = "Explore this dataset and surface the patterns that matter."
NEW_ANALYSIS_LABEL = "➕ New analysis"

_JUNK_DIRS = ("__MACOSX",)
_JUNK_NAMES = (".DS_Store", "Thumbs.db")

_STATE_ICONS = {"running": "🟡", "done": "🟢", "failed": "🔴", "stale": "⚪", "canceled": "⚪"}
_TIMELINE_STAGE_LABELS = {
    "ingest": "Upload preparation",
    "discover": "File discovery",
    "normalize": "Dataset normalization",
    "job": "Analysis worker",
    "m2": "Exploratory analysis (M2)",
    "m2_codegen": "M2 · code generation",
    "m2_execute": "M2 · code execution",
    "m3": "Hypothesis proposal (M3)",
    "persist": "Saving analysis artifacts",
    "route": "Follow-up request",
}


# ── run bookkeeping (pure helpers, unit-tested without streamlit) ────────────


def stage_zip(payload: bytes, run_dir: Path, *, limits: UploadLimits | None = None) -> Path:
    """Extract an uploaded zip into ``run_dir/data`` and return the directory
    the explorer should read.

    Guards against zip-slip (absolute member names or ``..`` escapes raise
    ``ValueError``), skips archive junk (``__MACOSX/``, ``.DS_Store``), and
    unwraps the common single-top-level-folder layout ("zip of a directory"):
    when everything sits under one folder, that folder is returned instead of
    ``data/`` itself.
    """
    return extract_archive(payload, run_dir / "data", limits=limits)


def build_explore_argv(
    data_dir: Path,
    out_dir: Path,
    *,
    question: str,
    outcome_col: str,
    backend: str,
    model: str,
    timeout_sec: int,
    holdout_frac: float = 0.0,
    holdout_confirm: bool = False,
    progress_path: Path | None = None,
    thread_id: str = "",
    turn_id: str = "",
) -> list[str]:
    """The exact ``evalvitals explore`` invocation for one uploaded run."""
    argv = [
        sys.executable, "-m", "evalvitals.cli", "explore", str(data_dir),
        "--backend", backend,
        "--out", str(out_dir),
        "-q", question,
        "--timeout-sec", str(int(timeout_sec)),
        "--max-attempts", "3",
        "--max-files", "600",  # normalized records + up to 500 media units
    ]
    if outcome_col:
        argv += ["--outcome-col", outcome_col]
    if model:
        argv += ["--model", model]
    if holdout_frac > 0:
        argv += ["--holdout-frac", str(round(holdout_frac, 4))]
    if holdout_confirm:
        argv += ["--holdout-confirm"]
    if progress_path:
        argv += ["--progress-path", str(progress_path), "--thread-id", thread_id, "--turn-id", turn_id]
    return argv


def launch_explore_job(
    run_dir: Path,
    data_dir: Path,
    *,
    question: str,
    outcome_col: str,
    backend: str,
    model: str,
    timeout_sec: int,
    mode: str = "explore",
    explore_share: float = 1.0,
    thread_id: str = "",
    turn_id: str = "",
    events_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Start the analysis as a detached subprocess and persist its job record.

    *mode* is ``"explore"`` (M2+M3 only) or ``"verify"`` (M2+M3, then the
    held-out share re-tests the frozen recipes + hypotheses). *explore_share*
    is the explore fraction; the remainder is held out — verified in
    ``verify`` mode, merely reserved in ``explore`` mode.

    The command goes through a tiny ``job.sh`` wrapper so the exit code lands
    in ``exit_code`` even if this server restarts while the run is in flight.
    """
    holdout_frac = max(0.0, round(1.0 - float(explore_share), 4))
    thread_id = thread_id or run_dir.name
    turn_id = turn_id or "initial"
    events_path = events_path or run_dir / "events.jsonl"
    out_dir = out_dir or run_dir / "output"
    EventSink(events_path, thread_id=thread_id, turn_id=turn_id).emit(
        "job", "queued", "Analysis job queued"
    )
    argv = build_explore_argv(
        data_dir, out_dir, question=question, outcome_col=outcome_col,
        backend=backend, model=model, timeout_sec=timeout_sec,
        holdout_frac=holdout_frac, holdout_confirm=(mode == "verify"),
        progress_path=events_path, thread_id=thread_id, turn_id=turn_id,
    )
    job_sh = run_dir / "job.sh"
    job_sh.write_text(
        "#!/usr/bin/env bash\n"
        + " ".join(shlex.quote(a) for a in argv)
        + " > explore.log 2>&1\necho $? > exit_code\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        ["bash", str(job_sh)], cwd=run_dir, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    job = {
        "pid": proc.pid,
        "argv": argv,
        "question": question,
        "outcome_col": outcome_col,
        "backend": backend,
        "model": model,
        "timeout_sec": int(timeout_sec),
        "mode": mode,
        "explore_share": float(explore_share),
        "holdout_frac": holdout_frac,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "events_path": str(events_path),
        "thread_id": thread_id,
        "turn_id": turn_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    return job


def launch_answer_job(
    turn_dir: Path,
    *,
    thread_dir: Path,
    report_path: Path,
    question: str,
    provider: str,
    model: str,
    timeout_sec: int,
    thread_id: str,
    turn_id: str,
) -> dict[str, Any]:
    """Start an artifact-grounded answer turn without re-running M2/M3."""
    turn_dir.mkdir(parents=True, exist_ok=True)
    events_path = thread_dir / "events.jsonl"
    EventSink(events_path, thread_id=thread_id, turn_id=turn_id).emit(
        "route", "queued", "Queued an artifact-grounded follow-up answer"
    )
    argv = [
        sys.executable, "-m", "evalvitals.analysis.workbench_worker", "answer",
        "--report", str(report_path), "--question", question,
        "--provider", provider, "--events", str(events_path),
        "--thread-dir", str(thread_dir), "--turn-dir", str(turn_dir),
        "--thread-id", thread_id, "--turn-id", turn_id,
        "--timeout-sec", str(int(timeout_sec)),
    ]
    if model:
        argv += ["--model", model]
    (turn_dir / "job.sh").write_text(
        "#!/usr/bin/env bash\n" + " ".join(shlex.quote(a) for a in argv)
        + " > explore.log 2>&1\necho $? > exit_code\n", encoding="utf-8"
    )
    proc = subprocess.Popen(
        ["bash", str(turn_dir / "job.sh")], cwd=turn_dir, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    job = {
        "pid": proc.pid, "argv": argv, "question": question, "backend": provider,
        "model": model, "timeout_sec": int(timeout_sec), "mode": "answer",
        "out_dir": str(turn_dir), "events_path": str(events_path),
        "thread_id": thread_id, "turn_id": turn_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (turn_dir / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    return job


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_status(run_dir: Path) -> dict[str, Any]:
    """Classify one run dir: ``done`` (report exists), ``running`` (pid alive),
    ``failed`` (exited without a report), or ``stale`` (no process, no exit
    code — e.g. the machine restarted mid-run)."""
    report_path = run_dir / "output" / "exploratory_report.json"
    exit_code: int | None = None
    exit_file = run_dir / "exit_code"
    if exit_file.exists():
        try:
            exit_code = int(exit_file.read_text().strip())
        except ValueError:
            exit_code = None

    canceled = False
    try:
        job_data = json.loads((run_dir / "job.json").read_text())
        events_path = Path(job_data.get("events_path") or run_dir / "events.jsonl")
        turn_id = str(job_data.get("turn_id") or "")
        events = _read_jsonl(events_path)
        canceled = any(e.get("turn_id") == turn_id and e.get("status") == "canceled" for e in events)
    except Exception:
        pass

    if report_path.exists() or (run_dir / "answer.md").exists():
        state = "done"
    elif canceled:
        state = "canceled"
    elif exit_code is not None:
        state = "failed"
    else:
        pid = 0
        try:
            pid = int(json.loads((run_dir / "job.json").read_text()).get("pid", 0))
        except Exception:
            pass
        state = "running" if _pid_alive(pid) else "stale"
    return {"state": state, "exit_code": exit_code, "report_path": str(report_path)}


def list_runs(workspace: Path) -> list[Path]:
    """Run dirs under the workspace (anything carrying a job.json), newest first."""
    runs = [d for d in workspace.iterdir() if d.is_dir() and (d / "job.json").exists()]
    return sorted(runs, key=lambda d: (d / "job.json").stat().st_mtime, reverse=True)


def archive_run(workspace: Path, run_dir: Path) -> Path:
    """Move one local workbench run into ``<workspace>/.trash``.

    This deliberately is not a recursive delete: the user can recover the
    complete upload, dataset bundle, report, and conversation from the trash.
    Only a direct child of *workspace* carrying ``job.json`` is accepted, so
    an attached result directory or an arbitrary path can never be moved.
    """
    root = workspace.resolve()
    source = run_dir.resolve()
    if source.parent != root or not source.is_dir() or not (source / "job.json").is_file():
        raise ValueError("only a local workbench run can be moved to trash")

    trash = root / ".trash"
    trash.mkdir(exist_ok=True)
    target = trash / source.name
    if target.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = trash / f"{source.name}_{suffix}"
        serial = 2
        while target.exists():
            target = trash / f"{source.name}_{suffix}_{serial}"
            serial += 1
    shutil.move(str(source), str(target))
    return target


def _run_name(upload_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(upload_name).stem).strip("_") or "upload"
    return f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _log_tail(run_dir: Path, n_lines: int = 60) -> str:
    log = run_dir / "explore.log"
    if not log.exists():
        return "(no output yet)"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:]) or "(no output yet)"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _timeline_steps(
    events: list[dict[str, Any]], *, turn_id: str = "", run_state: str = ""
) -> list[dict[str, Any]]:
    """Collapse event history into the latest state of each visible step.

    Events are append-only for auditability, so a literal event list naturally
    contains both ``started`` and ``completed`` rows. A user-facing timeline
    instead needs one current state per task. A few older runs lack matching
    terminal events; in those cases a later dependent stage (or a final report)
    safely establishes that the earlier task is no longer running.
    """
    scoped = [
        event for event in events
        if not turn_id or not event.get("turn_id") or str(event.get("turn_id")) == turn_id
    ]
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in scoped:
        stage = str(event.get("stage") or "job")
        if stage not in latest:
            order.append(stage)
        latest[stage] = dict(event)

    def _is_open(stage: str) -> bool:
        return str(latest.get(stage, {}).get("status") or "") in {"queued", "started"}

    def _complete_if_open(stage: str, reason: str) -> None:
        if _is_open(stage):
            latest[stage]["status"] = "completed"
            latest[stage]["message"] = f"{latest[stage].get('message') or stage} — complete ({reason})"

    # A dependent stage can only have begun after the preceding one finished.
    if "discover" in latest or "normalize" in latest:
        _complete_if_open("ingest", "preparation continued")
    if "m2_execute" in latest or "m2" in latest:
        _complete_if_open("m2_codegen", "analysis execution began")
    if "m3" in latest or "persist" in latest:
        _complete_if_open("m2", "next analysis stage began")
        _complete_if_open("m2_execute", "M2 finished")
    if "persist" in latest:
        _complete_if_open("m3", "artifacts were saved")

    # The report is rendered only after the run is terminal. Treat any old
    # dangling lifecycle event as finished rather than showing a false spinner.
    if run_state == "done":
        for stage in latest:
            _complete_if_open(stage, "final report is available")
    elif run_state in {"failed", "canceled"}:
        for stage in latest:
            if _is_open(stage):
                latest[stage]["status"] = run_state
                latest[stage]["message"] = (
                    f"{latest[stage].get('message') or stage} — "
                    f"{('canceled' if run_state == 'canceled' else 'did not finish')}"
                )

    return [latest[stage] for stage in order]


def _render_timeline(st: Any, run_dir: Path, *, turn_id: str = "", run_state: str = "") -> None:
    """Render current task states from durable events, never private reasoning."""
    events = _timeline_steps(_read_jsonl(run_dir / "events.jsonl"), turn_id=turn_id, run_state=run_state)
    if not events:
        st.caption("Waiting for the worker to publish its first stage…")
        return
    labels = {
        "queued": "Queued", "started": "Running", "completed": "Complete",
        "failed": "Failed", "canceled": "Canceled",
    }
    rows = []
    for event in events:
        status = str(event.get("status") or "")
        raw_stage = str(event.get("stage") or "job")
        stage = _TIMELINE_STAGE_LABELS.get(raw_stage, raw_stage.replace("_", " "))
        message = str(event.get("message") or "")
        status_cls = status if status in labels else "queued"
        rows.append(
            f'<div class="ev-timeline-row ev-timeline-{status_cls}">'
            '<span class="ev-timeline-dot"></span><div class="ev-timeline-copy">'
            f'<span class="ev-timeline-stage">{html.escape(stage)}</span>'
            f'<span class="ev-timeline-status">{labels.get(status, "Update")}</span>'
            f'<div class="ev-timeline-message">{html.escape(message)}</div>'
            "</div></div>"
        )
    st.markdown('<div class="ev-timeline">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def _render_agent_audit(st: Any, out_dir: Path) -> None:
    """Show only provider-verifiable skill-use evidence, never inferred use."""
    path = out_dir / "agent_audit.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        attempts = payload.get("attempts", []) if isinstance(payload, dict) else []
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(attempts, list) or not attempts:
        return

    with st.expander("Agent & skill audit", expanded=False):
        for index, audit in enumerate(attempts, start=1):
            if not isinstance(audit, dict):
                continue
            execution = audit.get("execution") if isinstance(audit.get("execution"), dict) else {}
            provider = str(audit.get("provider") or "unknown")
            st.markdown(
                f"**Attempt {index} · {provider}** — "
                f"{execution.get('status', 'unknown')} · "
                f"{execution.get('elapsed_sec', '?')} s"
            )
            requested = {str(name) for name in audit.get("skills_requested", [])}
            installed = {str(name) for name in audit.get("skills_installed", [])}
            invoked = {str(name) for name in audit.get("skills_invoked", [])}
            if requested:
                rows = []
                for skill in sorted(requested):
                    rows.append({
                        "skill": skill,
                        "installed": "yes" if skill in installed else "no",
                        "verified use": "yes" if skill in invoked else "no",
                    })
                st.dataframe(rows, hide_index=True, use_container_width=True)
            observation = str(audit.get("skill_observability") or "not_observable")
            if observation == "not_observable":
                st.caption("This provider exposes no machine-readable skill trace. "
                           "Installed is not proof of use.")
            elif requested and not invoked:
                st.caption("No provider-verifiable skill use was observed in this attempt.")
            evidence = audit.get("evidence")
            if isinstance(evidence, list) and evidence:
                st.caption("Evidence")
                st.json(evidence)
            if execution.get("error"):
                st.error(str(execution["error"]))
            elif execution.get("stderr"):
                st.code(str(execution["stderr"]), language="text")


def _inject_workbench_css(st: Any) -> None:
    """Small motion language for live events; respects reduced-motion settings."""
    st.markdown(
        """
        <style>
        .ev-timeline { border-left: 2px solid var(--ev-border); margin: .35rem 0 .7rem .45rem; padding-left: 1rem; }
        .ev-timeline-row { animation: ev-timeline-enter .32s ease-out both; min-height: 2.55rem; padding: .28rem .45rem .36rem .7rem; position: relative; border-radius: 8px; }
        .ev-timeline-row + .ev-timeline-row { margin-top: .1rem; }
        .ev-timeline-dot { background: var(--ev-muted); border: 2px solid var(--ev-panel); border-radius: 50%; box-shadow: 0 0 0 1px var(--ev-border); height: .62rem; left: -1.36rem; position: absolute; top: .72rem; width: .62rem; }
        .ev-timeline-stage { color: var(--ev-text); font-size: .84rem; font-weight: 720; text-transform: capitalize; }
        .ev-timeline-status { color: var(--ev-muted); font-size: .7rem; font-weight: 700; letter-spacing: .03em; margin-left: .4rem; text-transform: uppercase; }
        .ev-timeline-message { color: var(--ev-text-secondary); font-size: .82rem; line-height: 1.35; margin-top: .06rem; }
        .ev-timeline-completed .ev-timeline-dot { background: var(--ev-ok); }
        .ev-timeline-failed .ev-timeline-dot { background: var(--ev-fail); }
        .ev-timeline-canceled .ev-timeline-dot { background: var(--ev-muted); }
        .ev-timeline-started { background: linear-gradient(100deg, color-mix(in srgb, var(--ev-accent) 10%, transparent), color-mix(in srgb, var(--ev-accent) 20%, transparent), color-mix(in srgb, var(--ev-accent) 10%, transparent)); background-size: 220% 100%; animation: ev-timeline-enter .32s ease-out both, ev-timeline-shimmer 2.1s ease-in-out infinite; }
        .ev-timeline-started .ev-timeline-dot { background: var(--ev-accent); box-shadow: 0 0 0 1px var(--ev-accent), 0 0 0 0 color-mix(in srgb, var(--ev-accent) 55%, transparent); animation: ev-timeline-pulse 1.7s ease-out infinite; }
        @keyframes ev-timeline-enter { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes ev-timeline-shimmer { 0% { background-position: 100% 0; } 100% { background-position: -120% 0; } }
        @keyframes ev-timeline-pulse { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--ev-accent) 55%, transparent); } 70% { box-shadow: 0 0 0 .48rem transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }
        @media (prefers-reduced-motion: reduce) { .ev-timeline-row, .ev-timeline-started, .ev-timeline-started .ev-timeline-dot { animation: none !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _backend_label(value: str) -> str:
    return {"claude_code": "Claude Code", "codex": "Codex", "antigravity": "Antigravity"}.get(value, value)


def _render_thread_messages(st: Any, run_dir: Path) -> None:
    for message in _read_jsonl(run_dir / "messages.jsonl"):
        role = str(message.get("role") or "assistant")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.markdown(str(message.get("content") or ""))


def _active_turn_dir(thread_dir: Path) -> Path:
    """Return the current immutable turn, falling back to legacy root runs."""
    try:
        current = str(json.loads((thread_dir / "thread.json").read_text()).get("current_turn") or "")
    except Exception:
        current = ""
    candidate = thread_dir / "turns" / current
    return candidate if current and candidate.is_dir() else thread_dir


def _route_followup(question: str) -> str:
    """Auditable conservative router; uncertain requests take the analysis path."""
    low = question.lower()
    if any(word in low for word in ("hypothesis", "假设", "机制", "验证")):
        return "hypothesize"
    if any(word in low for word in (
        "chart", "plot", "compare", "filter", "slice", "correlation", "统计", "图", "比较", "筛选", "重新", "新")):
        return "analyze"
    return "answer"


def _next_turn_id(thread_dir: Path) -> str:
    turns = thread_dir / "turns"
    existing = [p for p in turns.glob("turn_*") if p.is_dir()] if turns.exists() else []
    return f"turn_{len(existing) + 1:03d}"


def _cancel_job(active_dir: Path, job: dict[str, Any], *, thread_id: str, turn_id: str, events_path: Path) -> bool:
    try:
        pid = int(job.get("pid") or 0)
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ValueError):
        return False
    EventSink(events_path, thread_id=thread_id, turn_id=turn_id).emit(
        "job", "canceled", "Cancellation requested by the user"
    )
    return True


# ── streamlit app ────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", nargs="?", default="evalvitals_web_runs")
    parser.add_argument("--backend", default="claude_code", choices=list(BACKENDS))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument(
        "--attach", action="append", default=[], metavar="DIR",
        help="Existing result directory (explore output or loop run) to list "
             "alongside uploads. Repeatable.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _local_state(path: Path) -> str:
    """Sidebar state for an attached (read-only) result directory."""
    if (path / "exploratory_report.json").exists() or (path / "fused_report.json").exists():
        return "done"
    if (path / "run_log.jsonl").exists() or any(path.glob("logs*/run_log.jsonl")):
        return "done"  # a loop run — renders through the loop story view
    return "stale"


def main() -> None:
    import streamlit as st

    from evalvitals.analysis import dashboard_app as dapp

    args = _parse_args()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    st.set_page_config(
        page_title="EvalVitals — Upload & Explore", layout="wide",
        initial_sidebar_state="expanded",
    )
    dapp._inject_css()
    _inject_workbench_css(st)

    new_label = NEW_ANALYSIS_LABEL
    runs = list_runs(workspace)
    states = {d.name: job_status(_active_turn_dir(d))["state"] for d in runs}
    # Attached read-only result dirs (e.g. an example's committed outputs) sit
    # in the same sidebar as uploads; option values carry an "@" prefix so a
    # local path can never collide with an upload's run name.
    attached = []
    for raw in args.attach:
        p = Path(raw).resolve()
        if p.is_dir() and p not in attached:
            attached.append(p)
    local_states = {f"@{p}": _local_state(p) for p in attached}

    def _label(v: str) -> str:
        if v == new_label:
            return v
        if v.startswith("@"):
            return f"📁 {_STATE_ICONS.get(local_states.get(v, 'stale'), '⚪')} {_pretty_name(Path(v[1:]))}"
        return f"{_STATE_ICONS.get(states.get(v, 'stale'), '⚪')} {_pretty_name(v)}"

    st.sidebar.markdown('<div class="ev-sidebar-title">EvalVitals</div>',
                        unsafe_allow_html=True)
    st.sidebar.caption("Investigate why your eval failed — and prove the fix")
    # Streamlit forbids changing a widget key after the widget has been
    # instantiated in the current run.  A just-created thread therefore sets
    # this pending value; it is consumed before constructing the radio on the
    # next rerun.
    pending_choice = st.session_state.pop("ev_pending_run_choice", None)
    if pending_choice is not None:
        st.session_state["ev_run_choice"] = pending_choice
    choice = st.sidebar.radio(
        "Runs",
        [new_label] + [f"@{p}" for p in attached] + [d.name for d in runs],
        key="ev_run_choice",
        format_func=_label,
    )
    st.sidebar.markdown("---")

    if choice == new_label:
        _render_new_analysis(st, workspace, args, new_label)
    elif choice.startswith("@"):
        _render_local(st, dapp, Path(choice[1:]))
    else:
        _render_run(st, dapp, workspace / choice)


def _render_new_analysis(st: Any, workspace: Path, args: argparse.Namespace,
                         new_label: str) -> None:
    st.markdown(
        '<div class="ev-hero"><h1>Upload &amp; Explore</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload a **.zip** containing tables (JSON / JSONL / CSV / TSV / "
        "Parquet / Excel) and/or images, PDFs, audio, or video. The workbench "
        "builds an auditable dataset bundle, then M2 analyses it and M3 proposes "
        "falsifiable hypotheses."
    )

    uploaded = st.file_uploader("Results archive", type=["zip"])
    question = st.text_area("Analysis question", value=DEFAULT_QUESTION, height=90)

    mode_label = st.radio(
        "Analysis mode",
        ["Explore + held-out verification", "Explore only (M2 + M3)"],
        horizontal=True,
        help="Verification (recommended): part of the rows is held out BEFORE "
             "exploration, then the frozen recipes and hypotheses are re-tested "
             "on rows the explorer never saw (e-BH + LLM judge) — the Held-out "
             "Verdicts tab fills in. Explore only: exploratory analysis and "
             "proposed hypotheses, with nothing confirmed; the Held-out "
             "Verdicts and Fix tabs stay greyed.",
    )
    verify = mode_label.startswith("Explore +")
    if verify:
        explore_share = st.slider(
            "Explore : verdict split", min_value=0.3, max_value=0.9, value=0.6,
            step=0.05, key="ev_share_verify", format="%.2f",
            help="Share of rows the explorer sees; the rest is the held-out "
                 "verdict half (stratified by outcome, deterministic).",
        )
        st.caption(f"explore **{explore_share:.0%}** : verdict **{1 - explore_share:.0%}**")
    else:
        explore_share = st.slider(
            "Explore : reserved split", min_value=0.5, max_value=1.0, value=1.0,
            step=0.05, key="ev_share_explore", format="%.2f",
            help="1.0 analyses everything in-sample. Below 1.0 the remainder "
                 "is held out untouched (saved to holdout_records.json) but "
                 "NOT verified in this mode.",
        )
        st.caption(
            f"explore **{explore_share:.0%}** : reserved **{1 - explore_share:.0%}**"
            + ("" if explore_share < 1.0 else " — all rows analysed in-sample")
        )

    outcome_col = st.text_input(
        "Outcome column", value="label",
        help="Name of the pass/fail (or target) column. Leave empty to "
             "auto-detect by name heuristics, or fall back to unsupervised EDA. "
             "Held-out verification needs it to stratify the split and grade "
             "signals.",
    )
    # Backend / model / timeout are operator knobs, not part of asking a
    # question. Keeping them collapsed makes the primary form four fields
    # instead of seven without taking the controls away.
    with st.expander("Advanced — engine settings", expanded=False):
        col2, col3, col4 = st.columns(3)
        backend = col2.selectbox(
            "Coding-agent backend", list(BACKENDS), index=list(BACKENDS).index(args.backend),
            format_func=_backend_label,
        )
        model = col3.text_input("Model (optional)", value=args.model,
                                help="Backend-specific model id, e.g. claude-opus-4-8.")
        timeout_sec = col4.number_input("Timeout (sec)", min_value=60, max_value=7200,
                                        value=int(args.timeout_sec), step=60)

    if st.button("Start analysis", type="primary", disabled=uploaded is None):
        payload = uploaded.getvalue()
        run_dir = ThreadStore(workspace).create(
            name=Path(uploaded.name).stem, provider=backend, model=model.strip()
        )
        ThreadStore.append_message(
            run_dir, "user", question.strip() or DEFAULT_QUESTION, turn_id="initial"
        )
        try:
            (run_dir / "upload.zip").write_bytes(payload)
            data_dir = stage_zip(payload, run_dir)
            sink = EventSink(run_dir / "events.jsonl", thread_id=run_dir.name, turn_id="initial")
            sink.emit("ingest", "started", "Extracting and normalizing uploaded data")
            bundle = ingest_directory(data_dir, run_dir / "dataset", sink=sink)
            sink.emit("ingest", "completed", "Upload is ready for analysis")
        except (ValueError, zipfile.BadZipFile) as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            st.error(f"Could not read the archive: {exc}")
            st.stop()
        launch_explore_job(
            run_dir, Path(bundle.root), question=question.strip() or DEFAULT_QUESTION,
            outcome_col=outcome_col.strip(), backend=backend,
            model=model.strip(), timeout_sec=int(timeout_sec),
            mode="verify" if verify else "explore",
            explore_share=float(explore_share), thread_id=run_dir.name, turn_id="initial",
        )
        st.session_state["ev_pending_run_choice"] = run_dir.name
        st.rerun()


_GENERIC_DIR_NAMES = {"outputs", "output", "reference_output", "results", "run", "runs"}


def _pretty_name(raw: str | Path) -> str:
    """Directory names are how runs are stored, not how they should read.

    ``outputs_attn_full`` -> ``Attn Full``; a trailing upload timestamp
    (``data_attn_2b_20260726_145427``) is dropped so the list stays scannable.
    A bundle stored under a generic folder (``<example>/reference_output``)
    takes the example's name instead, so two such bundles do not both render
    as the same word.
    """
    path = Path(raw)
    name = path.name
    if name.lower() in _GENERIC_DIR_NAMES and path.parent.name:
        name = path.parent.name
    name = re.sub(r"_\d{8}_\d{6}$", "", name.strip())
    name = re.sub(r"^(outputs?|evalvitals)[_-]*", "", name) or name
    name = name.replace("_", " ").replace("-", " ").strip()
    return name.title() if name else str(raw)


def _render_local(st: Any, dapp: Any, path: Path) -> None:
    """Render an attached (read-only) result directory — an explore output or
    a loop run — with the same views `evalvitals dashboard` would use."""
    from evalvitals.analysis.dashboard import load_run

    st.markdown(f"## 📁 {_pretty_name(path)}")
    st.caption("Reference result · read-only")

    session = load_run(path)
    if session.get("kind") == "loop" and session.get("story"):
        dapp._render_loop_story(Path(session["root"]), session["story"], session["runs"])
        return
    if session["runs"]:
        dapp.render_explore_report(Path(session["root"]), session["runs"][0])
        return
    st.warning("No exploratory_report.json / fused_report.json / run_log.jsonl "
               "found in this directory.")


def _render_followup_input(st: Any, thread_dir: Path, job: dict[str, Any]) -> None:
    question = st.chat_input("Ask a follow-up about this dataset")
    if not question:
        return
    turn_id = _next_turn_id(thread_dir)
    turn_dir = thread_dir / "turns" / turn_id
    turn_dir.mkdir(parents=True, exist_ok=True)
    ThreadStore.append_message(thread_dir, "user", question, turn_id=turn_id)
    ThreadStore.set_current_turn(thread_dir, turn_id)
    route = _route_followup(question)
    sink = EventSink(thread_dir / "events.jsonl", thread_id=thread_dir.name, turn_id=turn_id)
    sink.emit("route", "completed", f"Follow-up routed to {route}")
    provider = str(job.get("backend") or "claude_code")
    model = str(job.get("model") or "")
    timeout_sec = int(job.get("timeout_sec") or 1200)
    if route == "answer":
        report = Path(job.get("out_dir") or thread_dir / "output") / "exploratory_report.json"
        if not report.exists():
            candidates = [thread_dir / "output" / "exploratory_report.json"]
            candidates += sorted((thread_dir / "turns").glob("turn_*/output/exploratory_report.json"), reverse=True)
            report = next((p for p in candidates if p.exists()), report)
        launch_answer_job(
            turn_dir, thread_dir=thread_dir, report_path=report, question=question,
            provider=provider, model=model, timeout_sec=timeout_sec,
            thread_id=thread_dir.name, turn_id=turn_id,
        )
    else:
        # A new chart, slice, or hypothesis is deliberately a fresh immutable
        # M2/M3 version.  It reuses the normalized bundle rather than upload or
        # extraction, and has the previous report available through the thread.
        launch_explore_job(
            turn_dir, thread_dir / "dataset", question=question,
            outcome_col=str(job.get("outcome_col") or ""), backend=provider,
            model=model, timeout_sec=timeout_sec, mode="explore", explore_share=1.0,
            thread_id=thread_dir.name, turn_id=turn_id,
            events_path=thread_dir / "events.jsonl", out_dir=turn_dir / "output",
        )
    st.rerun()


def _render_run(st: Any, dapp: Any, run_dir: Path) -> None:
    active_dir = _active_turn_dir(run_dir)
    try:
        job = json.loads((active_dir / "job.json").read_text(encoding="utf-8"))
    except Exception:
        job = {}
    status = job_status(active_dir)
    state = status["state"]

    st.markdown(f"## {_STATE_ICONS.get(state, '⚪')} {run_dir.name}")
    bits = [f"state: **{state}**"]
    if job.get("started_at"):
        bits.append(f"started {job['started_at']}")
    if job.get("backend"):
        bits.append(f"backend `{job['backend']}`" + (f" · `{job['model']}`" if job.get("model") else ""))
    st.caption(" · ".join(bits))

    # Uploaded runs are local workspace children; attached result directories
    # intentionally do not enter this view and cannot be removed here.
    with st.expander("Manage this run", expanded=False):
        if state == "running":
            st.caption("Cancel the active turn before moving this run to trash.")
        else:
            st.caption("Moves this run, including its upload and artifacts, to the workspace trash. It can be recovered manually.")
        confirmed = st.checkbox(
            "I want to remove this run from the sidebar",
            key=f"ev_archive_confirm_{run_dir.name}",
            disabled=state == "running",
        )
        if st.button(
            "Move run to trash", type="secondary",
            key=f"ev_archive_{run_dir.name}",
            disabled=state == "running" or not confirmed,
        ):
            try:
                archive_run(run_dir.parent, run_dir)
            except (OSError, ValueError) as exc:
                st.error(f"Could not move this run to trash: {exc}")
            else:
                st.session_state["ev_pending_run_choice"] = NEW_ANALYSIS_LABEL
                st.rerun()

    with st.expander("Run parameters", expanded=False):
        st.markdown(f"**Question:** {job.get('question', '—')}")
        st.markdown(f"**Outcome column:** `{job.get('outcome_col') or '(auto)'}`")
        if job.get("mode"):
            share = float(job.get("explore_share", 1.0))
            mode_txt = ("explore + held-out verification"
                        if job["mode"] == "verify" else "explore only (M2+M3)")
            st.markdown(f"**Mode:** {mode_txt} — split "
                        f"{share:.0%} explore : {1 - share:.0%} "
                        f"{'verdict' if job['mode'] == 'verify' else 'reserved'}")
        st.code(" ".join(shlex.quote(a) for a in job.get("argv", [])) or "(unknown)",
                language="bash")

    _render_thread_messages(st, run_dir)
    with st.chat_message("assistant"):
        st.caption("Analysis timeline")
        _render_timeline(
            st, run_dir, turn_id=str(job.get("turn_id") or "initial"), run_state=state
        )

    if state == "running":
        st.info("Analysis in flight — M2 exploratory analysis + M3 hypothesis "
                "proposal. Closing this page does not stop it.")
        with st.expander("Worker log", expanded=False):
            st.code(_log_tail(active_dir), language="text")
        partial_path = Path(job.get("out_dir") or (active_dir / "output")) / "partial_report.json"
        if partial_path.exists():
            try:
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
            except Exception:
                partial = None
            if isinstance(partial, dict):
                st.caption("M2 is complete; M3 is still running. The partial analysis is available now.")
                dapp.render_explore_report(
                    partial_path.parent,
                    {"name": "partial", "dir": str(partial_path.parent), "report": partial},
                )
        if st.button("Cancel this turn", type="secondary"):
            if _cancel_job(
                active_dir, job, thread_id=str(job.get("thread_id") or run_dir.name),
                turn_id=str(job.get("turn_id") or "initial"), events_path=run_dir / "events.jsonl",
            ):
                st.rerun()
            else:
                st.error("Could not cancel this job; it may have already exited.")
        if st.toggle("Auto-refresh every 5 s", value=True):
            time.sleep(5)
            st.rerun()
        return

    if state == "failed":
        st.error(f"The analysis exited with code {status['exit_code']} before "
                 "producing a report.")
        st.code(_log_tail(active_dir, n_lines=120), language="text")
        st.caption("Common causes: the backend CLI is not on PATH, the API key "
                   "is missing, or the timeout is too low for the chosen model.")
        return

    if state == "stale":
        st.warning("No live process and no exit code — the server likely "
                   "restarted while this run was in flight. Re-run it by hand "
                   "with the command under *Run parameters* (`bash job.sh`).")
        st.code(_log_tail(active_dir), language="text")
        return

    if state == "canceled":
        st.warning("This turn was canceled. You can ask the question again to start a fresh version.")
        st.code(_log_tail(active_dir), language="text")
        _render_followup_input(st, run_dir, job)
        return

    # done — render with the exact same tabs as `evalvitals dashboard <out>`.
    if job.get("mode") == "answer":
        answer = active_dir / "answer.md"
        if answer.exists():
            with st.chat_message("assistant"):
                st.markdown(answer.read_text(encoding="utf-8", errors="replace"))
        else:
            st.warning("The answer job finished without writing an answer.")
        _render_followup_input(st, run_dir, job)
        return

    from evalvitals.analysis.dashboard import load_run

    out_dir = Path(job.get("out_dir") or (run_dir / "output"))
    session = load_run(out_dir)
    if not session["runs"]:
        st.error("exploratory_report.json exists but could not be parsed.")
        st.code(_log_tail(active_dir, n_lines=120), language="text")
        return
    turn = session["runs"][0]
    if not turn["report"].get("ok", True):
        st.warning("The explorer finished with an error — rendering whatever "
                   f"was produced. Error: {turn['report'].get('error') or 'unknown'}")
    _render_agent_audit(st, out_dir)
    with st.chat_message("assistant"):
        dapp.render_explore_report(Path(session["root"]), turn)
    _render_followup_input(st, run_dir, job)


if __name__ == "__main__":
    main()
