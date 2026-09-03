"""Run a user's existing evaluation codebase and feed its results into
exploratory analysis.

``run_codebase()`` bridges the run infrastructure (:mod:`evalrx.agent_runtime`)
and the analysis entry (:func:`evalrx.analysis.api.explore`): given a
path to a codebase, a CLI coding agent runs it inside an isolated copy,
harvests the per-case records it produces, and (by default) hands them
straight to ``explore()``.

Flow::

    run_codebase(path)
      -> copy repo into an isolated workspace (the original is never touched)
      -> CodegenRunner.run(prompt)     # the agent runs the eval in-place
      -> harvest workspace/records.json (one repair turn if empty/missing)
      -> explore(records)              # M2 + M3, unchanged

``run_codebase_cli`` is the CLI-facing counterpart: same pipeline, prints a
summary and optionally opens the dashboard.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalrx.agent_runtime.cli_types import CliAgentConfig
from evalrx.agent_runtime.codegen.runner import CodegenRunner
from evalrx.analysis.api import ExploreRunResult, explore
from evalrx.analysis.explorer import RECORDS_FILENAME, load_records_from_path
from evalrx.analysis.prompts.run_codebase import REPAIR_PROMPT_TEMPLATE, RUN_PROMPT_TEMPLATE

_IGNORE_DIR_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
_HARVEST_MAX_ROWS = 20_000


@dataclass
class CodebaseRunResult:
    """Outcome of one :func:`run_codebase` call."""

    records: list[dict[str, Any]] = field(default_factory=list)
    records_path: Path | None = None
    workspace: Path | None = None
    ran_ok: bool = False
    explore: ExploreRunResult | None = None
    error: str | None = None
    audit: dict | None = None


def _copy_ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _IGNORE_DIR_NAMES}


def _harvest(workspace: Path, records_name: str) -> list[dict[str, Any]]:
    target = workspace / records_name
    if not target.exists():
        return []
    return load_records_from_path(target, max_rows=_HARVEST_MAX_ROWS, max_files=1)


def run_codebase(
    path: "str | Path",
    *,
    out: "str | Path | None" = None,
    analyze: bool = True,
    provider: str = "claude_code",
    model: str = "",
    binary: str = "",
    outcome_col: str | None = None,
    records_name: str = RECORDS_FILENAME,
    timeout_sec: int = 1200,
    max_attempts: int = 2,
    question: str = "Explore this dataset and surface the patterns that matter.",
    progress_sink: Any = None,
    **explore_kwargs: Any,
) -> CodebaseRunResult:
    """Run a user's evaluation codebase at *path* and analyze its results.

    A CLI coding agent (*provider*) runs inside an isolated copy of *path* —
    the original directory is never modified — and is instructed to produce
    *records_name* (default ``"records.json"``, a JSON array or JSON-Lines
    file with one row per evaluation case, each carrying a ``label`` plus
    input/prediction/target fields). Those records are then handed to
    :func:`evalrx.analysis.api.explore` (M2 + M3) unless *analyze* is
    False.

    When *out* is given, the workspace lives under ``out/workspace`` and is
    kept on disk (harvested records plus explore artifacts are persisted
    there too), which is useful for inspecting what the agent did — including
    after a failed run. Without *out* everything runs in a temporary
    directory that is removed before returning; only the returned
    :class:`CodebaseRunResult` carries the records in memory.
    """
    src = Path(path).resolve()
    if not src.exists():
        return CodebaseRunResult(error=f"path does not exist: {src}")

    out_dir = Path(out).resolve() if out is not None else None
    cleanup: tempfile.TemporaryDirectory | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        workspace = out_dir / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="evalrx_run_codebase_")
        workspace = Path(cleanup.name) / "workspace"
    shutil.copytree(src, workspace, ignore=_copy_ignore)

    cli_config = CliAgentConfig(provider=provider, binary_path=binary, model=model, timeout_sec=timeout_sec)
    runner = CodegenRunner(cli_config)

    if progress_sink is not None:
        progress_sink.emit("run", "started", f"Running codebase at {src}")

    prompt = RUN_PROMPT_TEMPLATE.format(records_name=records_name, question=question)
    result = runner.run(prompt, workdir=workspace, timeout_sec=timeout_sec)
    audit = result.audit
    records = _harvest(workspace, records_name)

    attempts = 1
    while not records and attempts < max_attempts:
        reason = result.error or "no records found"
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(records_name=records_name, reason=reason)
        result = runner.run(repair_prompt, workdir=workspace, timeout_sec=timeout_sec)
        audit = result.audit or audit
        records = _harvest(workspace, records_name)
        attempts += 1

    if not records:
        error = result.error or f"the coding agent did not produce '{records_name}'"
        if progress_sink is not None:
            progress_sink.emit("run", "failed", error)
        if cleanup is not None:
            cleanup.cleanup()
        return CodebaseRunResult(workspace=workspace, ran_ok=False, error=error, audit=audit)

    if progress_sink is not None:
        progress_sink.emit("run", "completed", f"Harvested {len(records)} record(s)")

    run_result = CodebaseRunResult(
        records=records,
        records_path=workspace / records_name,
        workspace=workspace,
        ran_ok=True,
        audit=audit,
    )

    if analyze:
        run_result.explore = explore(
            records,
            question=question,
            out=out_dir,
            provider=provider,
            model=model,
            binary=binary,
            outcome_col=outcome_col,
            progress_sink=progress_sink,
            **explore_kwargs,
        )

    if out_dir is not None:
        (out_dir / records_name).write_text(json.dumps(records, default=str), encoding="utf-8")
    if cleanup is not None:
        cleanup.cleanup()

    return run_result


def run_codebase_cli(
    path: "str | Path",
    *,
    out: "str | Path" = "evalrx_run_codebase_output",
    coder_provider: str = "claude_code",
    coder_model: str = "",
    coder_binary: str = "",
    outcome_col: str | None = None,
    records_name: str = RECORDS_FILENAME,
    timeout_sec: int = 1200,
    max_attempts: int = 2,
    question: str = "Explore this dataset and surface the patterns that matter.",
    analyze: bool = True,
    dashboard: bool = False,
    dashboard_port: int | None = None,
    progress_sink: Any = None,
) -> int:
    """Run *path* and persist its harvested records + explore artifacts under *out*.

    Prints a short summary and returns a process exit code: ``0`` if records
    were harvested (and, when *analyze*, the exploration succeeded), else
    ``1``. When *dashboard* is set, the return value is the dashboard
    process's code instead.
    """
    out_dir = Path(out).resolve()
    result = run_codebase(
        path,
        out=out_dir,
        analyze=analyze,
        provider=coder_provider,
        model=coder_model,
        binary=coder_binary,
        outcome_col=outcome_col,
        records_name=records_name,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        question=question,
        progress_sink=progress_sink,
    )

    print(f"ran_ok: {result.ran_ok}")
    print(f"records harvested: {len(result.records)}")
    print(f"output: {out_dir}")
    if result.error:
        print(f"error: {result.error}")
    if result.explore is not None:
        report = result.explore.report
        print(f"explore ok: {report.ok}")
        if report.hypotheses:
            print("\nhypotheses (M3, proposed only — not validated):")
            for h in report.hypotheses[:8]:
                print(f"- {h.get('statement')}")
        if report.observations:
            print("\nobservations:")
            for obs in report.observations[:8]:
                print(f"- {obs}")

    ok = result.ran_ok and (result.explore is None or result.explore.ok)

    if ok:
        from evalrx.reporting.html_report import build_html_report

        build_html_report(out_dir, out_path=out_dir / "report.html")

    if dashboard:
        from evalrx.analysis.dashboard import launch_dashboard

        return launch_dashboard(out_dir, port=dashboard_port)
    return 0 if ok else 1
