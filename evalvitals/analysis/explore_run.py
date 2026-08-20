"""Single-shot exploratory analysis run (the ``evalvitals explore`` entry).

One call = one pipeline = one artifact directory. This is the non-interactive
replacement for the retired chat REPL: the explorer writes EDA code, the host
adjudicates any host-checkable candidate statistics, charts are rendered
deterministically, and everything is persisted (optionally a dashboard opens).

Flow::

    ExploratoryAnalysisAgent.explore_path        # M2: free-form EDA (a CLI coding agent)
      -> adjudicate_report              # host recomputes verdicts (in-sample)
      -> HypothesisAgent.propose        # M3: falsifiable hypotheses from M2's takeaways
      -> render_chart_specs             # spec + CSV -> PNG, host-side
      -> write_report_artifacts         # report.json + figures/ + tables/ + code
      -> launch_dashboard (optional)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evalvitals.analysis.api import explore
from evalvitals.analysis.explorer import RECORDS_FILENAME, load_records_from_path
from evalvitals.analysis.holdout import holdout_confirm as run_holdout_confirm
from evalvitals.analysis.holdout import split_records
from evalvitals.viz.renderer import render_chart_specs

# Appended to the analysis question when a held-out confirm phase will follow:
# recipes must be frozen and threshold-explicit or there is nothing to re-test.
HOLDOUT_QUESTION_SUFFIX = (
    " This is the EXPLORATION HALF of a held-out design: a disjoint validate "
    "split exists and will re-test whatever you propose. Make every candidate "
    "signal a FROZEN, threshold-explicit recipe (explicit numeric thresholds, "
    "no re-fitting) so it can be re-evaluated verbatim on the held-out half, "
    "and propose hypotheses precise enough to be graded against held-out "
    "statistics."
)


def _build_judge(model: str):
    """Judge for the held-out confirm phase (separate so tests can stub it)."""
    from evalvitals.agent_runtime.judges import ClaudeModel

    return ClaudeModel(model=model, effort="low")


def run_explore(
    path: str | Path,
    *,
    question: str = "Explore this dataset and surface the patterns that matter.",
    out: str | Path = "evalvitals_explore_output",
    coder_provider: str = "antigravity",
    coder_model: str = "",
    coder_binary: str = "",
    max_rows: int = 2000,
    max_files: int = 200,
    include_tool_calls: bool = False,
    timeout_sec: int = 120,
    max_attempts: int = 2,
    dashboard: bool = False,
    dashboard_port: int | None = None,
    skills: "list[str] | tuple[str, ...]" = (),
    allow_skills: bool = False,
    use_bundled_skills: bool = True,
    outcome_col: str | None = None,
    propose_hypotheses: bool = True,
    holdout_frac: float = 0.0,
    holdout_seed: int = 0,
    holdout_confirm: bool = False,
    judge_model: str = "claude-opus-4-8",
    progress_sink: Any = None,
) -> int:
    """Run one exploratory analysis and persist its artifacts.

    Returns a process exit code: ``0`` if the exploration succeeded, else ``1``.
    When *dashboard* is set, the return value is the dashboard process's code.

    Skills style the agent-authored figures (e.g. the bundled eval-chart-style
    and nature-figure skills). By default (*use_bundled_skills*) the package's
    bundled skills are applied on the claude/agy/codex backends; *skills* adds
    more dirs and *allow_skills* also enables globally-installed
    (`~/.claude/skills`) skills.

    *outcome_col* optionally names the target/label column explicitly (M1
    passes ``"label"``); omit it to let the agent auto-detect an outcome by
    name heuristics, or fall back to unsupervised EDA when there is none.

    *propose_hypotheses* runs M3 (``HypothesisAgent``) on M2's takeaways after
    a successful explore, using the same coding-agent backend; set False to
    skip it (e.g. to save the extra LLM call).

    *holdout_frac* > 0 splits the loaded records BEFORE exploration
    (deterministic, outcome-stratified; the explorer only sees the explore
    share). With *holdout_confirm* the held-out rows then re-test the frozen
    recipes + hypotheses (``analysis.holdout``) and ``confirm_report.json``
    lands next to the exploratory report — the dashboard's Held-out Verdicts
    tab fills in. Without it the rows are simply reserved on disk.
    """
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source: Any = path
    holdout_rows: "list[dict[str, Any]] | None" = None
    if holdout_frac and holdout_frac > 0:
        records = load_records_from_path(
            path, max_rows=max_rows, max_files=max_files,
            include_tool_calls=include_tool_calls,
        )
        explore_share, holdout_rows = split_records(
            records, holdout_frac, seed=holdout_seed, outcome_col=outcome_col
        )
        if holdout_rows is None:
            print(f"holdout: split impossible (n={len(records)}, frac={holdout_frac}) "
                  "— falling back to a plain in-sample run")
        else:
            source = explore_share
            print(f"holdout: exploring {len(explore_share)} rows, "
                  f"{len(holdout_rows)} held out "
                  + ("(verification follows)" if holdout_confirm
                     else "(reserved, not verified in this mode)"))
            if holdout_confirm:
                question = question.rstrip() + HOLDOUT_QUESTION_SUFFIX

    result = explore(
        source,
        question=question,
        out=out_dir,
        provider=coder_provider,
        model=coder_model,
        binary=coder_binary,
        outcome_col=outcome_col,
        propose_hypotheses=propose_hypotheses,
        skills=skills,
        allow_skills=allow_skills,
        use_bundled_skills=use_bundled_skills,
        max_rows=max_rows,
        max_files=max_files,
        include_tool_calls=include_tool_calls,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        progress_sink=progress_sink,
    )
    report = result.report

    print(f"ok: {report.ok}")
    print(f"attempts: {report.attempts}")
    print(f"rows loaded: {report.data_profile.get('loaded_rows', report.data_profile.get('n_rows'))}")
    print(f"output: {out_dir}")
    if report.error:
        print(f"error: {report.error}")
    if report.hypotheses:
        print("\nhypotheses (M3, proposed only — not validated):")
        for h in report.hypotheses[:8]:
            print(f"- {h.get('statement')}")
    if report.observations:
        print("\nobservations:")
        for obs in report.observations[:8]:
            print(f"- {obs}")
    if report.candidate_signals:
        print("\ncandidate signals:")
        for signal in report.candidate_signals[:8]:
            print(f"- {signal.name}: {signal.rationale}{_verdict_suffix(signal)}")
    if report.adjudication:
        adj = report.adjudication
        print(
            f"\nhost adjudication ({adj.get('split')}): "
            f"{adj.get('n_rejected', 0)}/{adj.get('n_host_adjudicated', 0)} reject "
            f"(e-BH family n={adj.get('n_in_family', 0)}, alpha={adj.get('alpha')})"
        )

    if holdout_rows is not None:
        # Audit trail either way: the exact rows the explorer never saw.
        (out_dir / "holdout_records.json").write_text(
            json.dumps(holdout_rows, default=str), encoding="utf-8"
        )
        split_meta = {
            "n_explore": len(source) if isinstance(source, list) else None,
            "n_holdout": len(holdout_rows),
            "holdout_frac": holdout_frac,
            "seed": holdout_seed,
            "stratified_by": outcome_col or "label",
        }
        if holdout_confirm:
            judge = None
            judge_meta = None
            try:
                judge = _build_judge(judge_model)
                judge_meta = {"model": judge_model, "effort": "low"}
            except Exception as exc:  # noqa: BLE001 — verdicts degrade to not_judged
                print(f"judge unavailable ({exc}) — hypotheses will be not_judged")
            confirm = run_holdout_confirm(
                report.to_dict(), holdout_rows,
                outcome_col=outcome_col or "label",
                judge=judge, judge_meta=judge_meta,
            )
            confirm["split_meta"] = split_meta
            (out_dir / "confirm_report.json").write_text(
                json.dumps(confirm, indent=1, default=str), encoding="utf-8"
            )
            cadj = confirm.get("adjudication") or {}
            print(
                f"\nheld-out adjudication: {cadj.get('n_rejected', 0)}/"
                f"{cadj.get('n_host_adjudicated', 0)} reject "
                f"(n={confirm.get('n_validate_rows')} rows, "
                f"{confirm.get('n_validate_fail')} failures)"
            )
            for v in confirm.get("hypothesis_verdicts", []):
                print(f" * [{v.get('verdict')}] {str(v.get('statement', ''))[:90]}")
        else:
            (out_dir / "holdout_split.json").write_text(
                json.dumps(split_meta, indent=1), encoding="utf-8"
            )

    if dashboard:
        from evalvitals.analysis.dashboard import launch_dashboard

        return launch_dashboard(out_dir, port=dashboard_port)
    return 0 if report.ok else 1


def _verdict_suffix(signal: Any) -> str:
    """One-line host verdict tag for a candidate signal, or '' if not adjudicated."""
    if not getattr(signal, "host_adjudicated", False):
        return "  [descriptive]" if getattr(signal, "sufficient", None) else ""
    verdict = "REJECT H0" if signal.reject else "inconclusive"
    parts = [verdict]
    if signal.e_value is not None:
        # In the e-BH family (has an e-value); reject already reflects survival.
        parts.append(f"e={signal.e_value:.2f}")
        parts.append("e-BH family")
    elif signal.ci is not None:
        parts.append(f"CI={signal.ci[0]:+.3f}..{signal.ci[1]:+.3f} (not FDR-corrected)")
    return "  [host: " + ", ".join(parts) + "]"


def write_report_artifacts(
    report: Any, out_dir: Path, *, report_filename: str = "exploratory_report.json"
) -> None:
    """Persist one explore run's artifacts: figures/, tables/, rendered charts,
    the report JSON, and the generated code/stdout/stderr.

    Charts are rendered here (host-side, from spec + CSV) so the persisted
    ``exploratory_report.json`` carries each chart's ``figure_path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_artifact_dirs(report, out_dir)

    # Render the explorer's chart specs to PNG from the copied tables/ CSVs.
    try:
        report.charts = render_chart_specs(
            getattr(report, "charts", None), out_dir / "tables", out_dir
        )
    except Exception:  # rendering is best-effort, never blocks persistence
        pass

    (out_dir / report_filename).write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    if report.code:
        (out_dir / "analysis.py").write_text(report.code, encoding="utf-8")
    if report.stdout:
        (out_dir / "stdout.txt").write_text(report.stdout, encoding="utf-8")
    if report.stderr:
        (out_dir / "stderr.txt").write_text(report.stderr, encoding="utf-8")
    if report.raw_outputs:
        (out_dir / "agent_raw_output.txt").write_text(
            "\n\n--- attempt ---\n\n".join(report.raw_outputs),
            encoding="utf-8",
        )
    if report.agent_audits:
        (out_dir / "agent_audit.json").write_text(
            json.dumps({"schema_version": 1, "attempts": report.agent_audits}, indent=2),
            encoding="utf-8",
        )

    # Clean up redundant nested sandbox directory so out_dir is the single clean source of truth
    workdir = Path(getattr(report, "workdir", "") or "")
    try:
        if workdir.exists() and workdir.resolve() != out_dir.resolve() and workdir.is_relative_to(out_dir):
            shutil.rmtree(workdir)
    except Exception:
        pass


def _copy_artifact_dirs(report: Any, out_dir: Path) -> None:
    workdir = Path(getattr(report, "workdir", "") or "")
    if not workdir.exists():
        return
    for name in ("figures", "tables"):
        src = workdir / name
        dest = out_dir / name
        if src.exists() and src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
    # Persist the raw loaded records alongside the report so the dashboard can
    # offer a "browse raw data" view — otherwise they only live in the sandbox
    # workdir, which isn't guaranteed to still exist by the time someone opens
    # the dashboard.
    records_src = workdir / RECORDS_FILENAME
    if records_src.exists():
        shutil.copy2(records_src, out_dir / RECORDS_FILENAME)
