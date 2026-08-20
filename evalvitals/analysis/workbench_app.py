"""The single EvalVitals web entry point.

It has two intentionally different workspaces under one product shell:

* **Data Analysis** is the user-operated M2/M3 flow (upload, explore,
  optionally validate, and ask follow-up questions).
* **Diagnostic Runs** is the read-only M1--M5 inspector for runs launched by
  the evaluation backend.

Both are backed by :mod:`run_view`, so the navigation and stage semantics do
not depend on whether the source was a report artifact or a legacy JSONL log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evalvitals.analysis.dashboard import load_run
from evalvitals.analysis.run_view import RunView, StageId, StageState, from_session
from evalvitals.analysis import stage_views


WORKSPACE_ANALYSIS = "Data Analysis"
WORKSPACE_DIAGNOSTICS = "Diagnostic Runs"


def _parse_args() -> argparse.Namespace:
    from evalvitals.analysis import upload_app as upload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", nargs="?", default="evalvitals_web_runs")
    parser.add_argument("--backend", default="claude_code", choices=list(upload.BACKENDS))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--attach", action="append", default=[], metavar="DIR")
    parser.add_argument(
        "--initial-workspace", choices=("auto", "analysis", "diagnostics"), default="auto",
        help="Initial workspace; auto chooses Diagnostic Runs for a diagnostic-only attachment.",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="Do not create a writable analysis workspace (used by `evalvitals dashboard`).",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _attached(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    for raw in args.attach:
        path = Path(raw).resolve()
        if path.is_dir() and path not in out:
            out.append(path)
    return out


def _workspace_default(args: argparse.Namespace, views: dict[Path, RunView]) -> str:
    if args.initial_workspace == "analysis":
        return WORKSPACE_ANALYSIS
    if args.initial_workspace == "diagnostics":
        return WORKSPACE_DIAGNOSTICS
    return (WORKSPACE_DIAGNOSTICS if views and all(view.is_diagnostic for view in views.values())
            else WORKSPACE_ANALYSIS)


def _state_icon(state: StageState) -> str:
    return {
        StageState.SUCCEEDED: "●", StageState.RUNNING: "◐", StageState.EMPTY: "○",
        StageState.PARTIAL: "◑", StageState.FAILED: "✕", StageState.NOT_STARTED: "○",
        StageState.UNAVAILABLE: "—",
    }[state]


def _status_label(state: StageState) -> str:
    return state.value.replace("_", " ")


def _render_stage_rail(st: Any, view: RunView) -> None:
    """Single action-ordered stage rail shared by both workspaces."""
    cells = "".join(
        '<div class="ev-metric-card">'
        f'<div class="ev-metric-label">{stage.id} · {stage.label}</div>'
        f'<div class="ev-metric-value">{_state_icon(stage.state)}</div>'
        f'<div class="ev-metric-caption">{_status_label(stage.state)}'
        f'{(" · " + str(stage.count)) if stage.count is not None else ""}</div></div>'
        for stage in view.stages
    )
    st.markdown(f'<div class="ev-kpi-row">{cells}</div>', unsafe_allow_html=True)


def _render_workspace_header(st: Any, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ev-header"><div><div class="ev-kicker">EvalVitals</div>'
        f'<h1>{title}</h1><div class="ev-path">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def _render_diagnostic_workspace(st: Any, dapp: Any, view: RunView) -> None:
    """Dedicated inspector for backend-launched M1--M5 runs."""
    _render_workspace_header(st, "Diagnostic Run", "A read-only record of measurement, evidence, validation, and repair.")
    st.subheader(view.title)
    _render_stage_rail(st, view)
    story = view.story or {}
    report = story.get("explore_report")
    artifact_dir = Path(story.get("explore_dir") or view.root)
    tabs = st.tabs([
        "Overview", "M1 Measure", "M2 Evidence", "M3 Hypotheses",
        "M5 Validate", "M4 Intervene & repair", "Cases & artifacts",
    ])
    with tabs[0]:
        lifecycle = story.get("run_start") or {}
        st.markdown("### Run summary")
        if lifecycle.get("protocol_description"):
            st.write(lifecycle["protocol_description"])
        details = [
            f"{key}: {lifecycle[key]}" for key in ("model", "protocol", "started_at")
            if lifecycle.get(key) is not None
        ]
        if details:
            st.caption(" · ".join(details))
        st.markdown("### What to read next")
        st.markdown("Start with M1 for raw measurements, M2 for evidence, then inspect each M3 hypothesis alongside M5 validation and any M4 repair attempt.")
    with tabs[1]:
        stage_views.render_measurement(st, view)
    with tabs[2]:
        if report:
            # Reuse the mature M2 renderer only inside its stage-specific view.
            dapp._render_standalone_analysis(report, artifact_dir, view.root)
        else:
            stage_views.render_stage_events(
                st,
                list(story.get("analyses") or []),
                "No M2 analysis record was persisted.",
            )
    with tabs[3]:
        stage_views.render_hypotheses(st, view)
    with tabs[4]:
        stage_views.render_stage_events(
            st,
            [event for event in story.get("surgeries") or [] if str(event.get("module", "")).lower() == "m5"],
            "No M5 validation was run for this diagnostic.",
        )
    with tabs[5]:
        m4 = [event for event in story.get("surgeries") or [] if str(event.get("module", "")).lower() == "m4"]
        stage_views.render_stage_events(st, m4 + list(story.get("fixes") or []), "No M4 intervention or repair was run for this diagnostic.")
    with tabs[6]:
        if report:
            dapp._render_raw_data_browser(report, artifact_dir)
            with st.expander("Derived tables", expanded=False):
                dapp._render_explore_tables(report, artifact_dir, full=True)
        else:
            st.info("This legacy log has no normalized records artifact alongside it.")
        with st.expander("Run log and artifact references", expanded=False):
            st.json({"log": story.get("log_path"), "explore_dir": story.get("explore_dir")})


def _render_exploratory_evidence(st: Any, dapp: Any, report: dict[str, Any], artifact_dir: Path) -> None:
    """The reusable M2 finding/evidence body, without legacy report chrome."""
    observations = [str(item) for item in report.get("observations") or []]
    caveats = [str(item) for item in report.get("caveats") or []]
    if observations or caveats:
        with st.expander("Scope and caveats", expanded=False):
            for observation in observations:
                st.markdown(f"- {observation}")
            for caveat in caveats:
                st.warning(caveat)

    takeaways = [
        item for item in report.get("takeaways") or []
        if isinstance(item, dict) and (item.get("title") or item.get("plain_title"))
    ]
    if not takeaways:
        st.info("No structured takeaways were recorded; showing the generated visual artifacts.")
        dapp._render_charts_and_plots(report, artifact_dir)
        return

    charts = dapp._chart_lookup(report)
    plots = dapp._plot_lookup(report)
    tables = report.get("tables") or {}
    referenced_tables: set[str] = set()
    dapp._render_findings_overview(takeaways)
    for index, takeaway in enumerate(takeaways, start=1):
        title = str(takeaway.get("title") or "")
        headline = dapp._takeaway_headline(takeaway)
        with st.container(border=True):
            st.markdown(f"### {index}. {headline}")
            chart_names = [str(name) for name in takeaway.get("chart_names") or []]
            found = [(name, "chart", charts[name]) for name in chart_names if name in charts]
            found += [(name, "plot", plots[name]) for name in chart_names if name in plots]
            if found:
                columns = st.columns(2)
                for column_index, (name, kind, item) in enumerate(found):
                    with columns[column_index % 2]:
                        if kind == "chart":
                            dapp._render_chart_card(item, artifact_dir, heading_level="caption", key_prefix=f"workspace{index}")
                            dapp._render_compact_visual_explanation(report, name=name, chart=item)
                        else:
                            dapp._render_plot_card(item, artifact_dir)
                            dapp._render_compact_visual_explanation(report, name=name)
            elif takeaway.get("analysis"):
                st.write(str(takeaway["analysis"]))
            dapp._render_takeaway_details(
                index, takeaway, title=title, headline=headline,
                table_names=[str(name) for name in takeaway.get("table_names") or []],
                tables=tables, turn_dir=artifact_dir, referenced_tables=referenced_tables,
            )


def _render_analysis_result_workspace(st: Any, dapp: Any, view: RunView) -> None:
    """M2--M3 report reader in the common action-ordered workspace shape."""
    report = view.report or {}
    artifact_dir = view.root
    _render_workspace_header(st, "Exploratory Analysis", "A user-operated M2–M3 analysis; validation and repair stay explicitly separate.")
    st.subheader(view.title)
    _render_stage_rail(st, view)
    tabs = st.tabs([
        "Overview", "M2 Evidence", "M3 Hypotheses", "M5 Validate",
        "M4 Intervene & repair", "Raw data & artifacts",
    ])
    with tabs[0]:
        st.markdown("### Research question")
        st.write(report.get("plain_question") or report.get("question") or "No question was saved.")
        takeaways = [item for item in report.get("takeaways") or [] if isinstance(item, dict)]
        st.markdown("### Reading guide")
        st.write(
            f"This report has {len(takeaways)} M2 finding(s) and "
            f"{view.stage(StageId.M3).count or 0} M3 hypothesis/hypotheses. "
            "M2 is descriptive; only M5 represents a held-out validation verdict."
        )
        with st.expander("Run provenance", expanded=False):
            st.caption(f"Bundle: {view.root}")
    with tabs[1]:
        _render_exploratory_evidence(st, dapp, report, artifact_dir)
    with tabs[2]:
        dapp._render_standalone_hypotheses(report, artifact_dir)
    with tabs[3]:
        if view.confirm:
            dapp._render_holdout_panel(view.confirm)
        else:
            st.info("No M5 held-out validation artifact was produced for this analysis.")
    with tabs[4]:
        if view.fix_report:
            dapp._render_fix_panel(view.fix_report)
        else:
            st.info("No M4 intervention or repair artifact was produced for this analysis.")
    with tabs[5]:
        dapp._render_raw_data_browser(report, artifact_dir)
        with st.expander("Derived tables", expanded=False):
            dapp._render_explore_tables(report, artifact_dir, full=True)
        with st.expander("Generated charts and figures", expanded=False):
            dapp._render_charts_and_plots(report, artifact_dir)


def _render_analysis_workspace(
    st: Any, dapp: Any, upload: Any, workspace: Path, args: argparse.Namespace,
    attached: list[Path], views: dict[Path, RunView], *, read_only: bool,
) -> None:
    local_runs = [] if read_only else upload.list_runs(workspace)
    explore_paths = [path for path in attached if views[path].kind == "explore"]
    values = ([upload.NEW_ANALYSIS_LABEL] if not read_only else []) + [f"@{path}" for path in explore_paths] + [run.name for run in local_runs]
    if not values:
        st.info("No M2–M3 analysis is attached. Launch `evalvitals web` without `--read-only` to upload a dataset.")
        return

    def label(value: str) -> str:
        if value == upload.NEW_ANALYSIS_LABEL:
            return "New analysis"
        if value.startswith("@"):
            return f"📁 {upload._pretty_name(Path(value[1:]))}"
        state = upload.job_status(upload._active_turn_dir(workspace / value))["state"]
        return f"{upload._STATE_ICONS.get(state, '⚪')} {upload._pretty_name(value)}"

    choice = st.sidebar.radio("Analyses", values, key="ev_analysis_choice", format_func=label)
    if choice == upload.NEW_ANALYSIS_LABEL:
        upload._render_new_analysis(st, workspace, args, upload.NEW_ANALYSIS_LABEL)
    elif choice.startswith("@"):
        path = Path(choice[1:])
        session = load_run(path)
        st.caption("Reference result · read-only")
        if session.get("runs"):
            _render_analysis_result_workspace(st, dapp, from_session(session))
        else:
            st.warning("The attached analysis has no readable report.")
    else:
        def render_uploaded_report(artifact_dir: Path, turn: dict[str, Any]) -> None:
            session = {
                "kind": "explore", "root": str(artifact_dir),
                "runs": [{**turn, "dir": str(artifact_dir)}],
            }
            _render_analysis_result_workspace(st, dapp, from_session(session))

        upload._render_run(st, dapp, workspace / choice, report_renderer=render_uploaded_report)


def _render_diagnostics_home(st: Any, dapp: Any, attached: list[Path], views: dict[Path, RunView]) -> None:
    paths = [path for path in attached if views[path].is_diagnostic]
    if not paths:
        _render_workspace_header(
            st, "Diagnostic Runs",
            "Inspect M1–M5 runs launched by the evaluation backend.",
        )
        st.info("Attach a backend run directory to inspect it here, e.g. `evalvitals web --attach path/to/outputs`.")
        return

    def label(value: str) -> str:
        view = views[Path(value)]
        completed = sum(stage.state == StageState.SUCCEEDED for stage in view.stages)
        return f"{view.title[:48]} · {completed}/5 stages"

    choice = st.sidebar.radio("Diagnostic runs", [str(path) for path in paths], key="ev_diagnostic_choice", format_func=label)
    view = views[Path(choice)]
    if view.kind == "casebench":
        session = load_run(view.root)
        dapp.render_case_study_run(view.root, session)
    else:
        _render_diagnostic_workspace(st, dapp, view)


def main() -> None:
    import streamlit as st
    from evalvitals.analysis import dashboard_app as dapp
    from evalvitals.analysis import upload_app as upload

    args = _parse_args()
    workspace = Path(args.workspace).resolve()
    if not args.read_only:
        workspace.mkdir(parents=True, exist_ok=True)
    attached = _attached(args)
    views = {path: from_session(load_run(path)) for path in attached}

    st.set_page_config(page_title="EvalVitals", layout="wide", initial_sidebar_state="expanded")
    dapp._inject_css()
    upload._inject_workbench_css(st)
    st.sidebar.markdown('<div class="ev-sidebar-title">EvalVitals</div>', unsafe_allow_html=True)
    st.sidebar.caption("Explore data or inspect a diagnostic run")
    default = _workspace_default(args, views)
    index = [WORKSPACE_ANALYSIS, WORKSPACE_DIAGNOSTICS].index(default)
    # Keep the run selector as the sidebar's primary radio control.  Besides
    # making the active run the most prominent choice, this preserves the
    # established keyboard interaction for the upload workbench.
    workspace_name = st.sidebar.selectbox("Workspace", [WORKSPACE_ANALYSIS, WORKSPACE_DIAGNOSTICS], index=index, key="ev_workspace")
    st.sidebar.markdown("---")
    if workspace_name == WORKSPACE_ANALYSIS:
        _render_analysis_workspace(st, dapp, upload, workspace, args, attached, views, read_only=args.read_only)
    else:
        _render_diagnostics_home(st, dapp, attached, views)


if __name__ == "__main__":
    main()
