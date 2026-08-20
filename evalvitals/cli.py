"""Top-level EvalVitals command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from evalvitals.analysis.dashboard import launch_dashboard, launch_upload_app
from evalvitals.analysis.explore_run import run_explore
from evalvitals.analysis.run_codebase import run_codebase_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalvitals",
        description="EvalVitals command-line interface.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print evalvitals' internal stage-by-stage narration "
             "(evalvitals.enable_console_logging()) to stdout. Applies to any subcommand.",
    )
    sub = parser.add_subparsers(dest="command")

    explore = sub.add_parser(
        "explore",
        help="Run a single-shot exploratory analysis over a results directory.",
        description="Run one Lambda-style exploratory analysis (no interactive REPL).",
    )
    explore.add_argument("path", nargs="?", help="File or directory of JSON/JSONL results.")
    explore.add_argument(
        "-q",
        "--question",
        default="Explore this dataset and surface the patterns that matter.",
        help="Natural-language analysis question for the local coding agent.",
    )
    explore.add_argument(
        "--outcome-col",
        default=None,
        help="Name of the target/outcome column, if any (e.g. 'label'). Omit to "
             "auto-detect by name heuristics, or fall back to unsupervised EDA "
             "when the data has no recognizable outcome.",
    )
    explore.add_argument(
        "--out",
        default="evalvitals_explore_output",
        help="Output directory for report/code/figures/tables.",
    )
    explore.add_argument(
        "--backend",
        "--coder-provider",
        dest="coder_provider",
        default="antigravity",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding-agent backend.",
    )
    explore.add_argument("--model", "--coder-model", dest="coder_model", default="")
    explore.add_argument("--coder-binary", default="")
    explore.add_argument("--max-rows", type=int, default=2000)
    explore.add_argument("--max-files", type=int, default=200)
    explore.add_argument("--include-tool-calls", action="store_true")
    explore.add_argument("--timeout-sec", type=int, default=120)
    explore.add_argument("--max-attempts", type=int, default=2)
    explore.add_argument(
        "--dashboard",
        action="store_true",
        help="Open the Streamlit dashboard on the output directory when done.",
    )
    explore.add_argument("--port", type=int, default=None, help="Optional dashboard port.")
    explore.add_argument(
        "--skill", action="append", default=[], metavar="DIR",
        help="Agent-Skill directory (with SKILL.md) to style agent-authored "
             "figures (e.g. nature-figure). Repeatable. claude/agy/codex backends.",
    )
    explore.add_argument(
        "--allow-skills", action="store_true",
        help="Enable the Skill tool so ~/.claude/skills are usable too "
             "(implied by --skill).",
    )
    explore.add_argument(
        "--no-skills", dest="use_bundled_skills", action="store_false", default=True,
        help="Do not apply the package's bundled skills (e.g. nature-figure).",
    )
    explore.add_argument(
        "--no-hypotheses", dest="propose_hypotheses", action="store_false", default=True,
        help="Skip M3 (falsifiable hypotheses proposed from the M2 takeaways). "
             "Runs by default after a successful explore.",
    )
    explore.add_argument(
        "--holdout-frac", type=float, default=0.0,
        help="Fraction of rows to hold out BEFORE exploration (outcome-"
             "stratified, deterministic). 0 disables (default).",
    )
    explore.add_argument(
        "--holdout-confirm", action="store_true",
        help="After exploring, re-test the frozen recipes + hypotheses on the "
             "held-out rows (writes confirm_report.json — the dashboard's "
             "Held-out Verdicts tab). Requires --holdout-frac > 0.",
    )
    explore.add_argument("--holdout-seed", type=int, default=0,
                         help="Seed for the held-out split (default 0).")
    explore.add_argument(
        "--judge-model", default="claude-opus-4-8",
        help="LLM judge grading each hypothesis against the held-out table "
             "(only used with --holdout-confirm).",
    )
    explore.add_argument("--progress-path", default="",
                         help="Append durable workbench progress events to this JSONL path.")
    explore.add_argument("--thread-id", default="", help=argparse.SUPPRESS)
    explore.add_argument("--turn-id", default="", help=argparse.SUPPRESS)

    run_codebase = sub.add_parser(
        "run-codebase",
        help="Run a user's evaluation codebase, then explore the results it produces.",
        description="A CLI coding agent runs the codebase at PATH inside an isolated copy, "
                    "harvests its per-case records (a records.json/.jsonl output contract), "
                    "and runs `evalvitals explore` (M2+M3) over them.",
    )
    run_codebase.add_argument("path", help="Directory containing the codebase to run.")
    run_codebase.add_argument(
        "-q", "--question",
        default="Explore this dataset and surface the patterns that matter.",
        help="Natural-language analysis question, also given to the run agent as task context.",
    )
    run_codebase.add_argument(
        "--outcome-col", default=None,
        help="Name of the target/outcome column, if any (e.g. 'label'). Omit to auto-detect.",
    )
    run_codebase.add_argument("--out", default="evalvitals_run_codebase_output",
                              help="Output directory for workspace/records/report/figures/tables.")
    run_codebase.add_argument(
        "--backend", "--coder-provider", dest="coder_provider", default="claude_code",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding-agent backend used both to run the codebase and to explore it.",
    )
    run_codebase.add_argument("--model", "--coder-model", dest="coder_model", default="")
    run_codebase.add_argument("--coder-binary", default="")
    run_codebase.add_argument("--records-name", default=None,
                              help="Output-contract filename the run agent must write "
                                   "(default: records.json).")
    run_codebase.add_argument("--timeout-sec", type=int, default=1200)
    run_codebase.add_argument("--max-attempts", type=int, default=2)
    run_codebase.add_argument(
        "--no-explore", dest="analyze", action="store_false", default=True,
        help="Only run the codebase and harvest records; skip the explore (M2+M3) step.",
    )
    run_codebase.add_argument(
        "--dashboard", action="store_true",
        help="Open the Streamlit dashboard on the output directory when done.",
    )
    run_codebase.add_argument("--port", type=int, default=None, help="Optional dashboard port.")

    dashboard = sub.add_parser(
        "dashboard",
        help="Open a Streamlit dashboard for an explore output or loop-run directory.",
        description="Open a Streamlit dashboard for EvalVitals single-run artifacts.",
    )
    dashboard.add_argument("run_dir", help="An explore output dir or a loop-run dir.")
    dashboard.add_argument("--port", type=int, default=None, help="Optional Streamlit port.")

    web = sub.add_parser(
        "web",
        help="Launch the upload-and-explore web workbench (upload a .zip, run M2+M3).",
        description="Serve a Streamlit page where users upload a .zip of results; "
                    "each upload becomes one `evalvitals explore` run (M2 exploratory "
                    "analysis + M3 hypotheses) and renders like `evalvitals dashboard`.",
    )
    web.add_argument(
        "workspace", nargs="?", default="evalvitals_web_runs",
        help="Directory where uploaded runs accumulate (created if missing).",
    )
    web.add_argument("--port", type=int, default=None, help="Optional Streamlit port.")
    web.add_argument(
        "--backend", "--coder-provider", dest="coder_provider", default="claude_code",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Default coding-agent backend pre-selected in the upload form.",
    )
    web.add_argument("--model", "--coder-model", dest="coder_model", default="",
                     help="Default model id pre-filled in the upload form.")
    web.add_argument("--timeout-sec", type=int, default=1200,
                     help="Default per-attempt explorer timeout in the upload form.")
    web.add_argument(
        "--attach", action="append", default=[], metavar="DIR",
        help="Existing result directory (explore output or loop run) to list "
             "in the sidebar alongside uploads. Repeatable.",
    )

    report_cmd = sub.add_parser(
        "report",
        help="Generate a self-contained, plain-language HTML diagnostic report for a run.",
        description="Renders a finished diagnostic run into a single self-contained HTML report with full M1-M5 trace, case browser, and plain-language explanations.",
    )
    report_cmd.add_argument("run_dir", nargs="?", default="outputs", help="Run directory holding run_log.jsonl or logs/")
    report_cmd.add_argument("--example-dir", default=None, help="Root holding data/ manifest.")
    report_cmd.add_argument("--out", "-o", default=None, help="Output HTML path (default: <run_dir>/report.html).")
    report_cmd.add_argument("--no-audio", action="store_true", help="Skip audio transcoding.")

    langfuse_cmd = sub.add_parser(
        "export-langfuse",
        help="Export a diagnostic run to Langfuse trace JSON format or sync live.",
        description="Map an EvalVitals run (M1-M5, fixes, scores) to Langfuse Traces, Spans, and Scores.",
    )
    langfuse_cmd.add_argument("run_dir", nargs="?", default="outputs", help="Run directory.")
    langfuse_cmd.add_argument("--out", "-o", default=None, help="Output JSON path.")
    langfuse_cmd.add_argument("--sync", action="store_true", help="Sync live to Langfuse server.")

    args = parser.parse_args(argv)
    if args.verbose:
        from evalvitals.logging_utils import enable_console_logging

        enable_console_logging()
    if args.command == "explore":
        if not args.path:
            parser.error("evalvitals explore requires a results path")
        progress_sink = None
        if args.progress_path:
            from evalvitals.analysis.workbench import EventSink

            progress_sink = EventSink(
                args.progress_path,
                thread_id=args.thread_id or "standalone",
                turn_id=args.turn_id or "explore",
            )
            progress_sink.emit("job", "started", "Analysis worker started")
        code = run_explore(
            args.path,
            question=args.question,
            out=args.out,
            coder_provider=args.coder_provider,
            coder_model=args.coder_model,
            coder_binary=args.coder_binary,
            max_rows=args.max_rows,
            max_files=args.max_files,
            include_tool_calls=args.include_tool_calls,
            timeout_sec=args.timeout_sec,
            max_attempts=args.max_attempts,
            dashboard=args.dashboard,
            dashboard_port=args.port,
            skills=args.skill,
            allow_skills=args.allow_skills,
            use_bundled_skills=args.use_bundled_skills,
            outcome_col=args.outcome_col,
            propose_hypotheses=args.propose_hypotheses,
            holdout_frac=args.holdout_frac,
            holdout_seed=args.holdout_seed,
            holdout_confirm=args.holdout_confirm,
            judge_model=args.judge_model,
            progress_sink=progress_sink,
        )
        if progress_sink is not None:
            progress_sink.emit(
                "job", "completed" if code == 0 else "failed",
                "Analysis worker completed" if code == 0 else "Analysis worker failed",
            )
        return code
    if args.command == "run-codebase":
        from evalvitals.analysis.explorer import RECORDS_FILENAME

        return run_codebase_cli(
            args.path,
            out=args.out,
            coder_provider=args.coder_provider,
            coder_model=args.coder_model,
            coder_binary=args.coder_binary,
            outcome_col=args.outcome_col,
            records_name=args.records_name or RECORDS_FILENAME,
            timeout_sec=args.timeout_sec,
            max_attempts=args.max_attempts,
            question=args.question,
            analyze=args.analyze,
            dashboard=args.dashboard,
            dashboard_port=args.port,
        )
    if args.command == "dashboard":
        return launch_dashboard(args.run_dir, port=args.port)
    if args.command == "web":
        return launch_upload_app(
            args.workspace,
            port=args.port,
            backend=args.coder_provider,
            model=args.coder_model,
            timeout_sec=args.timeout_sec,
            attach=args.attach,
        )
    if args.command == "report":
        from evalvitals.reporting.html_report import build_html_report

        build_html_report(
            run_dir=args.run_dir,
            example_dir=args.example_dir,
            out_path=args.out,
            no_audio=args.no_audio,
        )
        return 0
    if args.command == "export-langfuse":
        from evalvitals.reporting.langfuse_exporter import export_to_langfuse_bundle, sync_to_langfuse_live

        if args.sync:
            return 0 if sync_to_langfuse_live(args.run_dir) else 1
        out_p = args.out or (Path(args.run_dir) / "langfuse_trace.json")
        export_to_langfuse_bundle(args.run_dir, out_p)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
