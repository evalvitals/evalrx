"""Top-level EvalRX command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from evalrx.analysis.dashboard import launch_dashboard
from evalrx.analysis.explore_run import run_explore
from evalrx.analysis.run_codebase import run_codebase_cli


def serve_report(
    run_dir: str | Path | None,
    *,
    port: int,
    no_audio: bool = False,
    open_browser: bool = True,
    block: bool = True,
    runs_root: str | Path | None = None,
) -> int:
    """CLI seam for the dynamic UI (signature kept for compatibility/tests)."""
    del no_audio, block
    from evalrx.reporting.server import serve_dynamic_report

    return serve_dynamic_report(run_dir, port=port, open_browser=open_browser, runs_root=runs_root)


def _langfuse_cache(trace_id: str) -> Path:
    """Stable local cache for a Langfuse-backed static report, not run storage."""
    safe_id = "".join(char for char in trace_id if char.isalnum() or char in "-_")
    if not safe_id:
        raise ValueError("--trace-id must contain at least one letter or number")
    return Path(".evalrx-cache") / safe_id


def _resolve_report_source(source: str, trace_id: str | None, run_dir: str | None) -> str | None:
    if source == "auto":
        if trace_id:
            try:
                from evalrx.reporting.langfuse_source import LangfuseRunSource

                return str(LangfuseRunSource().materialize(trace_id, _langfuse_cache(trace_id)))
            except (ImportError, LookupError, RuntimeError):
                pass
        return run_dir
    if source == "local":
        return run_dir
    if not trace_id:
        raise ValueError("--trace-id is required with --source langfuse")
    from evalrx.reporting.langfuse_source import LangfuseRunSource

    return str(LangfuseRunSource().materialize(trace_id, _langfuse_cache(trace_id)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalrx",
        description="EvalRX command-line interface.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print evalrx' internal stage-by-stage narration "
             "(evalrx.enable_console_logging()) to stdout. Applies to any subcommand.",
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
        default="evalrx_explore_output",
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
        help="Deprecated alias for --serve-report.",
    )
    explore.add_argument("--serve-report", action="store_true",
                         help="Serve the generated static HTML report when done.")
    explore.add_argument("--port", type=int, default=None, help="Optional report-server port.")
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
                    "and runs `evalrx explore` (M2+M3) over them.",
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
    run_codebase.add_argument("--out", default="evalrx_run_codebase_output",
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
        help="Deprecated alias for --serve-report.",
    )
    run_codebase.add_argument("--serve-report", action="store_true",
                              help="Serve the generated static HTML report when done.")
    run_codebase.add_argument("--port", type=int, default=None, help="Optional report-server port.")

    dashboard = sub.add_parser(
        "dashboard",
        help="Deprecated alias for the dependency-free 'serve' command.",
        description="Deprecated alias for the dependency-free static report server.",
    )
    dashboard.add_argument("run_dir", help="An explore output dir or a loop-run dir.")
    dashboard.add_argument("--port", type=int, default=None, help="Optional report-server port.")

    serve = sub.add_parser(
        "serve",
        help="Publish (if needed) and serve the dynamic completed-run UI.",
        description="Generate (if needed) ReportData and serve the agent-composed React UI backed by Langfuse or a local run cache.",
    )
    serve.add_argument("run_dir", nargs="?", default=None, help="Run directory. Omit to start empty and drop a zipped run on the page.")
    serve.add_argument("--port", type=int, default=8501, help="Loopback port (default: 8501).")
    serve.add_argument("--no-audio", action="store_true", help=argparse.SUPPRESS)
    serve.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    serve.add_argument("--source", choices=["auto", "local", "langfuse"], default="auto", help="Run data source (default: Langfuse when --trace-id is set, otherwise local).")
    serve.add_argument("--trace-id", default=None, help="Langfuse trace id (required for --source langfuse).")
    serve.add_argument("--runs-root", default=None, help="Where the runs panel looks for other experiments (default: run_dir's parent, or the current directory).")

    report_cmd = sub.add_parser(
        "report",
        help="Explicitly export a portable self-contained HTML snapshot.",
        description="Uses the same React/json-render layout as `serve`; the dynamic UI remains the primary experience.",
    )
    report_cmd.add_argument("run_dir", nargs="?", default="outputs", help="Run directory holding run.json/M1..M5 (or the legacy run_log.jsonl), or logs/")
    report_cmd.add_argument("--example-dir", default=None, help="Root holding data/ manifest.")
    report_cmd.add_argument("--out", "-o", default=None, help="Output HTML path (default: <run_dir>/report.html).")
    report_cmd.add_argument("--no-audio", action="store_true", help="Skip audio transcoding.")
    report_cmd.add_argument("--embed-media", choices=["representative", "all", "none"], default="representative", help="Media to inline in the portable export (default: representative).")
    report_cmd.add_argument("--source", choices=["local", "langfuse"], default="local", help="Run data source.")
    report_cmd.add_argument("--trace-id", default=None, help="Langfuse trace id (required for --source langfuse).")

    publish_cmd = sub.add_parser(
        "publish-report",
        help="Compile and cache ReportData plus a validated json-render layout.",
    )
    publish_cmd.add_argument("run_dir", nargs="?", default="outputs", help="Completed run directory.")
    publish_cmd.add_argument("--example-dir", default=None, help="Optional benchmark/example root for legacy runs.")
    publish_cmd.add_argument("--source", choices=["auto", "local", "langfuse"], default="auto")
    publish_cmd.add_argument("--trace-id", default=None)

    langfuse_cmd = sub.add_parser(
        "export-langfuse",
        help="Export a diagnostic run to Langfuse trace JSON format or sync live.",
        description="Map an EvalRX run (M1-M4, fixes, scores) to Langfuse Traces, Spans, and Scores.",
    )
    langfuse_cmd.add_argument("run_dir", nargs="?", default="outputs", help="Run directory.")
    langfuse_cmd.add_argument("--out", "-o", default=None, help="Output JSON path.")
    langfuse_cmd.add_argument("--sync", action="store_true", help="Sync live to Langfuse server.")

    backfill_langfuse = sub.add_parser(
        "backfill-langfuse",
        help="Queue an existing run (run.json/M1..M5, or the legacy run_log.jsonl) for reliable Langfuse ingestion.",
    )
    backfill_langfuse.add_argument("run_dir", help="Run directory holding run.json/M1..M5 (or the legacy run_log.jsonl), or logs/.")
    backfill_langfuse.add_argument("--dry-run", action="store_true", help="Inspect the run without writing an outbox.")

    args = parser.parse_args(argv)
    if args.verbose:
        from evalrx.logging_utils import enable_console_logging

        enable_console_logging()
    if args.command == "explore":
        if not args.path:
            parser.error("evalrx explore requires a results path")
        progress_sink = None
        if args.progress_path:
            from evalrx.analysis.workbench import EventSink

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
            dashboard=args.dashboard or args.serve_report,
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
        from evalrx.analysis.explorer import RECORDS_FILENAME

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
            dashboard=args.dashboard or args.serve_report,
            dashboard_port=args.port,
        )
    if args.command == "dashboard":
        return launch_dashboard(args.run_dir, port=args.port)
    if args.command == "serve":
        try:
            source_dir = _resolve_report_source(args.source, args.trace_id, args.run_dir)
        except (ImportError, LookupError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        try:
            return serve_report(
                source_dir, port=args.port, no_audio=args.no_audio,
                open_browser=not args.no_browser, runs_root=args.runs_root,
            )
        except ImportError as exc:
            parser.error(str(exc))
    if args.command == "publish-report":
        from evalrx.reporting.dynamic import publish_report

        try:
            source_dir = _resolve_report_source(args.source, args.trace_id, args.run_dir)
            published = publish_report(source_dir, example_dir=args.example_dir)
        except (ImportError, LookupError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Published report data: {published.data_path}")
        print(f"Published report layout: {published.spec_path} ({published.generated_by})")
        return 0
    if args.command == "report":
        from evalrx.reporting.static_export import export_static_report

        try:
            source_dir = _resolve_report_source(args.source, args.trace_id, args.run_dir)
        except (ImportError, LookupError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))

        export_static_report(
            run_dir=source_dir,
            example_dir=args.example_dir,
            out_path=args.out,
            embed_media="none" if args.no_audio else args.embed_media,
        )
        return 0
    if args.command == "export-langfuse":
        from evalrx.reporting.langfuse_exporter import (
            export_to_langfuse_bundle,
            sync_to_langfuse_live,
        )

        if args.sync:
            return 0 if sync_to_langfuse_live(args.run_dir) else 1
        out_p = args.out or (Path(args.run_dir) / "langfuse_trace.json")
        export_to_langfuse_bundle(args.run_dir, out_p)
        return 0
    if args.command == "backfill-langfuse":
        from evalrx.observability import backfill_run_to_langfuse

        summary = backfill_run_to_langfuse(args.run_dir, dry_run=args.dry_run)
        print(
            "Langfuse backfill: "
            f"trace={summary['trace_id']} events={summary['events']} "
            f"published={summary['published']} pending={summary['pending']}"
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
