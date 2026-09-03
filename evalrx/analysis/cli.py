"""Command-line entrypoint for standalone single-shot exploration.

``evalrx-explore`` (and ``evalrx explore``) run one exploratory analysis
over JSON/JSONL logs: a CLI coding agent writes EDA code, the host adjudicates
any host-checkable candidate statistics, charts are rendered, and artifacts are
written. This replaces the retired interactive chat REPL — same backend, single
shot, no multi-turn state.
"""

from __future__ import annotations

import argparse

from evalrx.analysis.explore_run import run_explore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalrx-explore",
        description="Run Lambda-style local exploratory analysis over JSON/JSONL logs (single shot).",
    )
    parser.add_argument("path", help="File or directory of JSON/JSONL results.")
    parser.add_argument(
        "-q",
        "--question",
        default="Explore this dataset and surface the patterns that matter.",
        help="Natural-language analysis question for the local coding agent.",
    )
    parser.add_argument(
        "--outcome-col",
        default=None,
        help="Name of the target/outcome column, if any (e.g. 'label'). Omit to "
             "auto-detect by name heuristics, or fall back to unsupervised EDA "
             "when the data has no recognizable outcome.",
    )
    parser.add_argument(
        "--out",
        default="evalrx_explore_output",
        help="Output directory for report/code/figures/tables.",
    )
    parser.add_argument(
        "--backend",
        "--coder-provider",
        dest="coder_provider",
        default="antigravity",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding-agent backend.",
    )
    parser.add_argument("--coder-model", "--model", dest="coder_model", default="",
                        help="Optional model flag for the CLI agent.")
    parser.add_argument("--coder-binary", default="", help="Explicit path to the CLI binary.")
    parser.add_argument("--max-rows", type=int, default=2000, help="Maximum loaded records.")
    parser.add_argument("--max-files", type=int, default=200, help="Maximum JSON files to scan.")
    parser.add_argument(
        "--include-tool-calls",
        action="store_true",
        help="Also load tool_calls_*.json files. Off by default because these can dominate logs.",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Sandbox/agent timeout.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Code repair attempts.")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Open the Streamlit dashboard on the output directory when done.",
    )
    parser.add_argument("--port", type=int, default=None, help="Optional dashboard port.")
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="DIR",
        help="Path to an Agent-Skill directory (with SKILL.md) to style the "
             "agent-authored figures, e.g. a nature-figure skill. Repeatable. "
             "Only the claude/agy backends use skills.",
    )
    parser.add_argument(
        "--allow-skills",
        action="store_true",
        help="Enable the Skill tool so globally-installed (~/.claude/skills) "
             "skills are usable too. Implied when --skill is given.",
    )
    parser.add_argument(
        "--no-skills",
        dest="use_bundled_skills",
        action="store_false",
        default=True,
        help="Do not apply the package's bundled skills (e.g. nature-figure) "
             "for this run.",
    )
    args = parser.parse_args(argv)

    return run_explore(
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
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
