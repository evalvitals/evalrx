#!/usr/bin/env python3
"""Run the five-paper corpus through the public ``evalvitals explore`` interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "data" / "paper_records.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "five_paper_diagnosis"
QUESTION = """You are auditing EvalVitals itself using five research papers as evidence.
For each paper, identify the reported model-health or evaluation failure, the
evidence supporting it, plausible confounders, and the narrowest repair that
the evidence warrants. Treat `expected_stress_test` as a review rubric, not as
evidence. Do not infer a causal mechanism from a correlation alone. Compare
the five diagnoses and identify missing framework capabilities, unsafe repair
patterns, and cases that should end inconclusive. Preserve paper_id and page
numbers whenever you cite evidence."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        default="codex",
        choices=["codex", "claude_code", "opencode", "gemini_cli", "kimi_cli", "antigravity"],
        help="Authenticated coding-agent backend used by evalvitals explore.",
    )
    parser.add_argument("--model", default="", help="Optional backend model override.")
    # 300s cuts off a real run mid-analysis: a claude_code run against all
    # five papers took 21 turns / ~300s wall and was killed while writing its
    # final script ("error_during_execution"), having already done the
    # expensive evidence-verification work. vtcbench_diagnosis's equivalent
    # explore step (run_explore.sh) already defaults to 3600s for the same
    # class of task; match that here rather than re-guess a number.
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()

    if not args.records.exists():
        raise SystemExit("records are missing; run build_records.py after download_papers.py")

    command = [
        sys.executable,
        "-m",
        "evalvitals.cli",
        "explore",
        str(args.records),
        "--backend",
        args.backend,
        "--out",
        str(args.out),
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-rows",
        "2000",
        "--max-files",
        "1",
        "--question",
        QUESTION,
    ]
    if args.model:
        command.extend(["--model", args.model])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
