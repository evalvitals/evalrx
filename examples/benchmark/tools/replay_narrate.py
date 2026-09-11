#!/usr/bin/env python3
"""Replay a finished benchmark run's console narration from its V2 logs.

Usage:
    python replay_narrate.py <run-output-dir> [--color]

<run-output-dir> is an outputs/<model>/<dataset>[.<tag>]/ directory holding
logs/run.json + logs/M*/log.json (RunLoggerV2 layout) and, optionally,
summary.json + baseline.json for the runner's own context lines.

Nothing is simulated: every narrated line is rendered by the same
LoopNarrator the live run uses, fed the exact event records it logged.
"""
import argparse
import json
import os
import sys
from pathlib import Path

NARRATED = {"probe", "analysis", "explore", "diagnosis", "surgery", "experiment", "fix"}


def load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--color", action="store_true", help="force ANSI colors even when piped")
    args = ap.parse_args()
    if args.color:
        os.environ["EVALRX_FORCE_COLOR"] = "1"

    from evalrx.eval_agent.narration import LoopNarrator

    logs = args.run_dir / "logs"
    if not (logs / "run.json").exists():
        sys.exit(f"{logs}/run.json not found — not a RunLoggerV2 run (V1 layouts can't be replayed)")
    run = load(logs / "run.json")

    summary = load(args.run_dir / "summary.json") if (args.run_dir / "summary.json").exists() else {}
    baseline = load(args.run_dir / "baseline.json") if (args.run_dir / "baseline.json").exists() else {}

    # The runner's own context lines, reconstructed from the run's artifacts.
    if summary:
        print(f"[model] {summary.get('model')}, spec {summary.get('spec')}; backend={summary.get('backend')}; "
              f"dataset={summary.get('dataset')}; n={summary.get('n_cases')}")
    b = baseline.get("discovery") or baseline or {}
    if {"n_pass", "n_fail"} <= b.keys() or {"PASS", "FAIL"} <= b.keys():
        n_pass = b.get("n_pass", b.get("PASS"))
        n_fail = b.get("n_fail", b.get("FAIL"))
        acc = b.get("accuracy", summary.get("baseline_accuracy"))
        print(f"Baseline: PASS={n_pass}, FAIL={n_fail}, accuracy={acc}")
    elif summary:
        print(f"Baseline accuracy: {summary.get('baseline_accuracy')}")

    start = run.get("run_start") or {}
    model = summary.get("model") or start.get("model") or "?"
    print(f"\n{'=' * 64}\nVLDiagnoseLoop  model={model}  max_cycles={start.get('max_cycles', '?')}\n{'=' * 64}")

    narrator = LoopNarrator()
    narrator.on_run_start(start)

    events = []  # (event_seq, stage, key, record)
    for stage in ("M1", "M2", "M3", "M4", "M5"):
        doc_path = logs / stage / "log.json"
        if not doc_path.exists():
            continue
        for key, bucket in load(doc_path).items():
            if key not in NARRATED or not isinstance(bucket, list):
                continue
            for rec in bucket:
                events.append((rec.get("event_seq", 0), stage, key, rec))
    for rec in run.get("loop_end") or []:
        events.append((rec.get("event_seq", 0), "RUN", "loop_end", rec))
    events.sort(key=lambda e: e[0])

    for _, stage, key, rec in events:
        if stage == "RUN":
            narrator.on_run_event(key, rec)
        else:
            narrator.on_event(stage, key, rec)

    # Closing lines the runner prints from the loop report / summary.
    if summary:
        print(f"Diagnosis: stopped_by={summary.get('stopped_by')}, cycles={summary.get('cycles')}, "
              f"verified={summary.get('n_verified')}")
        for a in (summary.get("fix") or {}).get("attempted") or []:
            print(f"Fix [{a.get('tier')}] {a.get('name')}: fixed={a.get('fixed')}, "
                  f"repairs={a.get('repairs')}, breaks={a.get('breaks')}, effect={a.get('effect'):+}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
