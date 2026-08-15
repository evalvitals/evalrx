"""deco_hallu — STAGE 2 of a split run: M2-M5 + M4 + fix, on a FROZEN M1 result.

Loads outputs/m1_state.pkl (produced by run_m1.py) and runs the rest of the
pipeline — M2 stats analysis, M3 hypotheses, M5 hypothesis testing, then M4
surgery and the tiered fix module — WITHOUT re-running M1. M1 is short-circuited
by a thin replay agent that returns the frozen probe results, so the loop's
orchestration (and the M2-M5 stages) run exactly as in the single-shot run.py.

    python run_m1.py   --model qwen3-vl-2b-instruct --device cuda   # once
    python run_m2-5.py --model qwen3-vl-2b-instruct --device cuda   # iterate

The model is still loaded here (M2 codegen, M4 surgery and the fix module call
it), but M1's analyzer forward passes / generations are reused from the pickle.
Because only a single M1 pass was frozen, the loop runs one cycle (--max-cycles
defaults to 1). Shared setup (judge, protocol) is reused from run.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import run  # reuse build_judge / build_protocol / CFG

OUT = run.OUT
M1_STATE = OUT / "m1_state.pkl"

# ReplayProbeAgent (frozen-M1 stand-in) is shared from run.py so the staged
# scripts (run_m2-5 / run_analysis / run_confirm_fix) replay M1 identically.
ReplayProbeAgent = run.ReplayProbeAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=run.CFG["model"])
    ap.add_argument("--max-cycles", type=int, default=1,
                    help="only one M1 pass was frozen, so default 1 cycle")
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument("--judge-model", default=run.CFG.get("judge_model", "claude-opus-4-8"))
    ap.add_argument("--judge-effort", default=run.CFG.get("judge_effort", "low"))
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"],
                    help="coder backend for M2/M4/fix codegen (default claude)")
    ap.add_argument("--recipes", default="",
                    help="path to confirmed_recipes.json (Step 1 output) — bridged into M2")
    ap.add_argument("--explore-report", default="",
                    help="path to Step 1 fused_report.json — its charts/observations "
                         "(UNCONFIRMED) are shown to M3 only, never to M2/M5/fix")
    ap.add_argument("--max-validation-cases", type=int,
                    default=int(run.CFG.get("fix_validation_cases", 60)),
                    help="cap fix-validation cases (overrides config; 0 = full batch)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if not M1_STATE.exists():
        raise SystemExit(f"{M1_STATE} missing — run `python run_m1.py` first")
    with open(M1_STATE, "rb") as fh:
        state = pickle.load(fh)
    if state.get("model_key") != args.model:
        print(f"[WARN] frozen M1 was for {state.get('model_key')!r}, now {args.model!r} "
              f"— the loaded model must match the analyzers' model")
    probe_results = state["probe_results"]
    cases = state["cases"]
    print(f"loaded frozen M1: analyzers={list(probe_results)} cases={len(list(cases))} "
          f"failed={state.get('failed_analyzers') or '{}'}")

    from evalvitals import compose
    from evalvitals.core.capability import Capability
    from evalvitals.eval_agent import (
        CliAgentConfig,
        ExperimentWriterConfig,
        FixAgent,
        RunLogger,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.models.backends.base import RuntimeConfig

    judge = run.build_judge(args.judge_model, args.judge_effort)
    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})

    codegen = run.build_codegen(args.backend)

    # Optional: bridge the recipes confirmed by Step 1 (run_fused.py) into M2 as a
    # synthetic "explored" analyzer, so the LAMBDA-discovered composite signals
    # enter M2/M3/M5 via the standard findings["per_case"] contract.
    signal_recipes = []
    if args.recipes:
        from evalvitals.analysis.operationalize import SignalRecipe

        raw = json.loads(Path(args.recipes).read_text())
        signal_recipes = [SignalRecipe.from_dict(r) for r in raw]
        print(f"bridging {len(signal_recipes)} confirmed recipe(s): "
              f"{[r.name for r in signal_recipes]}")

    # Optional: feed Step 1's explorer mechanism notes (charts/observations) to M3.
    # Descriptive + UNCONFIRMED — they inform WHICH hypotheses M3 proposes; they do
    # NOT enter the M2 confirmatory family, M5 testing, or the fix gate.
    explore_report = None
    if args.explore_report:
        explore_report = json.loads(Path(args.explore_report).read_text())
        n_charts = len(explore_report.get("charts") or [])
        n_obs = len(explore_report.get("observations") or [])
        print(f"feeding explore context to M3: {n_charts} chart(s), {n_obs} observation(s)")

    run_logger = RunLogger(run_dir=OUT / "logs_m2_5", verbose=True)
    loop = VLDiagnoseLoop(
        model=model,
        probe_agent=ReplayProbeAgent(state),          # M1 short-circuited
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(judge=judge,
                           max_tier=str(run.CFG.get("fix_max_tier", "L3b")),
                           cli_config=codegen, run_logger=run_logger,
                           max_validation_cases=args.max_validation_cases,
                           exec_timeout_sec=int(run.CFG.get("fix_exec_timeout_sec", 900))),
        max_cycles=args.max_cycles,
        protocol=run.build_protocol(),
        run_logger=run_logger,
        signal_recipes=signal_recipes,
        explore_report=explore_report,
    )

    report = loop.run(cases)
    print(f"cycles={report.cycles} stopped_by={report.stopped_by} "
          f"verified={len(report.verified_hypotheses)}/{len(report.all_test_results)}")
    for t in report.all_test_results:
        stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
        print(f" - [{t.status}] conf={t.confidence:.2f} grade={t.evidence_grade} {stmt[:110]}")

    if not args.skip_m4:
        fix = loop.run_m4(report, cases)
        print("m4:", fix)
        outcome = loop.run_fix(report, cases)
        print("fix outcome:", getattr(outcome, "recommendation", None) or outcome)
    run_logger.close()


if __name__ == "__main__":
    main()
