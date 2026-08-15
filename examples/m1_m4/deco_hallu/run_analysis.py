"""deco_hallu — PHASE 1 of the decoupled run: analyse + propose, build the dashboard.

Runs M1 → M2 → M3 on the FROZEN M1 result (outputs/m1_state.pkl) WITHOUT
confirming any hypothesis (M5) and WITHOUT attempting a fix. The point is to get
the analysis story onto the dashboard first:

  M1 (replayed)  →  M2 rigorous stats + charts  →  M3 proposes hypotheses  →  STOP

What it produces:
  outputs/logs_analysis/run_log.jsonl     M1 + M2 (stats/charts) + M3 (proposed hyps)
  outputs/analysis/proposed_hypotheses.json   the hypotheses, human-readable
  outputs/analysis/analysis_state.pkl     {hypotheses, stats_report} for PHASE 2

Then point the dashboard at outputs/ to read "what we analysed → what we found →
hypotheses formed" — the hypotheses are PROPOSED, not yet confirmed (no M5/M4/Fix
verdicts). Confirmation + repair are deferred to run_confirm_fix.py, which reuses
the two artifacts above so the SAME hypotheses/stats are what gets confirmed.

    python run_m1.py        --model qwen3-vl-2b-instruct --device cuda   # once (GPU)
    python run_fused.py     --backend claude                            # Step 1 (no GPU)
    python run_analysis.py  --backend claude --recipes outputs/fused/confirmed_recipes.json --explore-report outputs/fused/fused_report.json   # PHASE 1 (no GPU)
    python -m evalvitals.cli dashboard outputs                         # the analysis report
    python run_confirm_fix.py --device cuda                            # PHASE 2 (GPU + claude)

No GPU here: M1 is replayed from the pickle, and M2/M3 only call the judge/coder
CLI — the VLM is never loaded (it is loaded in PHASE 2, which needs it for the fix).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import run  # reuse build_judge / build_protocol / build_codegen / ReplayProbeAgent / CFG

OUT = run.OUT
M1_STATE = OUT / "m1_state.pkl"
ANALYSIS_DIR = OUT / "analysis"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=run.CFG["model"])
    ap.add_argument("--judge-model", default=run.CFG.get("judge_model", "claude-opus-4-8"))
    ap.add_argument("--judge-effort", default=run.CFG.get("judge_effort", "low"))
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"],
                    help="coder backend for M2 stats codegen (default claude)")
    ap.add_argument("--recipes", default="",
                    help="path to confirmed_recipes.json (Step 1 output) — bridged into M2")
    ap.add_argument("--explore-report", default="",
                    help="path to Step 1 fused_report.json — its charts/observations "
                         "(UNCONFIRMED) are shown to M3 only, never to the M2 family")
    ap.add_argument("--dashboard", action="store_true",
                    help="launch the Streamlit dashboard on outputs/ when done")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if not M1_STATE.exists():
        raise SystemExit(f"{M1_STATE} missing — run `python run_m1.py` first")
    with open(M1_STATE, "rb") as fh:
        state = pickle.load(fh)
    if state.get("model_key") != args.model:
        print(f"[WARN] frozen M1 was for {state.get('model_key')!r}, now {args.model!r}")
    probe_results = state["probe_results"]
    cases = state["cases"]
    print(f"loaded frozen M1: analyzers={list(probe_results)} cases={len(list(cases))} "
          f"failed={state.get('failed_analyzers') or '{}'}")

    from evalvitals.eval_agent import (
        CliAgentConfig,
        RunLogger,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.hypothesis import hypothesis_to_dict
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent

    judge = run.build_judge(args.judge_model, args.judge_effort)
    codegen: CliAgentConfig = run.build_codegen(args.backend)

    # GPU-free: M1 is replayed and M2/M3 only need the model's repr string.
    model = run.FrozenModel(state.get("model_key", args.model))

    # Optional: bridge Step-1's confirmed recipes into M2's family (same as run_m2-5).
    signal_recipes = []
    if args.recipes:
        from evalvitals.analysis.operationalize import SignalRecipe

        raw = json.loads(Path(args.recipes).read_text())
        signal_recipes = [SignalRecipe.from_dict(r) for r in raw]
        print(f"bridging {len(signal_recipes)} confirmed recipe(s): "
              f"{[r.name for r in signal_recipes]}")

    # Optional: feed Step-1's charts/observations to M3 (UNCONFIRMED, M3-only).
    explore_report = None
    if args.explore_report:
        explore_report = json.loads(Path(args.explore_report).read_text())
        print(f"feeding explore context to M3: "
              f"{len(explore_report.get('charts') or [])} chart(s), "
              f"{len(explore_report.get('observations') or [])} observation(s)")

    run_logger = RunLogger(run_dir=OUT / "logs_analysis", verbose=True)
    loop = VLDiagnoseLoop(
        model=model,
        protocol=run.build_protocol(),
        probe_agent=run.ReplayProbeAgent(state),       # M1 short-circuited
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        run_logger=run_logger,
        signal_recipes=signal_recipes,
        explore_report=explore_report,
    )

    # ── PHASE 1: M1 → M2 → M3 (propose), no M5, no fix ──────────────────────
    report = loop.run_analysis(cases)
    run_logger.close()

    print(f"\nstopped_by={report.stopped_by} "
          f"proposed={len(report.all_hypotheses)} (UNCONFIRMED — no M5/M4/Fix yet)")
    for h in report.all_hypotheses:
        print(f" - [{h.predicted_failure_mode}] {h.statement[:110]}")

    # ── Persist the artifacts PHASE 2 reuses (same hypotheses + same stats) ──
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    hyp_dicts = [hypothesis_to_dict(h) for h in report.all_hypotheses]
    (ANALYSIS_DIR / "proposed_hypotheses.json").write_text(
        json.dumps(hyp_dicts, indent=2), encoding="utf-8")

    state_out = {"schema_version": 1, "model_key": state.get("model_key", args.model),
                 "hypotheses": hyp_dicts, "stats_report": None}
    # The M2 report is the exact evidence M5 will confirm against. Pickle it when
    # possible so PHASE 2 confirms the SAME statistics the dashboard showed; if it
    # cannot pickle, PHASE 2 regenerates M2 from the frozen M1 (a safe fallback).
    try:
        pickle.dumps(report.final_stats_report)
        state_out["stats_report"] = report.final_stats_report
    except Exception as exc:
        print(f"[WARN] stats_report not picklable ({exc}); PHASE 2 will regenerate M2")
    with open(ANALYSIS_DIR / "analysis_state.pkl", "wb") as fh:
        pickle.dump(state_out, fh)

    print(f"\nwrote {ANALYSIS_DIR / 'proposed_hypotheses.json'}")
    print(f"wrote {ANALYSIS_DIR / 'analysis_state.pkl'}  (hypotheses + stats for PHASE 2)")
    print(f"analysis logs -> {run_logger.run_dir}")
    print("\nView the analysis dashboard (proposed hypotheses, no verdicts yet):")
    print("  python -m evalvitals.cli dashboard outputs")
    print("\nPHASE 2 (confirm + fix, reuses the artifacts above):")
    print("  python run_confirm_fix.py --device cuda")

    if args.dashboard:
        from evalvitals.analysis.dashboard import launch_dashboard

        raise SystemExit(launch_dashboard(OUT))


if __name__ == "__main__":
    main()
