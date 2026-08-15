"""deco_hallu — PHASE 2 of the decoupled run: confirm the proposed hypotheses, then fix.

The deferred half of run_analysis.py. It REUSES that phase's artifacts —
outputs/analysis/analysis_state.pkl (the proposed hypotheses + the exact M2 stats
report the dashboard showed) — so the hypotheses confirmed here are the SAME ones
the analysis dashboard displayed, not a fresh re-proposal.

  reload {hypotheses, stats_report}  →  M5 confirm (HypothesisTester)  →  M4 + tiered Fix

What it produces:
  outputs/logs_confirm_fix/run_log.jsonl   M5 verdicts + M4 surgery + the fix attempts

After this, point the dashboard at outputs/ again: it now merges logs_analysis/
(stats + charts + proposed hypotheses) with logs_confirm_fix/ (the M5/M4/Fix
verdicts), so each proposed hypothesis gains its downstream verdict.

    python run_analysis.py     --backend claude ...     # PHASE 1 (no GPU) — writes the artifacts
    python run_confirm_fix.py  --model qwen3-vl-2b-instruct --device cuda   # this step
    python -m evalvitals.cli dashboard outputs

The VLM IS loaded here: the fix module (and M4 surgery) call it to validate
candidate repairs against the unmodified baseline (paired McNemar / e-BH guard).
M5 confirmation itself reuses the persisted M2 stats — no re-analysis needed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import run  # reuse build_judge / build_protocol / build_codegen / CFG

OUT = run.OUT
M1_STATE = OUT / "m1_state.pkl"
ANALYSIS_STATE = OUT / "analysis" / "analysis_state.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=run.CFG["model"])
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument("--judge-model", default=run.CFG.get("judge_model", "claude-opus-4-8"))
    ap.add_argument("--judge-effort", default=run.CFG.get("judge_effort", "low"))
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"],
                    help="coder backend for M4/fix codegen (default claude)")
    ap.add_argument("--max-validation-cases", type=int,
                    default=int(run.CFG.get("fix_validation_cases", 60)),
                    help="cap fix-validation cases (overrides config; 0 = full batch)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dashboard", action="store_true",
                    help="launch the Streamlit dashboard on outputs/ when done")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if not M1_STATE.exists():
        raise SystemExit(f"{M1_STATE} missing — run `python run_m1.py` first")
    if not ANALYSIS_STATE.exists():
        raise SystemExit(f"{ANALYSIS_STATE} missing — run `python run_analysis.py` first "
                         "(PHASE 1 must propose the hypotheses this phase confirms)")

    with open(M1_STATE, "rb") as fh:
        m1_state = pickle.load(fh)
    cases = m1_state["cases"]
    with open(ANALYSIS_STATE, "rb") as fh:
        analysis_state = pickle.load(fh)

    from evalvitals.eval_agent.hypothesis import hypothesis_from_dict

    hypotheses = [hypothesis_from_dict(d) for d in analysis_state.get("hypotheses", [])]
    stats_report = analysis_state.get("stats_report")  # may be None → regenerate
    print(f"reloaded {len(hypotheses)} proposed hypothesis(es) from PHASE 1"
          + (" (+ frozen M2 stats)" if stats_report is not None else " (M2 stats will be regenerated)"))
    for h in hypotheses:
        print(f" - [{h.predicted_failure_mode}] {h.statement[:110]}")
    if not hypotheses:
        print("no proposed hypotheses to confirm — nothing for PHASE 2 to do "
              "(this is an honest outcome, not an error).")
        return

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
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.models.backends.base import RuntimeConfig

    judge = run.build_judge(args.judge_model, args.judge_effort)
    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    codegen: CliAgentConfig = run.build_codegen(args.backend)

    run_logger = RunLogger(run_dir=OUT / "logs_confirm_fix", verbose=True)
    loop = VLDiagnoseLoop(
        model=model,
        protocol=run.build_protocol(),
        probe_agent=run.ReplayProbeAgent(m1_state),    # only used if stats regenerated
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(judge=judge,
                           max_tier=str(run.CFG.get("fix_max_tier", "L3b")),
                           cli_config=codegen, run_logger=run_logger,
                           max_validation_cases=args.max_validation_cases,
                           exec_timeout_sec=int(run.CFG.get("fix_exec_timeout_sec", 900))),
        run_logger=run_logger,
    )

    # ── PHASE 2a: confirm the reloaded hypotheses with M5 ───────────────────
    report = loop.run_confirm(cases, hypotheses, stats_report=stats_report)
    print(f"\nM5 confirm: verified={len(report.verified_hypotheses)}/{len(report.all_test_results)}")
    for t in report.all_test_results:
        stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
        print(f" - [{t.status}] conf={t.confidence:.2f} grade={t.evidence_grade} {stmt[:100]}")

    # ── PHASE 2b: M4 surgery + tiered fix on the confirmed hypotheses ───────
    if not args.skip_m4:
        fix = loop.run_m4(report, cases)
        print("m4:", fix)
        outcome = loop.run_fix(report, cases)
        print("fix outcome:", getattr(outcome, "recommendation", None) or outcome)
    run_logger.close()

    print(f"\nconfirm + fix logs -> {run_logger.run_dir}")
    print("\nView the full dashboard (analysis + verdicts now merged):")
    print("  python -m evalvitals.cli dashboard outputs")

    if args.dashboard:
        from evalvitals.analysis.dashboard import launch_dashboard

        raise SystemExit(launch_dashboard(OUT))


if __name__ == "__main__":
    main()
