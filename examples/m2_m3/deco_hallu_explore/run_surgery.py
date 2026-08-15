"""Pipeline phase 3 — surgery + tiered fix (L1→L3b) on the confirmed hypotheses.

Plan C: hand the held-out-vetted hypotheses to the diagnosis loop's repair
machinery (the same M5 confirm → M4 surgery → tiered FixAgent arc as the
deco_hallu loop example). Requires the loop example's frozen M1 state
(outputs/m1_state.pkl, qwen3-vl-2b) and a GPU — the fix module validates every
candidate repair against the unmodified baseline (paired McNemar / e-BH guard,
no-free-lunch on present-object probes).

Which hypotheses go in: verdicts "supported"/"partial" from phase 2, plus any
judged not_testable with needs_surgery=true (an interventional experiment is
exactly what they need). "refuted" ones stay out.

Writes:
  <pipeline-root>/3_surgery/logs/run_log.jsonl   full M5/M4/Fix record
  <pipeline-root>/1_explore/fix_report.json      distilled summary for the dashboard

    python run_surgery.py --pipeline-root outputs_pipeline --device cuda
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).parent
LOOP_DIR = HERE.parent.parent / "diagnosis_loops" / "deco_hallu"
sys.path.insert(0, str(LOOP_DIR))  # reuse the loop example's setup helpers


_ATTEMPT_FIELDS = ("tier", "name", "kind", "source", "n_pairs", "n_fixed",
                   "n_broken", "coverage", "e_value", "effect", "reject",
                   "verdict", "summary")


def _lean_fix(event: dict) -> dict:
    """Distill the logged fix event into the dashboard-facing structure:
    keep the per-candidate outcome numbers, drop payloads/case-id lists."""
    refine = event.get("refine_signal")
    if isinstance(refine, dict):
        refine = {"kind": refine.get("kind"),
                  "candidate": refine.get("candidate"),
                  "n_helped": len(refine.get("helped_cases") or []),
                  "n_hurt": len(refine.get("hurt_cases") or []),
                  "message": refine.get("message")}
    return {
        "max_tier": event.get("max_tier"),
        "fixed": event.get("fixed"),
        "best": event.get("best"),
        "ebh_survivors": event.get("ebh_survivors") or [],
        "repair_rounds": event.get("repair_rounds"),
        "recommendation": event.get("recommendation"),
        "refine_signal": refine,
        "routed": event.get("routed") or [],
        "attempted": [
            {k: a.get(k) for k in _ATTEMPT_FIELDS}
            for a in event.get("attempted") or [] if isinstance(a, dict)
        ],
    }


def _fix_event_from_logs(run_dir: Path) -> dict | None:
    log = Path(run_dir) / "run_log.jsonl"
    if not log.exists():
        return None
    event = None
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("event") == "fix":
            event = o  # keep the last fix event
    return event


def _to_loop_hypotheses(verdicts: list[dict], *, model_key: str,
                        include_refuted: bool = False) -> list:
    from evalvitals.eval_agent.hypothesis import hypothesis_from_dict

    out = []
    for v in verdicts:
        verdict = str(v.get("verdict", ""))
        if verdict == "refuted" and not include_refuted:
            continue
        if verdict == "not_testable" and not v.get("needs_surgery", False):
            continue
        out.append(hypothesis_from_dict({
            "statement": v.get("statement", ""),
            "target_model": model_key,
            # the explore-path hypothesis has no failure-mode tag; carry the
            # held-out verdict as the mode context instead of inventing one
            "predicted_failure_mode": f"holdout:{verdict or 'unjudged'}",
            "test_design": v.get("test_design", ""),
            "metadata": {"holdout_verdict": verdict,
                         "holdout_reasoning": v.get("reasoning", "")},
        }))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", default="outputs_pipeline")
    ap.add_argument("--model", default="qwen3-vl-2b-instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"])
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument("--include-refuted", action="store_true")
    ap.add_argument("--max-validation-cases", type=int, default=0,
                    help="cap fix-validation cases (0 = full batch)")
    ap.add_argument("--distill-only", action="store_true",
                    help="no GPU run: rebuild fix_report.json's fix section from "
                         "an existing 3_surgery/logs/run_log.jsonl")
    args = ap.parse_args()

    root = Path(args.pipeline_root)

    if args.distill_only:
        report_path = root / "1_explore" / "fix_report.json"
        prior = json.loads(report_path.read_text()) if report_path.exists() else {}
        event = _fix_event_from_logs(root / "3_surgery" / "logs")
        if event is None:
            raise SystemExit("no fix event found in 3_surgery/logs/run_log.jsonl")
        prior["fix"] = _lean_fix(event)
        report_path.write_text(json.dumps(prior, indent=1))
        print(f"re-distilled fix section -> {report_path}")
        return

    confirm_path = root / "1_explore" / "confirm_report.json"
    if not confirm_path.exists():
        raise SystemExit(f"{confirm_path} missing — run test_hypotheses.py first")
    confirm = json.loads(confirm_path.read_text())

    import run as loop_run  # the loop example's shared setup

    m1_state_path = loop_run.OUT / "m1_state.pkl"
    if not m1_state_path.exists():
        raise SystemExit(f"{m1_state_path} missing — run the loop example's run_m1.py first")
    with open(m1_state_path, "rb") as fh:
        m1_state = pickle.load(fh)
    if m1_state.get("model_key") not in (None, args.model):
        print(f"[WARN] frozen M1 is for {m1_state.get('model_key')!r}, surgery on {args.model!r}")
    cases = m1_state["cases"]

    hypotheses = _to_loop_hypotheses(confirm.get("hypothesis_verdicts") or [],
                                     model_key=args.model,
                                     include_refuted=args.include_refuted)
    print(f"surgery targets: {len(hypotheses)} hypothesis(es)")
    for h in hypotheses:
        print(f" - [{h.predicted_failure_mode}] {h.statement[:100]}")
    if not hypotheses:
        raise SystemExit("no hypothesis survived phase 2 with surgery eligibility — "
                         "nothing to repair (honest outcome).")

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

    judge = loop_run.build_judge(loop_run.CFG.get("judge_model", "claude-opus-4-8"),
                                 loop_run.CFG.get("judge_effort", "low"))
    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    codegen: CliAgentConfig = loop_run.build_codegen(args.backend)

    run_logger = RunLogger(run_dir=root / "3_surgery" / "logs", verbose=True)
    loop = VLDiagnoseLoop(
        model=model,
        protocol=loop_run.build_protocol(),
        probe_agent=loop_run.ReplayProbeAgent(m1_state),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(judge=judge,
                           max_tier=str(loop_run.CFG.get("fix_max_tier", "L3b")),
                           cli_config=codegen, run_logger=run_logger,
                           max_validation_cases=args.max_validation_cases,
                           exec_timeout_sec=int(loop_run.CFG.get("fix_exec_timeout_sec", 900))),
        run_logger=run_logger,
    )

    # ── M5: confirm the held-out-vetted hypotheses on the loop's evidence ────
    report = loop.run_confirm(cases, hypotheses)
    print(f"\nM5 confirm: verified={len(report.verified_hypotheses)}/{len(report.all_test_results)}")
    m5 = []
    for t in report.all_test_results:
        stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
        print(f" - [{t.status}] conf={t.confidence:.2f} grade={t.evidence_grade} {stmt[:100]}")
        m5.append({"statement": stmt, "status": str(t.status),
                   "confidence": float(t.confidence),
                   "evidence_grade": str(t.evidence_grade),
                   "holdout_verdict": getattr(t.hypothesis, "metadata", {}).get("holdout_verdict")})

    # ── M4 surgery + tiered fix (L1→L3b) ────────────────────────────────────
    m4_summary = None
    fix_summary = None
    if not args.skip_m4:
        m4 = loop.run_m4(report, cases)
        m4_summary = str(m4)
        print("m4:", m4_summary[:300])
        outcome = loop.run_fix(report, cases)
        print("fix outcome: best=", getattr(outcome, "best", None), "fixed=",
              getattr(outcome, "fixed", None))
    run_logger.close()

    if not args.skip_m4:
        # The run logger's fix event is the single source of truth for the
        # per-candidate outcomes; distill the dashboard-facing structure from it.
        event = _fix_event_from_logs(run_logger.run_dir)
        fix_summary = _lean_fix(event) if event else {"note": "no fix event logged"}

    out = {
        "phase": "surgery_fix",
        "model": args.model,
        "n_cases": len(cases),
        "hypotheses_in": [h.statement for h in hypotheses],
        "m5_results": m5,
        "m4": m4_summary,
        "fix": fix_summary,
        "logs": str(run_logger.run_dir),
    }
    out_path = root / "1_explore" / "fix_report.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")
    print(f"full M5/M4/Fix logs -> {run_logger.run_dir}")


if __name__ == "__main__":
    main()
