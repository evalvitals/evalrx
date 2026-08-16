"""deco_hallu — STAGE 1 of a split run: M1 only (analyzer selection + execution).

Runs just the ProbeAgent (M1) on the frozen hallucination batch, writes the full
human-readable M1 logs (logs_m1/ — selection prompt/response, per-analyzer
result.json, artifacts, failed analyzers), and freezes the M1 output to a pickle
so the analysis stages can be re-run against it WITHOUT re-doing M1's forward
passes / generations.

    python run_m1.py --model qwen3-vl-2b-instruct --device cuda
    # -> outputs/m1_state.pkl   (consumed by run_m2-5.py)
    # -> outputs/logs_m1/        (observable M1 record)

This is the expensive, model-bound part of the pipeline; isolating it lets you
iterate on M2-M5 / the fix module (run_m2-5.py) against a fixed M1 result.
Shared setup (manifest, judge, protocol) is reused from run.py.
"""

from __future__ import annotations

import argparse
import pickle

import run  # the existing single-shot script — reuse its setup functions

OUT = run.OUT
M1_STATE = OUT / "m1_state.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=run.CFG["model"])
    ap.add_argument("--max-analyzers", type=int, default=3)
    ap.add_argument("--judge-model", default=run.CFG.get("judge_model", "claude-opus-4-8"))
    ap.add_argument("--judge-effort", default=run.CFG.get("judge_effort", "low"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    from evalvitals import compose
    from evalvitals.core.capability import Capability
    from evalvitals.eval_agent import CliAgentConfig, RunLogger
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.models.backends.base import RuntimeConfig

    judge = run.build_judge(args.judge_model, args.judge_effort)
    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    cases, raw = run.load_manifest(args.model)
    print(f"cases={len(list(cases))} yields={raw['yields']} "
          f"hallu={raw['n_hallucination']} reject={raw['n_correct_reject']} "
          f"present={raw['n_present_detect']}")
    run.drift_check(model, cases)

    codegen_effort = str(run.CFG.get("codegen_effort", "") or "")
    codegen = CliAgentConfig(
        provider="claude_code",
        model=str(run.CFG.get("codegen_model", "claude-opus-4-8")),
        max_budget_usd=float(run.CFG.get("codegen_budget_usd", 2.0)),
        timeout_sec=int(run.CFG.get("codegen_timeout_sec", 240)),
        extra_args=(("--effort", codegen_effort) if codegen_effort else ()),
    )
    protocol = run.build_protocol()
    run_logger = RunLogger(run_dir=OUT / "logs_m1", verbose=True)
    probe_agent = ProbeAgent(judge=judge, max_analyzers=args.max_analyzers,
                             allow_codegen=True, codegen_config=codegen,
                             run_logger=run_logger)

    # ── M1 only ────────────────────────────────────────────────────────────
    probe_results = probe_agent.probe(model, cases, protocol=protocol)
    if not probe_results:
        raise SystemExit("M1 produced no probe results — nothing to freeze.")

    # Human-observable M1 record (selection IO, per-analyzer result.json,
    # artifacts, failed analyzers) — the full M1 output, on disk.
    run_logger.log_probe(
        0, probe_results, schema=probe_agent.last_schema,
        judge_prompt=getattr(probe_agent, "last_selection_prompt", ""),
        judge_raw=getattr(probe_agent, "last_selection_raw", ""),
        failed_analyzers=getattr(probe_agent, "_failed_analyzers", None) or None,
    )
    run_logger.close()

    # Machine handoff: freeze the exact M1 objects for run_m2-5.py. cases is
    # pickled alongside so the analysis stages see precisely the batch M1 saw
    # (Result.model is a repr string and artifacts are numpy/JSON, so the whole
    # state pickles cleanly without dragging in the HF model).
    state = {
        "schema_version": 1,
        "model_key": args.model,
        "cases": cases,
        "probe_results": probe_results,
        "schema": probe_agent.last_schema,
        "selection_prompt": getattr(probe_agent, "last_selection_prompt", ""),
        "selection_raw": getattr(probe_agent, "last_selection_raw", ""),
        "failed_analyzers": dict(getattr(probe_agent, "_failed_analyzers", {}) or {}),
        "generated_probes": [code for _, code in
                             getattr(probe_agent, "_generated_probes", []) or []],
    }
    with open(M1_STATE, "wb") as fh:
        pickle.dump(state, fh)

    print(f"\nM1 done — analyzers: {list(probe_results)}")
    for name, r in probe_results.items():
        keys = ", ".join(list(r.findings)[:6])
        print(f"  - {name}: {len(r.findings)} findings ({keys})")
    if state["failed_analyzers"]:
        print(f"  failed: {state['failed_analyzers']}")
    print(f"frozen M1 state -> {M1_STATE}")
    print(f"observable M1 logs -> {run_logger.run_dir}")
    print("next: python run_m2-5.py --model %s --device %s" % (args.model, args.device))


if __name__ == "__main__":
    main()
