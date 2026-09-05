"""VTC-Bench agent diagnosis — M5: paired fix experiments for the confirmed causes.

Held-out confirmed two failure structures (budget exhaustion without an
answer; more tool use → worse), and the judge marked the tool-description
hypothesis as testable ONLY by intervention.  This runs the interventions,
paired against the recorded baseline on the same 85 cases:

  L1_tool_desc    tool descriptions gain an explicit no-repeat warning
  L2_loop_policy  Agent(block_repeat_calls=True, force_final_answer=True)
  L1_plus_L2      both

Statistics per the house rules: paired McNemar via ``stats.compare``
(anytime-valid e-value correction), e-BH across the candidate-fix family,
no-free-lunch accounting (cases fixed vs cases broken), and a separate
read-out on the explore run's held-out rows.

Usage:
    .venv/bin/python examples/m2_m3/vtcbench_diagnosis/run_m5.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_m1 import SYSTEM, load_cases, mc_grader  # noqa: E402

from evalrx import compose  # noqa: E402
from evalrx.models.agent import run_batch  # noqa: E402
from evalrx.models.backends.openai_compat import openai_runtime  # noqa: E402
from evalrx.models.tools import detect_tool, zoom_in_tool  # noqa: E402
from evalrx.models.tools.perception import default_detect_engine  # noqa: E402
from evalrx.stats import compare  # noqa: E402
from evalrx.stats.ebh import ebh  # noqa: E402

NO_REPEAT_WARNING = (
    " Calling this tool again with the SAME arguments returns the exact same "
    "result you already received — never repeat an identical call; choose a "
    "different region/arguments or give your final answer."
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="counting")
    ap.add_argument("--base-url", default="http://localhost:8901/v1")
    ap.add_argument("--model", default="qwen3-vl-2b-instruct")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--detect-device", default="cuda:2")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--arms", default="L1_tool_desc,L2_loop_policy,L1_plus_L2",
                    help="comma-separated subset of fix arms to run (replication runs "
                         "typically test one pre-registered arm)")
    args = ap.parse_args()

    # -- baseline: the recorded M1 run (paired by doc_id) --------------
    # Fix arms only mean something against the UNFIXED agent; refuse a baseline
    # that was recorded with the (now default) loop policy already on.
    cfg_path = os.path.join(args.out, "run_config.json")
    if os.path.exists(cfg_path) and json.load(open(cfg_path)).get("loop_policy"):
        raise SystemExit(
            f"{cfg_path} says the recorded baseline already ran WITH the loop "
            "policy (the validated default). Re-record it first:\n"
            f"  run_m1.py --task {args.task} --out {args.out} --no-loop-policy"
        )
    records = json.load(open(os.path.join(args.out, "records.json")))
    baseline = {r["doc_id"]: r["label"] == "pass" for r in records}
    holdout_path = os.path.join(args.out, "explore", "holdout_records.json")
    holdout_ids = (
        {r["doc_id"] for r in json.load(open(holdout_path))}
        if os.path.exists(holdout_path)
        else set()  # no explore ran for this batch — skip the slice read-out
    )
    cases = load_cases(args.task, None)
    cases = [c for c in cases if c.metadata["doc_id"] in baseline]
    success_base = [baseline[c.metadata["doc_id"]] for c in cases]
    print(f"[0] {len(cases)} paired cases | baseline pass {sum(success_base)}/{len(cases)} "
          f"| held-out slice {sum(c.metadata['doc_id'] in holdout_ids for c in cases)}")

    engine = default_detect_engine(device=args.detect_device)
    vlm = compose(args.model, "api",
                  runtime=openai_runtime(base_url=args.base_url, max_tokens=512))

    def plain_tools(case):
        return [zoom_in_tool(case.inputs.image), detect_tool(case.inputs.image, engine=engine)]

    def warned_tools(case):
        return [dataclasses.replace(t, description=t.description + NO_REPEAT_WARNING)
                for t in plain_tools(case)]

    arms = {
        "L1_tool_desc":   dict(tools_factory=warned_tools, agent_kwargs={}),
        "L2_loop_policy": dict(tools_factory=plain_tools,
                               agent_kwargs={"block_repeat_calls": True, "force_final_answer": True}),
        "L1_plus_L2":     dict(tools_factory=warned_tools,
                               agent_kwargs={"block_repeat_calls": True, "force_final_answer": True}),
    }
    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in selected if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; choose from {list(arms)}")
    arms = {name: arms[name] for name in selected}

    report: dict = {"n": len(cases), "baseline_pass": sum(success_base), "alpha": args.alpha,
                    "arms": {}}
    evalues: list[float] = []
    arm_names: list[str] = []

    for name, cfg in arms.items():
        t0 = time.time()
        trajs = run_batch(vlm, cases, tools_factory=cfg["tools_factory"], system=SYSTEM,
                          max_turns=args.max_turns, concurrency=args.concurrency,
                          agent_kwargs=cfg["agent_kwargs"])
        success_fix = [bool(mc_grader(t, c)) for c, t in zip(cases, trajs)]
        stat = compare(success_base, success_fix, paired=True, alpha=args.alpha,
                       correction="evalue")
        fixed = sum(1 for a, b in zip(success_base, success_fix) if not a and b)
        broken = sum(1 for a, b in zip(success_base, success_fix) if a and not b)
        ho = [(a, b) for c, a, b in zip(cases, success_base, success_fix)
              if c.metadata["doc_id"] in holdout_ids]
        ho_stat = compare([a for a, _ in ho], [b for _, b in ho], paired=True,
                          alpha=args.alpha, correction="evalue") if ho else None
        forced = sum(1 for t in trajs if t.metrics.get("terminated") == "forced_final")
        blocked = sum(
            1 for t in trajs for s in t.steps if s.span.get("repeat_blocked")
        )
        report["arms"][name] = {
            "pass": sum(success_fix),
            "effect": stat.effect,
            "ci": list(stat.ci),
            "e_value": stat.e_value,
            "reject": stat.reject,
            "fixed_cases": fixed,
            "broken_cases": broken,
            "net": fixed - broken,
            "n_forced_final": forced,
            "n_repeat_blocked": blocked,
            "holdout_slice": None if ho_stat is None else {
                "n": len(ho), "effect": ho_stat.effect, "e_value": ho_stat.e_value,
                "reject": ho_stat.reject,
            },
            "per_case_pass": {c.metadata["doc_id"]: b for c, b in zip(cases, success_fix)},
        }
        evalues.append(stat.e_value or 0.0)
        arm_names.append(name)
        print(f"[{name}] {time.time()-t0:.0f}s pass {sum(success_fix)}/{len(cases)} "
              f"(baseline {sum(success_base)}) effect={stat.effect:+.3f} e={stat.e_value:.2f} "
              f"reject={stat.reject} | fixed {fixed} broken {broken} "
              f"| forced_final {forced} repeat_blocked {blocked}")

    family = ebh(evalues, alpha=args.alpha)
    report["family"] = {"method": "e-BH", "alpha": args.alpha,
                        "rejected": [arm_names[i] for i in family]}
    print(f"\n[e-BH over {len(arm_names)} candidate fixes] rejected: "
          f"{report['family']['rejected'] or 'NONE'}")
    out_path = os.path.join(args.out, "m5_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
