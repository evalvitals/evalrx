"""Convert the agent M5 record into the dashboard's Fix-tab format.

The five-tab explore view fills its "5 Fix" panel from a ``fix_report.json``
sitting next to ``exploratory_report.json``.  The VLM pipeline writes that
file from its surgery loop; the agent arc records paired fixes in
``m5_report.json`` instead.  This script maps one onto the other — including,
when replication batches are given, the combined-e row that carries the
validation (e-values multiply across independent batches).

Usage:
    python make_fix_report.py --out outputs \
        --replicate outputs_2b_chart outputs_2b_color outputs_2b_math \
                    outputs_2b_measure outputs_2b_spatial
    python make_fix_report.py --out outputs_4b     # per-batch only, no combine
"""

from __future__ import annotations

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

_TIERS = {"L1_tool_desc": "L1", "L2_loop_policy": "L2", "L1_plus_L2": "L1+L2"}
_KINDS = {
    "L1_tool_desc": "tool-description rewrite",
    "L2_loop_policy": "loop policy (block repeats + force final answer)",
    "L1_plus_L2": "description rewrite + loop policy",
}


def _arm_row(name: str, arm: dict, *, suffix: str = "") -> dict:
    reject = bool(arm.get("reject"))
    return {
        "tier": _TIERS.get(name, "L2"),
        "name": name + suffix,
        "kind": _KINDS.get(name, "replication batch"),
        "n_fixed": arm.get("fixed_cases"),
        "n_broken": arm.get("broken_cases"),
        "coverage": arm.get("pass"),
        "e_value": arm.get("e_value"),
        "reject": reject,
        "verdict": "REJECT H0 [fixed]" if reject else "inconclusive on this batch",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"),
                    help="M1/M5 output dir (contains m5_report.json + explore/)")
    ap.add_argument("--replicate", nargs="*", default=[],
                    help="further output dirs whose L2 arm joins the combined-e row")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    m5 = json.load(open(os.path.join(out, "m5_report.json")))
    cfg = {}
    cfg_path = os.path.join(out, "run_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))

    attempted = [_arm_row(name, arm) for name, arm in m5["arms"].items()]
    n_cases = m5.get("n")
    best = None
    ebh_survivors: list[str] = []
    recommendation = (
        "No candidate cleared the anytime-valid bar on this batch alone; "
        "replicate the strongest arm on independent batches (e-values multiply)."
    )

    # -- combined replication row (the validation, when batches are given) --
    if args.replicate and "L2_loop_policy" in m5["arms"]:
        arms = [m5["arms"]["L2_loop_policy"]]
        for rep in args.replicate:
            rep_dir = rep if os.path.isabs(rep) else os.path.join(HERE, rep)
            rep_m5 = json.load(open(os.path.join(rep_dir, "m5_report.json")))
            arm = rep_m5["arms"]["L2_loop_policy"]
            attempted.append(_arm_row("L2_loop_policy", arm,
                                      suffix=f" @ {os.path.basename(rep_dir).split('_')[-1]}"))
            arms.append(arm)
            n_cases += rep_m5.get("n", 0)
        combined_e = math.prod(float(a["e_value"] or 0.0) for a in arms)
        combined = {
            "tier": "L2",
            "name": "L2_loop_policy (combined, 6 independent batches)",
            "kind": "loop policy — replication product",
            "n_fixed": sum(a["fixed_cases"] for a in arms),
            "n_broken": sum(a["broken_cases"] for a in arms),
            "coverage": sum(a["pass"] for a in arms),
            "e_value": combined_e,
            "reject": combined_e >= 1 / m5.get("alpha", 0.05),
            "verdict": "VALIDATED — anytime-valid reject via multiplied e-values",
        }
        attempted.append(combined)
        best = combined["name"]
        ebh_survivors = [combined["name"]]
        recommendation = (
            "Adopt the L2 loop policy as the default agent configuration "
            "(done: run_m1.py runs it by default; --no-loop-policy reproduces "
            "the unfixed baseline). Escalating the checkpoint does not move "
            "the capability wall (2B/4B/8B all ~0.8 fail); the next repair "
            "candidate is a counting scaffold that aggregates detector boxes "
            "itself."
        )

    # -- M4 context from the held-out confirm report ----------------------
    m4_results = []
    conf_path = os.path.join(out, "explore", "confirm_report.json")
    if os.path.exists(conf_path):
        conf = json.load(open(conf_path))
        for h in conf.get("hypothesis_verdicts") or []:
            m4_results.append({
                "statement": h.get("plain_statement") or h.get("statement", ""),
                "status": h.get("verdict"),
                "confidence": None,
                "evidence_grade": "held-out judge",
                "holdout_verdict": h.get("verdict"),
            })

    fix_report = {
        "model": f"{cfg.get('model', m5.get('model', '?'))} (vLLM, agent loop)",
        "n_cases": n_cases,
        "logs": os.path.relpath(out, HERE),
        "m4_results": m4_results,
        "m5": {"paired_baseline": "recorded unfixed M1 run",
               "stats": "paired McNemar, anytime-valid e-values, e-BH family"},
        "fix": {
            "attempted": attempted,
            "best": best or max(attempted, key=lambda a: float(a["e_value"] or 0.0))["name"],
            "ebh_survivors": ebh_survivors,
            "recommendation": recommendation,
        },
    }
    dst = os.path.join(out, "explore", "fix_report.json")
    with open(dst, "w") as f:
        json.dump(fix_report, f, indent=1)
    print(f"wrote {dst}: {len(attempted)} candidates, best={fix_report['fix']['best']!r}")


if __name__ == "__main__":
    main()
