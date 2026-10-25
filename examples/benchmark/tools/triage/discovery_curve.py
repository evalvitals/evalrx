"""Budget-matched discovery curve of a multi-arm run directory (no model, no GPU).

For each arm, the number of validation-supported phenomena whose issues the arm had discovered within a
charged budget B. An issue counts as discovered at the charge of its step's last execution on its route (a
battery step: on the whole sample; a route signal: once the five route steps have run), so the curve is
conservative for a chunked step (its last chunk). Charges follow the report: every execution the arm read,
at the cost of its first execution (a read-back at its original cost, once); judge calls are not charged.

usage: python discovery_curve.py <run dir> [--arms arm:report_stem ...] [--grid B ...]
   e.g. python discovery_curve.py .../chartqa.triage18-chartqa-2b-mf128 --arms rules_full:triage_report \
            scored_full:triage_report.scored --grid 3000 4000 5000 6000 7095
"""
import argparse
import json
import math
from pathlib import Path

ROUTE_STEPS = {"answer_extraction_audit", "termination_audit", "modality_ablation", "self_consistency", "prompt_reword"}
DEFAULT_ARMS = ["rules_full:triage_report", "scored_full:triage_report.scored",
                "scored_x0_full:triage_report.scored", "scored_d_full:triage_report.scored_d"]
DEFAULT_GRID = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


def curve_of(run: Path, arm: str, stem: str, report: dict, frozen: dict, grid: list) -> dict:
    probing = run / "probing"
    phen_of = {iid: p["id"] for p in report["phenomena"] for iid in p["issues"]}
    supported = {p["id"] for p in report["phenomena"] if p["verdict"] == "supported"}
    issue_by_id = {i["issue_id"]: i for i in report["issues"]}
    r = json.loads((probing / f"{stem}.{arm}.json").read_text())
    seen, charge, cum = set(), 0.0, []
    for e in r["executions"]:
        aid = e.get("action_id")
        if aid is None:
            charge += float((e.get("cost") or {}).get("wall_sec") or 0.0)
        elif aid not in seen:
            seen.add(aid)
            f = probing / "observations" / f"{aid}.json"
            obs = json.loads(f.read_text()) if f.is_file() else {}
            charge += float((obs.get("cost") or e.get("cost") or {}).get("wall_sec") or 0.0)
        cum.append((e, charge))
    settled = max((c for e, c in cum if e.get("key") in ROUTE_STEPS), default=0.0)   # the routes are known
    found_at = {}
    for iid in frozen["arms"].get(arm) or []:
        i = issue_by_id[iid]
        if not i.get("step"):
            found_at[iid] = settled                       # a route signal: known once the route steps ran
            continue
        # a follow-up is keyed by route; a battery step runs on the whole sample (route None)
        last = [c for e, c in cum if e.get("key") == i["step"] and e.get("route") in (i["route"], None)]
        if not last:   # a reading frozen for a route from a step that ran on another route's cases
            last = [c for e, c in cum if e.get("key") == i["step"]]
        found_at[iid] = max(last) if last else None
    curve = [len({phen_of[iid] for iid, c in found_at.items()
                  if c is not None and c <= B and phen_of.get(iid) in supported}) for B in grid]
    return {"total_charged": round(r["cost"]["charged"]["wall_sec"]), "settled_at": round(settled), "curve": curve,
            "found_at": {phen_of.get(k, "?"): (round(v) if v is not None else None) for k, v in found_at.items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="the run directory (holding probing/ and validation/)")
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, metavar="ARM:STEM",
                   help="arm name and its report stem, e.g. rules_full:triage_report")
    p.add_argument("--grid", nargs="+", type=float, default=DEFAULT_GRID, metavar="B",
                   help="charged budgets (seconds) to count the supported phenomena found within")
    a = p.parse_args()
    report = json.loads((a.run / "validation" / "validation_report.json").read_text())
    frozen = json.loads((a.run / "validation" / "frozen.json").read_text())
    grid = [*a.grid, math.inf]
    for spec in a.arms:
        arm, stem = spec.split(":", 1)
        row = curve_of(a.run, arm, stem, report, frozen, grid)
        print(f"{arm:16s} charged={row['total_charged']:6d} settled={row['settled_at']:6d} "
              f"curve@{[int(b) for b in grid[:-1]] + ['total']}: {row['curve']}")
        print("    phenomenon -> found at:", dict(sorted(row["found_at"].items(), key=lambda kv: (kv[1] is None, kv[1] or 0))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
