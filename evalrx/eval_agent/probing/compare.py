"""Two schedulers' runs side by side: what each chose, in what order, at what evidence cost.

Both runs may share one ledger, so a scheduler that runs second reads most
observations back. Cost is therefore compared on the evidence each report
was built from, at the cost the observation had when it was first executed
(``cost.evidence``), not on what the second invocation happened to spend.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from evalrx.eval_agent.probing.explanations import CARDS, verdict


def discrimination(explained: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """How many pairs of *explained*'s last explanations (per route) with conflicting predictions
    *report*'s evidence tells apart: one of the pair misses on a reading they conflict on.

    Applies one run's explanations to another run's final ``outcomes``, so two
    schedulers are compared on the same explanations whichever of them asked."""
    sets = [s for s in (explained.get("explanations") or {}).get("sets") or [] if s["hypotheses"]]
    if not sets:
        return {"revision": None, "n_conflicting_pairs": 0, "n_resolved": 0, "pairs": []}
    last, outcomes = sets[-1], report.get("outcomes") or {}
    bands = {(key, m.field): m.band for key, ms in CARDS.items() for m in ms}
    pairs = []
    for a, b in combinations(last["hypotheses"], 2):
        if a["route"] != b["route"]:
            continue
        on = [(step, f) for step in sorted(set(a["predictions"]) & set(b["predictions"]))
              for f in sorted(set(a["predictions"][step]) & set(b["predictions"][step]))
              if a["predictions"][step][f] != b["predictions"][step][f]]
        if not on:
            continue
        resolved = [f"{step}.{f}" for step, f in on
                    if any(verdict(h["predictions"][step][f],
                                   ((outcomes.get(a["route"]) or {}).get(step) or {}).get(f, {}).get("ci95"),
                                   bands[(step, f)]) == "miss" for h in (a, b))]
        pairs.append({"route": a["route"], "pair": [a["id"], b["id"]], "on": [f"{s}.{f}" for s, f in on],
                      "resolved_on": resolved})
    return {"revision": last["revision"], "n_conflicting_pairs": len(pairs),
            "n_resolved": sum(1 for x in pairs if x["resolved_on"]), "pairs": pairs}


def _followed(report: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(d.get("route") or "-", d.get("rule_id") or "-", d.get("key") or d["kind"])
            for d in report["decisions"] if d["kind"] in ("expand", "follow_up", "extend", "battery")]


def compare(baseline: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Selection and cost differences of *other* against the *baseline* run (both triage reports)."""
    a, b = _followed(baseline), _followed(other)
    def open_units(r: dict[str, Any]) -> list[dict[str, Any]]:  # units some tool could still close here
        return [u for u in (r.get("coverage") or {}).values() if u.get("layer_state") != "exhausted"]

    gap = {name: sum(u["gap"] for u in open_units(r)) for name, r in (("baseline", baseline), ("other", other))}
    units = {name: len(open_units(r)) for name, r in (("baseline", baseline), ("other", other))}
    out = {
        "schedulers": [baseline.get("scheduler", "rules"), other.get("scheduler", "rules")],
        "same_sample": baseline["sample"]["case_ids"] == other["sample"]["case_ids"],
        "actions": {"baseline": a, "other": b},
        "only_baseline": [x for x in a if x not in b],
        "only_other": [x for x in b if x not in a],
        "same_order": a == b,
        "evidence_cost": {"baseline": (baseline.get("cost") or {}).get("evidence"),
                          "other": (other.get("cost") or {}).get("evidence")},
        "charged": {"baseline": (baseline.get("cost") or {}).get("charged"),
                    "other": (other.get("cost") or {}).get("charged")},
        "judge": {"baseline": (baseline.get("cost") or {}).get("judge"),
                  "other": (other.get("cost") or {}).get("judge")},
        "controller_sec": {"baseline": (baseline.get("cost") or {}).get("controller_sec"),
                           "other": (other.get("cost") or {}).get("controller_sec")},
        "coverage_gap": gap, "coverage_units": units,
        "alternatives": {name: [f"{d['route']}:{d['rule_id']}:{d['key']} (to {d['alternative_to']})"
                                for d in r["decisions"] if d.get("alternative_to") and d["kind"] == "follow_up"]
                         for name, r in (("baseline", baseline), ("other", other))},
        "explored": {name: {"decisions": sum(1 for d in r["decisions"] if d.get("choice") == "explore"),
                            **(r.get("explored") or {})} for name, r in (("baseline", baseline), ("other", other))},
        "stop": {"baseline": baseline["stop"], "other": other["stop"]},
    }
    source = next((r for r in (other, baseline) if r.get("explanations")), None)
    if source is not None:   # both runs' evidence, on the same explanations
        out["discrimination"] = {"baseline": discrimination(source, baseline), "other": discrimination(source, other)}
    return out


def markdown(diff: dict[str, Any]) -> str:
    a_name, b_name = diff["schedulers"]
    lines = [f"# Scheduler comparison: {a_name} vs {b_name}", "",
             f"Same final sample: {diff['same_sample']}. Same actions in the same order: {diff['same_order']}.", "",
             f"| | {a_name} | {b_name} |", "|---|---|---|"]
    rows = max(len(diff["actions"]["baseline"]), len(diff["actions"]["other"]))
    for i in range(rows):
        cells = [":".join(x[i]) if i < len(x) else "" for x in (diff["actions"]["baseline"], diff["actions"]["other"])]
        lines.append(f"| {i} | {cells[0]} | {cells[1]} |")
    ev = diff["evidence_cost"]
    lines += ["", f"Evidence cost (first execution of every observation read): {a_name} {ev['baseline']}; "
              f"{b_name} {ev['other']}.",
              f"Charged against the budget (evidence plus judge calls): {diff['charged']}; judge: {diff['judge']}.",
              f"Controller time outside the actions and the judge: {diff['controller_sec']}.",
              f"Coverage gap left on units some tool can close here (sum): {diff['coverage_gap']} of "
              f"{diff['coverage_units']} units.",
              f"Alternatives run on answered layers: {diff['alternatives']}; exploration: {diff['explored']}.",
              f"Stop: {diff['stop']}.", ""]
    if "discrimination" in diff:
        d = diff["discrimination"]
        lines += [f"Pairs of competing explanations (the explaining run's last set) each run's evidence tells "
                  f"apart: {a_name} {d['baseline']['n_resolved']}/{d['baseline']['n_conflicting_pairs']}, "
                  f"{b_name} {d['other']['n_resolved']}/{d['other']['n_conflicting_pairs']}.", ""]
    return "\n".join(lines)
