"""Two schedulers' runs side by side: what each chose, in what order, at what evidence cost.

Both runs may share one ledger, so a scheduler that runs second reads most
observations back. Cost is therefore compared on the evidence each report
was built from, at the cost the observation had when it was first executed
(``cost.evidence``), not on what the second invocation happened to spend.
"""

from __future__ import annotations

from typing import Any


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
    return {
        "schedulers": [baseline.get("scheduler", "rules"), other.get("scheduler", "rules")],
        "same_sample": baseline["sample"]["case_ids"] == other["sample"]["case_ids"],
        "actions": {"baseline": a, "other": b},
        "only_baseline": [x for x in a if x not in b],
        "only_other": [x for x in b if x not in a],
        "same_order": a == b,
        "evidence_cost": {"baseline": (baseline.get("cost") or {}).get("evidence"),
                          "other": (other.get("cost") or {}).get("evidence")},
        "controller_sec": {"baseline": (baseline.get("cost") or {}).get("controller_sec"),
                           "other": (other.get("cost") or {}).get("controller_sec")},
        "coverage_gap": gap, "coverage_units": units,
        "stop": {"baseline": baseline["stop"], "other": other["stop"]},
    }


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
              f"Controller time outside the actions: {diff['controller_sec']}.",
              f"Coverage gap left on units some tool can close here (sum): {diff['coverage_gap']} of "
              f"{diff['coverage_units']} units.",
              f"Stop: {diff['stop']}.", ""]
    return "\n".join(lines)
