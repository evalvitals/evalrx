"""The report's top-line reading: findings, charts, the repair, and the header
metrics.
"""

from __future__ import annotations

from typing import Any, Mapping

from evalrx.reporting.plain_labels import _plain_signal, _stat_label
from evalrx.reporting.report_inputs import _contract_stats
from evalrx.reporting.stage_views import _repair_name


def _findings(reader: Mapping[str, Any], raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, finding in enumerate(reader.get("key_findings") or []):
        if isinstance(finding, dict):
            result.append({"id": f"finding-{index + 1}", **finding})
    if result:
        return result
    hypotheses = (raw.get("m3") or {}).get("hypotheses") or []
    for index, hypothesis in enumerate(hypotheses[:3]):
        if isinstance(hypothesis, dict):
            result.append({
                "id": f"finding-{index + 1}",
                "title": hypothesis.get("failure_mode") or "Candidate mechanism",
                "summary": hypothesis.get("plain_statement") or hypothesis.get("statement") or "",
                "evidence_level": hypothesis.get("status") or "Proposed",
                "limitation": "See M4 for independent verification.",
            })
    return result


def _charts(run: Mapping[str, Any], raw: Mapping[str, Any],
            contract: "Mapping[str, Any] | None" = None) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    labels = run.get("label_distribution") or {}
    if isinstance(labels, dict) and labels:
        charts.append({
            "id": "outcomes", "kind": "donut", "title": "Evaluation outcomes",
            "series": [{"label": str(k).title(), "value": v} for k, v in labels.items() if isinstance(v, (int, float))],
        })
    # Prefer the contract's rows: `measured` is a name for the SUBJECT and is
    # unique across the report, so the axis cannot end up with two bars reading
    # "Mcnemar evalue" (the tool) or several truncating to the same sentence.
    contract_rows = _contract_stats(contract or {})
    effect_rows = []
    for item in contract_rows[:8]:
        if isinstance(item.get("effect"), (int, float)):
            signal = (item.get("config") or {}).get("signal")
            effect_rows.append({
                "label": item.get("measured") or _plain_signal(signal or item.get("tool")),
                # The analyzer's own sentence. Absent means undocumented, and the
                # UI must say that rather than paraphrase the identifier.
                "means": item.get("means") or "",
                "raw_label": str(signal or item.get("analysis_key") or item.get("tool") or ""),
                "value": item["effect"], "highlight": bool(item.get("fdr_corrected")),
            })
    if not effect_rows:
        stats = (raw.get("m2") or {}).get("stats") or []
        for item in stats[:8]:
            if isinstance(item, dict) and isinstance(item.get("effect"), (int, float)):
                signal = ((item.get("config") or {}).get("signal") if isinstance(item.get("config"), Mapping) else None)
                effect_rows.append({
                    "label": _stat_label(item.get("config"), signal, item.get("tool")),
                    "raw_label": str(signal or item.get("tool") or ""),
                    "value": item["effect"], "highlight": bool(item.get("reject")),
                })
    if effect_rows:
        charts.append({
            "id": "effects", "kind": "bar", "title": "Which behaviors appear most connected to errors?",
            "subtitle": "A larger bar means a stronger observed pattern. It does not prove the behavior caused the error.",
            "series": effect_rows,
        })
    return charts


def _repairs(raw: Mapping[str, Any], m5_detail: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    fix = raw.get("m5_fix") or {}
    if not fix.get("ran"):
        return []
    confirm = found if isinstance(found := fix.get("confirm"), Mapping) else {}
    scored = _m5_scored(m5_detail)
    # Early runs recorded the winning candidate as its bare name; later ones
    # record the whole candidate. Both name the same repair.
    best_name = _repair_name(fix.get("best"))
    return [{
        "id": "repair-1", "fixed": bool(fix.get("fixed")),
        "title": best_name or confirm.get("name") or (scored or {}).get("name") or "Targeted repair",
        "effect": (scored or confirm).get("effect"),
        "fixed_cases": int(scored["n_fixed"]) if scored else len(confirm.get("fixed_cases") or []),
        "broken_cases": int(scored.get("n_broken") or 0) if scored else len(confirm.get("broken_cases") or []),
    }]


def _m5_scored(m5_detail: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The repair M5 reports as its result, scored on the withheld cases.

    The stage's own confirmation -- the best candidate re-scored on the
    confirm pool -- is what M5 prints as "21 repaired, 6 broken", whether or
    not that candidate cleared the certification bar. With no named winner
    (it missed the bar, so the fix event names none) the stage still scored
    its confirm-phase candidates, and the strongest of those is what it
    prints. None when M5 scored nothing.
    """
    def _counted(item: Any) -> bool:
        value = item.get("n_fixed") if isinstance(item, Mapping) else None
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    detail = m5_detail if isinstance(m5_detail, Mapping) else {}
    confirmation = detail.get("confirmation")
    if isinstance(confirmation, Mapping) and confirmation.get("name") and _counted(confirmation):
        return confirmation
    scored = [item for item in detail.get("candidates") or [] if _counted(item)]
    if scored:
        return max(scored, key=lambda item: (float(item.get("effect") or 0.0), float(item["n_fixed"])))
    return None


def _cases_repaired(cases: list[dict[str, Any]], m5_detail: Mapping[str, Any] | None) -> int:
    """How many withheld cases the M5 repair turned from wrong to right.

    The header tile says what M5 printed (see :func:`_m5_scored`). Case status
    never carries a "fixed" value (it is pass / fail), so counting it left the
    tile at 0 on every run that repaired anything. Runs that scored no
    candidate fall back to the per-case repair outcomes, then to the old
    status count.
    """
    scored = _m5_scored(m5_detail)
    if scored:
        return int(scored["n_fixed"])
    per_case = sum(
        1 for case in cases
        if isinstance(case.get("repair"), Mapping) and str(case["repair"].get("status")) == "fixed"
    )
    if per_case:
        return per_case
    return sum(1 for case in cases if case.get("status") == "fixed")


def _metrics(
    run: Mapping[str, Any],
    raw: Mapping[str, Any],
    cases: list[dict[str, Any]],
    m5_detail: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels = run.get("label_distribution") or {}
    failed = labels.get("fail", labels.get("FAIL", 0)) if isinstance(labels, dict) else 0
    verified = len((raw.get("m4") or {}).get("results") or [])
    fixed = _cases_repaired(cases, m5_detail)
    return [
        {"id": "evaluated", "label": "Cases evaluated", "value": int(run.get("n_cases") or len(cases))},
        {"id": "failed", "label": "Initial failures", "value": int(failed or 0)},
        {"id": "verified", "label": "Mechanisms checked", "value": verified},
        {"id": "fixed", "label": "Cases repaired", "value": fixed},
    ]


def _explore_metrics(stage_detail: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An explore run evaluated no model: count what it found instead."""
    m2, m3 = stage_detail.get("m2") or {}, stage_detail.get("m3") or {}
    return [
        {"id": "takeaways", "label": "Patterns found", "value": len(m2.get("takeaways") or [])},
        {"id": "figures", "label": "Charts drawn", "value": len(m2.get("figures") or [])},
        {"id": "hypotheses", "label": "Causes proposed", "value": len(m3.get("hypotheses") or [])},
    ]
