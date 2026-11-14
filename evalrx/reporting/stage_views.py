"""Per-stage views of a run: the stage rail and the bounded ``stage_detail``
contract (:class:`evalrx.contract.stage_detail.StageDetail`) the UI renders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evalrx.reporting.plain_labels import PLAIN_LABELS, _plain_label, _plain_mapping, _stat_label
from evalrx.reporting.stage_examples import (
    _attach_example_media,
    _m4_examples,
    _m5_examples,
    _operation_examples,
    _repair_operation_previews,
)
from evalrx.reporting.stage_figures import _m2_figures


def _stages(raw: Mapping[str, Any], m5_detail: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    definitions = [
        ("M1", "Check behavior", "See what the model does differently when it succeeds or fails", raw.get("m1"), "results"),
        ("M2", "Find patterns", "Look for behaviors that reliably appear alongside errors", raw.get("m2"), "stats"),
        ("M3", "Suggest causes", "Turn the strongest patterns into ideas that can be tested", raw.get("m3"), "hypotheses"),
        ("M4", "Test on new cases", "Check those ideas on evidence not used to create them", raw.get("m4"), "results"),
        ("M5", "Try repairs", "Compare possible fixes with the unchanged model", raw.get("m5_fix"), "selection"),
    ]
    result = []
    for code, title, purpose, payload, evidence_key in definitions:
        stage = dict(payload or {})
        evidence = stage.get(evidence_key) or []
        ran = bool(stage.get("ran")) if "ran" in stage else bool(evidence or stage.get("duration"))
        status = "completed" if ran else "not-run"
        if code == "M5" and m5_detail and m5_detail.get("skipped"):
            status = "skipped"
        if code == "M5" and ran:
            status = "improved" if stage.get("fixed") else "no-improvement"
        result.append({
            "id": code.lower(), "code": code, "title": title, "purpose": purpose,
            "status": status, "evidence_count": len(evidence) if isinstance(evidence, list) else 1,
        })
    return result


def _stage_detail(
    raw: Mapping[str, Any], root: Path, cases: list[dict[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compile the mature stage-specific views into a bounded web contract.

    The shape is :class:`evalrx.contract.stage_detail.StageDetail`. It preserves
    the old workbench's load-bearing distinctions: M2 is descriptive, M3 is
    proposal-only, M4 owns held-out verdicts, and M5 compares every
    intervention against the unchanged baseline.
    """
    logs_dir = Path(raw.get("logs_dir") or root)
    m2 = dict(raw.get("m2") or {})
    explore = dict(m2.get("explore") or {})
    # M3 starts from M2's evidence, so both stages read the same figures and stats.
    figures = _m2_figures(raw, m2, explore, logs_dir, root)
    stats = [_display_stat(item) for item in m2.get("stats") or [] if isinstance(item, Mapping)]
    return {
        "m1": _m1_detail(dict(raw.get("m1") or {}), cases, events),
        "m2": _m2_detail(m2, explore, figures, stats, (raw.get("agents") or {}).get("explore")),
        "m3": _m3_detail(raw, explore, figures, stats),
        "m4": _m4_detail(dict(raw.get("m4") or {}), cases, events),
        "m5": _m5_detail(raw, cases, events),
    }


def _m1_detail(
    m1: Mapping[str, Any], cases: list[dict[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for result in m1.get("results") or []:
        if not isinstance(result, dict):
            continue
        rows = list(result.get("per_case") or [])
        probes.append({
            "id": str(result.get("name") or "probe"),
            # The plain question when the label table has one; otherwise the
            # glossary's name ("Self-Repair on Re-ask"), never a Title-Cased id.
            "title": (PLAIN_LABELS.get(str(result.get("name") or ""))
                      or str(result.get("display_name") or "")
                      or _plain_label(result.get("name") or "Probe")),
            # The glossary's own name for the check ("Self-Repair on Re-ask"),
            # as opposed to `title`, which is the plain question when one exists.
            "display_name": str(result.get("display_name") or ""),
            "raw_name": str(result.get("name") or ""),
            "question": str(result.get("question") or ""),
            "description": str(result.get("description") or ""),
            # None when the analyzer reported no case count at all (a batch-level
            # check such as self_consistency): "unknown" is not "zero".
            "n_cases": result.get("n") if result.get("n") is not None else (len(rows) or None),
            "metrics": list(result.get("headline") or []),
            "finding_summary": _plain_mapping(_compact_mapping(result.get("findings"))),
            "raw_finding_summary": _compact_mapping(result.get("findings")),
            "sample_rows": rows[:12],
            "n_sample_rows": len(rows),
        })
    # Examples are a first-class, bounded evidence type.  They are deliberately
    # derived from recorded case I/O and per-case measurements, never written
    # by the report agent.  A single case makes a measurement legible; it does
    # not establish the stage conclusion (the UI states that boundary too).
    logged_examples = [
        item for event in events if event.get("event") == "probe"
        for item in (event.get("examples") or []) if isinstance(item, Mapping)
    ]
    examples = [dict(item) for item in logged_examples[:2]]
    calls, n_calls = _m1_calls(events)
    return {
        "duration": m1.get("duration"), "probes": probes,
        "n_probes": len(probes), "n_measured": max((int(p.get("n_cases") or 0) for p in probes), default=0),
        "examples": _attach_example_media(examples, cases),
        # A trajectory is the only source allowed to say that an agent actually
        # used an intermediate operation (zoom, crop, search, ...).  Candidate
        # specs are kept separately (M5's operation previews) because a
        # proposed operation is not an executed one.
        "operations": _operation_examples(cases),
        # What each probe actually asked the model and what came back,
        # grouped by analyzer: the audit trail behind the numbers above.
        "calls": calls, "n_calls": n_calls,
    }


def _m2_detail(
    m2: Mapping[str, Any], explore: Mapping[str, Any],
    figures: list[dict[str, Any]], stats: list[dict[str, Any]],
    agent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    takeaways = [
        {key: item.get(key) for key in (
            "plain_title", "title", "analysis", "caveat", "chart_names", "table_names",
        )}
        for item in explore.get("takeaways") or [] if isinstance(item, dict)
    ]
    return {
        "mode": "descriptive", "conclusion": m2.get("conclusion") or "",
        "narrative": m2.get("narrative") or "", "stats": stats,
        "takeaways": takeaways, "figures": figures,
        "observations": list(explore.get("observations") or []),
        "caveats": list(explore.get("caveats") or []),
        "candidate_signals": list(explore.get("candidate_signals") or []),
        "recommended_tests": list(explore.get("recommended_confirmatory_tests") or []),
        "agent": _explore_agent(agent),
    }


#: The explore agent's CLI trajectory runs to megabytes on a long session;
#: the report carries its head, the run directory the whole file.
_TRANSCRIPT_CAP = 200_000


def _explore_agent(agent: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The analysis the explore agent wrote and ran, or None if it left nothing."""
    if not agent:
        return None
    transcript = str(agent.get("raw_streams") or agent.get("raw_output") or "")
    return {
        "code": agent.get("code"), "stdout": agent.get("stdout"), "stderr": agent.get("stderr"),
        "transcript": transcript[:_TRANSCRIPT_CAP] or None,
        "transcript_truncated": len(transcript) > _TRANSCRIPT_CAP,
        "audit": agent.get("audit"),
    }


def _m3_detail(
    raw: Mapping[str, Any], explore: Mapping[str, Any],
    figures: list[dict[str, Any]], stats: list[dict[str, Any]],
) -> dict[str, Any]:
    m3 = dict(raw.get("m3") or {})
    hypotheses = list(m3.get("hypotheses") or explore.get("hypotheses") or [])
    return {
        "mode": "proposal", "hypotheses": hypotheses,
        "candidate_signals": list(explore.get("candidate_signals") or []),
        "recommended_tests": list(explore.get("recommended_confirmatory_tests") or []),
        "evidence_figures": figures[:3], "evidence_stats": stats[:6],
    }


def _m4_detail(
    m4: Mapping[str, Any], cases: list[dict[str, Any]], events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results = list(m4.get("results") or [])
    event = dict(m4.get("event") or {})
    logged: Mapping[str, Any] = next((
        item for item in reversed(events)
        if item.get("event") == "surgery" and item.get("module") == "m4"
    ), {})
    return {
        "ran": bool(m4.get("ran") or results), "mode": "confirmatory",
        "results": results, "event": event,
        "examples": _attach_example_media(_m4_examples(results, {**event, **logged}, cases), cases),
    }


def _m5_detail(
    raw: Mapping[str, Any], cases: list[dict[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    m5 = dict(raw.get("m5_fix") or {})
    # extract_run_data may see an older sibling log when an example contains
    # multiple attempts.  The event stream is trace-filtered and is the source
    # of truth for the run currently being rendered.
    fix_event: Mapping[str, Any] = next((event for event in reversed(events) if event.get("event") == "fix"), {})
    skipped_event: Mapping[str, Any] = next((
        event for event in reversed(events)
        if event.get("event") == "stage_skipped" and str(event.get("stage")) == "M5"
    ), {})
    if fix_event:
        attempted = [item for item in fix_event.get("attempted") or [] if isinstance(item, Mapping)]
        # The fix event records `best` as the whole winning candidate; the
        # attempt of the same name carries what it scored on the confirm pool.
        winner = found if isinstance(found := fix_event.get("best"), Mapping) else {}
        best_name = str(winner.get("name") or "")
        best = next((item for item in attempted if item.get("name") == best_name), {}) if best_name else {}
        m5.update({
            "ran": True, "fixed": bool(fix_event.get("fixed")), "selection": attempted,
            "best": best or dict(winner), "recommendation": fix_event.get("recommendation"),
        })
    if skipped_event:
        m5.update({
            "ran": False,
            "skipped": True,
            "skip_reason": skipped_event.get("reason_code") or "not_attempted",
            "skip_detail": skipped_event.get("detail") or "",
        })
    candidates = [_repair_candidate(item) for item in m5.get("selection") or [] if isinstance(item, dict)]
    confirmation = _repair_candidate(m5.get("best") or {})
    if confirmation.get("name"):
        existing = next((i for i, item in enumerate(candidates) if item.get("name") == confirmation["name"]), None)
        if existing is None:
            candidates.append(confirmation)
        else:
            candidates[existing] = {**candidates[existing], **confirmation}
    return {
        "ran": bool(m5.get("ran")), "fixed": bool(m5.get("fixed")),
        "skipped": bool(m5.get("skipped")),
        "skip_reason": str(m5.get("skip_reason") or ""),
        "skip_detail": str(m5.get("skip_detail") or ""),
        "candidates": candidates, "confirmation": confirmation,
        "best": _repair_candidate(m5.get("best") or {}),
        # The FixAgent's own verdict on the run — action + the sentence that
        # justifies it ("underpowered: only 4 failing cases…"). This is the
        # line the health card's promotion gate prints.
        "recommendation": (dict(m5["recommendation"])
                           if isinstance(m5.get("recommendation"), Mapping) else None),
        "examples": _m5_examples(candidates, cases, list((raw.get("m1") or {}).get("results") or [])),
        "operation_previews": _repair_operation_previews(candidates, cases),
        "surgeries": list((raw.get("m5_surgery") or {}).get("surgeries") or []),
        "prompt_template": str(m5.get("prompt_template") or ""),
    }


def _display_stat(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep statistical evidence machine-auditable but primary labels human."""
    config = found if isinstance(found := value.get("config"), Mapping) else {}
    signal = str(config.get("signal") or value.get("signal") or value.get("tool") or "")
    details = found if isinstance(found := value.get("details"), Mapping) else {}
    # A PAIRED tool compares two strategies and carries no `signal`, so `signal`
    # falls through to the tool name and every such row was labelled with the
    # same words ("Mcnemar Evalue"). Three identically-labelled bars is not a
    # chart. The pair it actually compared is right there in the config, and it
    # is the only thing that tells the rows apart.
    groups = [str(name) for name in (config.get("strategies") or []) if name]
    label = _stat_label(config, signal, value.get("tool"))
    return {
        "label": label, "raw_signal": signal, "groups": groups,
        "effect": value.get("effect"), "ci": value.get("ci"),
        "reject": bool(value.get("reject")), "underpowered": bool(value.get("underpowered")),
        "summary": value.get("summary") or "", "p_value": value.get("p_value"),
        "tool": value.get("tool"), "e_value": value.get("e_value"),
        "fail_rate_signal": details.get("fail_rate_signal"),
        "fail_rate_control": details.get("fail_rate_control"),
    }


def _compact_mapping(value: Any, *, limit: int = 18) -> dict[str, Any]:
    """Retain readable aggregate findings, excluding unbounded per-case maps."""
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"per_case", "per_case_values", "case_values", "samples", "caveat"}:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            compact[str(key)] = item
        elif isinstance(item, list) and len(item) <= 12:
            compact[str(key)] = item
        elif isinstance(item, Mapping) and len(item) <= 12:
            nested = {str(k): v for k, v in item.items() if isinstance(v, (str, int, float, bool)) or v is None}
            if nested:
                compact[str(key)] = nested
        if len(compact) >= limit:
            break
    return compact


def _repair_headline(value: Mapping[str, Any]) -> str:
    """The candidate's own sentence, or the host's for a candidate it authored.

    Looking `visual_grounding` up in the fix agent's own table is not a
    consumer guessing at a slug -- it is the package that named the candidate
    saying what it does. Runs recorded before `headline` existed become
    readable this way; a judge-invented name the host never heard of still
    resolves to "" and renders blank.
    """
    headline = " ".join(str(value.get("headline") or "").split())
    if headline:
        return headline[:300]
    try:
        from evalrx.eval_agent.stages.fix.descriptions import BUILTIN_DESCRIPTIONS
    except Exception:
        return ""
    return BUILTIN_DESCRIPTIONS.get(str(value.get("name") or ""), "")


def _repair_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    fixed_cases = list(value.get("fixed_cases") or [])
    broken_cases = list(value.get("broken_cases") or [])
    payload = found if isinstance(found := value.get("payload"), Mapping) else {}
    return {
        "tier": value.get("tier"), "name": value.get("name"), "kind": value.get("kind"),
        # `ref` and `headline` are what a reader sees; `name` is only the join
        # key. Both are carried here so the legacy report path shows the same
        # identifier and the same sentence as the contract-backed views.
        "ref": value.get("ref") or "", "headline": _repair_headline(value),
        "source": value.get("source"), "verdict": value.get("verdict"),
        "n_pairs": value.get("n_pairs"), "n_baseline_correct": value.get("n_baseline_correct"),
        "n_candidate_correct": value.get("n_candidate_correct"),
        "n_fixed": value.get("n_fixed", len(fixed_cases)), "n_broken": value.get("n_broken", len(broken_cases)),
        "effect": value.get("effect"), "e_value": value.get("e_value"),
        "e_threshold": value.get("e_threshold"),
        "coverage": value.get("coverage"), "reject": value.get("reject"), "fixed": value.get("fixed"),
        "n_model_independent": value.get("n_model_independent"),
        "n_unstable": value.get("n_unstable"),
        "summary": value.get("summary"), "payload": dict(payload),
        "fixed_cases": fixed_cases[:20], "broken_cases": broken_cases[:20],
    }


_CALL_TEXT_CAP = 3000


_CALLS_PER_ANALYZER = 120


def _call_text(value: Any) -> str:
    """The prompt or the reply as text, cut so 500 calls stay a sane payload."""
    if isinstance(value, Mapping):
        value = value.get("prompt") if "prompt" in value else json.dumps(value, ensure_ascii=False)
    text = "" if value is None else str(value)
    return text if len(text) <= _CALL_TEXT_CAP else text[:_CALL_TEXT_CAP] + f"… [+{len(text) - _CALL_TEXT_CAP} chars]"


def _m1_calls(events: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """M1's model calls grouped by analyzer, oldest first, capped per analyzer."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for event in events:
        if str(event.get("event") or "") != "model_call" or str(event.get("stage") or "") != "M1":
            continue
        total += 1
        analyzer = str(event.get("analyzer") or event.get("role") or "model")
        bucket = grouped.setdefault(analyzer, [])
        if len(bucket) >= _CALLS_PER_ANALYZER:
            continue
        batch = event.get("batch_case_ids")
        bucket.append({
            "seq": event.get("event_seq"), "cycle": event.get("cycle"),
            "method": event.get("method") or event.get("operation") or "",
            "case_id": event.get("case_id"),
            "batch_case_ids": [str(item) for item in batch] if isinstance(batch, list) else [],
            "duration_sec": event.get("duration_sec"),
            "prompt": _call_text(event.get("inputs")),
            "output": _call_text(event.get("output")),
            "error": str(event.get("error")) if event.get("error") else None,
        })
    return grouped, total
