"""The worked examples each stage view shows: probe outputs (M1), held-out cases
(M4), and what a repair changed on real cases (M5).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evalrx.reporting.plain_labels import _plain_label


def _attach_example_media(examples: list[dict[str, Any]], cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    case_by_id = {str(case.get("id")): case for case in cases}
    for example in examples:
        if example.get("probe_title"):
            example["probe_title"] = _plain_label(example["probe_title"])
        if example.get("media_ids"):
            continue
        case = case_by_id.get(str(example.get("case_id") or ""))
        if case:
            example["media_ids"] = list(case.get("media_ids") or [])
    return examples


def _m4_examples(
    results: list[Any], event: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The validation cases the M4 event recorded, named for the hypothesis tested."""
    supplied = event.get("validation_examples") if isinstance(event, Mapping) else None
    if not isinstance(supplied, list):
        return []
    first: Mapping[str, Any] = next((item for item in results if isinstance(item, Mapping)), {})
    evidence = found if isinstance(found := first.get("evidence"), Mapping) else {}
    return [{
        **dict(item),
        "hypothesis": item.get("hypothesis") or first.get("hypothesis") or first.get("statement"),
        "test": item.get("test") or first.get("test_name") or evidence.get("chosen_tool"),
        "status": item.get("status") or first.get("status"),
    } for item in supplied if isinstance(item, Mapping)][:2]


def _operation_examples(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project recorded agent tool trajectories into compact, case-linked UI cards.

    This intentionally ignores prompt text that *mentions* a tool.  An action
    appears only if a persisted trajectory step carries a structured
    ``tool_call``.  That preserves the distinction between an actual zoom and
    an LLM merely proposing one.
    """
    result: list[dict[str, Any]] = []
    for case in cases:
        trajectory = case.get("trajectory")
        if not isinstance(trajectory, Mapping):
            continue
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            continue
        calls: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, Mapping) or not isinstance(step.get("tool_call"), Mapping):
                continue
            call = step["tool_call"]
            name = str(call.get("name") or "tool")
            calls.append({
                "order": int(step.get("idx") or len(calls) + 1),
                "action": _plain_operation(name), "raw_action": name,
                "parameters": dict(call.get("args") or {}) if isinstance(call.get("args"), Mapping) else {},
                "thought": str(step.get("content") or "")[:500],
            })
        if not calls:
            continue
        result.append({
            "id": f"operations-{case.get('id')}", "case_id": str(case.get("id") or ""),
            "input": case.get("prompt") or "", "expected": case.get("expected"),
            "observed": case.get("observed"), "outcome": case.get("status") or "unknown",
            "media_ids": list(case.get("media_ids") or []), "steps": calls[:6],
            "plain_reading": "These are the actions recorded in this agent's own trajectory for this case.",
        })
        if len(result) >= 2:
            break
    return result


def _repair_operation_previews(
    candidates: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Show what an image-operation repair *would* apply, without claiming execution.

    Older runs preserve a repair spec but not a per-call visual artifact.  The
    report can still pair its declared image operations with the original input
    image for a representative case, as long as the card is explicitly marked
    as a candidate preview.  Actual execution is represented by
    :func:`_operation_examples` instead.
    """
    image_cases = [case for case in cases if case.get("media_ids")]
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = found if isinstance(found := candidate.get("payload"), Mapping) else {}
        ops = [op for op in payload.get("image_ops") or [] if isinstance(op, Mapping) and op.get("tool")]
        if not ops:
            continue
        preferred_ids = [str(item) for item in (candidate.get("fixed_cases") or candidate.get("broken_cases") or [])]
        case = next((item for item in image_cases if str(item.get("id")) in preferred_ids), None)
        case = case or next((item for item in image_cases if str(item.get("status")) == "fail"), None)
        # A legacy run can retain its generated baseline answers in a separate
        # artifact whose ids no longer line up with the saved image manifest.
        # In that case an arbitrary corpus image is still useful to explain the
        # *declared transform*, provided the UI calls it a preview and never a
        # validated before/after result.
        case = case or (image_cases[0] if image_cases else None)
        if case is None:
            continue
        result.append({
            "id": f"repair-operations-{candidate.get('name')}", "candidate": candidate.get("name"),
            "case_id": str(case.get("id") or ""), "input": case.get("prompt") or "",
            "expected": case.get("expected"), "observed": case.get("observed"),
            "media_ids": list(case.get("media_ids") or []),
            "executed": bool(candidate.get("n_pairs")),
            "operations": [{
                "action": _plain_operation(op.get("tool")), "raw_action": str(op.get("tool")),
                "parameters": dict(op.get("params") or {}) if isinstance(op.get("params"), Mapping) else {},
            } for op in ops],
        })
        if len(result) >= 2:
            break
    return result


def _plain_operation(value: Any) -> str:
    names = {
        "image_zoom_in": "Zoom in on part of the image", "zoom_center": "Zoom into the chart center",
        "crop": "Crop the visual evidence", "crop_case_bbox": "Crop the relevant chart region",
        "sharpen": "Sharpen chart details", "contrast": "Increase visual contrast",
        "equalize": "Balance the image contrast", "detect": "Locate an object or region",
    }
    raw = str(value or "")
    return names.get(raw, _plain_label(raw))


def _m5_examples(
    candidates: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], probe_results: list[Any]
) -> list[dict[str, Any]]:
    """Reconstruct a truthful before/after repair example — confirmed or not.

    A candidate that never cleared the significance bar is still what the
    search actually tried, and a reader debugging a failed repair needs to
    see a real case, not just "no candidate passed the repair gate". Prefer a
    `fixed` winner (statistically significant, net-positive); when none
    exists, fall back to the strongest attempt by effect size — the same
    fallback `M5Detail` already uses for its own "best repair" KPI — and tag
    the example `confirmed: False` so the UI can say plainly it is one
    candidate's attempt, not an accepted repair. When that candidate never
    flipped a case to correct either, fall back once more to a case it broke:
    still evidence, just of the failure mode rather than the fix.
    """
    winner = next((item for item in candidates if item.get("fixed")), None)
    confirmed = winner is not None
    if winner is None:
        winner = max(
            (item for item in candidates if isinstance(item.get("effect"), (int, float))),
            key=lambda item: item["effect"],
            default=(candidates[0] if candidates else None),
        )
    if winner is None:
        return []
    case_by_id = {str(case.get("id")): case for case in cases}
    baseline_by_id: dict[str, Any] = {}
    for probe in probe_results:
        if not isinstance(probe, Mapping):
            continue
        for row in probe.get("per_case") or []:
            if not isinstance(row, Mapping):
                continue
            case_id = str(row.get("sample_id") or row.get("case_id") or "")
            output = row.get("extracted_answer", row.get("modal_answer"))
            if case_id and output not in (None, ""):
                baseline_by_id.setdefault(case_id, output)

    def _build(case_id: str, kind: str) -> "dict[str, Any] | None":
        case = case_by_id.get(case_id)
        if not case:
            return None
        # `case["observed"]` is the Stage-0 baseline, written once per case id
        # (RunLoggerV2.log_cases dedupes by id) and never updated — it is NOT
        # this candidate's repaired answer, no matter which case_record last
        # touched it. The only place a candidate's own per-case output lives
        # is `case["repair"]`, which build_report_data attached from the fix
        # event's confirmation attempts (see _repair_outcomes). Confirm
        # it is THIS candidate's repair, not some other one that also touched
        # this case, before trusting it as "repaired".
        repair_hit = case.get("repair") if isinstance(case.get("repair"), Mapping) else None
        repaired = (
            repair_hit.get("output")
            if repair_hit and repair_hit.get("candidate") == winner.get("name")
            else None
        )
        if repaired in (None, ""):
            return None  # no recorded per-case output for this candidate — nothing truthful to show
        baseline = baseline_by_id.get(case_id)
        if baseline in (None, ""):
            baseline = case.get("observed")  # the genuine Stage-0 answer, as a last resort
        if confirmed:
            reading = ("This case was counted as fixed in the paired repair evaluation. "
                       "The repair decision still depends on all tested cases and its "
                       "regression checks.")
        elif kind == "fixed":
            reading = ("This candidate flipped this case from wrong to right, but the "
                       "repair overall did not clear the significance bar against luck — "
                       "read it as one attempt worth inspecting, not a confirmed fix.")
        else:
            reading = ("This candidate did not clear the significance bar, and did not "
                       "flip any case to correct either — this is a case it broke instead, "
                       "shown so the failure mode is inspectable.")
        return {
            "id": f"m5-{winner.get('name', 'repair')}-{case_id}", "case_id": case_id,
            "kind": kind, "confirmed": confirmed,
            "repair_name": winner.get("name") or "Recorded repair", "input": case.get("prompt") or "",
            "expected": case.get("expected"), "baseline_output": baseline,
            "repaired_output": repaired, "media_ids": list(case.get("media_ids") or []),
            "plain_reading": reading,
            "baseline_available": baseline not in (None, ""),
        }

    for case_id in [str(item) for item in winner.get("fixed_cases") or []]:
        example = _build(case_id, "fixed")
        if example:
            return [example]
    for case_id in [str(item) for item in winner.get("broken_cases") or []]:
        example = _build(case_id, "broken")
        if example:
            return [example]
    return []
