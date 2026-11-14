"""What a run recorded, read for the report: its typed contract payloads, its
cases and their media, M5's per-case repair outcomes, and the partitions of
the frozen batch each case sat in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _contract_payloads(root: Path) -> dict[str, Any]:
    """The validated per-stage payloads a run emitted, keyed by span id.

    Passed through verbatim. ``stage_detail`` above is a hand-built view whose
    every field access is a defensive ``.get(x) or y`` — it has to be, because
    it reads whatever the run log happened to contain. These payloads were
    validated against the wire models on the way out, so a frontend can decode
    them with the generated TypeScript instead of re-deriving the shape.

    Absent for a run produced before contract emission existed, or one where the
    ``contract`` extra was not installed. ``{}`` says "this run emitted none",
    which a reader must not confuse with "this run had no stages".
    """
    out: dict[str, Any] = {}
    directory = root / "contract"
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = _load_json(path)
        if payload is None:
            continue
        # "c0.m1.json" -> "c0.m1"; ".invalid.json" keeps its suffix so a reader
        # can see that a stage failed validation rather than silently missing it.
        out[path.name[: -len(".json")]] = payload
    return out


def _repair_outcomes(events: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """What each repair candidate answered, per case.

    ``case_id -> {candidate, tier, status, output}``. The counts in M5 say twelve
    cases were repaired and one broken; this is what lets a reader open those
    thirteen and see what actually changed. Without it a repair is a number.

    Read from the fix event's confirmation attempts, which carry each
    candidate's answer per case and the cases it fixed and broke. The
    selection sweep chose the candidate, and its per-case results are not
    evidence about it.
    """
    out: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "fix":
            continue
        for attempt in event.get("attempted") or []:
            outputs = attempt.get("outputs") if isinstance(attempt, Mapping) else None
            if not isinstance(outputs, Mapping):
                continue
            fixed = {str(cid) for cid in attempt.get("fixed_cases") or []}
            broken = {str(cid) for cid in attempt.get("broken_cases") or []}
            for cid, output in outputs.items():
                cid = str(cid)
                out[cid] = {
                    "candidate": attempt.get("name"),
                    "tier": attempt.get("tier"),
                    "status": "fixed" if cid in fixed else "broken" if cid in broken else "unchanged",
                    "output": output,
                }
    return out


def _contract_model_name(contract: Mapping[str, Any]) -> str:
    """The model's name from M1's payload, or "" for a run that emitted none."""
    for key in sorted(contract):
        if not key.endswith(".m1"):
            continue
        ref = (contract[key] or {}).get("model") or {}
        name = str(ref.get("name") or "").strip()
        if name:
            return name
    return ""


def _contract_stats(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """M2's statistical rows from the newest cycle that has any."""
    for key in sorted(contract, reverse=True):
        if not key.endswith(".m2"):
            continue
        rows = (contract[key] or {}).get("stats_results")
        if isinstance(rows, list) and rows:
            return rows
    return []


def _media_index(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per distinct media file, preferring the copy inside the run.

    ``inputs.<slot>`` is where the producer read the file from — an absolute path
    on that host, and meaningless on any other. The run logger already copied the
    file into the run and recorded that relative path, so a run unzipped
    elsewhere still has its media; index by that, and by the original path only
    for a file the logger could not copy.
    """
    media: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        inputs = found if isinstance(found := case.get("inputs"), dict) else {}
        # Slots and copies are recorded in the same order, so they zip up.
        local = [p for p in (case.get("_media_paths") or []) if isinstance(p, str)]
        i = 0
        for kind in ("audio", "image", "video"):
            source = inputs.get(kind)
            if not isinstance(source, str) or not source or source.startswith("<"):
                continue
            path = local[i] if i < len(local) else source
            i += 1
            if path in seen:
                continue
            seen.add(path)
            media.append({"id": f"media-{len(media) + 1}", "kind": kind,
                          "path": path, "source": source})
    return media


def _normalise_case(case: Mapping[str, Any], media: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = found if isinstance(found := case.get("inputs"), dict) else {}
    # Match on either key: the index prefers the in-run copy, and a file the
    # logger could not copy still resolves through the producer's path.
    media_by_path: dict[str, str] = {}
    for item in media:
        media_by_path[item["path"]] = item["id"]
        if item.get("source"):
            media_by_path[item["source"]] = item["id"]
    media_ids = []
    for path in (case.get("_media_paths") or []):
        if isinstance(path, str) and path in media_by_path and media_by_path[path] not in media_ids:
            media_ids.append(media_by_path[path])
    for kind in ("audio", "image", "video"):
        value = inputs.get(kind)
        if isinstance(value, str) and value in media_by_path \
                and media_by_path[value] not in media_ids:
            media_ids.append(media_by_path[value])
    return {
        "id": str(case.get("id") or f"case-{id(case)}"),
        "status": str(case.get("label") or "unknown").lower(),
        "prompt": inputs.get("prompt") or "",
        "expected": case.get("expected"),
        "observed": case.get("observed"),
        "choices": case.get("choices") or [],
        "tags": case.get("tags") or [],
        "task": (case.get("metadata") or {}).get("category") or "",
        "media_ids": media_ids,
        "trajectory": case.get("trajectory"),
        # explore / confirm / test, as the loop logged it; None when the run
        # held nothing back.
        "split": case.get("_split") if isinstance(case.get("_split"), str) else None,
    }


# What each partition was used for, in the words the loop figure draws them
# with: D_E on top, D_H, D_C at the bottom. The code is the subscript.
_EXPLORE_ROLE = ("M1-M3 mined patterns and hypotheses here. Anything measured on "
                 "these cases is a lead, not a verdict.")


_HELDOUT_ROLE = ("Withheld from M1-M3. M4 adjudicated the hypotheses here and M5 "
                 "developed its repair candidates on the same cases.")


_CONFIRM_ROLE = ("Withheld from every adaptive decision. The frozen repair was "
                 "scored exactly once here.")


_TWO_WAY_ROLE = ("Withheld from M1-M3. M4 adjudicated the hypotheses here and the "
                 "repair was scored on these same cases: one withheld pool "
                 "serving as both the held-out and the confirm set.")


def _assign_partitions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise the partitions the run put its cases in.

    The rows come back in drawing order, each with the subscript the loop
    figure uses (E, H, C), its size and what the run used it for. With a
    two-way split the withheld pool is both the held-out set M4 adjudicates on
    and the confirm set M5 scores on, so it is one row labelled as both.

    Returns ``[]`` when there is no split to show: the run held nothing back.
    """
    present = {str(case.get("split")) for case in cases if case.get("split")}
    if not present:
        return []
    three_way = "test" in present

    def row(split: str, code: str, label: str, role: str) -> dict[str, Any]:
        return {"split": split, "code": code, "label": label,
                "n": sum(1 for case in cases if case.get("split") == split), "role": role}

    rows = [row("explore", "E", "Explore", _EXPLORE_ROLE)]
    if three_way:
        rows.append(row("confirm", "H", "Held-out", _HELDOUT_ROLE))
        rows.append(row("test", "C", "Confirm", _CONFIRM_ROLE))
    elif "confirm" in present:
        rows.append(row("confirm", "H/C", "Held-out / Confirm", _TWO_WAY_ROLE))
    n_unknown = sum(1 for case in cases if not case.get("split"))
    if n_unknown:
        rows.append({"split": "", "code": "?", "label": "Unrecorded", "n": n_unknown,
                     "role": "The run did not record which partition these came from."})
    return rows
