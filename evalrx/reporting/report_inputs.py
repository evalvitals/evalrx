"""What a run recorded, read for the report: its typed contract payloads, its
cases and their media, M5's per-case repair outcomes, and the partitions of
the frozen batch each case sat in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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
    for key, payload in out.items():
        if key.endswith("m5_fix"):
            _backfill_repair_identity(payload)
    return out


def _backfill_repair_identity(payload: Any) -> None:
    """Give a pre-`ref` run the same numbering a current run would emit.

    Runs recorded before these fields existed carry neither, and a consumer
    left to number rows itself gives the frozen candidate one number in the
    sweep and another on its card. The assignment here is the emitter's, done
    the emitter's way -- selection order first, one number per distinct name --
    so an old report and a new one read alike.

    It only ever FILLS BLANKS. What a producer actually wrote is never
    overwritten, and a `headline` is only ever taken from the fix agent's table
    of its own candidates: a judge-invented name nobody described stays blank,
    because inventing a sentence for it here would be this file guessing.
    """
    if not isinstance(payload, dict):
        return
    try:
        from evalrx.eval_agent.stages.fix.descriptions import BUILTIN_DESCRIPTIONS
    except Exception:
        BUILTIN_DESCRIPTIONS = {}
    refs: dict[str, str] = {}
    for group in ("selection", "attempted"):
        for row in payload.get(group) or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            ref = refs.get(name)
            if ref is None:
                ref = f"R{len(refs) + 1}"
                refs[name] = ref
            row.setdefault("ref", "")
            if not row["ref"]:
                row["ref"] = ref
            row.setdefault("headline", "")
            if not row["headline"]:
                row["headline"] = BUILTIN_DESCRIPTIONS.get(name, "")


def _repair_outcomes(root: Path, contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """What each repair candidate answered, per case.

    ``case_id -> {candidate, tier, status, output}``. The counts in M5 say twelve
    cases were repaired and one broken; this is what lets a reader open those
    thirteen and see what actually changed. Without it a repair is a number.

    Read from each attempt's ``trial_root/outputs.jsonl`` rather than the wire,
    because it is one row per case per candidate — the contract points at the
    file instead of inlining it, and this is the reader following the pointer.
    Confirmation attempts only: the selection sweep chose the candidate and its
    per-case results are not evidence about it.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(contract):
        if not key.endswith("m5_fix"):
            continue
        for attempt in (contract[key] or {}).get("attempted") or []:
            trial = str(attempt.get("trial_root") or "")
            if not trial:
                continue
            path = (root / trial / "outputs.jsonl")
            if not path.is_file():
                path = Path(trial) / "outputs.jsonl"   # a run read where it was produced
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                cid = str(row.get("case_id") or "")
                if not cid:
                    continue
                out[cid] = {
                    "candidate": attempt.get("name"),
                    "tier": attempt.get("tier"),
                    "status": row.get("status"),
                    "output": row.get("output"),
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
    file to ``artifacts/case_media/`` and recorded that relative path, so a run
    unzipped elsewhere still has its media; index by that when it exists and keep
    the original only as the fallback for runs that predate the copy.
    """
    media: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        inputs = found if isinstance(found := case.get("inputs"), dict) else case
        # Slots and copies are recorded in the same order, so they zip up.
        local = [p for p in (case.get("_media_paths") or []) if isinstance(p, str)]
        i = 0
        for kind in ("audio", "image", "video"):
            source = inputs.get(kind) or inputs.get(f"{kind}_path")
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
    prompt = inputs.get("prompt") or case.get("instruction") or case.get("prompt") or ""
    # Match on either key: the index prefers the in-run copy, and a run without
    # copies still resolves through the producer's original path.
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
        value = inputs.get(kind) or case.get(f"{kind}_path") or case.get(kind)
        if isinstance(value, str) and value in media_by_path \
                and media_by_path[value] not in media_ids:
            media_ids.append(media_by_path[value])
    return {
        "id": str(case.get("id") or case.get("case_id") or f"case-{id(case)}"),
        "status": str(case.get("status") or case.get("label") or "unknown").lower(),
        "prompt": prompt,
        "expected": case.get("expected"),
        "observed": case.get("observed", case.get("output")),
        "choices": case.get("choices") or [],
        "tags": case.get("tags") or case.get("probe_flags") or [],
        "task": case.get("task") or (case.get("metadata") or {}).get("category") or "",
        "media_ids": media_ids,
        "trajectory": case.get("trajectory"),
        # explore / confirm / test, as the loop logged it; None until
        # _assign_partitions has had a chance to infer it.
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


def _assign_partitions(
    cases: list[dict[str, Any]],
    *,
    logged_ids: "Sequence[str]",
    n_explore: Any,
    final_ids: "set[str]",
) -> list[dict[str, Any]]:
    """Put every case in its partition and summarise the partitions.

    ``case["split"]`` is filled in place (``explore`` / ``confirm`` / ``test``)
    and the summary rows come back in drawing order, each with the subscript
    the loop figure uses (E, H, C), its size and what the run used it for.

    A run that tagged its case records is read as recorded. A run from before
    the tag existed is inferred from what its log still says: ``run_start``'s
    ``n_cases`` is the explore partition (the loop narrows ``data`` to it before
    recording the start), and the loop logs explore first, then confirm, then
    test -- so the first ``n_explore`` logged ids are explore and the rest were
    withheld. Whether the withheld pool was further divided (train/val/test
    mode) is recovered from M5's per-case scoring: the cases it scored the
    frozen repair on are the test partition when they are a strict subset of
    what was withheld, and the whole pool otherwise. Cases the report merged
    in from a manifest rather than the log stay untagged -- their order says
    nothing.

    Returns ``[]`` when there is no split to show: no case carries one and none
    can be inferred (a run without a confirm split, or a legacy manifest).
    """
    by_id = {str(case.get("id")): case for case in cases}
    inferred = False
    if not any(case.get("split") for case in cases):
        ids = [cid for cid in logged_ids if cid and cid in by_id]
        try:
            n_head = int(n_explore or 0)
        except (TypeError, ValueError):
            n_head = 0
        if not ids or n_head <= 0 or n_head >= len(ids):
            return []
        withheld = ids[n_head:]
        final = {cid for cid in final_ids if cid in set(withheld)}
        three_way = bool(final) and len(final) < len(withheld)
        for cid in ids[:n_head]:
            by_id[cid]["split"] = "explore"
        for cid in withheld:
            by_id[cid]["split"] = "test" if three_way and cid in final else "confirm"
        inferred = True

    present = {str(case.get("split")) for case in cases if case.get("split")}
    if not present:
        return []
    three_way = "test" in present

    def row(split: str, code: str, label: str, role: str) -> dict[str, Any]:
        return {"split": split, "code": code, "label": label,
                "n": sum(1 for case in cases if case.get("split") == split),
                "role": role, "inferred": inferred}

    rows = [row("explore", "E", "Explore", _EXPLORE_ROLE)]
    if three_way:
        rows.append(row("confirm", "H", "Held-out", _HELDOUT_ROLE))
        rows.append(row("test", "C", "Confirm", _CONFIRM_ROLE))
    elif "confirm" in present:
        rows.append(row("confirm", "H/C", "Held-out / Confirm", _TWO_WAY_ROLE))
    n_unknown = sum(1 for case in cases if not case.get("split"))
    if n_unknown:
        rows.append({"split": "", "code": "?", "label": "Unrecorded", "n": n_unknown,
                     "role": "The run did not record which partition these came from.",
                     "inferred": False})
    return rows


def _merge_recorded_case_evidence(cases: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Supplement legacy manifests with the run's recorded baseline I/O.

    Older benchmark exports often keep prompts/media in a manifest while
    baseline answers live separately in ``report/discovery_cases.json``.  This
    is a generic run artifact, not a task-specific fallback.  New JSONL case
    records already contain both and therefore simply win when populated.
    """
    snapshots: list[Mapping[str, Any]] = []
    for base in (root, root.parent):
        value = _load_json(base / "report" / "discovery_cases.json")
        if isinstance(value, list):
            snapshots = [item for item in value if isinstance(item, Mapping)]
            if snapshots:
                break
    if not snapshots:
        return cases
    existing = {str(case.get("id")): case for case in cases}
    for raw_case in snapshots:
        case_id = str(raw_case.get("id") or raw_case.get("case_id") or "")
        if not case_id:
            continue
        update = _normalise_case(raw_case, [])
        target = existing.get(case_id)
        if target is None:
            cases.append(update)
            existing[case_id] = update
            continue
        for key in ("prompt", "expected", "observed"):
            if not target.get(key) and update.get(key):
                target[key] = update[key]
        if target.get("status") in {"", "unknown", "unchanged"} and update.get("status"):
            target["status"] = update["status"]
    return cases
