"""The cases of a run, one row each, for the Case Studio.

A row joins a case's benchmark-manifest entry with what the run did to it: the
fix candidate's own output and status when one was confirmed, otherwise the
baseline output and the fixed / broken lists of the best fix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_manifest(example_dir: Path, logs_dir: Path) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    """Find and load benchmark manifest (.jsonl / .json)."""
    candidates = []
    if example_dir:
        candidates.extend(sorted(example_dir.glob("data/*.jsonl")))
        candidates.extend(sorted(example_dir.glob("data/*.json")))
        candidates.extend(sorted(example_dir.glob("*.jsonl")))
    candidates.extend(sorted(logs_dir.glob("data/*.jsonl")))
    candidates.extend(sorted(logs_dir.parent.glob("data/*.jsonl")))

    for p in candidates:
        if p.name.endswith(".result.json") or p.name.endswith("results.json"):
            continue
        try:
            rows = {}
            if p.suffix == ".jsonl":
                for line in p.open(encoding="utf-8"):
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        cid = str(obj.get("id") or obj.get("case_id") or "")
                        if cid:
                            rows[cid] = obj
            elif p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                recs = data if isinstance(data, list) else (data.get("records") or data.get("cases") or [])
                for obj in recs:
                    if isinstance(obj, dict):
                        cid = str(obj.get("id") or obj.get("case_id") or "")
                        if cid:
                            rows[cid] = obj
            if rows:
                return p, rows
        except Exception:
            continue
    return None, {}


def _case_row(cid: str, entry: dict[str, Any], status: Any, output: Any, flags: list[str]) -> dict[str, Any]:
    """One case row from its manifest ``entry``, whatever the manifest calls its fields."""
    image = entry.get("image") or entry.get("image_path") or entry.get("image_file") or ""
    if isinstance(image, (list, tuple)):
        image = image[0] if image else ""
    metadata = entry.get("metadata") or {}
    return {
        "id": cid,
        "status": status,
        "output": str(output),
        "instruction": (
            entry.get("instruction") or entry.get("question") or entry.get("prompt") or entry.get("query") or ""
        ),
        "choices": entry.get("choices") or entry.get("options") or entry.get("candidates") or [],
        "expected": str(entry.get("expected") or entry.get("answer") or entry.get("label") or entry.get("gt") or ""),
        "audio_path": (
            entry.get("audio_path") or entry.get("audio") or entry.get("wav_path") or entry.get("audio_file") or ""
        ),
        "image_path": image,
        "duration": float(entry.get("duration_sec") or 0.0),
        "task": metadata.get("mmau_task") or metadata.get("category") or "",
        "probe_flags": flags,
    }


def _rows_from_outputs(
    outputs: Path, manifest: dict[str, dict[str, Any]], probe_flags: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Rows for the cases an outputs.jsonl names; a line that cannot be read is skipped."""
    rows = []
    for line in outputs.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            cid = str(record.get("case_id") or record.get("id") or "")
            if not cid:
                continue
            rows.append(_case_row(
                cid, manifest.get(cid) or {}, record.get("status", "unchanged"),
                record.get("output", ""), probe_flags.get(cid, []),
            ))
        except Exception:
            pass
    return rows


def _rows_from_manifest(
    logs_dir: Path, manifest: dict[str, dict[str, Any]], confirmed_fix: dict[str, Any],
    probe_flags: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Every manifest case, with its baseline output and the confirmed fix's verdict on it."""
    fixed = set(confirmed_fix.get("fixed_cases") or [])
    broken = set(confirmed_fix.get("broken_cases") or [])
    baseline: dict[str, dict[str, Any]] = {}
    for path in [logs_dir / "artifacts" / "baseline.json", *logs_dir.glob("**/baseline.json")]:
        if path.exists():
            try:
                for case in json.loads(path.read_text()).get("cases") or []:
                    if isinstance(case, dict) and case.get("id"):
                        baseline[str(case["id"])] = case
            except Exception:
                pass
    return [
        _case_row(
            cid, entry, "fixed" if cid in fixed else "broken" if cid in broken else "unchanged",
            baseline.get(cid, {}).get("output", ""), probe_flags.get(cid, []),
        )
        for cid, entry in manifest.items()
    ]


_STATUS_ORDER = {"fixed": 0, "broken": 1, "unchanged": 2, "untested": 3}


def case_rows(
    logs_dir: Path, fix_dir: Path | None, manifest: dict[str, dict[str, Any]],
    confirmed_fix: dict[str, Any], probe_flags: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """The run's cases, fixed ones first.

    The confirmed fix's outputs.jsonl is preferred, then any other one under
    the run; with none (or none readable), the manifest is scored against the
    baseline instead.
    """
    outputs = []
    if fix_dir and (fix_dir / "outputs.jsonl").exists():
        outputs.append(fix_dir / "outputs.jsonl")
    outputs.extend(sorted(logs_dir.glob("**/outputs.jsonl")))
    if logs_dir.parent != logs_dir:
        outputs.extend(sorted(logs_dir.parent.glob("**/outputs.jsonl")))

    rows = _rows_from_outputs(outputs[0], manifest, probe_flags) if outputs else []
    if not rows and manifest:
        rows = _rows_from_manifest(logs_dir, manifest, confirmed_fix, probe_flags)
    rows.sort(key=lambda row: (_STATUS_ORDER.get(row["status"], 9), row["id"]))
    return rows
