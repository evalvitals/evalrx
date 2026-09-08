"""Read tidy RunLoggerV2 documents through the existing event-view contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STAGES = ("M1", "M2", "M3", "M4", "M5")


def resolve_v2_root(root: str | Path) -> Path | None:
    """Return the directory containing V2 ``run.json`` and stage folders."""
    root = Path(root).resolve()
    candidates = (root, root / "logs")
    for candidate in candidates:
        if (candidate / "run.json").is_file() and any(
            (candidate / stage / "log.json").is_file() for stage in _STAGES
        ):
            return candidate
    return None


def read_v2_events(root: str | Path) -> list[dict[str, Any]]:
    """Flatten V2 bucketed documents into the legacy read-only event view.

    The V2 files remain authoritative and are never rewritten by this adapter.
    It exists so current report/UI code can migrate independently from the
    storage layout.
    """
    v2_root = resolve_v2_root(root)
    if v2_root is None:
        return []
    try:
        run = json.loads((v2_root / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    trace_id = str((run.get("run_start") or {}).get("trace_id") or run.get("trace_id") or "")
    events: list[dict[str, Any]] = []

    def append(event: str, payload: Any, *, stage: str = "RUN") -> None:
        if not isinstance(payload, dict):
            return
        record = dict(payload)
        record.setdefault("event", event)
        record.setdefault("stage", stage)
        # Older V2 bundles encode this only in the folder name. Existing
        # case-study/report consumers still use the V1 module discriminator.
        if event == "surgery":
            record.setdefault("module", stage.lower())
        elif event == "fix":
            record.setdefault("module", "fix")
        if trace_id:
            record.setdefault("trace_id", trace_id)
        events.append(record)

    append("run_start", run.get("run_start"))
    for row in run.get("cases") or []:
        append("case_record", row)
    for key in (
        "report_published", "diagnose_reports", "loop_end",
        "agent_decisions", "agent_tool_calls", "unrouted",
    ):
        event_name = {
            "diagnose_reports": "diagnose_report",
            "agent_decisions": "agent_decision",
            "agent_tool_calls": "agent_tool",
            "unrouted": "unrouted",
        }.get(key, key)
        for row in run.get(key) or []:
            append(event_name, row)

    for stage in _STAGES:
        path = v2_root / stage / "log.json"
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        for event, rows in doc.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                append("model_call" if event == "model_calls" else event, row, stage=stage)

    # New bundles persist a global sequence at write time, even when calls
    # finish concurrently or the wall clock moves backwards. Legacy bundles
    # retain timestamp ordering; never replace a persisted event identity.
    if all(isinstance(event.get("event_seq"), int) for event in events):
        events.sort(key=lambda event: event["event_seq"])
    else:
        events.sort(key=lambda event: str(event.get("ts") or ""))
        seq = max((event.get("event_seq", 0) for event in events), default=0)
        for event in events:
            if "event_seq" not in event:
                seq += 1
                event["event_seq"] = seq
    return events


# Inverse of the RUN-stage bucket mapping `read_v2_events` flattens — event
# name -> run.json key. These names are always RUN-level regardless of any
# "stage" tag on the recovered event.
_RUN_EVENT_TO_KEY = {
    "case_record": "cases",
    "diagnose_report": "diagnose_reports",
    "agent_decision": "agent_decisions",
    "agent_tool": "agent_tool_calls",
    "report_published": "report_published",
    "loop_end": "loop_end",
}

# A source that recovers events out-of-band (e.g. Langfuse observations
# reconstructed from a durable-outbox delivery, which predates per-event
# "stage" tagging) may hand back events with no "stage" field at all. Route
# by the event's own name to its canonical stage rather than dumping it into
# run.json's "unrouted" bucket, where `resolve_v2_root` would never find it
# (it requires at least one real M<n>/log.json to recognize a V2 root).
_NAME_TO_STAGE = {
    "probe": "M1",
    "analysis": "M2", "explore": "M2",
    "diagnosis": "M3",
    "experiment": "M5",  # module=... on the row disambiguates M4 vs M5 below
    "fix": "M5",
}


def _fallback_stage(name: "str | None", row: dict[str, Any]) -> "str | None":
    """Best-effort stage for an event with no "stage" tag of its own."""
    if name == "surgery" or name == "tool_codegen" or name == "experiment":
        module = str(row.get("module") or "").lower()
        if module.startswith("m") and module[1:].isdigit():
            return f"M{module[1:]}"
        return "M5" if name == "experiment" else "M4"
    return _NAME_TO_STAGE.get(name or "")


def write_v2_bundle(root: str | Path, events: list[dict[str, Any]]) -> None:
    """Reconstruct ``run.json`` + ``M<n>/log.json`` from a flat event list.

    The exact inverse of :func:`read_v2_events`'s bucketing, for callers that
    recover events from an out-of-band source (e.g. Langfuse) and need to
    write a V2-shaped cache directory back for the report/dashboard's
    normal V2-only read path to find.
    """
    root = Path(root)
    run_doc: dict[str, Any] = {}
    stage_docs: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for event in events:
        row = {k: v for k, v in event.items() if k not in ("event", "stage")}
        stage = event.get("stage")
        name = event.get("event")
        if name == "run_start":
            run_doc["run_start"] = row
            continue
        if name in _RUN_EVENT_TO_KEY:
            run_doc.setdefault(_RUN_EVENT_TO_KEY[name], []).append(row)
            continue
        if stage not in _STAGES:
            stage = _fallback_stage(name, row) or stage
        if stage not in _STAGES:
            run_doc.setdefault("unrouted", []).append(row)
            continue
        key = "model_calls" if name == "model_call" else (name or "unrouted")
        stage_docs.setdefault(stage, {}).setdefault(key, []).append(row)

    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        json.dumps(run_doc, ensure_ascii=False, default=str), encoding="utf-8",
    )
    for stage, doc in stage_docs.items():
        stage_dir = root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "log.json").write_text(
            json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8",
        )


__all__ = ["read_v2_events", "resolve_v2_root", "write_v2_bundle"]
