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


__all__ = ["read_v2_events", "resolve_v2_root"]
