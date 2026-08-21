"""Stable event and artifact envelopes shared by writers and future readers."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any  # noqa: I001 - kept with Path for Python 3.10 compatibility notes

OBSERVABILITY_SCHEMA_VERSION = 1
_EVENT_NAMESPACE = uuid.UUID("3a09f2ed-28b1-4dd4-914f-b8da031edab2")


def make_event_envelope(
    event: dict[str, Any], *, trace_id: str, event_seq: int,
    artifact_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize a run-log record into an idempotent observability event."""
    event_type = str(event.get("event") or "event")
    event_id = str(uuid.uuid5(_EVENT_NAMESPACE, f"{trace_id}:{event_seq}:{event_type}"))
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "event_id": event_id,
        "trace_id": trace_id,
        "event_seq": event_seq,
        "timestamp": event.get("ts"),
        "event_type": event_type,
        "stage": _stage_for(event_type),
        "cycle": event.get("cycle"),
        "parent_id": event.get("span_id"),
        "status": _status_for(event),
        "payload": event,
        "artifact_refs": artifact_refs or [],
    }


def artifact_manifest(path: str | Path, *, run_dir: str | Path, role: str = "artifact") -> dict[str, Any]:
    """Describe an artifact without reading it into the event payload."""
    artifact_path = Path(path)
    root = Path(run_dir).resolve()
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        relative = str(artifact_path.relative_to(root))
    except ValueError:
        relative = artifact_path.name
    return {
        "artifact_id": f"sha256:{digest.hexdigest()}",
        "role": role,
        "path": relative,
        "filename": artifact_path.name,
        "mime_type": mimetypes.guess_type(artifact_path.name)[0] or "application/octet-stream",
        "size_bytes": artifact_path.stat().st_size,
    }


def artifact_manifests_for_event(event: dict[str, Any], *, run_dir: str | Path) -> list[dict[str, Any]]:
    """Find file references in a run event and turn them into content manifests.

    Event fields intentionally contain relative paths, never inline binary
    content.  This small adapter gives the Langfuse writer a complete upload
    manifest without changing the established JSONL event shapes.
    """
    root = Path(run_dir).resolve()
    candidates: list[tuple[str, str]] = []

    def visit(value: Any, key: str = "artifact", path_like: bool = False) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                child_key = str(child_key)
                visit(child, child_key, path_like or child_key.endswith("path") or child_key.endswith("_paths") or child_key == "record")
        elif isinstance(value, list):
            for child in value:
                visit(child, key, path_like)
        elif isinstance(value, str) and path_like:
            candidates.append((key, value))

    visit(event)
    manifests: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for role, relative in candidates:
        path = (root / relative).resolve()
        if root not in path.parents:
            continue
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        try:
            manifests.append(artifact_manifest(path, run_dir=root, role=role))
        except OSError:
            continue
    return manifests


def canonical_json(value: Any) -> str:
    """Canonical form used by parity checks and tests."""
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True, separators=(",", ":"))


def _stage_for(event_type: str) -> str:
    return {
        "probe_search": "PRE_M1", "probe": "M1", "analysis": "M2",
        "explore": "M2", "diagnosis": "M3", "surgery": "M5",
        "fix": "M4", "experiment": "M4",
    }.get(event_type, "RUN")


def _status_for(event: dict[str, Any]) -> str:
    if event.get("ok") is False or event.get("status") in {"failed", "error"}:
        return "failed"
    return "completed"
