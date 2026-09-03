"""Langfuse-backed run reader for the EvalRX HTML report.

Langfuse's observations API is row-oriented.  EvalRX writes each durable
JSONL event as an EVENT observation, so this adapter recovers the ordered event
stream without making the renderer understand Langfuse-specific response types.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


class LangfuseRunSource:
    """Read EvalRX event envelopes from Langfuse Observations API v2."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                raise ImportError(
                    "Langfuse report input needs `pip install evalrx[observability]`."
                ) from exc
            client = Langfuse()
        self.client = client

    def events(self, trace_id: str) -> list[dict[str, Any]]:
        """Return the exact EvalRX events for *trace_id*, ordered by sequence."""
        return [_public_event(event) for event in self._raw_events(trace_id)]

    def _raw_events(self, trace_id: str) -> list[dict[str, Any]]:
        cursor: str | None = None
        recovered: list[tuple[int, dict[str, Any]]] = []
        # The SDK uses the canonical 32-hex Langfuse trace id.  Our local UUID
        # carries dashes, so accept either spelling at the CLI boundary.
        langfuse_trace_id = trace_id.replace("-", "")
        while True:
            try:
                response = self.client.api.observations.get_many(
                    trace_id=langfuse_trace_id,
                    fields="io,metadata,time",
                    expand_metadata="event_id,event_seq,stage,cycle,artifact_refs",
                    limit=1000,
                    cursor=cursor,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not query Langfuse observations. Set LANGFUSE_PUBLIC_KEY, "
                    "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST (if self-hosted)."
                ) from exc
            for observation in response.data:
                event = _event_from_observation(observation)
                if event is not None:
                    metadata = _as_dict(getattr(observation, "metadata", None))
                    recovered.append((int(metadata.get("event_seq") or event.get("event_seq") or 0), event))
            cursor = getattr(getattr(response, "meta", None), "cursor", None)
            if not cursor:
                break
        recovered.sort(key=lambda item: item[0])
        return [event for _, event in recovered]

    def materialize(self, trace_id: str, destination: str | Path) -> Path:
        """Write a renderer-compatible cache; it is not a source of truth."""
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        events = self._raw_events(trace_id)
        if not events:
            raise LookupError(f"No EvalRX events found for Langfuse trace {trace_id!r}")
        for event in events:
            self._materialize_artifacts(event, root)
        log_path = root / "run_log.jsonl"
        log_path.write_text(
            "\n".join(json.dumps(_public_event(event), ensure_ascii=False, default=str) for event in events) + "\n",
            encoding="utf-8",
        )
        log_path.chmod(0o600)
        return root

    def _materialize_artifacts(self, event: dict[str, Any], root: Path) -> None:
        attachments = event.pop("_langfuse_artifacts", [])
        refs = event.pop("_langfuse_artifact_refs", [])
        if not attachments or not refs or not hasattr(self.client, "resolve_media_references"):
            return
        resolved = self.client.resolve_media_references(
            obj=attachments, resolve_with="base64_data_uri",
        )
        ref_by_id = {str(ref.get("artifact_id")): ref for ref in refs if isinstance(ref, dict)}
        for attachment in resolved:
            if not isinstance(attachment, dict):
                continue
            ref = ref_by_id.get(str(attachment.get("artifact_id")))
            content = attachment.get("content")
            if ref is None or not isinstance(content, str):
                continue
            _write_data_uri(root, str(ref.get("path") or ""), content)


def _event_from_observation(observation: Any) -> dict[str, Any] | None:
    """Extract our payload from an ObservationV2 or a lightweight test double."""
    input_data = _as_dict(getattr(observation, "input", None))
    event = input_data.get("event")
    if not isinstance(event, dict):
        return None
    result = dict(event)
    attachments = input_data.get("artifacts")
    metadata = _as_dict(getattr(observation, "metadata", None))
    refs = metadata.get("artifact_refs")
    if isinstance(attachments, list):
        result["_langfuse_artifacts"] = attachments
    if isinstance(refs, list):
        result["_langfuse_artifact_refs"] = refs
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if not key.startswith("_langfuse_")}


def _write_data_uri(root: Path, relative_path: str, data_uri: str) -> None:
    """Write a resolved Langfuse media value inside the renderer cache only."""
    if not data_uri.startswith("data:") or "," not in data_uri:
        return
    target = (root / relative_path).resolve()
    if root.resolve() not in target.parents:
        return
    try:
        payload = base64.b64decode(data_uri.split(",", 1)[1])
    except ValueError:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    target.chmod(0o600)
