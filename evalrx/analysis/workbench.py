"""Durable primitives for the browser workbench.

The original upload app treated an archive as a directory of JSON files.  This
module is deliberately independent from Streamlit: it validates and extracts
an archive, turns the supported tabular and media files into an auditable
``DatasetBundle``, and writes append-only events/messages for a data thread.
Keeping these operations here makes the browser UI a thin observer rather than
the source of truth for an analysis run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TABULAR_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".parquet", ".xlsx", ".xls"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
SUPPORTED_SUFFIXES = TABULAR_SUFFIXES | IMAGE_SUFFIXES | PDF_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES


@dataclass(frozen=True)
class UploadLimits:
    """Conservative local-workbench limits; callers may explicitly raise them."""

    max_archive_bytes: int = 1024 * 1024 * 1024
    max_extracted_bytes: int = 4 * 1024 * 1024 * 1024
    max_files: int = 10_000
    max_compression_ratio: float = 100.0
    max_path_depth: int = 32


@dataclass
class Event:
    seq: int
    timestamp: str
    thread_id: str
    turn_id: str
    stage: str
    status: str
    message: str
    attempt: int | None = None
    artifact_refs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventSink:
    """Append-only, restart-safe progress events for one data thread."""

    def __init__(self, path: str | Path, *, thread_id: str, turn_id: str) -> None:
        self.path = Path(path)
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = _last_event_seq(self.path)

    def emit(
        self,
        stage: str,
        status: str,
        message: str,
        *,
        attempt: int | None = None,
        artifact_refs: Iterable[str | Path] = (),
        metrics: dict[str, Any] | None = None,
    ) -> Event:
        with self._lock:
            self._seq += 1
            event = Event(
                seq=self._seq,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                stage=stage,
                status=status,
                message=message,
                attempt=attempt,
                artifact_refs=[str(p) for p in artifact_refs],
                metrics=metrics or {},
            )
            _append_jsonl(self.path, event.to_dict())
            return event


@dataclass
class DatasetBundle:
    """Normalized view of a local archive plus references to original media."""

    root: str
    source_dir: str
    records_path: str
    media_units_path: str
    manifest_path: str
    n_records: int
    n_media_units: int
    files: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThreadStore:
    """Small filesystem store for a local, single-user analysis conversation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create(self, *, name: str, provider: str, model: str = "") -> Path:
        slug = _safe_name(name)
        thread_id = f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        thread = self.root / thread_id
        thread.mkdir(parents=True, exist_ok=False)
        meta = {
            "id": thread_id,
            "name": name,
            "provider": provider,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "current_turn": "",
        }
        (thread / "thread.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return thread

    @staticmethod
    def append_message(thread: str | Path, role: str, content: str, *, turn_id: str = "") -> None:
        _append_jsonl(Path(thread) / "messages.jsonl", {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "role": role,
            "content": content,
            "turn_id": turn_id,
        })

    @staticmethod
    def set_current_turn(thread: str | Path, turn_id: str) -> None:
        path = Path(thread) / "thread.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["current_turn"] = turn_id
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_archive(payload: bytes, destination: str | Path, *, limits: UploadLimits | None = None) -> Path:
    """Safely extract a ZIP and return its data root.

    In addition to zip-slip, this rejects symlinks, duplicate targets, overly
    deep paths, excessive file counts, and archives with suspicious expansion.
    """
    limits = limits or UploadLimits()
    if len(payload) > limits.max_archive_bytes:
        raise ValueError("archive exceeds the configured upload size limit")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    seen: set[Path] = set()
    total = 0
    n_files = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > limits.max_files:
            raise ValueError("archive contains too many files")
        for info in infos:
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if "__MACOSX" in parts or parts[-1] in {".DS_Store", "Thumbs.db"}:
                continue
            is_link = (info.external_attr >> 16) & 0o170000 == 0o120000
            if (not name or name.startswith("/") or ".." in parts or is_link
                    or len(parts) > limits.max_path_depth):
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
            if info.file_size and info.compress_size and info.file_size / info.compress_size > limits.max_compression_ratio:
                raise ValueError(f"archive member has an unsafe compression ratio: {info.filename!r}")
            total += info.file_size
            if total > limits.max_extracted_bytes:
                raise ValueError("archive exceeds the configured extracted size limit")
            target = (destination / name).resolve()
            if not target.is_relative_to(root) or target in seen:
                raise ValueError(f"unsafe or duplicate path in archive: {info.filename!r}")
            seen.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            n_files += 1
    if not n_files:
        raise ValueError("archive contains no files")
    children = [p for p in destination.iterdir() if p.name not in {"__MACOSX", ".DS_Store"}]
    return children[0] if len(children) == 1 and children[0].is_dir() else destination


def ingest_directory(
    source_dir: str | Path,
    bundle_dir: str | Path,
    *,
    media_unit_limit: int = 500,
    sink: EventSink | None = None,
) -> DatasetBundle:
    """Create an auditable normalized bundle without sending data off-machine."""
    source = Path(source_dir).resolve()
    bundle = Path(bundle_dir).resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    if sink:
        sink.emit("discover", "started", "Discovering uploaded files")
    files = [p for p in sorted(source.rglob("*")) if p.is_file() and not _is_junk(p)]
    file_entries = [_file_entry(p, source) for p in files]
    unsupported = [e["path"] for e in file_entries if e["suffix"] not in SUPPORTED_SUFFIXES]
    if sink:
        sink.emit("discover", "completed", f"Discovered {len(files)} files", metrics={"files": len(files)})
        sink.emit("normalize", "started", "Normalizing tabular records and media metadata")

    records: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path, entry in zip(files, file_entries):
        suffix = entry["suffix"]
        if suffix in TABULAR_SUFFIXES:
            try:
                for row in _read_tabular(path):
                    row.setdefault("_source_file", entry["path"])
                    records.append(row)
            except Exception as exc:  # data quality is reported, not fatal for a mixed archive
                warnings.append(f"could not read {entry['path']}: {exc}")
        elif suffix in IMAGE_SUFFIXES | PDF_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES:
            bundled_media = _materialize_media(path, source, bundle)
            entry["bundle_path"] = str(bundled_media.relative_to(bundle))
            media.extend(_media_units(path, source, entry, media_unit_limit - len(media)))
            if len(media) >= media_unit_limit:
                warnings.append(f"media unit limit ({media_unit_limit}) reached; remaining media was not profiled")
                break

    records_path = bundle / "records.json"
    media_path = bundle / "media_units.jsonl"
    manifest_path = bundle / "bundle.json"
    records_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    for unit in media:
        _append_jsonl(media_path, unit)
    bundle_data = DatasetBundle(
        root=str(bundle), source_dir=str(source), records_path=str(records_path),
        media_units_path=str(media_path), manifest_path=str(manifest_path),
        n_records=len(records), n_media_units=len(media), files=file_entries,
        warnings=warnings + ([f"unsupported files skipped: {', '.join(unsupported[:8])}"] if unsupported else []),
    )
    manifest_path.write_text(json.dumps(bundle_data.to_dict(), indent=2), encoding="utf-8")
    if sink:
        sink.emit("normalize", "completed", "Dataset bundle is ready", artifact_refs=[manifest_path, records_path, media_path], metrics={"records": len(records), "media_units": len(media)})
    return bundle_data


def load_bundle(path: str | Path) -> DatasetBundle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetBundle(**data)


def _read_tabular(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _read_json_rows(path)
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ",")]
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dashboard extra supplies pandas
        raise RuntimeError("pandas is required to read Parquet and Excel files") from exc
    frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_excel(path)
    return [dict(r) for r in frame.to_dict(orient="records")]


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [dict(v) if isinstance(v, dict) else {"value": v} for v in value]
        if isinstance(value, dict):
            for key in ("cases", "records", "results", "rows", "items", "data", "examples", "samples"):
                if isinstance(value.get(key), list):
                    meta = {k: v for k, v in value.items() if k != key and not isinstance(v, (list, dict))}
                    return [{**meta, **dict(v)} if isinstance(v, dict) else {**meta, "value": v} for v in value[key]]
            return [value]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(dict(value) if isinstance(value, dict) else {"value": value})
    return rows


def _media_units(path: Path, source: Path, entry: dict[str, Any], remaining: int) -> list[dict[str, Any]]:
    if remaining <= 0:
        return []
    parent_id = hashlib.sha256(entry["path"].encode()).hexdigest()[:16]
    unit = {"id": parent_id, "parent_id": None, "kind": _media_kind(entry["suffix"]), **entry}
    suffix = entry["suffix"]
    if suffix in IMAGE_SUFFIXES:
        try:
            from PIL import Image
            with Image.open(path) as image:
                unit.update(width=image.width, height=image.height, mode=image.mode)
        except Exception as exc:
            unit["warning"] = f"image metadata unavailable: {exc}"
    elif suffix in PDF_SUFFIXES:
        return _pdf_units(path, unit, remaining)
    elif suffix in AUDIO_SUFFIXES | VIDEO_SUFFIXES:
        unit.update(_ffprobe_metadata(path))
        duration = unit.get("duration_sec")
        if isinstance(duration, (float, int)) and duration > 0 and remaining > 1:
            segment_sec = 30.0 if suffix in AUDIO_SUFFIXES else 10.0
            children = []
            start = 0.0
            while start < float(duration) and len(children) < remaining - 1:
                end = min(start + segment_sec, float(duration))
                children.append({
                    "id": f"{parent_id}:{len(children):04d}", "parent_id": parent_id,
                    "kind": f"{unit['kind']}_segment", **entry,
                    "start_sec": start, "end_sec": end,
                })
                start = end
            return [unit, *children]
    return [unit]


def _materialize_media(path: Path, source: Path, bundle: Path) -> Path:
    """Expose original media to a coding-agent workdir without a fragile symlink."""
    target = bundle / "media" / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    try:
        os.link(path, target)
    except OSError:
        shutil.copy2(path, target)
    return target


def _pdf_units(path: Path, unit: dict[str, Any], remaining: int) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        unit["pages"] = len(reader.pages)
        children = []
        for page_no, page in enumerate(reader.pages[:max(0, remaining - 1)], start=1):
            text = page.extract_text() or ""
            children.append({
                "id": f"{unit['id']}:p{page_no}", "parent_id": unit["id"], "kind": "pdf_page",
                **{k: v for k, v in unit.items() if k not in {"id", "parent_id", "kind"}},
                "page": page_no, "text_excerpt": text[:1000],
            })
        return [unit, *children]
    except Exception:
        unit.update(pages=None, warning="PDF text extraction requires the optional pypdf dependency")
        return [unit]


def _ffprobe_metadata(path: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], capture_output=True, text=True, timeout=15, check=False)
        data = json.loads(proc.stdout or "{}")
        duration = (data.get("format") or {}).get("duration")
        return {"duration_sec": float(duration) if duration is not None else None}
    except Exception:
        return {"duration_sec": None, "warning": "ffprobe metadata unavailable"}


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    return {"path": str(path.relative_to(root)), "suffix": suffix, "mime": mimetypes.guess_type(path.name)[0] or "application/octet-stream", "bytes": path.stat().st_size}


def _media_kind(suffix: str) -> str:
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "video"


def _is_junk(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name in {".DS_Store", "Thumbs.db"}


def _safe_name(value: str) -> str:
    clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in value).strip("_")
    return clean or "dataset"


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _last_event_seq(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                return int(json.loads(line).get("seq", 0))
    except Exception:
        pass
    return 0
