"""RunLoggerV2 — a tidy, from-scratch redesign of the diagnose-loop logger.

Coexists with :mod:`~evalrx.eval_agent.run_logger` (``RunLogger``) — nothing
there is touched, deleted, or reused destructively; a caller opts into this
one explicitly by constructing it instead. See ``RUN_LOGGER_V2.md`` (next to
this file) for the full design rationale, layout, and trade-offs; the four
rules that shaped it:

1. Few files. One JSON document per pipeline stage, not a scattered pile of
   ``prompts/*.txt`` + ``artifacts/*.json`` + ``experiments/*.py`` + ...
2. Same-type logging in one JSON. Every event of a given kind (all ``probe``
   events, all ``model_call`` events, ...) lives in ONE array, in ONE file —
   not one small file per call.
3. M1..M5 each get their own folder. A reader who only cares about M3 opens
   exactly one folder.
4. Nothing but JSON, except real binary artifacts (images, audio, tensors). Code,
   stdout, prompts, markdown summaries — all of that is now a STRING VALUE
   inside the JSON, not a sibling ``.py``/``.txt``/``.md`` file.

Public API mirrors ``RunLogger`` method-for-method (same names, same
signatures, same call-site behavior for return values like ``log_probe``'s
``list[Path]``) so an existing ``VLDiagnoseLoop(run_logger=...)`` /
``ProbeAgent(run_logger=...)`` / ``AutoDiagnoseLoop(run_logger=...)`` can use
this class by construction alone — no other code in ``loop.py``,
``probe_agent.py``, or any ``stages/*.py`` needs to change to try it.

Known, deliberate scope cuts (see the design doc for why each is safe):
  - RunContext integration uses an external ephemeral runtime tree. Generated
    text/code is captured into stage JSON and the runtime tree is removed at
    finalization instead of becoming a forest of trial files.
  - No human-readable Markdown summaries (``record.md``, ``outcome.md``) —
    the same information is in the JSON for a renderer to build one from.
  - No opt-in JSON-Schema self-validation (``EVALRX_VALIDATE_LOG``) — this is
    a new structure with its own doc instead of ``log_schema.py``.
  - Verbose console narration is a plain one-line-per-event summary, not
    ``RunLogger``'s multi-line stage narration.
Native Langfuse/OpenTelemetry mirroring (:class:`DiagnosticTracer`) IS kept,
reused unchanged — it is orthogonal to file layout.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Reused, not reimplemented: pure, stateless content-shaping helpers that
# already produce exactly the values this module wants to embed. Importing
# them is the ONLY coupling to run_logger.py — nothing there is modified, and
# these functions have no file-writing side effects of their own.
from evalrx.eval_agent.run_logger import (
    _artifact_to_numpy,
    _case_snapshot,
    _iter_cases,
    _probe_examples,
    _save_artifact_figure,
)

if TYPE_CHECKING:
    from evalrx.analysis.analysis_module import AnalysisReport
    from evalrx.core.result import Result
    from evalrx.eval_agent.hypothesis import Hypothesis
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport
    from evalrx.eval_agent.stages.diagnosis import DiagnosisResult
    from evalrx.eval_agent.stages.surgery import InterventionResult

RUN_LOGGER_V2_VERSION = 1

_STAGES = ("M1", "M2", "M3", "M4", "M5")

#: A tag string not matching this falls back to _STAGE_ALIASES, then to the
#: run-level "unrouted" bucket (never silently dropped — see _resolve_stage).
_STAGE_RE = re.compile(r"m([1-5])", re.IGNORECASE)

#: Tags used somewhere in the codebase that carry no "m<N>" substring at all
#: (checked against every literal `module=`/`stage=` value passed to a
#: log_* method as of this writing — see RUN_LOGGER_V2.md's "routing" table).
_STAGE_ALIASES: dict[str, str] = {
    "fix_pipeline": "M5",
    "fix": "M5",
    "explore": "M2",
}

#: Extensions treated as genuine binary media — the one thing rule 4 still
#: allows as a separate file. Everything else becomes a JSON string value.
_MEDIA_EXTS = frozenset({
    ".npy", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".wav", ".mp3", ".flac", ".ogg", ".mp4", ".avi", ".mov",
})

#: Text/code file suffixes worth inlining from a sandbox workspace snapshot.
#: Mirrors RunLogger._SNAPSHOT_SUFFIXES's intent (skip weights/binaries) —
#: redefined locally rather than imported so this module has no dependency
#: on RunLogger's internals, only its free functions (see the imports above).
_INLINE_SUFFIXES = frozenset(
    {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".log", ".toml"}
)
_INLINE_MAX_BYTES = 2_000_000  # skip (note-only) any single file larger than this


def _resolve_stage(tag: "str | None") -> "str | None":
    """"m1_probe" / "M4_SURGERY" / "codegen_m2_stats" / "fix_pipeline" -> "M1".."M5".

    Returns ``None`` when *tag* matches nothing — the caller must not drop
    the event in that case; route it to the run-level "unrouted" bucket
    instead (see ``RunLoggerV2._route``). A tag is never assumed unroutable
    without trying both the regex AND the alias table.
    """
    if not tag:
        return None
    m = _STAGE_RE.search(tag)
    if m:
        return f"M{m.group(1)}"
    return _STAGE_ALIASES.get(tag.strip().lower())


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write *obj* as JSON to *path* such that a reader never sees a partial file.

    Writes to a sibling temp file first, then ``os.replace`` (atomic on the
    same filesystem) — a crash mid-write leaves the OLD complete file in
    place, never a truncated one. This runs on every single logged event
    (see the design doc's "durability" section for the cost trade-off that
    was chosen deliberately here, not overlooked).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _inline_workspace(
    workdir: "str | Path", media_dir: Path, *, run_dir: "Path | None" = None,
) -> "dict[str, Any] | None":
    """Read a sandbox working directory into a JSON-safe dict, inlining text.

    Returns ``{"files": {relative_path: content_or_note}, "media": [rel_paths],
    "skipped": n}`` or ``None`` when *workdir* does not exist. Text-like files
    (see ``_INLINE_SUFFIXES``) are read and inlined verbatim; recognised media
    extensions are COPIED into *media_dir* (a real binary artifact, rule 4's
    one exception) with a path reference left in ``"media"``; anything else is
    skipped with a one-line note so its existence is still visible.
    """
    import hashlib
    import shutil

    src = Path(workdir)
    if not src.exists() or not src.is_dir():
        return None
    files: dict[str, Any] = {}
    media: list[str] = []
    skipped = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(src))
        suffix = f.suffix.lower()
        if suffix in _MEDIA_EXTS:
            media_dir.mkdir(parents=True, exist_ok=True)
            # A content/path digest, not Python's str hash() — hash() is
            # salted per-process (PYTHONHASHSEED), so the same workspace file
            # would get a different artifact name on every run, breaking the
            # re-run diffing this file exists to support.
            digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:8]
            dest = media_dir / f"{f.stem}_{digest}{suffix}"
            try:
                shutil.copy2(f, dest)
                try:
                    media.append(str(dest.relative_to(run_dir)) if run_dir else str(dest))
                except ValueError:
                    media.append(str(dest))
            except Exception:  # noqa: BLE001
                skipped += 1
            continue
        if suffix not in _INLINE_SUFFIXES:
            files[rel] = f"<skipped: {suffix or 'no extension'}, not a recognised text type>"
            skipped += 1
            continue
        try:
            if f.stat().st_size > _INLINE_MAX_BYTES:
                files[rel] = f"<skipped: {f.stat().st_size} bytes, over the inline cap>"
                skipped += 1
                continue
            files[rel] = f.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            files[rel] = f"<could not read: {exc}>"
            skipped += 1
    return {"files": files, "media": media, "skipped": skipped}


class _V2JsonFormatter:
    """Renders one plain one-line console summary per event, for ``verbose=True``.

    Intentionally simple relative to ``RunLogger``'s multi-line narration —
    this module's ask was file layout, not console UX; see the design doc.
    """

    @staticmethod
    def line(stage: "str | None", event: str, payload: "dict[str, Any]") -> str:
        where = f"[{stage}]" if stage else "[run]"
        cycle = payload.get("cycle")
        tail = f" cycle={cycle}" if cycle is not None else ""
        return f"{where} {event}{tail}"


class RunLoggerV2:
    """A tidy, from-scratch M1..M5 logger. See the module docstring + design doc.

    Args:
        run_dir:  Directory to write into. Created if missing. Defaults to
                  ``runs_v2/<YYYYMMDD_HHMMSS>/`` relative to cwd.
        verbose:  Print a one-line summary of every event to stdout.
        trace_id: Ties every event to one Langfuse trace; auto-generated if
                  omitted.
        observability_mode: Forwarded to :class:`DiagnosticTracer` unchanged.

    Layout written under *run_dir*::

        run.json          run-wide: run_start, cases, report_published,
                           loop_end, agent_decisions, agent_tool_calls,
                           unrouted (see _resolve_stage)
        M1/log.json       probe, model_calls, tool_codegen, tool_registry,
                           stage_skipped — all M1-tagged events
        M2/log.json       analysis, explore, ...
        M3/log.json       diagnosis, ...
        M4/log.json       surgery (hypothesis-verification kind), ...
        M5/log.json       surgery (intervention kind), experiment, fix, ...
        M*/artifacts/     binary media for that stage only (rule 4's exception)
        media/            case-level baseline media (images/audio referenced
                           by FailureCase.inputs)
        artifacts/         run-global named JSON (save_artifact_json)
    """

    def __init__(
        self,
        run_dir: "str | Path | None" = None,
        *,
        verbose: bool = False,
        trace_id: "str | None" = None,
        observability_mode: "str | None" = None,
        context: "Any | None" = None,
    ) -> None:
        if run_dir is None:
            run_dir = Path("runs_v2") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_json_path = self.run_dir / "run.json"

        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.current_cycle: int = -1
        self.verbose = verbose
        self._context = context
        self._closed = False
        # Producers use this capability flag to keep text/code in log events
        # instead of writing V1-style sibling files into trial directories.
        self.inline_text_artifacts = True
        self.preserve_full_model_io = True

        # One in-memory doc per stage + one run-level doc. Every log_* method
        # appends to the relevant bucket(s), then atomically rewrites exactly
        # the doc(s) it touched — see _atomic_write_json.
        self._lock = threading.RLock()
        self._run_doc: dict[str, Any] = {
            "trace_id": self.trace_id,
            "run_start": None,
            "cases": [],
            "report_published": [],
            "diagnose_reports": [],
            "manifest": None,
            "loop_end": [],
            "agent_decisions": [],
            "agent_tool_calls": [],
            "unrouted": [],
        }
        self._stage_docs: dict[str, dict[str, Any]] = {s: {} for s in _STAGES}
        self._logged_case_ids: set[str] = set()
        self._model_call_seq = 0
        self._codegen_seq = 0
        # See log_model_call / log_probe: calls are recorded immediately into
        # M1's doc AND buffered here so log_probe can replay them into
        # Langfuse nested under the right probe span once it exists.
        self._pending_model_calls: "dict[int, list[dict[str, Any]]]" = {}

        from evalrx.observability.tracer import DiagnosticTracer
        # The SQLite delivery queue is runtime state, not part of the tidy run
        # artifact.  Keep it outside the run tree; langfuse_trace.json remains
        # the durable, portable JSON trace bundled with the run.
        outbox_dir = Path(tempfile.gettempdir()) / "evalrx-v2-outbox"
        self._outbox_path = outbox_dir / f"{self.trace_id}.sqlite3"
        self.tracer = DiagnosticTracer(
            run_dir=self.run_dir, mode=observability_mode, auto_sync=True,
            outbox_path=self._outbox_path,
        )
        self.tracer.trace_id = self.trace_id

        self._flush_run()

    # ------------------------------------------------------------------
    # Internal: doc access, routing, durability
    # ------------------------------------------------------------------

    def _stage_dir(self, stage: str) -> Path:
        d = self.run_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _stage_artifacts_dir(self, stage: str) -> Path:
        d = self._stage_dir(stage) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _flush_run(self) -> None:
        _atomic_write_json(self.run_json_path, self._run_doc)

    def _flush_stage(self, stage: str) -> None:
        _atomic_write_json(self._stage_dir(stage) / "log.json", self._stage_docs[stage])

    def _bucket(self, stage: str, key: str) -> list:
        return self._stage_docs[stage].setdefault(key, [])

    def _append_stage(self, tag: "str | None", key: str, record: "dict[str, Any]") -> str:
        """Route *record* by *tag* into the right stage bucket; flush; return the stage."""
        stage = _resolve_stage(tag)
        with self._lock:
            if stage is None:
                warnings.warn(
                    f"RunLoggerV2: could not route event {key!r} (tag={tag!r}) to a "
                    "stage — filed under run.json['unrouted'] instead of being lost.",
                    stacklevel=3,
                )
                self._run_doc["unrouted"].append({"key": key, "tag": tag, **record})
                self._flush_run()
                return "unrouted"
            self._bucket(stage, key).append(record)
            self._flush_stage(stage)
        if self.verbose:
            print(_V2JsonFormatter.line(stage, key, record))
        return stage

    def _append_run(self, key: str, record: "dict[str, Any]") -> None:
        """Append *record* to the (always list-valued) run.json bucket *key*.

        ``run_start`` is the one run.json field that isn't a list — it's set
        directly by ``log_run_start``, never through here.
        """
        with self._lock:
            self._run_doc[key].append(record)
            self._flush_run()
        if self.verbose:
            print(_V2JsonFormatter.line(None, key, record))

    @property
    def managed_json_paths(self) -> "tuple[Path, ...]":
        """Atomic JSON documents that may be rewritten while quarantine runs."""
        return (self.run_json_path, *(self.run_dir / s / "log.json" for s in _STAGES))

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    def _save_media(self, stage: str, stem: str, artifact: Any) -> "str | None":
        """Save a numeric artifact (tensor/array) + a rendered figure, if any.

        Mirrors ``RunLogger._save_artifact`` in spirit but writes under this
        stage's ``artifacts/`` dir. Returns the ``.npy`` path (run-relative)
        or ``None`` when *artifact* isn't a recognised numeric type — the
        one deliberate use of the reused, stateless helpers from run_logger.py.
        """
        try:
            import numpy as np

            arr = _artifact_to_numpy(artifact)
            if arr is not None:
                art_dir = self._stage_artifacts_dir(stage)
                path = art_dir / f"{stem}.npy"
                np.save(path, arr)
                _save_artifact_figure(art_dir, stem, arr)
                return str(path.relative_to(self.run_dir))
            return None
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLoggerV2: could not save artifact {stem!r}: {exc}")
            return None

    def _save_case_media(self, path: Path) -> "str | None":
        """Copy external case media into ``media/`` (content-hash-deduped); return rel path."""
        import hashlib
        import shutil

        media_dir = self.run_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        copied = media_dir / f"{digest.hexdigest()[:16]}_{path.name}"
        if not copied.exists():
            shutil.copy2(path, copied)
        return str(copied.relative_to(self.run_dir))

    def _portable_path(self, value: "str | Path") -> str:
        path = Path(value)
        try:
            return str(path.resolve().relative_to(self.run_dir.resolve()))
        except (OSError, ValueError):
            return str(value)

    # ------------------------------------------------------------------
    # Run provenance
    # ------------------------------------------------------------------

    def log_run_start(self, config: "dict[str, Any] | None" = None) -> None:
        import platform

        entry: dict[str, Any] = {"ts": self._ts(), "trace_id": self.trace_id}
        if config:
            entry.update(config)
        entry.setdefault("python_version", platform.python_version())
        try:
            from evalrx import __version__ as _ver  # type: ignore
            entry.setdefault("evalrx_version", _ver)
        except Exception:  # noqa: BLE001
            pass
        commit = self._git_commit()
        if commit:
            entry.setdefault("git_commit", commit)
        with self._lock:
            self._run_doc["run_start"] = entry
            self._flush_run()
        if self.verbose:
            print(_V2JsonFormatter.line(None, "run_start", entry))

        model_name = str(entry.get("model") or "Target Model")
        proto = entry.get("protocol") or {}
        proto_desc = proto.get("description", "") if isinstance(proto, dict) else str(proto)
        bench_name = str(entry.get("benchmark_name") or proto_desc or "Benchmark")
        self.tracer.start_trace(
            model=model_name, benchmark=bench_name,
            n_cases=int(entry.get("n_cases", 0) or 0), metadata=entry,
        )

    @staticmethod
    def _git_commit() -> "str | None":
        import subprocess

        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            commit = out.stdout.strip()
            if commit:
                return commit
        except Exception:  # noqa: BLE001
            pass
        return os.environ.get("EVALRX_GIT_COMMIT") or None

    def log_cases(self, cases: "Any") -> None:
        """Persist complete case I/O; media is copied into ``media/`` (rule 4's exception)."""
        for case in cases:
            case_id = str(getattr(case, "id", "") or "")
            if not case_id or case_id in self._logged_case_ids:
                continue
            if hasattr(case, "to_dict"):
                payload = case.to_dict()
            elif isinstance(case, dict):
                payload = dict(case)
            else:
                payload = {"id": case_id, "value": str(case)}
            payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
            media_paths: list[str] = []
            inputs = getattr(case, "inputs", None)
            for kind in ("image", "audio", "video"):
                value = getattr(inputs, kind, None)
                if not isinstance(value, (str, Path)):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    run_relative = self.run_dir / path
                    path = run_relative if run_relative.is_file() else path.resolve()
                if not path.is_file():
                    continue
                saved = self._save_case_media(path)
                if saved:
                    media_paths.append(saved)
            self._append_run("cases", {
                "ts": self._ts(), "case_id": case_id, "case": payload, "media_paths": media_paths,
            })
            self._logged_case_ids.add(case_id)

    def log_report_published(self, envelope: "dict[str, Any]") -> None:
        generated = envelope.get("generated_by") or {}
        self._append_run("report_published", {
            "ts": self._ts(),
            "report_schema_version": int(envelope.get("schema_version") or 1),
            "catalog_version": str(envelope.get("catalog_version") or ""),
            "json_render_version": str(envelope.get("json_render_version") or ""),
            "source_event_seq": int(envelope.get("source_event_seq") or 0),
            "sha256": str(envelope.get("sha256") or ""),
            "generated_by": generated,
            "report_paths": ["report/report_data.json", "report/report_spec.json"],
        })

    def log_diagnose_report(
        self,
        report: Any,
        cases: "list[Any]",
        *,
        discovery: "list[dict[str, Any]] | None" = None,
    ) -> None:
        """Inline the standard post-diagnosis report into ``run.json``.

        V1 renders several JSON and Markdown siblings.  V2 keeps one detailed
        machine-readable record; a UI can render prose from this data.
        """
        hyps_src = getattr(report, "all_hypotheses", None)
        if hyps_src is None:
            hyps_src = getattr(report, "final_hypotheses", [])
        hypotheses = [
            {
                "statement": h.statement,
                "plain_statement": getattr(h, "plain_statement", ""),
                "failure_mode": h.predicted_failure_mode,
                "status": h.status.value if h.status else None,
            }
            for h in hyps_src
        ]
        m4_results = [
            {
                "hypothesis": tr.hypothesis.statement,
                "failure_mode": tr.hypothesis.predicted_failure_mode,
                "status": tr.status.value,
                "effect_size": tr.effect_size,
                "confidence": tr.confidence,
                "protocol_consistent": tr.is_consistent_with_protocol,
                "verdict": tr.verdict,
                "evidence": tr.evidence,
            }
            for tr in getattr(report, "all_test_results", [])
        ]
        self._append_run("diagnose_reports", {
            "ts": self._ts(),
            "cycles": report.cycles,
            "stopped_by": getattr(report, "stopped_by", None),
            "resolved": getattr(report, "resolved", None),
            "n_cases": len(cases),
            "n_hypotheses": len(hypotheses),
            "n_verified": len(getattr(report, "verified_hypotheses", [])),
            "hypotheses": hypotheses,
            "m4_results": m4_results,
            "discovery": list(discovery or []),
        })

    def log_manifest(self, *, run_id: str, config: "dict[str, Any]") -> None:
        """Record final run provenance and a compact file index in ``run.json``."""
        files = [
            str(path.relative_to(self.run_dir))
            for path in sorted(self.run_dir.rglob("*"))
            if path.is_file() and not path.name.startswith(".")
        ]
        with self._lock:
            self._run_doc["manifest"] = {
                "ts": self._ts(), "run_id": run_id, "config": dict(config), "files": files,
            }
            self._flush_run()

    def log_model_exchange(
        self,
        stage: str,
        *,
        role: str,
        operation: str,
        inputs: Any,
        output: Any = None,
        error: "str | None" = None,
        duration_sec: "float | None" = None,
        metadata: "dict[str, Any] | None" = None,
        cycle: "int | None" = None,
    ) -> None:
        """Persist one exact model/agent input-output exchange in its stage."""
        def json_safe(value: Any) -> Any:
            import dataclasses

            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            elif hasattr(value, "to_dict") and callable(value.to_dict):
                try:
                    value = value.to_dict()
                except Exception:  # noqa: BLE001
                    pass
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))

        entry: dict[str, Any] = {
            "ts": self._ts(),
            "cycle": self.current_cycle if cycle is None else cycle,
            "role": role,
            "operation": operation, "inputs": json_safe(inputs), "output": json_safe(output),
            "error": error, "metadata": json_safe(dict(metadata or {})),
        }
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 4)
        self._append_stage(stage, "model_calls", entry)

    def save_artifact_json(self, stem: str, obj: Any) -> "str | None":
        """Write *obj* as JSON under the run-global ``artifacts/`` dir; return rel path."""
        try:
            d = self.run_dir / "artifacts"
            d.mkdir(parents=True, exist_ok=True)
            path = d / stem
            path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
            return str(path.relative_to(self.run_dir))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLoggerV2: could not save artifact {stem!r}: {exc}")
            return None

    # ------------------------------------------------------------------
    # M1 — target-model calls (see model_instrumentation.InstrumentedModel)
    # ------------------------------------------------------------------

    def log_model_call(
        self,
        *,
        cycle: int,
        analyzer: str,
        call_index: int,
        method: str,
        inputs: Any,
        kwargs: "dict[str, Any]",
        output: Any,
        duration_sec: float,
        error: "str | None",
        case_id: "str | None" = None,
        batch_case_ids: "list[str] | None" = None,
        n_batch_cases: "int | None" = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle, "analyzer": analyzer,
            "call_index": call_index, "method": method,
            "case_id": case_id, "batch_case_ids": batch_case_ids or [],
            "n_batch_cases": n_batch_cases if n_batch_cases is not None else len(batch_case_ids or []),
            "inputs": inputs, "kwargs": kwargs, "output": output,
            "duration_sec": round(duration_sec, 4),
        }
        if error is not None:
            record["error"] = error
        with self._lock:
            self._model_call_seq += 1
            record["seq"] = self._model_call_seq
            self._bucket("M1", "model_calls").append(record)
            self._flush_stage("M1")
            self._pending_model_calls.setdefault(cycle, []).append(record)
        if self.verbose:
            print(_V2JsonFormatter.line("M1", "model_call", record))

    # ------------------------------------------------------------------
    # M1 — probe
    # ------------------------------------------------------------------

    def log_probe(
        self,
        cycle: int,
        results: "dict[str, Result]",
        schema: "Any | None" = None,
        *,
        cases: "Any | None" = None,
        judge_prompt: "str | None" = None,
        judge_raw: "str | None" = None,
        duration_sec: "float | None" = None,
        failed_analyzers: "dict[str, str] | None" = None,
    ) -> "list[Path]":
        """M1: one entry in M1/log.json's "probe" list. See RunLogger.log_probe
        for the field-level rationale this mirrors; ``artifact_paths``/results
        are inlined here instead of living in separate ``.result.json`` files."""
        artifact_paths: dict[str, str] = {}
        overlay_pngs: list[Path] = []
        result_docs: dict[str, Any] = {}
        for name, result in results.items():
            for art_name, artifact in getattr(result, "artifacts", {}).items():
                stem = f"c{cycle}_{name}_{art_name}"
                rel = self._save_media("M1", stem, artifact)
                if rel is not None:
                    artifact_paths[f"{name}/{art_name}"] = rel
            image_overlays = getattr(result, "image_overlays", None)
            if image_overlays is not None:
                try:
                    overlay_pngs.extend(
                        image_overlays(self._stage_artifacts_dir("M1"), f"c{cycle}_{name}")
                    )
                except Exception as exc:  # noqa: BLE001 - viz must never break the probe
                    warnings.warn(f"RunLoggerV2: image_overlays failed for {name}: {exc}")
            to_dict = getattr(result, "to_dict", None)
            if callable(to_dict):
                try:
                    doc = to_dict()
                    summary = getattr(result, "summary", None)
                    if callable(summary):
                        doc["summary"] = summary()
                    result_docs[name] = doc
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"RunLoggerV2: could not serialise result {name!r}: {exc}")

        with self._lock:
            pending_calls = [c for bucket in self._pending_model_calls.values() for c in bucket]
            self._pending_model_calls.clear()

        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle,
            "analyzers": list(results),
            "findings": {name: r.findings for name, r in results.items()},
            "results": result_docs,
            "artifact_paths": artifact_paths,
            "n_model_calls": len(pending_calls),
        }
        examples = _probe_examples(results, cases)
        if examples:
            entry["examples"] = examples
        if failed_analyzers:
            entry["failed_analyzers"] = dict(failed_analyzers)
        if schema is not None:
            entry["selection_rationale"] = getattr(schema, "rationale", "")
            selected = getattr(schema, "selected_analyzers", None)
            if selected is not None:
                entry["selected_analyzers"] = list(selected)
        if judge_prompt:
            entry["judge_prompt"] = judge_prompt
        if judge_raw:
            entry["judge_response"] = judge_raw
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_stage("M1", "probe", entry)
        if judge_prompt or judge_raw:
            self.log_model_exchange(
                "M1", role="analyzer_selection_judge", operation="generate",
                inputs=judge_prompt or "", output=judge_raw or "", cycle=cycle,
                duration_sec=duration_sec,
            )

        png_figures: list[Path] = list(overlay_pngs)
        for rel_npy in artifact_paths.values():
            if not rel_npy.endswith(".npy"):
                continue
            png = self.run_dir / (rel_npy[: -len(".npy")] + ".png")
            if png.exists():
                png_figures.append(png)

        m1_span = self.tracer.start_span(
            name=f"M1: Multi-Dimensional Checkup (Cycle {cycle})",
            stage="M1",
            input_data={"analyzers": list(results.keys()), "selected_analyzers": entry.get("selected_analyzers", [])},
            metadata={"duration_sec": duration_sec, "artifacts": {
                "artifact_paths": artifact_paths, "figures": [str(p) for p in png_figures],
            }},
        )
        if judge_prompt or judge_raw:
            self.tracer.log_generation(
                name="M1 Analyzer Selection", model="judge",
                prompt=judge_prompt or "", completion=judge_raw or "", span_id=m1_span,
            )
        for name, r in results.items():
            findings = getattr(r, "findings", {}) or {}
            per_case = findings.get("per_case") or []
            probe_span = self.tracer.start_span(
                name=f"Probe: {name}", stage=f"M1_{name}", input_data={"probe": name},
                parent_id=m1_span,
                metadata={
                    "n_scored": len(per_case) if per_case else (findings.get("n_cases") or findings.get("n_scored")),
                },
            )
            calls_for_analyzer = [c for c in pending_calls if c.get("analyzer") == name]
            for call in calls_for_analyzer[:50]:
                self.tracer.log_generation(
                    name=f"{name} · {call.get('method')} #{call.get('call_index')}",
                    model="target_model", prompt=call.get("inputs"),
                    completion=call.get("error") or call.get("output"),
                    span_id=probe_span,
                    metadata={"duration_sec": call.get("duration_sec"), "error": call.get("error")},
                )
            self.tracer.end_span(probe_span, output_data={
                "findings": findings, "n_model_calls": len(calls_for_analyzer),
            })
        orphaned = {c["analyzer"] for c in pending_calls} - set(results)
        for name in orphaned:
            failed_span = self.tracer.start_span(
                name=f"Probe: {name} (failed)", stage=f"M1_{name}", input_data={"probe": name},
                parent_id=m1_span, metadata={"note": "analyzer raised before producing a Result"},
            )
            calls_for_analyzer = [c for c in pending_calls if c.get("analyzer") == name]
            for call in calls_for_analyzer[:50]:
                self.tracer.log_generation(
                    name=f"{name} · {call.get('method')} #{call.get('call_index')}",
                    model="target_model", prompt=call.get("inputs"),
                    completion=call.get("error") or call.get("output"),
                    span_id=failed_span,
                    metadata={"duration_sec": call.get("duration_sec"), "error": call.get("error")},
                )
            self.tracer.end_span(failed_span, status="failed", output_data={
                "n_model_calls": len(calls_for_analyzer),
            })
        self.tracer.end_span(m1_span, output_data={"n_probes": len(results)})
        return png_figures

    # ------------------------------------------------------------------
    # M2 — analysis + explore
    # ------------------------------------------------------------------

    def log_analysis(
        self, cycle: int, report: "AnalysisReport", *, duration_sec: "float | None" = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle,
            "severity": report.severity,
            "n_findings": len(report.findings),
            "findings": [str(f) for f in report.findings],
            "narrative": report.narrative,
            "descriptive_only": bool(getattr(report, "descriptive_only", False)),
        }
        stats_tool = getattr(report, "stats_tool", None)
        if stats_tool:
            entry["stats_tool"] = stats_tool
        fallback_reason = getattr(report, "llm_fallback_reason", None)
        if fallback_reason:
            entry["llm_fallback_reason"] = fallback_reason
        conclusion = getattr(report, "conclusion", None)
        if conclusion:
            entry["conclusion"] = conclusion
        evidence_chain = getattr(report, "evidence_chain", None)
        if evidence_chain:
            entry["evidence_chain"] = list(evidence_chain)
        stats_tool_results = getattr(report, "stats_tool_results", None)
        if stats_tool_results:
            entry["stats_tool_results"] = list(stats_tool_results)
        visualizations = getattr(report, "visualizations", None)
        if visualizations:
            entry["visualizations"] = list(visualizations)
        stats_plan = getattr(report, "stats_plan", None)
        if stats_plan:
            entry["stats_plan"] = stats_plan
        stats_results = getattr(report, "stats_results", None)
        if stats_results:
            entry["stats_results"] = [r.to_dict() for r in stats_results]
        corrected = getattr(report, "corrected_rejections", None)
        if corrected:
            entry["corrected_rejections"] = corrected
        figures = getattr(report, "figures", None)
        if figures:
            entry["figures"] = [self._portable_path(f) for f in figures]
        llm_prompt = getattr(report, "llm_prompt", None)
        llm_raw = getattr(report, "llm_raw", None)
        if llm_prompt:
            entry["judge_prompt"] = llm_prompt
        if llm_raw:
            entry["judge_response"] = llm_raw
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_stage("M2", "analysis", entry)
        if llm_prompt or llm_raw:
            self.log_model_exchange(
                "M2", role="statistics_judge", operation="generate",
                inputs=llm_prompt or "", output=llm_raw or "", cycle=cycle,
                duration_sec=duration_sec,
            )

        m2_span = self.tracer.start_span(
            name=f"M2: Screening & Confirmatory Signals (Cycle {cycle})", stage="M2",
            input_data={"severity": report.severity, "n_findings": len(report.findings)},
            metadata={"duration_sec": duration_sec, "stats_tool": stats_tool,
                      "artifacts": {"figures": [str(f) for f in (figures or [])]}},
        )
        if llm_prompt or llm_raw:
            self.tracer.log_generation(
                name="M2 Statistical Screening Analysis", model="judge",
                prompt=llm_prompt or "", completion=llm_raw or "", span_id=m2_span,
            )
        for s in (stats_results or []):
            s_dict = s.to_dict() if hasattr(s, "to_dict") else (s if isinstance(s, dict) else {})
            sig_name = s_dict.get("config", {}).get("signal") or s_dict.get("tool") or "signal"
            eff = s_dict.get("effect")
            if eff is not None:
                self.tracer.log_score(
                    name=f"m2_effect_{sig_name}", value=float(eff),
                    comment=f"p={s_dict.get('p_value')}", span_id=m2_span,
                )
        self.tracer.end_span(m2_span, output_data={"conclusion": conclusion or ""})

    def log_explore(
        self, cycle: int, report: "Any | None", *,
        out_dir: "Path | str | None" = None, duration_sec: "float | None" = None,
    ) -> None:
        ok = bool(getattr(report, "ok", False)) if report is not None else False
        charts = list(getattr(report, "charts", None) or []) if report is not None else []
        rendered = [
            str(c.get("figure_path")) for c in charts
            if isinstance(c, dict) and c.get("figure_path")
        ]
        workspace = None
        if out_dir is not None:
            workspace = _inline_workspace(
                out_dir, self._stage_artifacts_dir("M2"), run_dir=self.run_dir,
            )
            if workspace and workspace.get("media"):
                rendered = [
                    path for path in workspace["media"]
                    if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
                ]
        tables = getattr(report, "tables", None) or {}
        adjudication = dict(getattr(report, "adjudication", None) or {}) if report is not None else {}
        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle, "ok": ok,
            "n_observations": len(getattr(report, "observations", None) or []) if report is not None else 0,
            "n_charts": len(charts), "n_charts_rendered": len(rendered),
            "n_tables": len(tables) if isinstance(tables, dict) else len(list(tables or [])),
            "n_candidate_signals": len(getattr(report, "candidate_signals", None) or []) if report is not None else 0,
            "n_hypotheses": len(getattr(report, "hypotheses", None) or []) if report is not None else 0,
            "adjudication": {
                k: adjudication[k] for k in (
                    "method", "alpha", "split", "n_host_adjudicated", "n_rejected",
                    "n_in_family", "n_descriptive_only",
                ) if k in adjudication
            },
            "observations": [str(o) for o in (getattr(report, "observations", None) or [])[:12]] if report is not None else [],
            "caveats": [str(c) for c in (getattr(report, "caveats", None) or [])[:8]] if report is not None else [],
            "figures": rendered,
            "workspace_snapshot": workspace,
            "attempts": int(getattr(report, "attempts", 0) or 0) if report is not None else 0,
        }
        error = str(getattr(report, "error", "") or "") if report is not None else "explorer produced no report"
        if error:
            entry["error"] = error
        if out_dir is not None and workspace is None:
            entry["out_dir"] = str(out_dir)
            report_path = Path(out_dir) / "exploratory_report.json"
            if report_path.exists():
                entry["report_path"] = str(report_path)
        if report is not None and getattr(report, "code", None):
            entry["code"] = str(report.code)
        if report is not None and getattr(report, "raw_outputs", None):
            entry["raw_outputs"] = [str(r) for r in report.raw_outputs]
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_stage("M2", "explore", entry)
        for call in list(getattr(report, "model_calls", None) or []):
            self.log_model_exchange(
                "M2",
                role=str(call.get("role") or "explore_coder"),
                operation=str(call.get("operation") or "generate"),
                inputs=call.get("inputs"), output=call.get("output"),
                error=call.get("error"), duration_sec=call.get("duration_sec"),
                metadata=dict(call.get("metadata") or {}), cycle=cycle,
            )

        exp_span = self.tracer.start_span(
            name=f"Explore: Free-form EDA (Cycle {cycle})", stage="EXPLORE",
            input_data={"ok": ok, "attempts": entry.get("attempts", 0)},
            metadata={"duration_sec": duration_sec, "artifacts": {
                "out_dir": str(out_dir) if out_dir is not None else None,
                "report_path": entry.get("report_path"), "figures": rendered,
            }},
        )
        if report is not None:
            for i, raw in enumerate(getattr(report, "raw_outputs", None) or []):
                self.tracer.log_generation(
                    name=f"Explore Coder Agent (attempt {i + 1})", model="coder_agent",
                    prompt=None, completion=str(raw), span_id=exp_span,
                )
            if getattr(report, "code", None):
                self.tracer.log_generation(
                    name="Explore Analysis Code (analysis.py)", model="coder_agent",
                    prompt=None, completion=str(report.code), span_id=exp_span,
                )
        self.tracer.end_span(exp_span, output_data={
            "n_observations": entry.get("n_observations", 0),
            "n_candidate_signals": entry.get("n_candidate_signals", 0),
        })

    # ------------------------------------------------------------------
    # M3 — diagnosis
    # ------------------------------------------------------------------

    def log_diagnosis(
        self, cycle: int, diag: "DiagnosisResult", *,
        duration_sec: "float | None" = None, explore_figures: "list[str] | None" = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle,
            "model_name": diag.model_name,
            "n_hypotheses": len(diag.hypotheses),
            "hypotheses": [
                {
                    "statement": h.statement, "plain_statement": h.plain_statement,
                    "failure_mode": h.predicted_failure_mode,
                    "status": h.status.value if h.status else None,
                    "test_design": h.test_design,
                    "critic": (h.metadata or {}).get("critic"),
                    "critic_reason": (h.metadata or {}).get("critic_reason"),
                }
                for h in diag.hypotheses
            ],
            "raw_judge_output": diag.raw_judge_output,
            "n_critic_kept": int(getattr(diag, "n_critic_kept", 0) or 0),
            "n_critic_rejected": int(getattr(diag, "n_critic_rejected", 0) or 0),
        }
        critic_raw = getattr(diag, "critic_raw_output", "") or ""
        if critic_raw:
            entry["critic_raw_output"] = critic_raw
            entry["critic_prompt"] = getattr(diag, "critic_prompt", "") or ""
        proposed = list(getattr(diag, "proposed_hypotheses", None) or [])
        if proposed:
            entry["proposed_hypotheses"] = [
                {"statement": h.statement, "plain_statement": h.plain_statement,
                 "failure_mode": h.predicted_failure_mode, "test_design": h.test_design}
                for h in proposed
            ]
        review_decisions = list(getattr(diag, "review_decisions", None) or [])
        if review_decisions:
            entry["review"] = {
                "n_kept": sum(d.get("decision") == "keep" for d in review_decisions),
                "n_rejected": sum(d.get("decision") == "reject" for d in review_decisions),
                "decisions": review_decisions,
            }
        referenced = getattr(diag, "referenced_charts", None)
        if referenced:
            entry["referenced_charts"] = list(referenced)
        if getattr(diag, "explore_context_used", False):
            entry["explore_context_used"] = True
        if getattr(diag, "failure_modes_used", False):
            entry["failure_modes_used"] = True
        if explore_figures:
            entry["explore_figures"] = list(explore_figures)
        m3_prompt = getattr(diag, "prompt", None) or ""
        if m3_prompt:
            entry["judge_prompt"] = m3_prompt
        review_prompt = getattr(diag, "review_prompt", None) or ""
        review_raw = getattr(diag, "review_raw", None) or ""
        if review_prompt or review_raw:
            entry["review_prompt"] = review_prompt
            entry["review_response"] = review_raw
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_stage("M3", "diagnosis", entry)
        detailed_calls = list(getattr(diag, "model_calls", None) or [])
        if detailed_calls:
            for call in detailed_calls:
                self.log_model_exchange(
                    "M3",
                    role=str(call.get("role") or "diagnosis_judge"),
                    operation=str(call.get("operation") or "generate"),
                    inputs=call.get("inputs"), output=call.get("output"),
                    error=call.get("error"), duration_sec=call.get("duration_sec"),
                    metadata=dict(call.get("metadata") or {}), cycle=cycle,
                )
        else:
            # Compatibility for DiagnosisResult values created by older callers.
            if m3_prompt or diag.raw_judge_output:
                self.log_model_exchange(
                    "M3", role="diagnosis_judge", operation="generate",
                    inputs=m3_prompt, output=diag.raw_judge_output or "", cycle=cycle,
                    duration_sec=duration_sec,
                )
            if critic_raw or review_prompt or review_raw:
                self.log_model_exchange(
                    "M3", role="hypothesis_critic", operation="generate",
                    inputs=entry.get("critic_prompt") or review_prompt,
                    output=critic_raw or review_raw, cycle=cycle,
                )

        m3_span = self.tracer.start_span(
            name=f"M3: Root-Cause Diagnosis (Cycle {cycle})", stage="M3",
            input_data={"model_name": diag.model_name, "n_hypotheses": len(diag.hypotheses)},
            metadata={"duration_sec": duration_sec},
        )
        self.tracer.log_generation(
            name="AI Doctor Diagnostician",
            model=str(self.tracer.trace_metadata.get("judge") or "diagnosis_judge"),
            prompt=m3_prompt, completion=diag.raw_judge_output or "", span_id=m3_span,
            metadata={"hypotheses": [h.statement for h in diag.hypotheses]},
        )
        if review_prompt or review_raw:
            self.tracer.log_generation(
                name="M3 Adversarial Evidence Review",
                model=str(self.tracer.trace_metadata.get("judge") or "diagnosis_judge"),
                prompt=review_prompt, completion=review_raw, span_id=m3_span,
                metadata={"decisions": review_decisions},
            )
        self.tracer.end_span(m3_span, output_data={
            "n_hypotheses": len(diag.hypotheses),
            "n_proposed": len(proposed) or len(diag.hypotheses),
            "review": entry.get("review"),
        })

    # ------------------------------------------------------------------
    # M4/M5 — hypothesis verification & intervention
    # ------------------------------------------------------------------

    def log_surgery(
        self, cycle: int, hypothesis: "Hypothesis", iv: "InterventionResult", *,
        validation_cases: "Any | None" = None, duration_sec: "float | None" = None,
        judge_prompt: "str | None" = None, judge_raw: "str | None" = None,
    ) -> None:
        """M4 (hypothesis verification) or M5 (intervention) — split exactly as
        RunLogger does: "m4_test_name" present in ``iv.evidence`` means M4."""
        is_m4 = "m4_test_name" in (iv.evidence or {})
        stage = "M4" if is_m4 else "M5"
        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle,
            "hypothesis": hypothesis.statement,
            "failure_mode": hypothesis.predicted_failure_mode,
            "status": iv.status.value, "fixed": iv.fixed,
            "confidence_score": iv.confidence_score,
            "evidence_dimensions": iv.evidence_dimensions,
            "evidence": iv.evidence,
            "n_refocused_cases": len(iv.new_data) if iv.new_data else None,
        }
        if is_m4:
            candidates = _iter_cases(validation_cases)
            if candidates:
                snapshots = [_case_snapshot(case) for case in candidates]
                snapshot = next(
                    (item for item in snapshots if str(item.get("outcome", "")).lower() == "fail"),
                    snapshots[0],
                )
                entry["validation_examples"] = [{
                    "id": f"m4-{snapshot.get('id')}", "kind": "validation_case",
                    "case_id": snapshot.get("id"), **snapshot,
                    "plain_reading": "This is one case in the independent validation pool. The verdict is determined from the full pool, not this case alone.",
                    "evidence_scope": "one case in the independent validation pool",
                }]
        if judge_prompt:
            entry["judge_prompt"] = judge_prompt
        if judge_raw:
            entry["judge_response"] = judge_raw
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_stage(stage, "surgery", entry)
        if judge_prompt or judge_raw:
            self.log_model_exchange(
                stage, role="protocol_consistency_judge", operation="generate",
                inputs=judge_prompt or "", output=judge_raw or "", cycle=cycle,
                duration_sec=duration_sec,
            )

        stage_title = "M4 Adjudication" if is_m4 else "M5 Intervention"
        surg_span = self.tracer.start_span(
            name=f"{stage_title}: {hypothesis.statement[:60]}",
            stage="M4" if is_m4 else "M5_SURGERY",
            input_data={"hypothesis": hypothesis.statement, "failure_mode": hypothesis.predicted_failure_mode},
            metadata={"status": iv.status.value, "fixed": iv.fixed, "confidence_score": iv.confidence_score},
        )
        if judge_prompt or judge_raw:
            self.tracer.log_generation(
                name=f"{stage_title}: Protocol Consistency Judge", model="judge",
                prompt=judge_prompt or "", completion=judge_raw or "", span_id=surg_span,
            )
        if iv.confidence_score is not None:
            self.tracer.log_score(
                name="adjudication_confidence", value=float(iv.confidence_score),
                comment=f"status={iv.status.value}, fixed={iv.fixed}", span_id=surg_span,
            )
        self.tracer.end_span(surg_span, output_data={"evidence": iv.evidence or {}})

    def log_experiment(
        self, cycle: int, hypothesis: "Hypothesis", iv: "InterventionResult", *,
        module: str = "m5",
    ) -> None:
        """The experiment the agent wrote and ran to test *hypothesis*.

        Everything ``RunLogger`` writes to ``experiments/``/``workspace/`` as
        separate files is inlined here as JSON string values instead: the
        generated source (``exp["files"]``/``exp["code"]``), stdout/stderr,
        the coder agent's raw narration, and the validation log are already
        plain strings in *iv.experiment* — RunLogger's only job with them was
        deciding a filename; here they go straight into the entry.
        """
        exp = getattr(iv, "experiment", None) or {}
        files = exp.get("files") or {}
        if not files and exp.get("code"):
            files = {"main.py": exp["code"]}

        # An "experiment" is always M4/M5-shaped content by definition, so it
        # gets the same resolve-with-M5-floor treatment as its workspace
        # media, rather than trusting `_append_stage`'s generic unroutable
        # fallback (which would file a genuinely M5-ish record under
        # run.json["unrouted"] if *module* is ever something the M1-M5 regex
        # and alias table don't recognize).
        stage = _resolve_stage(module) or "M5"

        workspace = None
        workdir = exp.get("workdir")
        if workdir:
            workspace = _inline_workspace(
                workdir, self._stage_artifacts_dir(stage), run_dir=self.run_dir,
            )

        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cycle, "module": module,
            "hypothesis": hypothesis.statement,
            "failure_mode": hypothesis.predicted_failure_mode,
            "status": iv.status.value if iv.status else None,
            "fixed": iv.fixed,
            "provider": exp.get("provider"), "verdict": exp.get("verdict"),
            "metrics": exp.get("metrics"), "returncode": exp.get("returncode"),
            "timed_out": exp.get("timed_out"), "cli_usage": exp.get("cli_usage"),
            "llm_calls": exp.get("llm_calls"), "sandbox_runs": exp.get("sandbox_runs"),
            "code": files,
            "stdout": exp.get("stdout"), "stderr": exp.get("stderr"),
            "blueprint": exp.get("blueprint"), "cli_raw_output": exp.get("cli_raw_output"),
            "validation_log": list(exp.get("validation_log") or []) or None,
            "workspace_snapshot": workspace,
            "trial_root": exp.get("trial_root"),
        }
        self._append_stage(stage, "experiment", entry)
        for call in exp.get("model_calls") or []:
            if isinstance(call, dict):
                self.log_model_exchange(
                    stage,
                    role=str(call.get("role") or "experiment_writer"),
                    operation=str(call.get("operation") or "generate"),
                    inputs=call.get("inputs"), output=call.get("output"),
                    error=call.get("error"), duration_sec=call.get("duration_sec"),
                    metadata=call.get("metadata"),
                )

        exp_span = self.tracer.start_span(
            name=f"{stage}: Experiment — {hypothesis.statement[:60]}",
            stage=f"{stage}_EXPERIMENT",
            input_data={"hypothesis": hypothesis.statement, "failure_mode": hypothesis.predicted_failure_mode},
            metadata={"provider": exp.get("provider"), "returncode": exp.get("returncode")},
        )
        cli_raw = exp.get("cli_raw_output")
        if cli_raw:
            self.tracer.log_generation(
                name=f"{module.upper()} Coder Agent", model=str(exp.get("provider") or "coder_agent"),
                prompt=None, completion=str(cli_raw), span_id=exp_span,
            )
        vlog = exp.get("validation_log")
        if vlog:
            self.tracer.log_generation(
                name=f"{module.upper()} Validation Log", model=str(exp.get("provider") or "coder_agent"),
                prompt=None, completion="\n".join(str(x) for x in vlog), span_id=exp_span,
            )
        self.tracer.end_span(exp_span, output_data={
            "status": entry.get("status"), "fixed": entry.get("fixed"), "verdict": entry.get("verdict"),
        })

    def log_fix(self, outcome: "Any") -> None:
        """Post-loop fix module: the tiered repair attempt + recommendation.

        Unlike ``RunLogger.log_fix``, per-case outputs are NOT popped out to a
        sibling ``outputs.jsonl`` — they stay inline in ``M5/log.json`` under
        each candidate's own ``"outputs"`` key. Fewer files was the whole
        point; a bulkier single JSON is the intended trade for that.
        """
        d = outcome.to_dict()
        best_ref = d.get("best")
        if isinstance(best_ref, dict):
            best = best_ref
        elif isinstance(best_ref, str):
            best = next((a for a in d.get("attempted") or [] if a.get("name") == best_ref), {})
        else:
            best = {}
        entry: dict[str, Any] = {"ts": self._ts(), "cycle": -1}
        entry.update(d)
        entry["best"] = best
        self._append_stage("M5", "fix", entry)

        fix_span = self.tracer.start_span(
            name="M5: Targeted Repair & Confirmation", stage="M5_FIX",
            input_data={"candidates_evaluated": len(d.get("attempted", []))},
        )
        if best.get("effect") is not None:
            self.tracer.log_score(
                name="repair_net_accuracy_gain", value=float(best.get("effect", 0.0)),
                comment=f"cured={best.get('n_fixed')}, broken={best.get('n_broken')}",
                span_id=fix_span,
            )
        if (best.get("payload") or {}).get("prompt_template"):
            self.tracer.log_generation(
                name="Winning Repair Patch", model="evalrx_repair",
                prompt="Repair Candidate Search",
                completion=best["payload"]["prompt_template"], span_id=fix_span,
            )
        self.tracer.end_span(fix_span, output_data={"selected": best.get("name")})

    # ------------------------------------------------------------------
    # AgenticDiagnoseLoop dispatch layer — run-level, not stage content
    # ------------------------------------------------------------------

    def log_agent_decision(
        self, step: int, *, action: str, params: "dict[str, Any] | None" = None,
        rationale: str = "", valid: bool = True, repair_attempts: int = 0,
        fallback_used: bool = False, judge_prompt: "str | None" = None,
        judge_raw: "str | None" = None, duration_sec: "float | None" = None,
        judge_calls: "list[dict[str, Any]] | None" = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": self._ts(), "step": step, "action": action, "params": params or {},
            "rationale": rationale, "valid": valid, "repair_attempts": repair_attempts,
            "fallback_used": fallback_used,
        }
        if judge_prompt:
            entry["judge_prompt"] = judge_prompt
        if judge_raw:
            entry["judge_response"] = judge_raw
        if judge_calls:
            entry["model_calls"] = json.loads(json.dumps(judge_calls, default=str))
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_run("agent_decisions", entry)

        decision_span = self.tracer.start_span(
            name=f"Agent Decision: step {step}", stage="AGENT_DECISION",
            input_data={"action": action, "params": params or {}},
            metadata={"valid": valid, "repair_attempts": repair_attempts, "fallback_used": fallback_used},
        )
        if judge_prompt or judge_raw:
            self.tracer.log_generation(
                name="Agent Decision Judge", model="judge",
                prompt=judge_prompt or "", completion=judge_raw or "", span_id=decision_span,
            )
        self.tracer.end_span(decision_span, output_data={"action": action, "rationale": rationale})

    def log_agent_tool(
        self, step: int, *, tool: str, ok: bool, summary: str = "",
        error: "str | None" = None, duration_sec: "float | None" = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": self._ts(), "step": step, "tool": tool, "ok": ok, "summary": summary,
        }
        if error is not None:
            entry["error"] = error
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._append_run("agent_tool_calls", entry)

    def log_stage_skipped(self, stage: str, reason_code: str, *, cycle: int = -1, detail: str = "") -> None:
        entry = {"ts": self._ts(), "stage": stage, "cycle": cycle, "reason_code": reason_code, "detail": detail}
        self._append_stage(stage, "stage_skipped", entry)
        span = self.tracer.start_span(
            name=f"{stage}: skipped", stage=stage,
            input_data={"reason_code": reason_code}, metadata={"detail": detail},
        )
        self.tracer.end_span(span, output_data={"reason_code": reason_code}, status="skipped")

    def log_loop_end(
        self, report: "AutoDiagnoseReport", *,
        tokens_used: "int | None" = None, timings: "dict[str, float] | None" = None,
    ) -> None:
        entry: dict[str, Any] = {"ts": self._ts(), "cycles": report.cycles}
        if tokens_used is not None:
            entry["tokens_used"] = tokens_used
        if timings:
            entry["timings_sec"] = {k: round(v, 3) for k, v in timings.items()}
            entry["total_duration_sec"] = round(sum(timings.values()), 3)
        if hasattr(report, "resolved"):
            entry["resolved"] = report.resolved
            hyps = getattr(report, "final_hypotheses", [])
            entry["n_hypotheses"] = len(hyps)
            entry["final_hypotheses"] = [
                {"statement": h.statement, "plain_statement": h.plain_statement,
                 "failure_mode": h.predicted_failure_mode, "status": h.status.value if h.status else None}
                for h in hyps
            ]
        if hasattr(report, "stopped_by"):
            entry["stopped_by"] = report.stopped_by
            all_hyps = getattr(report, "all_hypotheses", [])
            verified = getattr(report, "verified_hypotheses", [])
            entry["n_hypotheses"] = len(all_hyps)
            entry["n_verified"] = len(verified)
            entry["verified_hypotheses"] = [
                {"statement": tr.hypothesis.statement, "verdict": getattr(tr, "verdict", None)}
                for tr in verified
            ]
        self._append_run("loop_end", entry)

    # ------------------------------------------------------------------
    # Tool synthesis (M1/M2 probes+stats tools, generated on demand)
    # ------------------------------------------------------------------

    def log_tool_codegen(
        self, *, module: str, name: str, need: str, source: str, ok: bool,
        code: str = "", prompt: str = "", raw_output: str = "", raw_stream: str = "",
        error: str = "", stdout: str = "", cycle: "int | None" = None,
        extra: "dict[str, Any] | None" = None,
    ) -> None:
        cyc = self.current_cycle if cycle is None else cycle
        entry: dict[str, Any] = {
            "ts": self._ts(), "cycle": cyc, "module": module, "tool_name": name,
            "need": need, "source": source, "ok": ok, "error": error or None,
            "code": code or None, "prompt": prompt or None,
            "raw_output": raw_output or None, "raw_stream": raw_stream or None,
            "stdout": stdout or None,
        }
        if extra:
            entry.update(extra)
        self._append_stage(module, "tool_codegen", entry)
        if prompt or raw_stream or raw_output:
            self.log_model_exchange(
                _resolve_stage(module) or "M5", role="tool_codegen",
                operation=source, inputs=prompt or "",
                output=raw_stream or raw_output or code,
                error=error or None, cycle=cyc,
                metadata={"tool_name": name, "ok": ok},
            )

        cg_span = self.tracer.start_span(
            name=f"Tool Codegen: {module}/{name}", stage=f"CODEGEN_{module}",
            input_data={"need": need, "source": source},
            metadata={"ok": ok, "error": error or None},
        )
        if prompt or raw_output:
            self.tracer.log_generation(
                name=f"Tool Synthesis: {name}", model=source,
                prompt=prompt or "", completion=raw_stream or raw_output or code, span_id=cg_span,
            )
        self.tracer.end_span(cg_span, output_data={"ok": ok, "code_chars": len(code or "")})

    def log_tool_registry(self, cycle: int, module: str, generated: "list[Any]") -> None:
        if not generated:
            return
        tools = [
            {
                "name": getattr(g, "name", "tool"), "need": getattr(g, "need", ""),
                "source": getattr(g, "source", ""), "code": getattr(g, "code", "") or None,
            }
            for g in generated
        ]
        self._append_stage(module, "tool_registry", {
            "ts": self._ts(), "cycle": cycle, "module": module, "n_tools": len(tools), "tools": tools,
        })

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush every doc one last time and persist the Langfuse trace bundle."""
        if self._closed:
            return
        with self._lock:
            self._flush_run()
            for stage in _STAGES:
                self._flush_stage(stage)
        self.tracer.end_trace({
            "spans": len(self.tracer.spans),
            "generations": len(self.tracer.generations),
            "scores": len(self.tracer.scores),
        })
        self.tracer.flush()
        try:
            self.tracer.export_bundle(self.run_dir / "langfuse_trace.json")
        except Exception:  # noqa: BLE001
            pass
        # Offline runs need no retry queue once the complete JSON trace bundle
        # has been exported.  A live run keeps a non-empty queue for retry.
        if self.tracer.mode == "offline" or self.tracer.outbox.pending_count() == 0:
            try:
                self._outbox_path.unlink(missing_ok=True)
                self._outbox_path.parent.rmdir()
            except OSError:
                pass
        self._closed = True

    def __enter__(self) -> "RunLoggerV2":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
