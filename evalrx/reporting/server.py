"""Local FastAPI application for dynamic completed-run reports."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalrx.reporting.dynamic import (
    CATALOG_VERSION,
    JSON_RENDER_VERSION,
    REPORT_SCHEMA_VERSION,
    build_report_data,
    fallback_spec,
    load_published_report,
    publish_report,
    report_is_current,
)

try:  # FastAPI stays optional, but a route's annotations must resolve against
    # module globals — a function-local import of UploadFile would not be seen.
    from fastapi import File, UploadFile
except ImportError:  # pragma: no cover - create_app reports the missing extra
    File = UploadFile = None  # type: ignore[assignment]

# A dropped archive is a whole run directory, which for a media-carrying
# benchmark is mostly audio/video. The workbench default (1 GiB) is the same
# ceiling the upload app used, and it stays generous enough for those.
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_JUNK_DIRS = {"__MACOSX", ".git", "node_modules", ".venv"}


class ReportSession:
    """The run currently on screen, and where dropped archives are unpacked.

    ``serve`` starts either on a run directory or on nothing at all, and an
    uploaded archive replaces whatever is loaded. Holding that in one object
    lets every endpoint read the current run instead of closing over the one
    that happened to exist at startup.
    """

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.root: Path | None = None
        self.data: dict[str, Any] = {}
        self.envelope: dict[str, Any] = {}
        self.media_by_id: dict[str, Any] = {}
        self.label: str = ""
        self._workspace: Path | None = None
        if run_dir is not None:
            self.load(run_dir)

    @property
    def loaded(self) -> bool:
        return self.root is not None

    def load(self, run_dir: str | Path) -> None:
        root = _resolve_report_root(Path(run_dir).resolve())
        data, envelope = _compile_report(root)
        self.root = root
        self.data = data
        self.envelope = envelope
        self.media_by_id = {str(item.get("id")): item for item in data.get("media", [])}
        self.label = _run_label(root)

    @property
    def payload(self) -> dict[str, Any]:
        return {"data": self.data, "layout": self.envelope}

    def ingest_archive(self, payload: bytes, *, filename: str = "run.zip") -> Path:
        """Unpack a dropped run archive and make it the report on screen."""
        from evalrx.analysis.workbench import UploadLimits, extract_archive

        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="evalrx-report-"))
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).stem).strip("-") or "run"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extracted = extract_archive(
            payload,
            self._workspace / f"{stem[:60]}-{stamp}",
            limits=UploadLimits(max_archive_bytes=MAX_ARCHIVE_BYTES),
        )
        found = find_run_root(extracted)
        if found is None:
            raise ValueError(
                "this archive has no run in it: expected a directory holding "
                "run_log.jsonl or report/report_data.json"
            )
        self.load(found)
        return found


def _run_label(root: Path) -> str:
    # "logs" alone names nothing; a run written to <run>/logs is better
    # identified by the directory the reader actually zipped up.
    return f"{root.parent.name}/{root.name}" if root.name == "logs" else root.name


# Directories a run scan never descends into: version control and dependency
# noise, plus a found run's own internals (raw case media, the coder's
# sandbox, cached transcodes) — none of those hold a *further* run, and
# `data/` in particular can be hundreds of megabytes of dataset files.
_SCAN_SKIP_DIRS = _JUNK_DIRS | {"data", "sandbox", ".media_cache"}

#: Runs panel display cap, applied after sorting by recency (see /api/runs) —
#: distinct from discover_runs' own max_results, which is a much larger
#: traversal safety valve, not a display limit.
RUNS_DISPLAY_LIMIT = 200


def discover_runs(scan_root: Path, *, max_depth: int = 8, max_results: int = 2000) -> list[Path]:
    """Find run directories under ``scan_root``, without walking into any of them.

    A run directory is one holding ``run_log.jsonl`` directly at its root —
    the layout both the legacy logger and V2 ``RunContext`` write (matching
    ``find_run_root``'s rank-0 marker). Once a run is found its own subtree
    (data, sandbox, report, ...) is not searched further, and directories
    named `outputs*`/`logs` are walked through since a run commonly sits a
    few levels under an example's output directory.

    ``max_results`` is a safety valve for a pathological tree, not the panel's
    display limit — hitting it means the *walk* stops, in whatever order
    ``os.walk`` reached matches, so a caller that wants "most recent N" must
    sort the full result and slice afterward rather than rely on this cutoff
    to have kept the newest ones.
    """
    scan_root = scan_root.resolve()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(scan_root):
        current = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d not in _SCAN_SKIP_DIRS and not d.startswith("."))
        if "run_log.jsonl" in filenames:
            found.append(current)
            dirnames[:] = []  # a run's own subtree holds no further runs
            if len(found) >= max_results:
                break
            continue
        if len(current.relative_to(scan_root).parts) >= max_depth:
            dirnames[:] = []  # too deep to be worth descending further
    return found


def _run_summary(root: Path, run_id: str, scan_root: Path) -> dict[str, Any]:
    """Cheap, read-only metadata for the runs panel — never compiles a report."""
    log_path = root / "run_log.jsonl"
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        mtime = None
    dataset = model = None
    published = False
    data_path = root / "report" / "report_data.json"
    if data_path.is_file():
        published = True
        try:
            setting = json.loads(data_path.read_text(encoding="utf-8")).get("setting") or {}
            dataset, model = setting.get("dataset"), setting.get("model")
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    try:
        rel_path = str(root.relative_to(scan_root))
    except ValueError:
        rel_path = str(root)
    return {
        "id": run_id,
        # Relative to the scan root, not `_run_label`'s "<parent>/logs": two
        # sibling experiments both nest their run under an `outputs/logs`
        # child, and the topbar's one-run label collapses them to identical,
        # unidentifiable rows in a list that shows several at once.
        "path": rel_path,
        # The resolved absolute path, matching what /api/session reports for
        # the run currently on screen — how the panel knows which entry to
        # highlight as active.
        "root": str(root),
        "dataset": dataset,
        "model": model,
        "published": published,
        "modified_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None,
    }


def find_run_root(base: Path) -> Path | None:
    """Locate the run inside an unpacked archive, shallowest candidate first.

    Archives are zipped from wherever the run happened to sit, so the run log
    can be at the top level or several directories down beside its example's
    data. Depth is the tiebreak that matches how people zip: the run they meant
    to share is the one nearest the surface.
    """
    candidates: list[tuple[int, int, Path]] = []
    for rank, pattern in (
        (0, "run_log.jsonl"), (1, "report/report_data.json"), (2, "run.json"),
    ):
        for hit in base.rglob(pattern.rsplit("/", 1)[-1]):
            if rank == 1 and hit.parent.name != "report":
                continue
            if rank == 0 or rank == 2:
                run_root = hit.parent
            else:
                run_root = hit.parent.parent
            if rank == 2 and not any(
                (run_root / stage / "log.json").is_file()
                for stage in ("M1", "M2", "M3", "M4", "M5")
            ):
                continue
            if any(part in _JUNK_DIRS for part in run_root.relative_to(base).parts):
                continue
            candidates.append((len(run_root.relative_to(base).parts), rank, run_root))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], str(item[2])))[2]


def _compile_report(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the run's report cache when writable, else compile in memory."""
    try:
        if not report_is_current(root):
            publish_report(root)
        return load_published_report(root)
    except PermissionError:
        # Historical/shared runs are often intentionally read-only. Serving a
        # report must not require mutating its evidence directory.
        data = build_report_data(root)
        envelope = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "format": "json-render",
            "json_render_version": JSON_RENDER_VERSION,
            "catalog_version": CATALOG_VERSION,
            "trace_id": data["trace_id"],
            "source_event_seq": data["source_event_seq"],
            "generated_by": {"mode": "deterministic-read-only", "model": None},
            "spec": fallback_spec(data),
        }
        return data, envelope


def create_app(
    run_dir: str | Path | None = None, *, frontend_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> Any:
    """Create the loopback report API and serve the packaged React application.

    ``run_dir`` may be omitted: the app then starts empty and waits for a run
    archive to be dropped on the page.

    ``runs_root`` is where the runs panel looks for sibling experiments to
    list; it defaults to ``run_dir``'s parent (so opening one run in an
    example's ``outputs/`` surfaces the others beside it) or, with no
    ``run_dir``, the current directory.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise ImportError(
            "The dynamic UI needs `pip install evalrx[ui]`."
        ) from exc

    session = ReportSession(run_dir)
    app = FastAPI(title="EvalRX Report", docs_url="/api/docs", redoc_url=None)

    # State for the runs panel: which directory it scans, and the ids handed
    # to the browser for the last listing (an id is a hash of a resolved
    # path, so `open` needs this to map it back — the scan itself is not
    # repeated on open).
    scan_state: dict[str, Path] = {
        "root": Path(runs_root).resolve() if runs_root is not None
        else (Path(run_dir).resolve().parent if run_dir is not None else Path.cwd())
    }
    run_index: dict[str, Path] = {}

    def current() -> ReportSession:
        if not session.loaded:
            raise HTTPException(status_code=404, detail="No run is loaded; drop a run .zip to open one.")
        return session

    @app.get("/api/session")
    def get_session() -> dict[str, Any]:
        return {
            "loaded": session.loaded,
            "label": session.label,
            "trace_id": session.data.get("trace_id") if session.loaded else None,
            "accepts_upload": True,
            # Lets the runs panel highlight the run already on screen (e.g.
            # one passed on the command line) against its own listing, which
            # keys entries by this same resolved path.
            "root": str(session.root) if session.loaded else None,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        """List experiments found on disk under the scan root, most recent first."""
        scan_root = scan_state["root"]
        if not scan_root.is_dir():
            raise HTTPException(status_code=404, detail=f"{scan_root} is not a directory")
        found = discover_runs(scan_root)
        # Sort the full (safety-valve-capped, not display-capped) find before
        # truncating to what the panel actually shows — discover_runs' own
        # cutoff stops the walk in traversal order, which is not recency.
        found.sort(
            key=lambda path: (path / "run_log.jsonl").stat().st_mtime
            if (path / "run_log.jsonl").is_file() else 0,
            reverse=True,
        )
        items = []
        for path in found[:RUNS_DISPLAY_LIMIT]:
            run_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
            run_index[run_id] = path
            items.append(_run_summary(path, run_id, scan_root))
        return {"root": str(scan_root), "items": items, "truncated": len(found) > RUNS_DISPLAY_LIMIT}

    @app.post("/api/runs/{run_id}/open")
    def open_run(run_id: str) -> dict[str, Any]:
        """Load a run found by /api/runs and make it the report on screen.

        Opens by id rather than by a client-supplied path, so the browser can
        only load a run this server itself already found on a prior listing.
        """
        path = run_index.get(run_id)
        if path is None or not path.is_dir():
            raise HTTPException(status_code=404, detail="Unknown run id; refresh the runs list and try again")
        try:
            session.load(path)
        except Exception as exc:  # a run mid-write or otherwise unreadable
            raise HTTPException(status_code=422, detail=f"could not open this run: {exc}") from exc
        return {"label": session.label, **session.payload}

    @app.post("/api/upload")
    async def upload_run(file: UploadFile = File(...)) -> dict[str, Any]:
        """Accept a zipped run directory and swap it in as the current report."""
        payload = await file.read()
        try:
            session.ingest_archive(payload, filename=file.filename or "run.zip")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # a malformed run reads as a bad upload, not a crash
            raise HTTPException(status_code=422, detail=f"could not read this run: {exc}") from exc
        return {"label": session.label, **session.payload}

    @app.get("/api/report")
    def get_report() -> dict[str, Any]:
        return current().payload

    @app.get("/api/stages/{stage_id}")
    def get_stage(stage_id: str) -> dict[str, Any]:
        data = current().data
        stage = next((item for item in data.get("stages", []) if item.get("id") == stage_id.lower()), None)
        if stage is None:
            raise HTTPException(status_code=404, detail="Unknown stage")
        related = [event for event in data.get("debug", {}).get("events", []) if str(event.get("stage", "")).lower() in {stage_id.lower(), stage.get("code", "").lower()}]
        return {"stage": stage, "events": related}

    @app.get("/api/cases")
    def get_cases(
        status: str | None = None,
        task: str | None = None,
        search: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        rows = list(current().data.get("cases", []))
        if status:
            rows = [case for case in rows if str(case.get("status", "")).lower() == status.lower()]
        if task:
            rows = [case for case in rows if str(case.get("task", "")).lower() == task.lower()]
        if search:
            needle = search.lower()
            rows = [case for case in rows if needle in " ".join(str(case.get(key, "")) for key in ("id", "prompt", "expected", "observed", "task")).lower()]
        return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset:offset + limit]}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        case = next((item for item in current().data.get("cases", []) if str(item.get("id")) == case_id), None)
        if case is None:
            raise HTTPException(status_code=404, detail="Unknown case")
        return case

    @app.get("/api/debug")
    def get_debug(
        event: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = list(current().data.get("debug", {}).get("events", []))
        if event:
            rows = [row for row in rows if row.get("event") == event]
        return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset:offset + limit]}

    @app.get("/api/media/{media_id}")
    def get_media(media_id: str) -> Any:
        item = current().media_by_id.get(media_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Unknown media")
        # A media id is selected from the report's fixed evidence index, not a
        # caller-provided path.  It may safely resolve into the example's
        # adjacent data directory (where VLM images / ALLM audio conventionally
        # live), while the generic artifact endpoint remains run-confined.
        path = _resolve_indexed_media(session.root, str(item.get("path") or ""))
        if path is None:
            raise HTTPException(status_code=404, detail="Media is not available in the local cache")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    @app.get("/api/artifact")
    def get_artifact(path: str = Query(...)) -> Any:
        """Serve a report-referenced local figure, constrained to this run."""
        resolved = _resolve_media(current().root, path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Artifact is not available in this run")
        return FileResponse(resolved, media_type=mimetypes.guess_type(resolved.name)[0])

    packaged = Path(frontend_dir).resolve() if frontend_dir else Path(__file__).with_name("web_dist")
    if packaged.is_dir() and (packaged / "index.html").exists():
        assets = packaged / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> Any:
            candidate = (packaged / path).resolve()
            if packaged in candidate.parents and candidate.is_file():
                return FileResponse(candidate, headers={"Cache-Control": "no-store"})
            # The packaged app is a single-file bundle.  Keeping it out of the
            # browser cache makes a restarted local report server immediately
            # pick up new interactive behavior instead of a stale UI shell.
            return FileResponse(packaged / "index.html", headers={"Cache-Control": "no-store"})
    else:
        @app.get("/")
        def missing_frontend() -> dict[str, str]:
            return {"message": "API ready; build evalrx/reporting/web to install the UI."}

    return app


def _resolve_report_root(requested_root: Path) -> Path:
    """Accept either a run-log directory or its enclosing output directory.

    Agentic examples conventionally write their immutable event stream in an
    ``outputs_*/logs`` child.  The CLI takes the enclosing output directory so
    that it also remains convenient for legacy flat runs; prefer the nested
    directory whenever it contains a run log or a published report.
    """
    # An explicit directory with its own event stream always wins.  Some
    # examples retain a later ``logs/`` sub-run beside an earlier successful
    # top-level run; silently preferring it makes the UI show the wrong repair.
    if (requested_root / "run_log.jsonl").is_file():
        return requested_root
    if (requested_root / "run.json").is_file():
        return requested_root
    nested_logs = requested_root / "logs"
    if nested_logs.is_dir() and (
        (nested_logs / "run_log.jsonl").is_file()
        or (nested_logs / "report" / "report_data.json").is_file()
        or (nested_logs / "run.json").is_file()
    ):
        return nested_logs
    return requested_root


def serve_dynamic_report(
    run_dir: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
    runs_root: str | Path | None = None,
) -> int:
    """Publish if necessary and run the completed-report UI on loopback.

    With no ``run_dir`` the server comes up empty and waits for a zipped run to
    be dropped on the page, which is how a run that was produced on another
    machine gets looked at here. ``runs_root`` overrides where the runs panel
    looks for sibling experiments (see ``create_app``).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("The dynamic UI needs `pip install evalrx[ui]`.") from exc
    app = create_app(run_dir, runs_root=runs_root)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    print(f"Serving dynamic diagnostic report at http://{host}:{port}")
    if run_dir is None:
        print("No run loaded yet — drop a zipped run directory on the page to open one.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def _resolve_media(root: Path, value: str) -> Path | None:
    candidate = Path(value)
    allowed_roots = [root, root.parent]
    possibilities = [candidate] if candidate.is_absolute() else [
        root / candidate,
        root.parent / candidate,
        root.parent / "data" / candidate,
    ]
    for possibility in possibilities:
        try:
            resolved = possibility.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        # Absolute case refs are allowed locally only when they belong to the
        # run or its enclosing example. Langfuse caches always materialize under
        # the run root and therefore pass the same check.
        if any(resolved == base or base in resolved.parents for base in allowed_roots):
            return resolved
    return None


def _resolve_indexed_media(root: Path, value: str) -> Path | None:
    """Resolve report-indexed case media without opening arbitrary paths.

    Case assets may live at ``<example>/data`` while a run writes its log to
    ``<example>/outputs*/logs``.  The media index was built from saved case
    records, so this endpoint permits only that nearby data directory in
    addition to the usual run-owned roots.  ``/api/artifact`` deliberately
    does not use this broader resolver.
    """
    resolved = _resolve_media(root, value)
    if resolved is not None:
        return resolved
    candidate = Path(value)
    possibilities = [candidate] if candidate.is_absolute() else [
        root.parent / "data" / candidate,
        root.parent.parent / "data" / candidate,
    ]
    allowed_roots = [root.parent / "data", root.parent.parent / "data"]
    for possibility in possibilities:
        try:
            path = possibility.resolve()
        except OSError:
            continue
        if path.is_file() and any(base.resolve() in path.parents for base in allowed_roots if base.is_dir()):
            return path
    return None
