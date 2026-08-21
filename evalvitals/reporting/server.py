"""Local FastAPI application for dynamic completed-run reports."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from evalvitals.reporting.dynamic import (
    CATALOG_VERSION,
    JSON_RENDER_VERSION,
    REPORT_SCHEMA_VERSION,
    build_report_data,
    fallback_spec,
    load_published_report,
    publish_report,
    report_is_current,
)


def create_app(run_dir: str | Path, *, frontend_dir: str | Path | None = None) -> Any:
    """Create the loopback report API and serve the packaged React application."""
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise ImportError(
            "The dynamic UI needs `pip install evalvitals[ui]`."
        ) from exc

    root = _resolve_report_root(Path(run_dir).resolve())
    try:
        if not report_is_current(root):
            publish_report(root)
        data, envelope = load_published_report(root)
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
    media_by_id = {str(item.get("id")): item for item in data.get("media", [])}
    app = FastAPI(title="EvalVitals Report", docs_url="/api/docs", redoc_url=None)

    @app.get("/api/report")
    def get_report() -> dict[str, Any]:
        return {"data": data, "layout": envelope}

    @app.get("/api/stages/{stage_id}")
    def get_stage(stage_id: str) -> dict[str, Any]:
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
        rows = list(data.get("cases", []))
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
        case = next((item for item in data.get("cases", []) if str(item.get("id")) == case_id), None)
        if case is None:
            raise HTTPException(status_code=404, detail="Unknown case")
        return case

    @app.get("/api/debug")
    def get_debug(
        event: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = list(data.get("debug", {}).get("events", []))
        if event:
            rows = [row for row in rows if row.get("event") == event]
        return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset:offset + limit]}

    @app.get("/api/media/{media_id}")
    def get_media(media_id: str) -> Any:
        item = media_by_id.get(media_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Unknown media")
        # A media id is selected from the report's fixed evidence index, not a
        # caller-provided path.  It may safely resolve into the example's
        # adjacent data directory (where VLM images / ALLM audio conventionally
        # live), while the generic artifact endpoint remains run-confined.
        path = _resolve_indexed_media(root, str(item.get("path") or ""))
        if path is None:
            raise HTTPException(status_code=404, detail="Media is not available in the local cache")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    @app.get("/api/artifact")
    def get_artifact(path: str = Query(...)) -> Any:
        """Serve a report-referenced local figure, constrained to this run."""
        resolved = _resolve_media(root, path)
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
            return {"message": "API ready; build evalvitals/reporting/web to install the UI."}

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
    nested_logs = requested_root / "logs"
    if nested_logs.is_dir() and (
        (nested_logs / "run_log.jsonl").is_file()
        or (nested_logs / "report" / "report_data.json").is_file()
    ):
        return nested_logs
    return requested_root


def serve_dynamic_report(
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
) -> int:
    """Publish if necessary and run the completed-report UI on loopback."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("The dynamic UI needs `pip install evalvitals[ui]`.") from exc
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    print(f"Serving dynamic diagnostic report at http://{host}:{port}")
    uvicorn.run(create_app(run_dir), host=host, port=port, log_level="warning")
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
