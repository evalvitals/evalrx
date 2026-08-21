"""Local FastAPI application for dynamic completed-run reports."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from evalvitals.reporting.dynamic import load_published_report, publish_report, report_is_current


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

    root = Path(run_dir).resolve()
    if not report_is_current(root):
        publish_report(root)
    data, envelope = load_published_report(root)
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
        path = _resolve_media(root, str(item.get("path") or ""))
        if path is None:
            raise HTTPException(status_code=404, detail="Media is not available in the local cache")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    packaged = Path(frontend_dir).resolve() if frontend_dir else Path(__file__).with_name("web_dist")
    if packaged.is_dir() and (packaged / "index.html").exists():
        assets = packaged / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> Any:
            candidate = (packaged / path).resolve()
            if packaged in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(packaged / "index.html")
    else:
        @app.get("/")
        def missing_frontend() -> dict[str, str]:
            return {"message": "API ready; build evalvitals/reporting/web to install the UI."}

    return app


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
