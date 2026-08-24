"""Explicit single-file export using the same React/json-render renderer."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Literal

from evalvitals.reporting.dynamic import (
    load_published_report,
    publish_report,
    report_is_current,
)
from evalvitals.reporting.server import _resolve_media, _resolve_report_root

EmbedMedia = Literal["representative", "all", "none"]


def export_static_report(
    run_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    example_dir: str | Path | None = None,
    embed_media: EmbedMedia = "representative",
    model: object | None = None,
) -> Path:
    """Write a portable HTML snapshot without introducing another renderer."""
    # `serve` accepts either a run directory or its logs/ child and resolves
    # between them; the export did not, so the same path produced two different
    # reports — the export compiled from the enclosing directory and missed
    # everything keyed to the log dir (the run manifest, and with it the agent
    # that drove the run). One resolver for both entry points.
    root = _resolve_report_root(Path(run_dir).resolve())
    if model is not None or not report_is_current(root):
        publish_report(root, example_dir=example_dir, model=model)
    data, layout = load_published_report(root)
    _embed_media(root, data, mode=embed_media)
    _embed_stage_figures(root, data)
    template = Path(__file__).with_name("web_dist") / "index.html"
    if not template.exists():
        raise FileNotFoundError("Packaged report renderer is missing; build evalvitals/reporting/web first.")
    payload = json.dumps({"data": data, "layout": layout}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    bootstrap = f"<script>window.__EVALVITALS_REPORT__={payload};</script>"
    html = template.read_text(encoding="utf-8").replace("</head>", f"{bootstrap}</head>", 1)
    destination = Path(out_path).resolve() if out_path else root / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


def _embed_media(root: Path, data: dict, *, mode: EmbedMedia) -> None:
    if mode == "none":
        return
    allowed: set[str]
    if mode == "all":
        allowed = {str(item.get("id")) for item in data.get("media", [])}
    else:
        allowed = {
            str(media_id)
            for case in data.get("cases", [])[:4]
            for media_id in case.get("media_ids", [])
        }
    for item in data.get("media", []):
        if str(item.get("id")) not in allowed:
            continue
        path = _resolve_media(root, str(item.get("path") or ""))
        if path is None:
            continue
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        item["data_uri"] = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _embed_stage_figures(root: Path, data: dict) -> None:
    """Keep cited M2 evidence visible in a portable single-file report."""
    details = data.get("stage_detail") or {}
    figures = (details.get("m2") or {}).get("figures") or []
    for figure in figures:
        path = _resolve_media(root, str(figure.get("path") or ""))
        if path is None:
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        figure["data_uri"] = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
