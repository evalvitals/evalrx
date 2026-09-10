"""Explicit single-file export using the same React/json-render renderer."""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from evalrx.reporting.dynamic import (
    load_published_report,
    publish_report,
    report_is_current,
)
from evalrx.reporting.server import _resolve_media, _resolve_report_root

EmbedMedia = Literal["representative", "all", "none"]


def export_static_report(
    run_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    example_dir: str | Path | None = None,
    embed_media: EmbedMedia = "representative",
    model: object | None = None,
    audio_bitrate: str | None = "48k",
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
    _embed_media(root, data, mode=embed_media, audio_bitrate=audio_bitrate)
    _embed_stage_figures(root, data)
    template = Path(__file__).with_name("web_dist") / "index.html"
    if not template.exists():
        raise FileNotFoundError("Packaged report renderer is missing; build evalrx/reporting/web first.")
    payload = json.dumps({"data": data, "layout": layout}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    bootstrap = f"<script>window.__EVALRX_REPORT__={payload};</script>"
    html = template.read_text(encoding="utf-8").replace("</head>", f"{bootstrap}</head>", 1)
    destination = Path(out_path).resolve() if out_path else root / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


#: Lossless audio the runs record (16 kHz mono WAV, ~380 KB per 10 s clip).
#: Inlined as-is, an audio run's every-case export lands well past GitHub's
#: 100 MB per-file limit (mmau/demo192: 256 clips, 97 MB, ~130 MB in base64);
#: at 48 kbps mono MP3 the same clips fit in ~20 MB and play in every browser.
_TRANSCODE_SUFFIXES = (".wav", ".flac", ".aiff", ".aif")


def _transcode_audio(path: Path, bitrate: str, workdir: Path) -> "tuple[Path, str] | None":
    """MP3 rendition of an audio file, or None when ffmpeg is unavailable/fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    out = workdir / (path.stem + ".mp3")
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1",
           "-codec:a", "libmp3lame", "-b:a", bitrate, str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return None
    return (out, "audio/mpeg") if out.is_file() and out.stat().st_size > 0 else None


def _embed_media(root: Path, data: dict, *, mode: EmbedMedia,
                 audio_bitrate: str | None = "48k") -> None:
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
    with tempfile.TemporaryDirectory(prefix="evalrx_audio_") as tmp:
        workdir = Path(tmp)
        for item in data.get("media", []):
            if str(item.get("id")) not in allowed:
                continue
            path = _resolve_media(root, str(item.get("path") or ""))
            if path is None:
                continue
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if audio_bitrate and path.suffix.lower() in _TRANSCODE_SUFFIXES:
                rendition = _transcode_audio(path, audio_bitrate, workdir)
                if rendition is not None:
                    path, mime = rendition
            item["data_uri"] = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


_FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")


def _embed_stage_figures(root: Path, data: dict) -> None:
    """Keep every cited stage figure visible in a portable single-file report.

    Figure references are not confined to one list: M2 publishes
    ``stage_detail.m2.figures``, M3 cites some of the same files again as
    ``stage_detail.m3.evidence_figures``, and each entry is its own dict even
    when the underlying file is shared. Walk the whole stage_detail tree and
    embed anything that looks like a figure, caching by path so a file shared
    across stages is read (and stored) once.
    """
    cache: dict[str, str] = {}

    def embed(node: object) -> None:
        if isinstance(node, dict):
            raw = node.get("path")
            if isinstance(raw, str) and raw.lower().endswith(_FIGURE_SUFFIXES) \
                    and not node.get("data_uri"):
                if raw not in cache:
                    path = _resolve_media(root, raw)
                    if path is not None:
                        mime = mimetypes.guess_type(path.name)[0] or "image/png"
                        cache[raw] = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
                if raw in cache:
                    node["data_uri"] = cache[raw]
            for value in node.values():
                embed(value)
        elif isinstance(node, list):
            for value in node:
                embed(value)

    embed(data.get("stage_detail") or {})
