"""Style-token loading for deterministic host-rendered figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalvitals.agent_assets.skills import BUNDLED_SKILLS_DIR

# Mirrored fallback for when package data is unavailable. The live values are
# read from the bundled nature-figure skill when present.
NATURE_RC_FALLBACK: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}
# Mirrors the bundled nature-figure skill's DEFAULT_COLORS, whose values are
# synced to the CVD-validated dataviz palette (see that skill's README notice)
# — one palette across agent PNGs, host plotly charts, and these spec PNGs.
NATURE_COLORS_FALLBACK = ["#2a78d6", "#008300", "#e34948", "#1baf7a", "#4a3aa7", "#898781"]
SCREEN_RC: dict[str, Any] = {
    "font.size": 9.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
}

NATURE_SKILL_DIR = BUNDLED_SKILLS_DIR / "nature-figure"
_NATURE_STYLE_CACHE: dict[str, Any] | None = None


def _balanced(text: str, open_idx: int, open_ch: str, close_ch: str) -> str | None:
    """Return the balanced ``open_ch..close_ch`` group starting at *open_idx*."""
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_idx : j + 1]
    return None


def _literal_after(text: str, marker: str, open_ch: str, close_ch: str):
    """ast.literal_eval the first balanced literal after *marker* (or None)."""
    import ast

    k = text.find(marker)
    if k < 0:
        return None
    o = text.find(open_ch, k)
    if o < 0:
        return None
    blob = _balanced(text, o, open_ch, close_ch)
    if blob is None:
        return None
    try:
        return ast.literal_eval(blob)
    except Exception:
        return None


def load_nature_style(skill_dir: str | Path | None = None) -> dict[str, Any]:
    """Load nature-figure rcParams and palette from bundled skill assets.

    Returns a cached ``{"rc": ..., "colors": ...}`` dict. Never raises; falls
    back to mirrored tokens if bundled files are absent or unparsable.
    """
    global _NATURE_STYLE_CACHE
    if skill_dir is None and _NATURE_STYLE_CACHE is not None:
        return _NATURE_STYLE_CACHE

    rc = dict(NATURE_RC_FALLBACK)
    colors = list(NATURE_COLORS_FALLBACK)
    root = Path(skill_dir) if skill_dir is not None else NATURE_SKILL_DIR
    try:
        import re

        py = (root / "static" / "fragments" / "backend" / "python.md").read_text(
            encoding="utf-8"
        )
        loaded_rc = _literal_after(py, "rcParams.update(", "{", "}")
        if isinstance(loaded_rc, dict):
            rc = {str(k): v for k, v in loaded_rc.items() if k != "font.size"}

        api = (root / "references" / "api.md").read_text(encoding="utf-8")
        palette = _literal_after(api, "PALETTE = {", "{", "}")
        m = api.find("DEFAULT_COLORS")
        if isinstance(palette, dict) and m >= 0:
            blob = _balanced(api, api.find("[", m), "[", "]") or ""
            keys = re.findall(r'PALETTE\["([^"]+)"\]', blob)
            resolved = [palette[k] for k in keys if k in palette]
            if resolved:
                colors = resolved
    except Exception:
        pass

    style = {"rc": {**rc, **SCREEN_RC}, "colors": colors}
    if skill_dir is None:
        _NATURE_STYLE_CACHE = style
    return style

