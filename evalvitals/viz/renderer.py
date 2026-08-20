"""Host-side, deterministic chart rendering for exploratory artifacts.

The explorer (a CLI coding agent) only *proposes* chart specifications — a small
dict ``{name, kind, data, x, y, title}`` where ``data`` points to a CSV table it
wrote. This module renders those specs to PNG **on the host**, from the spec and
the CSV alone: it NEVER executes LLM-authored plotting code. That keeps the
visual layer auditable and reproducible (same spec + same CSV → same figure).

Rendering is best-effort and never raises into a pipeline:
- ``matplotlib`` missing → every spec is returned with a ``render_skipped`` note
  and a free-text ``description``; callers fall back to the text.
- A spec that can't be rendered (missing CSV, unknown columns, empty table) is
  returned annotated, not dropped — discovery output is preserved.

This is the single visualization core shared by every single-shot entry
(``evalvitals explore``, the fused pipeline, and the in-loop M3 view).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from evalvitals.viz.style import NATURE_COLORS_FALLBACK, load_nature_style

logger = logging.getLogger(__name__)

_KINDS = {"bar", "line", "scatter", "timeseries"}


def render_chart_specs(
    charts: list[dict[str, Any]] | None,
    tables_dir: str | Path | None,
    out_dir: str | Path,
) -> list[dict[str, Any]]:
    """Render each chart spec to a PNG under *out_dir*, deterministically.

    Args:
        charts:     Explorer chart specs. Each is ``{name, kind, data, x, y,
                    title}`` where ``data`` is a CSV path (relative to
                    *tables_dir*, or absolute).
        tables_dir: Directory the spec ``data`` CSVs live in (the explorer's
                    ``tables/``). ``None`` resolves CSVs relative to *out_dir*'s
                    parent.
        out_dir:    Directory to write ``figures/`` PNGs into (created lazily).

    Returns:
        A new list of chart dicts, each a shallow copy of the input spec plus:
        - ``figure_path``: absolute PNG path when rendered;
        - ``description``: a one-line textual summary (always set);
        - ``render_skipped``: reason string when no PNG was produced.

        The input list is not mutated; ordering is preserved.
    """
    specs = [dict(c) for c in (charts or []) if isinstance(c, dict)]
    if not specs:
        return specs

    out_dir = Path(out_dir)
    tdir = Path(tables_dir) if tables_dir else None
    plt = _import_matplotlib()
    style = load_nature_style()

    rendered: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        rows, load_err = _load_table(spec.get("data"), tdir, out_dir)
        x = spec.get("x")
        y = spec.get("y")
        spec.setdefault("description", _describe(spec, rows, x, y))

        if plt is None:
            spec["render_skipped"] = "matplotlib not installed (pip install 'evalvitals[viz]')"
            rendered.append(spec)
            continue

        ok, reason = _can_render(rows, x, y)
        if not ok:
            spec["render_skipped"] = load_err or reason
            rendered.append(spec)
            continue

        try:
            png = _render_one(plt, spec, rows, x, y, out_dir, idx, style)
            spec["figure_path"] = str(png)
            spec.pop("render_skipped", None)
        except Exception as exc:  # rendering must never sink the caller
            logger.warning("render_chart_specs: chart %d failed: %s", idx, exc)
            spec["render_skipped"] = f"render error: {exc}"
        rendered.append(spec)

    return rendered


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless, no display needed
        # Arial/Helvetica are usually absent on servers; DejaVu Sans is the
        # deterministic fallback. Silence the per-figure findfont warning spam.
        import logging as _logging

        import matplotlib.pyplot as plt
        _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
        return plt
    except Exception:
        return None


def _load_table(
    data: Any, tables_dir: Path | None, out_dir: Path
) -> tuple[list[dict[str, str]] | None, str]:
    """Resolve and read the spec's CSV into a list of row dicts.

    Returns ``(rows, "")`` on success, ``(None, reason)`` otherwise.
    """
    if not data:
        return None, "no data table referenced"
    if not isinstance(data, str):
        # inline list-of-dicts table (rare; the explorer usually writes CSV)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [{str(k): str(v) for k, v in r.items()} for r in data], ""
        return None, "unsupported inline data shape"

    path = Path(data)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if tables_dir is not None:
            candidates.append(tables_dir / path)            # data="tables/foo.csv"
            candidates.append(tables_dir / path.name)       # data="foo.csv"
            candidates.append(tables_dir / "tables" / path.name)  # tables_dir=sandbox
        candidates.append(out_dir / path)
        candidates.append(out_dir / "tables" / path.name)
        candidates.append(out_dir.parent / path)
    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        return None, f"CSV not found: {data}"
    if resolved.suffix.lower() != ".csv":
        return None, f"not a CSV: {resolved.name}"
    try:
        # errors="replace" tolerates non-UTF-8 cells (VL logs carry latin-1/binary
        # text); the broad except covers csv.Error (oversized field) so a malformed
        # CSV degrades to a (None, reason) annotation instead of raising — the
        # module's "never raises into a pipeline" contract.
        with resolved.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeError, csv.Error) as exc:
        return None, f"could not read {resolved.name}: {exc}"
    if not rows:
        return None, "empty CSV"
    return rows, ""


def _can_render(rows: list[dict[str, str]] | None, x: Any, y: Any) -> tuple[bool, str]:
    if not rows:
        return False, "no table data"
    if not x or not y:
        return False, "spec missing x or y column"
    cols = rows[0].keys()
    if x not in cols:
        return False, f"x column {x!r} not in table"
    if y not in cols:
        return False, f"y column {y!r} not in table"
    return True, ""


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_bar_colors(xs_raw: list[Any], default_palette: list[str]) -> list[str]:
    """Map categories to semantic palette colors (FAIL-red, PASS-green, else palette)."""
    colors = []
    for idx, label in enumerate(xs_raw):
        s = str(label).strip().lower()
        if any(w in s for w in ("fail", "broken", "error", "loss", "regression")):
            colors.append("#d03b3b")
        elif any(w in s for w in ("pass", "fixed", "cured", "correct", "gain", "survivor")):
            colors.append("#0ca30c")
        elif any(w in s for w in ("inconclusive", "warn", "unverified", "middle")):
            colors.append("#fab219")
        else:
            colors.append(default_palette[idx % len(default_palette)])
    return colors


def _render_one(plt, spec, rows, x, y, out_dir, idx, style) -> Path:
    kind = str(spec.get("kind", "bar")).lower()
    if kind not in _KINDS:
        kind = "bar"

    xs_raw = [r.get(x, "") for r in rows]
    ys = [_to_float(r.get(y, "")) for r in rows]
    # Drop rows whose y is non-numeric so the plot stays well-defined.
    pairs = [(xr, yv) for xr, yv in zip(xs_raw, ys) if yv is not None]
    if not pairs:
        raise ValueError(f"y column {y!r} has no numeric values")
    xs_raw, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    xs_num = [_to_float(v) for v in xs_raw]
    x_is_num = all(v is not None for v in xs_num)

    rc = (style or {}).get("rc", {})
    colors = (style or {}).get("colors") or NATURE_COLORS_FALLBACK
    primary = colors[0]
    title = str(spec.get("title") or spec.get("name") or f"chart_{idx}")

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(spec.get("name") or spec.get("title") or f"chart_{idx}")
    png = figures / f"{idx:02d}_{name}.png"

    # Clean label helpers
    clean_x = str(x).replace("_", " ").title() if x else ""
    clean_y = str(y).replace("_", " ").title() if y else ""

    # Apply the nature-figure style in a scoped rc_context
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")

        if kind == "scatter":
            pt_colors = _get_bar_colors(xs_raw, colors) if not x_is_num else [primary] * len(ys)
            ax.scatter(xs_num if x_is_num else range(len(xs_raw)), ys, s=36,
                       c=pt_colors, edgecolor="white", linewidth=0.6, zorder=3, alpha=0.85)
            if not x_is_num:
                ax.set_xticks(range(len(xs_raw)))
                ax.set_xticklabels([str(v) for v in xs_raw], rotation=35, ha="right", fontsize=9)
        elif kind in {"line", "timeseries"}:
            x_vals = xs_num if x_is_num else range(len(xs_raw))
            ax.plot(x_vals, ys, marker="o", color=primary, linewidth=2.0, markersize=5.5, zorder=3)
            ax.fill_between(x_vals, ys, color=primary, alpha=0.08, zorder=2)
            if not x_is_num:
                ax.set_xticks(range(len(xs_raw)))
                ax.set_xticklabels([str(v) for v in xs_raw], rotation=35, ha="right", fontsize=9)
        else:  # bar
            positions = range(len(xs_raw))
            bar_colors = _get_bar_colors(xs_raw, colors)
            bars = ax.bar(positions, ys, color=bar_colors, width=0.62, zorder=3, edgecolor="white", linewidth=0.5)
            ax.set_xticks(list(positions))
            ax.set_xticklabels([str(v) for v in xs_raw], rotation=35, ha="right", fontsize=9)
            # Format value labels on top of bars
            is_pct = all(0.0 <= val <= 1.0 for val in ys) and max(ys, default=0) <= 1.0 and ("rate" in clean_y.lower() or "pct" in clean_y.lower() or "share" in clean_y.lower())
            fmt = "%.1f%%" if is_pct else ("%.2f" if any(isinstance(v, float) and not v.is_integer() for v in ys) else "%d")
            labels = [fmt % (v * 100 if is_pct else v) for v in ys]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8.5, color="#334155")

        if kind in {"bar", "line", "timeseries", "scatter"}:
            ax.grid(axis="y", linewidth=0.6, color="#e2e8f0", alpha=0.7, zorder=0)
            ax.set_axisbelow(True)

        ax.set_xlabel(clean_x, fontsize=10, fontweight="bold", color="#1e293b", labelpad=6)
        ax.set_ylabel(clean_y, fontsize=10, fontweight="bold", color="#1e293b", labelpad=6)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color="#0f172a", pad=10)
        fig.tight_layout()
        fig.savefig(png, dpi=160, metadata={"Software": "evalvitals", "Creation Time": None})
        plt.close(fig)
    return png


def _safe_filename(name: Any) -> str:
    text = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name)).strip("_")
    return (text or "chart")[:48]


def _describe(spec: dict[str, Any], rows, x, y) -> str:
    """One-line textual summary of a chart, used when the image can't render
    and as the caption M3 sees alongside the attached PNG."""
    existing = spec.get("description")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    kind = str(spec.get("kind", "bar")).lower()
    title = str(spec.get("title") or spec.get("name") or "chart")
    if x and y:
        body = f"{kind} of {y} by {x}"
    elif x:
        body = f"{kind} over {x}"
    else:
        body = f"{kind} chart"
    n = len(rows) if rows else 0
    return f"{title}: {body}" + (f" ({n} rows)" if n else "")
