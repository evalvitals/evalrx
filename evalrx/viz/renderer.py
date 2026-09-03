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
(``evalrx explore``, the fused pipeline, and the in-loop M3 view).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

from evalrx.viz.labels import display_name
from evalrx.viz.style import (
    NATURE_COLORS_FALLBACK,
    SEMANTIC_PALETTE,
    load_nature_style,
    outcome_color,
)

logger = logging.getLogger(__name__)

_KINDS = {"bar", "line", "scatter", "timeseries", "forest"}

# Chart-type policy (agent-side counterpart: the eval-chart-style skill).
# "A bar's filled area means amount accumulated from zero — use bars only for
# counts." Rates, proportions, means and effect sizes must not be bars.
_COUNT_Y = re.compile(r"(count$|^n$|_n$|n_cases|ncases|num_|_num$|samples|draws|frequency|^total$)", re.I)
_EFFECT_Y = re.compile(r"(effect|separation|smd|odds|ratio|coefficient|importance|weight|score|mean|avg|average|delta|diff|lift|gain)", re.I)


def _is_count_like(y: Any, ys: list[float]) -> bool:
    """True only when the y column plausibly holds raw counts."""
    name = str(y or "")
    if _EFFECT_Y.search(name):
        return False  # a mean/effect column is never a count even if integer-valued
    if not _COUNT_Y.search(name):
        return False
    return all(float(v).is_integer() for v in ys)


def _demote_bar(spec: dict[str, Any], y: Any, ys: list[float]) -> str:
    """Pick the policy-compliant kind a 'bar' spec must be rendered as instead."""
    return "forest" if _EFFECT_Y.search(str(y or "")) else "line"

#: Column names that mean "this y is a count" (composition / count bars).
_COUNT_COLUMNS = {"count", "counts", "n", "n_cases", "cases", "num", "total", "freq", "frequency"}
#: Column names that carry the per-group / per-bin sample size.
_N_COLUMNS = ("n", "n_cases", "n_rows", "support", "cases", "count")
#: (low, high) column-name pairs the explorer may use for a 95% interval.
_CI_COLUMNS = (("ci_low", "ci_high"), ("ci_lo", "ci_hi"), ("ci95_low", "ci95_high"),
               ("lower", "upper"), ("lo", "hi"), ("ymin", "ymax"))
_RATE_WORDS = ("rate", "pct", "percent", "share", "frac", "fraction", "proportion", "prob")
#: Up to this many groups a group->value comparison is a dot + CI / lollipop,
#: never bars (eval-chart-style §0: a bar's area means "accumulated from 0").
_DOT_MAX_GROUPS = 3


def _count_like(y: Any, ys: list[float]) -> bool:
    """Count-valued y for the bar policy: a known count column (``count``,
    ``n``, ...), an ``n_<group>`` / ``num_<x>`` numerator column, or
    :func:`_is_count_like`'s name test — and integer values in every case."""
    name = str(y or "").lower()
    if not ys or not all(float(v).is_integer() for v in ys):
        return False
    return name in _COUNT_COLUMNS or name.startswith(("n_", "num_")) or _is_count_like(y, ys)


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

        if plt is None:
            spec["render_skipped"] = "matplotlib not installed (pip install 'evalrx[viz]')"
            spec.setdefault("description", _describe(spec, rows, x, y))
            rendered.append(spec)
            continue

        ok, reason = _can_render(rows, x, y)
        if not ok:
            spec["render_skipped"] = load_err or reason
            spec.setdefault("description", _describe(spec, rows, x, y))
            rendered.append(spec)
            continue

        try:
            png = _render_one(plt, spec, rows, x, y, out_dir, idx, style)
            spec["figure_path"] = str(png)
            spec.pop("render_skipped", None)
        except Exception as exc:  # rendering must never sink the caller
            logger.warning("render_chart_specs: chart %d failed: %s", idx, exc)
            spec["render_skipped"] = f"render error: {exc}"
        # After rendering, so the caption names the form actually drawn.
        spec.setdefault("description", _describe(spec, rows, x, y))
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
    """Draw one spec. The chart FORM follows the eval-chart-style policy, not
    the spec's ``kind`` alone (the explorer may only say bar/line/scatter):

    * ``kind=bar`` whose y is a count over a handful of classes (class
      balance)           -> ONE 100%-stacked composition strip;
    * ``kind=bar`` comparing a rate/mean across <= 3 groups
                         -> horizontal dot + 95% CI (Wilson from ``n`` for a
                            rate, or the CSV's own ci_low/ci_high), lollipop
                            when no interval can be formed — never two bars;
    * other bars         -> bars, FAIL/PASS hues on outcome axes, n annotated;
    * line / scatter     -> as before, aliased axes, n annotated on lines.

    Every PNG is deterministic (same spec + CSV -> byte-identical file) and the
    chosen form is recorded in ``spec["rendered_as"]``.
    """
    kind = str(spec.get("kind", "bar")).lower()
    if kind not in _KINDS:
        kind = "bar"
    xs_raw = [r.get(x, "") for r in rows]
    ys = [_to_float(r.get(y, "")) for r in rows]
    keep = [i for i, yv in enumerate(ys) if yv is not None]
    if not keep:
        raise ValueError(f"y column {y!r} has no numeric values")
    rows = [rows[i] for i in keep]
    xs_raw = [xs_raw[i] for i in keep]
    ys = [ys[i] for i in keep]
    xs_num = [_to_float(v) for v in xs_raw]
    x_is_num = all(v is not None for v in xs_num)
    ns = _n_values(rows, y)
    scale = _rate_scale(y, ys)
    rate = scale is not None
    rc = (style or {}).get("rc", {})
    colors = (style or {}).get("colors") or NATURE_COLORS_FALLBACK
    accent = SEMANTIC_PALETTE["ACCENT"]
    title = str(spec.get("title") or spec.get("name") or f"chart_{idx}")
    xlabel, ylabel = display_name(x), display_name(y)
    spec["axis_labels"] = {"x": xlabel, "y": ylabel}
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(spec.get("name") or spec.get("title") or f"chart_{idx}")
    png = figures / f"{idx:02d}_{name}.png"

    form = kind
    if kind == "bar" and not x_is_num:
        if _is_composition(spec, xs_raw, y, ys):
            form = "composition"
        elif 2 <= len(xs_raw) <= _DOT_MAX_GROUPS and str(y).lower() not in _COUNT_COLUMNS:
            form = "dot_ci"
    if kind == "bar" and form == "bar" and not _count_like(y, ys):
        # Chart-type policy enforcement (eval-chart-style): a bar may only
        # encode raw counts. A rate/mean/effect bar that is neither a class
        # composition nor a <= 3-group comparison is demoted to the
        # policy-compliant kind — line for binned rates, forest (horizontal
        # dot plot) for ranked effects — so the rule holds even when the
        # agent ignores it.
        kind = _demote_bar(spec, y, ys)
        form = kind
        spec["kind"] = kind
        spec["render_note"] = (
            f"kind=bar demoted to {kind}: y column {str(y)!r} is not a count "
            "(bars are for counts only — eval-chart-style policy)"
        )
        logger.warning(
            "render_chart_specs: %r demoted bar -> %s (%r is not a count)",
            spec.get("name"), kind, y,
        )

    # Apply the nature-figure style in a scoped rc_context (no global leak; fully
    # deterministic -> same spec + CSV yields byte-identical PNGs).
    with plt.rc_context(rc):
        if form == "composition":
            fig, ax = plt.subplots(figsize=(6.4, 1.7))
            _draw_composition(ax, xs_raw, ys, colors)
            ax.set_title(title, fontweight="bold", loc="left")
        elif form == "dot_ci":
            fig, ax = plt.subplots(figsize=(6.4, 1.6 + 0.55 * len(xs_raw)))
            form = _draw_dot_ci(ax, rows, xs_raw, ys, ns, scale, accent)
            ax.set_xlabel(ylabel)
            ax.set_title(title, fontweight="bold", loc="left")
        elif kind == "forest":
            # Horizontal dot plot for ranked effects / means: position encodes
            # the value, no filled area faking "accumulated amount". Sorted so
            # the strongest is on top.
            fig, ax = plt.subplots(figsize=(6.4, 1.6 + 0.45 * len(xs_raw)))
            order = sorted(range(len(ys)), key=lambda i: ys[i])
            labels = [str(xs_raw[i]) for i in order]
            vals = [ys[i] for i in order]
            ypos = list(range(len(vals)))
            ax.hlines(ypos, [min(0.0, v) for v in vals], vals,
                      color=SEMANTIC_PALETTE["GRID"], linewidth=1.8, zorder=2)
            ax.axvline(0, color=SEMANTIC_PALETTE["AXIS"], linewidth=0.9, zorder=1)
            pt_colors = [outcome_color(lab) or accent for lab in labels]
            ax.scatter(vals, ypos, s=68, c=pt_colors, edgecolor="white",
                       linewidth=0.8, zorder=3)
            has_neg = any(v < 0 for v in vals)
            for yi, v in zip(ypos, vals):
                ax.annotate(f"{v:+.2f}" if has_neg else f"{v:.2f}", (v, yi),
                            textcoords="offset points", xytext=(7, -3.5),
                            fontsize=8.5, color=SEMANTIC_PALETTE["TEXT"])
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels)
            ax.grid(axis="x", linewidth=0.6, alpha=0.25, zorder=0)
            ax.set_axisbelow(True)
            # The value axis is horizontal: swap the axis labels.
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
            ax.set_title(title, fontweight="bold", loc="left")
        else:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            if kind == "scatter":
                ax.scatter(xs_num if x_is_num else range(len(xs_raw)), ys, s=26,
                           color=accent, edgecolor="white", linewidth=0.4, zorder=3)
                if not x_is_num:
                    ax.set_xticks(range(len(xs_raw)))
                    ax.set_xticklabels([str(v) for v in xs_raw], rotation=45, ha="right")
            elif kind in {"line", "timeseries"}:
                px = xs_num if x_is_num else list(range(len(xs_raw)))
                ax.plot(px, ys, marker="o", color=accent, linewidth=1.8, markersize=5, zorder=3)
                if not x_is_num:
                    ax.set_xticks(range(len(xs_raw)))
                    ax.set_xticklabels([str(v) for v in xs_raw], rotation=45, ha="right")
                if ns is not None:
                    for xv, yv, nv in zip(px, ys, ns):
                        ax.annotate(f"n={nv:g}", (xv, yv), textcoords="offset points",
                                    xytext=(0, 7), ha="center", fontsize=7,
                                    color=SEMANTIC_PALETTE["AXIS"])
            else:  # bar (counts, or more than _DOT_MAX_GROUPS groups)
                positions = list(range(len(xs_raw)))
                bar_colors = [outcome_color(v) or accent for v in xs_raw]
                ax.bar(positions, ys, color=bar_colors, width=0.72, zorder=3)
                ax.set_xticks(positions)
                ax.set_xticklabels([str(v) for v in xs_raw], rotation=45, ha="right")
                is_count = str(y).lower() in _COUNT_COLUMNS
                for pos, yv, nv in zip(positions, ys, ns or [None] * len(ys)):
                    label = (f"{yv:g}" if is_count else (f"n={nv:g}" if nv is not None else ""))
                    if label:
                        ax.annotate(label, (pos, yv), textcoords="offset points",
                                    xytext=(0, 3), ha="center", fontsize=7,
                                    color=SEMANTIC_PALETTE["AXIS"])
            if kind in {"bar", "line", "timeseries"}:
                ax.grid(axis="y", linewidth=0.6, alpha=0.25, zorder=0)
                ax.set_axisbelow(True)
            if rate and kind != "scatter":
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(scale, decimals=0))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        fig.savefig(png, dpi=160, metadata={"Software": "evalrx", "Creation Time": None})
        plt.close(fig)
    spec["rendered_as"] = form
    return png


# -- chart-form helpers -------------------------------------------------------

def _n_values(rows, y) -> "list[float] | None":
    """Per-row sample size from an ``n``-like column (not the y column itself)."""
    cols = rows[0].keys() if rows else []
    for cand in _N_COLUMNS:
        if cand in cols and cand != y:
            vals = [_to_float(r.get(cand, "")) for r in rows]
            if all(v is not None for v in vals):
                return vals
    return None


def _ci_values(rows) -> "tuple[list[float], list[float]] | None":
    cols = rows[0].keys() if rows else []
    for lo, hi in _CI_COLUMNS:
        if lo in cols and hi in cols:
            los = [_to_float(r.get(lo, "")) for r in rows]
            his = [_to_float(r.get(hi, "")) for r in rows]
            if all(v is not None for v in los + his):
                return los, his
    return None


def _rate_scale(y, ys) -> "float | None":
    """1.0 when *y* is a fraction-valued rate column, 100.0 when it is a
    percent-valued one, ``None`` when it is not a rate (counts, means of
    unbounded quantities, effect sizes)."""
    name = str(y).lower()
    if name in _COUNT_COLUMNS or not any(w in name for w in _RATE_WORDS):
        return None
    if all(0.0 <= v <= 1.0 for v in ys):
        return 1.0
    if all(0.0 <= v <= 100.0 for v in ys) and ("pct" in name or "percent" in name):
        return 100.0
    return None


def _numerator_values(rows, y, ns) -> "list[float] | None":
    """The successes column behind a rate (``n_fail``, ``n_with_audit``, ``k``,
    ...): any ``n_<something>`` / ``k`` / ``successes`` column whose values never
    exceed ``n``. Lets the host form a Wilson interval from the data instead of
    guessing one."""
    if ns is None or not rows:
        return None
    for col in rows[0].keys():
        low = col.lower()
        if col == y or low in _N_COLUMNS or low in _COUNT_COLUMNS:
            continue
        if not (low.startswith("n_") or low in ("k", "successes", "hits", "events")):
            continue
        vals = [_to_float(r.get(col, "")) for r in rows]
        if all(v is not None and 0 <= v <= n for v, n in zip(vals, ns)):
            return vals
    return None


def _is_composition(spec, xs_raw, y, ys) -> bool:
    """Class balance: a count per class over a handful of classes."""
    if str(y).lower() not in _COUNT_COLUMNS or not (2 <= len(xs_raw) <= 6):
        return False
    if any(v < 0 for v in ys) or sum(ys) <= 0:
        return False
    text = f"{spec.get('name', '')} {spec.get('title', '')}".lower()
    outcome_axis = all(outcome_color(v) is not None for v in xs_raw)
    return outcome_axis or "balance" in text or "composition" in text


def _wilson(k: float, n: float, z: float = 1.959964) -> "tuple[float, float]":
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _draw_composition(ax, labels, values, colors) -> None:
    total = float(sum(values)) or 1.0
    left = 0.0
    for i, (lab, val) in enumerate(zip(labels, values)):
        color = outcome_color(lab) or colors[i % len(colors)]
        ax.barh([0], [val], left=left, color=color, height=0.6, zorder=3)
        frac = val / total
        text = f"{lab} {val:g} ({round(frac * 100)}%)"
        if frac >= 0.14:
            ax.text(left + val / 2, 0, text, ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold")
        else:
            ax.text(left + val / 2, 0.42, text, ha="center", va="bottom",
                    color=SEMANTIC_PALETTE["TEXT"], fontsize=7.5)
        left += val
    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, 0.9)
    ax.axis("off")


def _draw_dot_ci(ax, rows, labels, values, ns, scale, accent) -> str:
    """Horizontal dot (+ 95% CI when one can be formed) per group; stem from 0.

    The interval comes from the CSV's own ``ci_low``/``ci_high`` when present,
    else — for a RATE with a numerator column (``n_fail``, ``n_with_audit``,
    ``k`` ...) and ``n`` — from a Wilson score interval on those counts. A
    ``mean_*`` of per-case values is not a binomial proportion, so no interval
    is invented for it: the dot stands alone (lollipop) with its n. Returns the
    form actually drawn: ``"dot_ci"`` or ``"lollipop"``."""
    rate = scale is not None
    ci = _ci_values(rows)
    if ci is None and rate:
        ks = _numerator_values(rows, None, ns)
        if ks is not None:
            los, his = [], []
            for k, n in zip(ks, ns):
                lo, hi = _wilson(k, n)
                los.append(lo * scale)
                his.append(hi * scale)
            ci = (los, his)
    form = "dot_ci" if ci is not None else "lollipop"
    ypos = list(range(len(labels)))[::-1]  # first CSV row on top
    for yp, lab, val in zip(ypos, labels, values):
        color = outcome_color(lab) or accent
        ax.plot([0, val], [yp, yp], color=SEMANTIC_PALETTE["GRID"], linewidth=2.2, zorder=2)
        if ci is not None:
            i = labels.index(lab)
            ax.plot([ci[0][i], ci[1][i]], [yp, yp], color=color, linewidth=1.6, zorder=3,
                    solid_capstyle="butt")
            ax.plot([ci[0][i], ci[0][i]], [yp - 0.12, yp + 0.12], color=color, linewidth=1.2)
            ax.plot([ci[1][i], ci[1][i]], [yp - 0.12, yp + 0.12], color=color, linewidth=1.2)
        ax.scatter([val], [yp], s=60, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        shown = f"{round(val * 100 / scale)}%" if rate else f"{val:.2f}"
        if ns is not None:
            shown += f"  (n={ns[labels.index(lab)]:g})"
        ax.annotate(shown, (val, yp), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color=SEMANTIC_PALETTE["TEXT"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([str(v) for v in labels])
    ax.set_ylim(-0.7, len(labels) - 0.3)
    hi = max([v for v in values] + (list(ci[1]) if ci is not None else []))
    lo = min([0.0] + [v for v in values] + (list(ci[0]) if ci is not None else []))
    span = (hi - lo) or 1.0
    ax.set_xlim(min(0.0, lo - 0.05 * span), hi + 0.18 * span)
    ax.grid(axis="x", linewidth=0.6, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if rate:
        from matplotlib.ticker import PercentFormatter
        ax.xaxis.set_major_formatter(PercentFormatter(scale, decimals=0))
    return form


def _safe_filename(name: Any) -> str:
    text = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name)).strip("_")
    return (text or "chart")[:48]


_FORM_WORDS = {
    "composition": "composition strip of",
    "dot_ci": "dot + 95% CI of",
    "lollipop": "dot (lollipop) of",
    "forest": "forest plot (ranked dots) of",
}


def _describe(spec: dict[str, Any], rows, x, y) -> str:
    """One-line textual summary of a chart, used when the image can't render
    and as the caption M3 sees alongside the attached PNG. Names the FORM the
    host actually drew when that is known (``rendered_as``)."""
    existing = spec.get("description")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    kind = str(spec.get("kind", "bar")).lower()
    form = _FORM_WORDS.get(str(spec.get("rendered_as", "")), f"{kind} of")
    title = str(spec.get("title") or spec.get("name") or "chart")
    if x and y:
        body = f"{form} {y} by {x}"
    elif x:
        body = f"{kind} over {x}"
    else:
        body = f"{kind} chart"
    n = len(rows) if rows else 0
    return f"{title}: {body}" + (f" ({n} rows)" if n else "")
