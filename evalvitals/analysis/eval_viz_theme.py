"""
eval_viz_theme.py
=================
Drop-in styling + chart layer for FAIL-vs-PASS LLM eval analysis.

Goal: one import fixes the three recurring problems —
  (1) inconsistent color / font / truncated names  -> global template + semantic
      palette + short-name registry
  (2) overflow clipping + misallocated space        -> sizing helpers (full-row vs
      compact) and chart builders that pick distribution plots over two-bar plots
  (3) inconsistent number / bin formatting           -> fmt() and human_bins()

Targets Plotly and static report figures. A matplotlib rcParams equivalent is
at the bottom for non-Plotly callers.

Usage
-----
    import eval_viz_theme as viz
    viz.apply()                       # register + set default template once, at startup

    fig = viz.violin_by_outcome(df, signal="relative_attention_max_relative_weight",
                                outcome="label")
    st.plotly_chart(fig, use_container_width=True)

Every builder returns a plotly Figure. Nothing renders on its own.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from evalvitals.viz.labels import display_name, raw_hint

# ---------------------------------------------------------------------------
# 1. SEMANTIC PALETTE  — color encodes ROLE, never decoration.
#    Reuse these names everywhere; never hardcode a hex in a chart.
#    Values are the validated dataviz-skill reference palette (dataviz
#    skill's references/palette.md): status colors are fixed across modes,
#    only ink/surface/grid shift. apply() swaps PALETTE in place between
#    _LIGHT/_DARK — see its docstring for how dark mode is detected.
# ---------------------------------------------------------------------------
_LIGHT = {
    "FAIL":          "#d03b3b",   # status: critical
    "PASS":          "#0ca30c",   # status: good
    "SIGNIFICANT":   "#0ca30c",   # status: good — survived FDR / REJECT H0
    "INCONCLUSIVE":  "#fab219",   # status: warning — did not survive
    "ACCENT":        "#2a78d6",   # categorical slot 1 / sequential base hue
    "LEAKY":         "#898781",   # muted ink: target-leaking signal, do not elevate
    "GRID":          "#e1e0d9",   # hairline gridline
    "AXIS":          "#898781",   # muted ink (axis/tick labels)
    "TEXT":          "#0b0b0b",   # primary ink
    "SURFACE":       "#fcfcfb",   # chart surface — sequential-scale zero end
    "BAND":          "rgba(42,120,214,0.12)",  # CI / reference band fill
    "DIVERGE_NEG":   "#2a78d6",   # diverging pole (negative) — blue
    "DIVERGE_POS":   "#e34948",   # diverging pole (positive) — red
    "DIVERGE_MID":   "#f0efec",   # diverging neutral midpoint
}
_DARK = {
    "FAIL":          "#d03b3b",
    "PASS":          "#0ca30c",
    "SIGNIFICANT":   "#0ca30c",
    "INCONCLUSIVE":  "#fab219",
    "ACCENT":        "#3987e5",
    "LEAKY":         "#898781",
    "GRID":          "#2c2c2a",
    "AXIS":          "#898781",
    "TEXT":          "#ffffff",
    "SURFACE":       "#1a1a19",
    "BAND":          "rgba(57,135,229,0.20)",
    "DIVERGE_NEG":   "#3987e5",
    "DIVERGE_POS":   "#e66767",
    "DIVERGE_MID":   "#383835",
}
PALETTE = dict(_LIGHT)  # mutated in place by apply(); chart builders read this dict live

FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

# Map any outcome label -> role color. Extend if labels differ. Rebuilt by
# apply() alongside PALETTE so it never goes stale after a mode switch.
OUTCOME_COLORS = {
    # NB: True hashes to 1 and False to 0, so the int keys cover the bool case too.
    "FAIL": PALETTE["FAIL"], "fail": PALETTE["FAIL"], 1: PALETTE["FAIL"],
    "PASS": PALETTE["PASS"], "pass": PALETTE["PASS"], 0: PALETTE["PASS"],
}

def outcome_color(v):
    return OUTCOME_COLORS.get(v, PALETTE["ACCENT"])

def _detect_dark() -> bool:
    """Static report figures use the portable light palette by default."""
    return False


# ---------------------------------------------------------------------------
# 2. SHORT-NAME REGISTRY  — never print a raw signal column on an axis/header.
#    short() returns the display alias; full() / tooltip kept for hover.
# ---------------------------------------------------------------------------
SHORT_NAMES = {
    "relative_attention_focus_share":        "Attention focus",
    "relative_attention_max_relative_weight":"Max attention",
    "relative_attention_mean_relative_weight":"Mean attention",
    "generated_probe1_false_detection":      "Sanity check",
    "low_focus_share":                       "High focus",
    "probe1_positive":                       "Probe positive",
}

def short(name: str) -> str:
    """Display alias. Falls back to a deterministic abbreviation."""
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    return display_name(name, compact=True)

def full(name: str) -> str:
    hint = raw_hint(name)
    label = display_name(name)
    return f"{label} ({hint})" if hint else label


# ---------------------------------------------------------------------------
# 3. NUMBER + BIN FORMATTING  — one global precision policy, by quantity kind.
# ---------------------------------------------------------------------------
def fmt(x, kind: str = "effect") -> str:
    """
    kind: 'effect'|'corr'  -> 2 dp     (e.g. 0.87)
          'stat'|'p'       -> 3 dp     (e.g. 0.082)
          'pct'            -> integer % (e.g. 41%)
          'count'          -> integer
          'val'            -> adaptive: 2 sig-ish, trims long floats
    n=25 cannot support 4 dp — never print 0.8665.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if kind in ("effect", "corr"):
        return f"{x:.2f}"
    if kind in ("stat", "p"):
        return f"{x:.3f}"
    if kind == "pct":
        return f"{round(x * 100) if x <= 1 else round(x)}%"
    if kind == "count":
        return f"{int(round(x))}"
    # 'val'
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"

def human_bins(edges) -> list[str]:
    """
    Turn pandas.cut float edges into readable labels.
    [113.844, 233.066, 791.774] -> ['114–233', '233–792']
    Picks decimals from the data range so small ranges keep precision.
    """
    edges = list(edges)
    span = max(edges) - min(edges)
    dec = 0 if span >= 10 else (1 if span >= 1 else 2)
    def f(v):
        return f"{v:.{dec}f}"
    return [f"{f(a)}–{f(b)}" for a, b in zip(edges[:-1], edges[1:])]


# ---------------------------------------------------------------------------
# 4. GLOBAL TEMPLATE  — register once; every chart inherits font, grid, margins.
# ---------------------------------------------------------------------------
def apply(set_default: bool = True, dark: "bool | None" = None) -> str:
    """Register the plotly template. *dark* picks the light/dark palette
    variant; ``None`` auto-detects via Streamlit's theme context (see
    :func:`_detect_dark`), defaulting to light outside Streamlit.

    Mutates the module-level ``PALETTE``/``OUTCOME_COLORS`` dicts in place
    (rather than rebinding them) so every chart builder — which all read
    ``PALETTE["..."]`` live, not a captured snapshot — picks up the new
    values immediately.
    """
    if dark is None:
        dark = _detect_dark()
    PALETTE.clear()
    PALETTE.update(_DARK if dark else _LIGHT)
    OUTCOME_COLORS.clear()
    OUTCOME_COLORS.update({
        "FAIL": PALETTE["FAIL"], "fail": PALETTE["FAIL"], 1: PALETTE["FAIL"],
        "PASS": PALETTE["PASS"], "pass": PALETTE["PASS"], 0: PALETTE["PASS"],
    })
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=FONT, size=13, color=PALETTE["TEXT"]),
        title=dict(font=dict(size=15), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PALETTE["ACCENT"], PALETTE["FAIL"], PALETTE["PASS"],
                  PALETTE["SIGNIFICANT"], PALETTE["INCONCLUSIVE"]],
        margin=dict(l=64, r=24, t=48, b=72),
        xaxis=dict(showgrid=False, zeroline=False, ticks="outside",
                   linecolor=PALETTE["AXIS"], tickcolor=PALETTE["AXIS"],
                   tickfont=dict(size=12), automargin=True),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["GRID"], zeroline=False,
                   linecolor="rgba(0,0,0,0)", tickfont=dict(size=12),
                   automargin=True),
        # legend below the plot so it never collides with the left-aligned title
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=12)),
    )
    pio.templates["evalvitals"] = tmpl
    if set_default:
        pio.templates.default = "evalvitals"
    return "evalvitals"


# ---------------------------------------------------------------------------
# 5. SIZING  — kills overflow clipping and the value/space inversion.
#    Distribution + scatter get a full row; counts get compact.
# ---------------------------------------------------------------------------
SIZES = {
    "full":    dict(height=420, width=None),   # use_container_width=True
    "wide":    dict(height=380, width=None),   # scatter / joint
    "compact": dict(height=300, width=None),   # counts, single comparison
    "strip":   dict(height=300, width=None),   # small-multiples cell
}

def _size(fig: go.Figure, kind: str) -> go.Figure:
    s = SIZES.get(kind, SIZES["full"])
    fig.update_layout(height=s["height"], autosize=True)
    return fig


# ---------------------------------------------------------------------------
# 6. CHART BUILDERS  — each one encodes a "use this, not a bar" decision.
# ---------------------------------------------------------------------------

def violin_by_outcome(df, signal, outcome="label", points="all"):
    """
    REPLACES the two-bar 'mean by outcome'.
    Shows the full distribution + every point — so bimodality and
    outlier-driven means are visible instead of hidden. Use for any
    continuous signal compared across FAIL/PASS.
    """
    fig = go.Figure()
    for grp in ["FAIL", "PASS"]:
        sub = df[df[outcome].astype(str).str.upper() == grp]
        if sub.empty:
            continue
        fig.add_trace(go.Violin(
            y=sub[signal], name=grp, line_color=outcome_color(grp),
            fillcolor=outcome_color(grp), opacity=0.55, meanline_visible=True,
            points=points, jitter=0.35, marker=dict(size=4, opacity=0.6),
            box_visible=True, spanmode="hard",
        ))
    fig.update_layout(title=f"{short(signal)} by outcome  ·  full {full(signal)}",
                      yaxis_title=short(signal), xaxis_title="", showlegend=False,
                      violingap=0.25)
    return _size(fig, "full")


def forest_effects(rows, label_key="signal", effect_key="effect",
                   lo_key="ci_lo", hi_key="ci_hi", sig_key="significant",
                   leaky_key="leaky"):
    """
    REPLACES the green/grey effect-size bar chart.
    Horizontal dot + CI (forest style): keeps the REJECT-H0 color coding AND
    shows the confidence interval the bar chart dropped. Greys out any signal
    flagged leaky so a target-leaking signal is never visually #1.
    rows: list of dicts. ci_lo/ci_hi/leaky optional.
    """
    rows = sorted(rows, key=lambda r: r.get(effect_key, 0))
    fig = go.Figure()
    for r in rows:
        leaky = r.get(leaky_key, False)
        if leaky:
            color = PALETTE["LEAKY"]
        elif r.get(sig_key, True):
            color = PALETTE["SIGNIFICANT"]
        else:
            color = PALETTE["INCONCLUSIVE"]
        y = short(r[label_key]) + ("  (leaky)" if leaky else "")
        lo, hi = r.get(lo_key), r.get(hi_key)
        if lo is not None and hi is not None:
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[y, y], mode="lines",
                line=dict(color=color, width=3), showlegend=False,
                hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[r[effect_key]], y=[y], mode="markers+text",
            marker=dict(color=color, size=11),
            text=[fmt(r[effect_key], "effect")], textposition="middle right",
            textfont=dict(size=11), showlegend=False,
            hovertext=full(r[label_key]), hoverinfo="text"))
    fig.add_vline(x=0, line=dict(color=PALETTE["AXIS"], width=1, dash="dot"))
    fig.update_layout(title="Effect size (failure association) — green = REJECT H₀",
                      xaxis_title="effect size", yaxis_title="")
    return _size(fig, "full")


def logistic_failrate(df, signal, outcome="label", n_grid=80, band=True):
    """
    REPLACES 'Fail rate by <binned signal>' line charts.
    Keeps the x-axis continuous and fits a logistic fail-rate curve with a
    sample-density rug — no arbitrary bin edges, no fake monotone line.
    """
    x = df[signal].to_numpy(dtype=float)
    y = (df[outcome].astype(str).str.upper() == "FAIL").to_numpy(dtype=float)
    # simple logistic fit via numpy (no sklearn dependency)
    b0, b1 = _fit_logistic(x, y)
    gx = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    gy = 1 / (1 + np.exp(-(b0 + b1 * gx)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines",
                             line=dict(color=PALETTE["FAIL"], width=2.5),
                             name="P(FAIL)"))
    # density rug
    fig.add_trace(go.Scatter(x=x, y=np.full_like(x, -0.04),
                             mode="markers",
                             marker=dict(color=PALETTE["AXIS"], size=4,
                                         opacity=0.4, symbol="line-ns-open"),
                             name="samples", hoverinfo="x"))
    fig.update_layout(title=f"Fail rate vs {short(signal)} (logistic fit)",
                      xaxis_title=short(signal), yaxis_title="P(FAIL)",
                      yaxis=dict(range=[-0.08, 1.05]), showlegend=False)
    return _size(fig, "full")


def joint_scatter(df, x_sig, y_sig, outcome="label"):
    """
    For two continuous signals: scatter colored by outcome WITH marginal
    distributions. This is the chart that was clipped off-page — give it a
    full row, not a narrow column.
    """
    d = df.copy()
    d["_grp"] = d[outcome].astype(str).str.upper()
    fig = px.scatter(d, x=x_sig, y=y_sig, color="_grp",
                     color_discrete_map={"FAIL": PALETTE["FAIL"], "PASS": PALETTE["PASS"]},
                     marginal_x="histogram", marginal_y="histogram",
                     labels={x_sig: short(x_sig), y_sig: short(y_sig), "_grp": "outcome"})
    fig.update_traces(marker=dict(size=7, opacity=0.75),
                      selector=dict(type="scatter"))
    fig.update_traces(marker=dict(size=7, opacity=0.75),
                      selector=dict(type="scattergl"))
    fig.update_layout(title=f"{short(x_sig)} vs {short(y_sig)} by outcome",
                      legend_title_text="")
    return _size(fig, "wide")


def composition_bar(labels, values, title="Class balance"):
    """Parts-of-a-whole as ONE slim 100%-stacked horizontal bar — the right
    encoding for a handful of category counts. Two tall bars for two numbers is
    chart-junk (and usually redundant with the text); a single stacked strip
    reads the proportion at a glance. Each segment labels its count + percent."""
    total = sum(values) or 1
    fig = go.Figure()
    for lab, val in zip(labels, values):
        fig.add_trace(go.Bar(
            y=["balance"], x=[val], orientation="h", name=str(lab),
            marker_color=outcome_color(lab),
            text=f"{lab} {fmt(val, 'count')} ({fmt(val / total, 'pct')})",
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=13)))
    fig.update_layout(barmode="stack", title=title, showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      height=120, margin=dict(l=8, r=8, t=44, b=8))
    return fig


def counts_bar(df, outcome="label"):
    """Discrete class counts → a single 100%-stacked composition strip (see
    `composition_bar`); avoids two fat bars for two numbers."""
    vc = df[outcome].astype(str).str.upper().value_counts()
    order = [g for g in ["FAIL", "PASS"] if g in vc.index]
    return composition_bar(order, [int(vc[g]) for g in order])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fit_logistic(x, y, iters=100, lr=None):
    """Tiny IRLS-free gradient logistic fit; standardizes x for stability."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    mu, sd = x.mean(), x.std() or 1.0
    xs = (x - mu) / sd
    b0, b1 = 0.0, 0.0
    lr = lr or 0.3
    for _ in range(iters):
        z = b0 + b1 * xs
        p = 1 / (1 + np.exp(-z))
        g0 = np.mean(p - y)
        g1 = np.mean((p - y) * xs)
        b0 -= lr * g0
        b1 -= lr * g1
    # unstandardize back to raw-x coefficients
    return b0 - b1 * mu / sd, b1 / sd


# ---------------------------------------------------------------------------
# 7. STATISTICAL BUILDERS  — distribution diagnostics, variable relationships,
#    model evaluation, decision analysis. All numpy-only (no scipy/sklearn).
#    Each takes a per-case tidy DataFrame (one row per case) + signal column(s).
# ---------------------------------------------------------------------------

def _norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation). Vectorized."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.zeros_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    ql = np.sqrt(-2 * np.log(np.clip(p[lo], 1e-16, 1)))
    out[lo] = (((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
    qh = np.sqrt(-2 * np.log(np.clip(1 - p[hi], 1e-16, 1)))
    out[hi] = -(((((c[0]*qh+c[1])*qh+c[2])*qh+c[3])*qh+c[4])*qh+c[5]) / ((((d[0]*qh+d[1])*qh+d[2])*qh+d[3])*qh+1)
    qm = p[mid] - 0.5
    rm = qm * qm
    out[mid] = (((((a[0]*rm+a[1])*rm+a[2])*rm+a[3])*rm+a[4])*rm+a[5])*qm / (((((b[0]*rm+b[1])*rm+b[2])*rm+b[3])*rm+b[4])*rm+1)
    return out


def _clean(df, signal, outcome="label"):
    d = df[[signal, outcome]].copy()
    d[signal] = pd.to_numeric(d[signal], errors="coerce")
    return d.dropna()


def _groups(d, signal, outcome="label"):
    out = {}
    for grp in ("FAIL", "PASS"):
        vals = d[d[outcome].astype(str).str.upper() == grp][signal].to_numpy(float)
        if len(vals):
            out[grp] = vals
    return out


# ---- distribution diagnostics --------------------------------------------

def hist_by_outcome(df, signal, outcome="label", nbins=24):
    """Overlaid FAIL/PASS histograms — the actual frequency shape, not a mean."""
    d = _clean(df, signal, outcome)
    fig = go.Figure()
    for grp, vals in _groups(d, signal, outcome).items():
        fig.add_trace(go.Histogram(x=vals, name=grp, nbinsx=nbins,
                                    marker_color=outcome_color(grp), opacity=0.6))
    fig.update_layout(barmode="overlay", title=f"{short(signal)} — distribution by outcome",
                      xaxis_title=short(signal), yaxis_title="count")
    return _size(fig, "wide")


def kde_by_outcome(df, signal, outcome="label", n_grid=200):
    """Gaussian-KDE density per outcome (Scott's bandwidth) — smooth shape +
    bimodality without bin-edge artifacts."""
    d = _clean(df, signal, outcome)
    groups = _groups(d, signal, outcome)
    allv = d[signal].to_numpy(float)
    if allv.size < 2:
        return hist_by_outcome(df, signal, outcome)
    gx = np.linspace(np.nanmin(allv), np.nanmax(allv), n_grid)
    fig = go.Figure()
    for grp, vals in groups.items():
        if len(vals) < 2:
            continue
        bw = vals.std(ddof=1) * (len(vals) ** (-1 / 5)) or 1.0
        dens = np.exp(-0.5 * ((gx[:, None] - vals[None, :]) / bw) ** 2).sum(1) / (len(vals) * bw * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(x=gx, y=dens, mode="lines", name=grp, fill="tozeroy",
                                 line=dict(color=outcome_color(grp), width=2),
                                 fillcolor=outcome_color(grp).replace(")", ",0.18)").replace("#", "rgba(") if False else None))
        fig.data[-1].update(opacity=0.85)
    fig.update_layout(title=f"{short(signal)} — density by outcome (KDE)",
                      xaxis_title=short(signal), yaxis_title="density")
    return _size(fig, "wide")


def ecdf_by_outcome(df, signal, outcome="label"):
    """Empirical CDF per outcome — reads exact quantiles; the gap between curves
    is the distributional separation (a visual KS)."""
    d = _clean(df, signal, outcome)
    fig = go.Figure()
    for grp, vals in _groups(d, signal, outcome).items():
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=grp,
                                 line=dict(color=outcome_color(grp), width=2, shape="hv")))
    fig.update_layout(title=f"{short(signal)} — empirical CDF by outcome",
                      xaxis_title=short(signal), yaxis_title="cumulative proportion",
                      yaxis=dict(range=[0, 1.02]))
    return _size(fig, "wide")


def qq_normal(df, signal, outcome="label"):
    """Normal Q-Q plot (per outcome) — points on the line ⇒ ~normal; curvature ⇒
    skew/heavy tails, the cue that a mean is the wrong summary."""
    d = _clean(df, signal, outcome)
    fig = go.Figure()
    allmin, allmax = [], []
    for grp, vals in _groups(d, signal, outcome).items():
        v = np.sort(vals)
        n = len(v)
        if n < 3:
            continue
        z = _norm_ppf((np.arange(1, n + 1) - 0.5) / n)
        theo = z * v.std(ddof=1) + v.mean()
        fig.add_trace(go.Scatter(x=theo, y=v, mode="markers", name=grp,
                                 marker=dict(color=outcome_color(grp), size=6, opacity=0.7)))
        allmin.append(min(theo.min(), v.min()))
        allmax.append(max(theo.max(), v.max()))
    if allmin:
        lo, hi = min(allmin), max(allmax)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                 line=dict(color=PALETTE["AXIS"], width=1, dash="dot")))
    fig.update_layout(title=f"{short(signal)} — normal Q–Q by outcome",
                      xaxis_title="theoretical quantile", yaxis_title="observed quantile")
    return _size(fig, "wide")


# ---- variable relationships ----------------------------------------------

def corr_heatmap(df, signals, method="pearson"):
    """Correlation matrix across signals — shows redundancy/structure the
    per-signal-vs-label views miss. Diverging scale centered at 0."""
    cols = [s for s in signals if s in df.columns]
    M = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)
    labels = [short(c) for c in cols]
    fig = go.Figure(go.Heatmap(
        z=M.values, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0, PALETTE["DIVERGE_NEG"]], [0.5, PALETTE["DIVERGE_MID"]], [1, PALETTE["DIVERGE_POS"]]],
        text=[[fmt(v, "corr") for v in row] for row in M.values],
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="r", thickness=12)))
    fig.update_layout(title=f"Signal correlation ({method})", xaxis_title="", yaxis_title="")
    return _size(fig, "wide")


def bubble(df, x_sig, y_sig, size_sig, outcome="label"):
    """Scatter with a third variable encoded as marker size."""
    d = df[[x_sig, y_sig, size_sig, outcome]].apply(
        lambda c: pd.to_numeric(c, errors="coerce") if c.name != outcome else c).dropna()
    s = d[size_sig].to_numpy(float)
    srange = (s - s.min()) / (np.ptp(s) or 1)
    fig = go.Figure()
    d["_g"] = d[outcome].astype(str).str.upper()
    for grp in ("FAIL", "PASS"):
        sub = d[d["_g"] == grp]
        if sub.empty:
            continue
        ss = (srange[d["_g"].to_numpy() == grp]) * 28 + 6
        fig.add_trace(go.Scatter(x=sub[x_sig], y=sub[y_sig], mode="markers", name=grp,
                                 marker=dict(color=outcome_color(grp), size=ss, opacity=0.6,
                                             line=dict(color="white", width=1))))
    fig.update_layout(title=f"{short(x_sig)} vs {short(y_sig)} (size = {short(size_sig)})",
                      xaxis_title=short(x_sig), yaxis_title=short(y_sig))
    return _size(fig, "wide")


def quadrant(df, x_sig, y_sig, outcome="label", x_split=None, y_split=None):
    """Scatter split into four quadrants at (median, median) — turns two signals
    into a decision map; quadrant fail-rates are annotated."""
    d = df[[x_sig, y_sig, outcome]].apply(
        lambda c: pd.to_numeric(c, errors="coerce") if c.name != outcome else c).dropna()
    xv, yv = d[x_sig].to_numpy(float), d[y_sig].to_numpy(float)
    xs = float(np.median(xv)) if x_split is None else x_split
    ys = float(np.median(yv)) if y_split is None else y_split
    fig = go.Figure()
    d["_g"] = d[outcome].astype(str).str.upper()
    for grp in ("FAIL", "PASS"):
        sub = d[d["_g"] == grp]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub[x_sig], y=sub[y_sig], mode="markers", name=grp,
                                     marker=dict(color=outcome_color(grp), size=7, opacity=0.7)))
    fig.add_vline(x=xs, line=dict(color=PALETTE["AXIS"], width=1, dash="dot"))
    fig.add_hline(y=ys, line=dict(color=PALETTE["AXIS"], width=1, dash="dot"))
    fig.update_layout(title=f"{short(x_sig)} × {short(y_sig)} quadrants (split at medians)",
                      xaxis_title=short(x_sig), yaxis_title=short(y_sig))
    return _size(fig, "wide")


# ---- model evaluation -----------------------------------------------------

def _roc(score, y):
    """ROC points + AUC for a continuous score vs binary label (1=positive)."""
    score = np.asarray(score, float)
    y = np.asarray(y, float)
    m = ~(np.isnan(score) | np.isnan(y))
    score, y = score[m], y[m]
    P, N = y.sum(), (1 - y).sum()
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    order = np.argsort(-score)
    ys = y[order]
    tpr = np.concatenate([[0], np.cumsum(ys) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - ys) / N])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def roc_curves(df, signals, label_col="is_fail"):
    """Overlaid ROC curves — does each signal discriminate FAIL? AUC in legend.
    A signal with AUC≈1.0 that *is* the label re-measured is leakage, not skill."""
    sigs = [s for s in (signals if isinstance(signals, (list, tuple)) else [signals]) if s in df.columns]
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color=PALETTE["AXIS"], width=1, dash="dot")))
    palette = [PALETTE["ACCENT"], PALETTE["FAIL"], PALETTE["SIGNIFICANT"], PALETTE["PASS"], PALETTE["LEAKY"]]
    for i, s in enumerate(sigs):
        fpr, tpr, auc = _roc(pd.to_numeric(df[s], errors="coerce").to_numpy(float), y)
        if auc < 0.5:  # orient so the curve reads as discrimination strength
            fpr, tpr, auc = _roc(-pd.to_numeric(df[s], errors="coerce").to_numpy(float), y)
            name = f"{short(s)} (AUC {fmt(auc,'corr')}, inverted)"
        else:
            name = f"{short(s)} (AUC {fmt(auc,'corr')})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                 line=dict(color=palette[i % len(palette)], width=2.5)))
    fig.update_layout(title="ROC — signal discrimination of FAIL",
                      xaxis_title="false positive rate", yaxis_title="true positive rate",
                      xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]))
    return _size(fig, "wide")


def confusion_matrix(y_true, y_pred, pos_label="Yes", neg_label="No", title="Confusion matrix"):
    """2×2 confusion heatmap from already-binary truth/pred arrays (1=positive).
    Cells annotated with counts; diagonal = correct."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[m], yp[m]
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    # rows = predicted (Yes,No), cols = actual (Yes,No)
    Z = [[tp, fp], [fn, tn]]
    fig = go.Figure(go.Heatmap(
        z=Z, x=[f"actual {pos_label}", f"actual {neg_label}"],
        y=[f"pred {pos_label}", f"pred {neg_label}"],
        colorscale=[[0, PALETTE["SURFACE"]], [1, PALETTE["ACCENT"]]],
        text=[[f"TP {tp}", f"FP {fp}"], [f"FN {fn}", f"TN {tn}"]],
        texttemplate="%{text}", textfont=dict(size=14), showscale=False))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="")
    return _size(fig, "compact")


def _fit_logistic_std(x, y, iters=300, lr=0.3):
    """Univariate logistic slope in *standardized-x* units (comparable across
    signals). Returns the standardized coefficient b1."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    xs = (x - x.mean()) / x.std()
    b0, b1 = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(b0 + b1 * xs)))
        b0 -= lr * np.mean(p - y)
        b1 -= lr * np.mean((p - y) * xs)
    return b1


def coef_plot(df, signals, label_col="is_fail", n_boot=400, seed=0):
    """Standardized univariate logistic coefficients (signal→FAIL) with bootstrap
    95% CI, forest-style. Comparable across signals; dotted 0 line = no effect."""
    rng = np.random.default_rng(seed)
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    rows = []
    for s in signals:
        if s not in df.columns:
            continue
        x = pd.to_numeric(df[s], errors="coerce").to_numpy(float)
        m = ~(np.isnan(x) | np.isnan(y))
        xm, ym = x[m], y[m]
        if len(xm) < 5:
            continue
        b = _fit_logistic_std(xm, ym)
        boots = []
        idx = np.arange(len(xm))
        for _ in range(n_boot):
            bi = rng.choice(idx, len(idx), replace=True)
            boots.append(_fit_logistic_std(xm[bi], ym[bi]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"signal": s, "effect": b, "ci_lo": lo, "ci_hi": hi,
                     "significant": (lo > 0) or (hi < 0)})
    fig = forest_effects(rows)
    fig.update_layout(title="Std. logistic coefficient (signal → FAIL) ± 95% CI",
                      xaxis_title="standardized log-odds per SD")
    return fig


def calibration_curve(y_true, y_prob, n_bins=10, title="Calibration"):
    """Reliability diagram: mean predicted probability vs observed frequency per
    bin. Generic — pass model P(FAIL) once logit-capturing inference is run."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_prob, float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[m], yp[m]
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (yp >= lo) & (yp < hi if hi < 1 else yp <= hi)
        if sel.sum():
            xs.append(yp[sel].mean())
            ys.append(yt[sel].mean())
            ns.append(int(sel.sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color=PALETTE["AXIS"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="observed",
                             line=dict(color=PALETTE["FAIL"], width=2.5),
                             marker=dict(size=[6 + 24 * n / max(ns) for n in ns] if ns else 8)))
    fig.update_layout(title=title, xaxis_title="mean predicted probability",
                      yaxis_title="observed frequency",
                      xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]))
    return _size(fig, "wide")


# ---- combined multi-signal comparisons (collapse small-multiples) ---------

def groupstats_strip(df, signals, outcome="label"):
    """ONE chart for ALL signals' FAIL-vs-PASS group means, standardized (z-scored
    on the pooled distribution) so different-scale signals share an axis. One row
    per signal: a dumbbell from PASS mean to FAIL mean. Replaces N two-bar/dumbbell
    panels (skill §2). Hover shows the raw means."""
    fig = go.Figure()
    drew_pass = drew_fail = False
    for s in signals:
        if s not in df.columns:
            continue
        x = pd.to_numeric(df[s], errors="coerce")
        mu, sd = x.mean(), x.std(ddof=0)
        if not sd or np.isnan(sd):
            continue
        z = (x - mu) / sd
        lab = df[outcome].astype(str).str.upper()
        fz = z[lab == "FAIL"].mean()
        pz = z[lab == "PASS"].mean()
        fr = x[lab == "FAIL"].mean()
        pr = x[lab == "PASS"].mean()
        y = short(s)
        fig.add_trace(go.Scatter(x=[pz, fz], y=[y, y], mode="lines",
                                 line=dict(color=PALETTE["AXIS"], width=2),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[pz], y=[y], mode="markers", name="PASS",
                                 marker=dict(color=PALETTE["PASS"], size=13),
                                 legendgroup="PASS", showlegend=not drew_pass,
                                 hovertext=f"PASS mean {fmt(pr,'val')}", hoverinfo="text"))
        fig.add_trace(go.Scatter(x=[fz], y=[y], mode="markers", name="FAIL",
                                 marker=dict(color=PALETTE["FAIL"], size=13),
                                 legendgroup="FAIL", showlegend=not drew_fail,
                                 hovertext=f"FAIL mean {fmt(fr,'val')}", hoverinfo="text"))
        drew_pass = drew_fail = True
    fig.add_vline(x=0, line=dict(color=PALETTE["AXIS"], width=1, dash="dot"))
    fig.update_layout(title="Group means by signal (standardized) — FAIL vs PASS",
                      xaxis_title="standardized mean (pooled z-score)", yaxis_title="")
    return _size(fig, "wide")


def failrate_percentile(df, signals, label_col="is_fail", n_bins=5):
    """ONE chart overlaying every signal's fail-rate curve on a shared x-axis of
    within-signal percentile (so different units line up). Replaces N fail-rate-by-
    bin panels; marker size ∝ bin n."""
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    palette = [PALETTE["ACCENT"], PALETTE["FAIL"], PALETTE["SIGNIFICANT"], PALETTE["PASS"], PALETTE["LEAKY"]]
    fig = go.Figure()
    qmid = (np.arange(n_bins) + 0.5) / n_bins * 100
    for i, s in enumerate(signals):
        if s not in df.columns:
            continue
        x = pd.to_numeric(df[s], errors="coerce").to_numpy(float)
        m = ~(np.isnan(x) | np.isnan(y))
        xx, yy = x[m], y[m]
        if len(xx) < n_bins:
            continue
        ranks = xx.argsort().argsort() / max(len(xx) - 1, 1)  # 0..1 percentile
        rates, ns = [], []
        for b in range(n_bins):
            sel = (ranks >= b / n_bins) & (ranks < (b + 1) / n_bins if b < n_bins - 1 else ranks <= 1.0)
            rates.append(float(yy[sel].mean()) if sel.any() else np.nan)
            ns.append(int(sel.sum()))
        sizes = [8 + 22 * (n / max(ns)) for n in ns] if max(ns) else [10] * n_bins
        fig.add_trace(go.Scatter(x=qmid, y=rates, mode="lines+markers", name=short(s),
                                 line=dict(color=palette[i % len(palette)], width=2.5),
                                 marker=dict(size=sizes, color=palette[i % len(palette)])))
    fig.update_layout(title="Fail rate vs signal percentile (all signals)",
                      xaxis_title="within-signal percentile", yaxis_title="fail rate",
                      yaxis=dict(range=[-0.05, 1.05]))
    return _size(fig, "wide")


# ---- decision analysis ----------------------------------------------------

def pareto(labels, values, title="Pareto — contribution by factor"):
    """Bars (desc) + cumulative % line — ranks which factors account for most of
    the total (the 'vital few')."""
    pairs = sorted(zip(labels, [float(v) for v in values]), key=lambda t: -t[1])
    labs = [short(str(lab)) for lab, _ in pairs]
    vals = [v for _, v in pairs]
    total = sum(vals) or 1.0
    cum = np.cumsum(vals) / total * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labs, y=vals, marker_color=PALETTE["ACCENT"], name="count",
                         text=[fmt(v, "count") for v in vals], textposition="outside"))
    fig.add_trace(go.Scatter(x=labs, y=cum, mode="lines+markers", name="cumulative %",
                             yaxis="y2", line=dict(color=PALETTE["FAIL"], width=2)))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="count",
                      yaxis2=dict(title="cumulative %", overlaying="y", side="right",
                                  range=[0, 105], showgrid=False))
    return _size(fig, "wide")


# ---------------------------------------------------------------------------
# matplotlib equivalent (only if a caller uses matplotlib instead of plotly)
# ---------------------------------------------------------------------------
def matplotlib_rcparams() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial"],
        "font.size": 12,
        "axes.edgecolor": PALETTE["AXIS"],
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": PALETTE["GRID"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "none", "axes.facecolor": "none",
        "axes.titlesize": 14, "axes.titlelocation": "left",
        "legend.frameon": False,
    }
