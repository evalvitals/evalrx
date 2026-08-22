---
name: eval-chart-style
version: 0.5.0
description: >
  Chart-type policy + house style for FAIL-vs-PASS LLM/VLM eval analysis
  figures. Use whenever you plot eval results inside an EvalVitals analysis
  sandbox — matplotlib PNGs under figures/ or deterministic chart specs — and
  the point is to NOT default to a bar chart. This skill decides WHAT chart to
  draw and the semantic palette; for journal-grade polish (fonts, export, QA)
  additionally apply the nature-figure skill on the figures you draw.
---

# Eval Chart Style

House chart-type policy for FAIL-vs-PASS eval analysis. It exists to stop three
recurring failures: plotting a **mean as a bar** when the claim depends on the
*distribution*; inconsistent FAIL/PASS colors between charts; and raw machine
output (column ids, `pandas.cut` edges) leaking into axes and titles.

It is the agent-side counterpart of the host module
`evalvitals.analysis.eval_viz_theme` (plotly). If plotly is available you may
`from evalvitals.analysis import eval_viz_theme as viz; viz.apply()` and use its
builders; otherwise apply the rules below directly in matplotlib.

## 0. Chart-type policy (the most important rule)

**A bar's filled area means "amount accumulated from zero." Use bars only for
counts.** For a mean, a proportion, a measurement, or an effect size, a bar
hides the distribution and fakes certainty. Pick by *what the reader must see*:

| What you're showing | DON'T | DO |
|---|---|---|
| Discrete class counts (FAIL/PASS n) | — | grouped bars, n annotated |
| One variable's own distribution (EDA) | — | histogram (continuous) / count bars (categorical) |
| A continuous signal across outcomes | two-bar "mean by outcome" | violin/box + jittered points, one panel |
| Several signals' effects ranked (incl. regression odds ratios) | green/grey bars | horizontal **dot + CI** (forest) |
| Fail rate vs a continuous signal | bare binned bars | binned rate **line** (or logistic curve) + n per bin |
| Predicted probability vs a key predictor | coefficient table only | probability **curve** + CI band, others held at reference |
| Model discrimination / calibration | a lone accuracy bar | **ROC curve** + calibration (reliability) plot |
| Two continuous signals jointly | scatter squeezed small | full-width scatter colored by outcome (+ marginals if easy) |
| Rates across two categoricals | grouped bars ×k | **heatmap** with annotated cells |
| Distribution shape / tail claims | histogram only | ECDF or KDE overlay by outcome |
| Paired/intervention outcomes | two bars | paired slope or discordant counts |

Why distributions: with small FAIL n a mean is outlier-driven, and any
"two sub-populations" story is a *bimodality claim* — a bar can show neither.

**This policy wins.** When another installed skill (e.g. a statistics-method
skill like outcome-driver-analysis) calls for a specific chart at some step,
keep its statistical intent but render it under THIS table's chart-type policy
and §1's palette — e.g. its "side-by-side boxplot" becomes violin/box +
jittered points here.

**Suppress degenerate charts.** A binary/constant signal's fail-rate curve
collapses to one or two dots — don't render it; note it in caveats instead.
Aim for *diversity that matches the data*: across the figure set prefer
violin + ECDF + heatmap + forest + scatter over eight bar charts.

## 1. Semantic palette (color encodes role, not decoration)

These are the light-mode values of the CVD-validated palette the host theme
(`eval_viz_theme`) renders with — using them keeps agent-drawn PNGs and
host-rendered charts on ONE palette. Lock them and reuse in every figure; the
reader must never re-learn who is who:

- FAIL → critical red `#d03b3b`; PASS → good green `#0ca30c` (same side, same
  hue in every chart). These are *status* colors — reserved for outcome/verdict
  roles, never for "series 4".
- Supported / survived-adjudication → the same good green `#0ca30c`;
  inconclusive / did-not-survive → warning amber `#fab219`. Unknown or
  not-yet-adjudicated defaults to amber, never green. Amber is low-contrast on
  light surfaces by design: a status color always ships with a text label,
  never color alone.
- Leakage/sanity signals → muted ink `#898781`, never a "winner" color.

Non-outcome dimensions get their own color logic — do NOT squeeze every panel
into the FAIL/PASS pair plus grey:

- **Categorical series** (checkpoint/model lines, object classes, several
  signals overlaid on one axis): use this fixed slot order, never cycled —
  1 `#2a78d6` blue, 2 `#1baf7a` aqua, 3 `#eda100` yellow, 4 `#008300` green\*,
  5 `#4a3aa7` violet, 6 `#e34948` red\*, 7 `#e87ba4` magenta, 8 `#eb6834`
  orange. (\*Skip the green and red slots whenever PASS-green/FAIL-red appear
  on the same panel — the outcome hues must stay unambiguous.) A 9th series is
  never a new hue: fold it into "Other" or split into small multiples. Color
  follows the entity, not its rank — filtering series must not repaint the
  survivors.
- **Ordered dimensions** (model size 2B→4B→8B, ordered bins, dose/strength
  ladders): ONE hue with a luminance ramp (light → dark = small → large), not
  distinct hues — the ordering should be readable from the ramp alone. Use the
  blue ramp `#86b6ef → #2a78d6 → #104281` (validated: monotone lightness, the
  light end still clears the surface).
- Single-series / neutral measurement → accent blue `#2a78d6`.
- Precedence: semantic role colors always win — wherever the outcome appears,
  FAIL/PASS keep their hues; the series/ramp colors are for panels sliced by
  something other than the outcome. Heatmaps: diverging data (signed effects)
  → blue `#2a78d6` ↔ neutral grey `#f0efec` ↔ red `#e34948` around 0 (the
  categorical red, deliberately NOT the FAIL red — a signed heatmap must not
  impersonate the outcome; never put a hue at the midpoint); magnitude-only
  data → a single-hue sequential ramp from near-white toward `#2a78d6`.

Never print a raw column id (`generated_probe1_false_detection`) on an axis,
tick, or title — use a short human alias (≤ ~12 chars) and keep the raw name in
a caption/hover/table only.

Never print a raw filename (`violin_by_probe.png`) as a figure's on-dashboard
title either — the host strips `.png` and aliases the stem, but only for
`figures/<name>.png` paths it recognizes; give figures descriptive stems
(`attention_focus_by_probe`, not `plot3`) so the alias is readable.

### 1a. Apply the house rcParams before drawing anything

Colors alone don't make a matplotlib PNG match the host plotly charts and the
dashboard's card chrome — font, spines, and grid do. At the top of every
script that draws a figure under `figures/`, before the first `plt.subplots`
or `plt.figure` call:

```python
from evalvitals.analysis.eval_viz_theme import matplotlib_rcparams
import matplotlib.pyplot as plt
plt.rcParams.update(matplotlib_rcparams())
```

If `evalvitals` isn't importable from the sandbox, inline the same rcParams
rather than skipping this step — dropping it is what produces default-navy
bars and visible top/right spines next to the host's themed charts.

## 2. Layout

- Distribution plots and scatters get full width; never squeeze a scatter into
  a narrow subplot.
- One title per chart; don't repeat the section heading inside the figure.
- Label every axis with the *short* name + unit; annotate the zero/reference
  line on effect axes.

## 3. Numbers and bins

- effects/correlations → 2 dp; p-values/statistics → 3 dp; percents → integer
  (`41%`); counts → integer.
- **Never display raw `pandas.cut` edges** (`(113.844, 233.066]`). Render
  human bins: `114–233`. Annotate per-bin n when bins are sparse.

## 4. Statistical honesty

- A signal that *is* the outcome re-measured (fail rate 1.00/0.00, effect ≈
  perfect) is target leakage: demote it to a greyed sanity-check, never rank
  it #1.
- Pick ONE primary effect metric for headline figures; relegate the rest to a
  table, and label which metric every axis shows.
- Descriptive is descriptive: no "confirmed/validated" wording in titles.

## Scope note for EvalVitals sandboxes

Host-rendered chart *specs* stay deterministic (`kind` ∈ bar/line/scatter with
pre-aggregated CSVs) **and §0 applies to them too** — the host picks the form
from the table, so shape the CSV for the form you want:

- class balance (count per class) → ONE composition strip; give `x=outcome`,
  `y=count`;
- a rate or mean across ≤ 3 groups → horizontal **dot + 95% CI**, a lollipop
  when no interval can be formed; give an `n` column and, for a rate, the
  numerator (`n_fail`, `n_with_audit`, `k`) or explicit `ci_low`/`ci_high`.
  A `mean_*` of per-case values never gets an invented interval;
- a numeric signal vs the outcome → the **binned fail-rate line**
  (`bin -> fail_rate, n`); never a two-group "mean of the signal" spec — draw
  the distribution view (violin/box + points) yourself under `figures/`;
- bars remain only for counts and for ranked/many-category tables.

Outcome groups named exactly `FAIL`/`PASS` get the §1 hues; axis labels are
aliased from column ids (§1). The PNGs you draw under `figures/` are still
where violins, ECDFs, heatmaps, forests, and paired-slope figures belong.
Styling only: never change the data, the analysis, or the final result JSON.
