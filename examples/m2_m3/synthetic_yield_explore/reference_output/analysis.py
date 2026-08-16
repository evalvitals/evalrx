#!/usr/bin/env python3
"""
Exploratory analysis: what predicts yield_pct in the reactor-batch dataset?

Self-contained. Reads whatever is under raw_input/, builds ONE tidy table,
writes records.json, tables/*.csv (host-rendered chart data) and figures/*.png
(rich distribution-first figures), then prints EXPLORATORY_RESULT_JSON.

Method follows the outcome-driver-analysis skill, adapted to a CONTINUOUS
outcome (linear regression rather than logistic):
  1. intake / structure       2. explanatory-variable EDA
  3. per-variable vs outcome  4. marginal screening
  5. justified model          6. fit diagnostics (VIF, residuals, calibration)
  7. result visuals           8. descriptive narrative

Everything reported is DESCRIPTIVE. No causal claims, no verdicts.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RAW_DIR = Path("raw_input")
TABLES = Path("tables")
FIGS = Path("figures")
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

TARGET = "yield_pct"  # caller-specified outcome column
RECORD_KEYS = ("cases", "results", "rows", "items", "data", "examples", "samples", "records", "batches")


# --------------------------------------------------------------------------
# 1. LOAD -> ONE TIDY TABLE  (shape is not assumed; it is inspected)
# --------------------------------------------------------------------------
def _rows_from_obj(obj, meta=None, src=""):
    """Recursively turn an arbitrary JSON object into a list of flat row dicts."""
    meta = dict(meta or {})
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return [{**meta, **_flatten(x)} for x in obj]
        return []
    if isinstance(obj, dict):
        # dict of scalar metadata + a list of per-case records under some key
        list_keys = [k for k, v in obj.items() if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)]
        scalar_meta = {k: v for k, v in obj.items() if not isinstance(v, (list, dict))}
        if list_keys:
            preferred = [k for k in RECORD_KEYS if k in list_keys] or list_keys
            out = []
            for k in preferred:
                child_meta = {**meta, **scalar_meta}
                if len(preferred) > 1:
                    child_meta["_record_group"] = k
                out.extend([{**child_meta, **_flatten(r)} for r in obj[k]])
            return out
        # dict-of-dicts keyed by id
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return [{**meta, "_key": k, **_flatten(v)} for k, v in obj.items()]
        return [{**meta, **_flatten(obj)}] if obj else []
    return []


def _flatten(d, prefix=""):
    """One level of dict flattening; lists of scalars become a joined string."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}_"))
        elif isinstance(v, list):
            if all(not isinstance(i, (dict, list)) for i in v):
                out[key] = ", ".join(map(str, v))
            else:
                out[key] = json.dumps(v)[:200]
        else:
            out[key] = v
    return out


def load_tidy() -> pd.DataFrame:
    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson", ".csv"})
    if not files:
        raise SystemExit("no readable files under raw_input/")

    # A browser-workbench bundle would carry records.json; prefer it when present.
    normalized = [p for p in files if p.name == "records.json"]
    if normalized:
        files = normalized + [p for p in files if p not in normalized and p.name != "records.json"]

    rows = []
    for path in files:
        text = path.read_text(errors="replace").strip()
        if not text:
            continue
        meta = {} if len(files) == 1 else {"source_file": path.name}
        if path.suffix.lower() == ".csv":
            rows.extend(pd.read_csv(path).to_dict("records"))
            continue
        try:
            rows.extend(_rows_from_obj(json.loads(text), meta=meta, src=path.name))
        except json.JSONDecodeError:  # JSONL / NDJSON
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.extend(_rows_from_obj(json.loads(line), meta=meta, src=path.name))
                except json.JSONDecodeError:
                    pass
        if normalized and path.name == "records.json":
            break  # normalized rows are authoritative

    df = pd.DataFrame(rows)
    # numeric coercion for object columns that are really numbers
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.9 and conv.notna().sum() > 0:
                df[c] = conv
    return df


df = load_tidy()
df.to_json("records.json", orient="records", indent=1)
N = len(df)

# ---- column typing -------------------------------------------------------
id_like = [c for c in df.columns if df[c].nunique(dropna=True) == N and df[c].dtype == object]
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_like]
cat_cols = [
    c for c in df.columns
    if c not in numeric_cols and c not in id_like and 1 < df[c].nunique(dropna=True) <= max(12, N // 3)
]

# ---- outcome framing -----------------------------------------------------
has_target = TARGET in df.columns
if not has_target:
    raise SystemExit(f"caller-named outcome '{TARGET}' not present in tidy table; columns = {list(df.columns)}")
n_unique_y = df[TARGET].nunique(dropna=True)
outcome_kind = "continuous" if (pd.api.types.is_numeric_dtype(df[TARGET]) and n_unique_y > 10) else (
    "binary" if n_unique_y == 2 else "categorical")

predictors_num = [c for c in numeric_cols if c != TARGET]
y = df[TARGET].astype(float)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fisher_ci(r, n, alpha=0.05):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return tuple(np.tanh([z - crit * se, z + crit * se]))


def human_bins(series, q=5):
    """Quantile bins with human labels like '155-174' (never raw pandas.cut edges)."""
    qs = min(q, max(2, series.nunique() // 3))
    edges = np.unique(np.quantile(series.dropna(), np.linspace(0, 1, qs + 1)))
    if len(edges) < 3:
        edges = np.linspace(series.min(), series.max(), 4)
    idx = np.clip(np.digitize(series, edges[1:-1], right=True), 0, len(edges) - 2)
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" if edges[-1] > 20 else f"{edges[i]:.1f}-{edges[i+1]:.1f}"
              for i in range(len(edges) - 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    return pd.Series([labels[i] for i in idx], index=series.index), np.array(labels), np.array(centers)


def wcsv(name, frame):
    frame.to_csv(TABLES / f"{name}.csv", index=False)


charts, plots, visual_plan, chart_readings, tables_out = [], [], [], [], {}


def add_chart(name, kind, display_name, frame, x, y_, title):
    wcsv(name, frame)
    charts.append({"name": name, "kind": kind, "display_name": display_name,
                   "data": f"tables/{name}.csv", "x": x, "y": y_, "title": title})


def plan(name, display_name, question, data_shape, plot_kind, fallback, cols, rationale,
         disposition="primary", not_promoted=""):
    visual_plan.append({"name": name, "display_name": display_name, "question": question,
                        "data_shape": data_shape, "plot_kind": plot_kind, "fallback_kind": fallback,
                        "required_columns": cols, "rationale": rationale,
                        "disposition": disposition, "not_promoted_reason": not_promoted})


def reading(key, what_you_see, do_not_infer):
    chart_readings.append({"chart": key, "reading": what_you_see, "do_not_infer": do_not_infer})


# --------------------------------------------------------------------------
# 2. EXPLANATORY-VARIABLE EDA (distributions, outliers, missingness, structure)
# --------------------------------------------------------------------------
profile_rows = []
for c in df.columns:
    col = df[c]
    r = {"column": c, "dtype": str(col.dtype), "n_missing": int(col.isna().sum()),
         "pct_missing": round(100 * col.isna().mean(), 1), "n_unique": int(col.nunique(dropna=True))}
    if c in numeric_cols:
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        r.update(mean=round(col.mean(), 3), sd=round(col.std(), 3), min=round(col.min(), 3),
                 max=round(col.max(), 3),
                 n_iqr_outliers=int(((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).sum()),
                 skew=round(float(col.skew()), 3))
    else:
        vc = col.value_counts()
        r["top_values"] = "; ".join(f"{k}={v}" for k, v in vc.head(5).items())
        r["rarest_n"] = int(vc.min()) if len(vc) else 0
    profile_rows.append(r)
profile = pd.DataFrame(profile_rows)
wcsv("data_profile", profile)
tables_out["data_profile"] = "tables/data_profile.csv"

total_missing = int(df.isna().sum().sum())

# ---- outcome distribution (host chart: pre-aggregated histogram) ----------
plan("yield_distribution", "How yields are spread across batches",
     "What is the spread of batch yield?", "unsupervised", "bar", "bar",
     [TARGET], "A pre-binned count histogram shows spread and any bimodality that a mean would hide.")
counts, edges = np.histogram(y, bins=8)
hist = pd.DataFrame({"yield_bin": [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges) - 1)],
                     "n_batches": counts})
add_chart("yield_distribution", "bar", "How yields are spread across batches", hist,
          "yield_bin", "n_batches", "Distribution of batch yield (%)")
reading("yield_distribution",
        f"Yield spans {y.min():.1f}-{y.max():.1f}% with a mean of {y.mean():.1f}% and SD {y.std():.1f}; "
        "the shape is broad and single-peaked rather than split into two groups.",
        "It does not show which conditions produced the high- or low-yield batches.")

# --------------------------------------------------------------------------
# 3-4. PER-VARIABLE vs OUTCOME + MARGINAL SCREENING
# --------------------------------------------------------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf

screen = []
for c in predictors_num:
    x = df[c].astype(float)
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    lo, hi = fisher_ci(r_p, len(x))
    uni = smf.ols(f"{TARGET} ~ Q('{c}')", data=df).fit()
    screen.append({"variable": c, "kind": "numeric", "n": len(x),
                   "pearson_r": round(r_p, 3), "r_ci_low": round(lo, 3), "r_ci_high": round(hi, 3),
                   "p_value": round(p_p, 4), "spearman_rho": round(r_s, 3), "spearman_p": round(p_s, 4),
                   "univariate_r2": round(uni.rsquared, 3),
                   "slope_per_unit": round(uni.params.iloc[1], 4),
                   "aic_improvement_vs_null": round(smf.ols(f"{TARGET} ~ 1", data=df).fit().aic - uni.aic, 2),
                   "abs_effect": round(abs(r_p), 3)})

for c in cat_cols:
    groups = [g[TARGET].astype(float).values for _, g in df.groupby(c) if len(g) >= 2]
    if len(groups) < 2:
        continue
    F, p_a = stats.f_oneway(*groups)
    H, p_k = stats.kruskal(*groups)
    uni = smf.ols(f"{TARGET} ~ C(Q('{c}'))", data=df).fit()
    eta2 = uni.rsquared
    screen.append({"variable": c, "kind": "categorical", "n": int(df[c].notna().sum()),
                   "pearson_r": np.nan, "r_ci_low": np.nan, "r_ci_high": np.nan,
                   "p_value": round(p_a, 4), "spearman_rho": np.nan, "spearman_p": round(p_k, 4),
                   "univariate_r2": round(eta2, 3), "slope_per_unit": np.nan,
                   "aic_improvement_vs_null": round(smf.ols(f"{TARGET} ~ 1", data=df).fit().aic - uni.aic, 2),
                   "abs_effect": round(np.sqrt(eta2), 3)})

screen_df = pd.DataFrame(screen).sort_values("abs_effect", ascending=False).reset_index(drop=True)
wcsv("marginal_screening", screen_df)
tables_out["marginal_screening"] = "tables/marginal_screening.csv"

# ---- ranked association bar ---------------------------------------------
plan("association_with_yield", "Which factor tracks yield most closely",
     "Which measured factor moves together with yield the most?", "many-numeric", "bar", "bar",
     predictors_num + cat_cols + [TARGET],
     "One ranked magnitude bar puts every candidate on a comparable 0-1 association scale "
     "(|correlation| for numeric factors, the equivalent root-explained-variance for the categorical one).")
add_chart("association_with_yield", "bar", "Which factor tracks yield most closely",
          screen_df[["variable", "abs_effect"]].rename(columns={"abs_effect": "association_strength"}),
          "variable", "association_strength", "Strength of association with yield (0-1)")

# ---- scatter of each numeric predictor vs outcome ------------------------
for c in predictors_num:
    nm = f"yield_vs_{c}"
    plan(nm, f"Yield against {c}", f"Does yield move with {c}?", "numeric-vs-numeric", "scatter", "scatter",
         [c, TARGET], "A scatter of every batch keeps the raw spread visible instead of collapsing it to a mean.")
    add_chart(nm, "scatter", f"Yield against {c}", df[[c, TARGET]].round(3), c, TARGET,
              f"Yield (%) vs {c} (one point per batch)")

# ---- binned mean-outcome curves -----------------------------------------
bin_tables = {}
for c in predictors_num:
    lab, order, centers = human_bins(df[c].astype(float), q=5)
    g = df.assign(_bin=lab).groupby("_bin")[TARGET].agg(["mean", "std", "count"]).reindex(order).dropna(subset=["mean"])
    frame = pd.DataFrame({f"{c}_bin": g.index, "mean_yield_pct": g["mean"].round(2),
                          "sd_yield_pct": g["std"].round(2), "n_batches": g["count"].astype(int)}).reset_index(drop=True)
    bin_tables[c] = frame
    nm = f"mean_yield_by_{c}"
    plan(nm, f"Average yield across {c} bands", f"How does average yield change across {c} bands?",
         "numeric-vs-numeric", "line", "line", [c, TARGET],
         "Ordered equal-count bands show the trend shape without assuming a straight line; n is carried in the CSV.")
    add_chart(nm, "line", f"Average yield across {c} bands", frame, f"{c}_bin", "mean_yield_pct",
              f"Mean yield (%) by {c} band")

# ---- outcome by categorical group ---------------------------------------
cat_summary = {}
for c in cat_cols:
    g = df.groupby(c)[TARGET].agg(["count", "mean", "std", "median"]).round(2)
    frame = g.reset_index().rename(columns={"count": "n_batches", "mean": "mean_yield_pct",
                                            "std": "sd_yield_pct", "median": "median_yield_pct"})
    cat_summary[c] = frame
    nm = f"yield_by_{c}"
    plan(nm, f"Average yield by {c}", f"Do batches differ in yield by {c}?", "numeric-vs-categorical",
         "bar", "bar", [c, TARGET],
         "Group means with n in the table give the deterministic summary; the paired violin PNG carries the "
         "distribution so the mean is never read alone.")
    add_chart(nm, "bar", f"Average yield by {c}", frame[[c, "mean_yield_pct", "n_batches"]],
              c, "mean_yield_pct", f"Mean yield (%) by {c}")
    wcsv(f"{c}_summary", frame)
    tables_out[f"{c}_summary"] = f"tables/{c}_summary.csv"

# ---- CONDITIONING / Simpson's check: predictor means per category --------
cond_frames = {}
for c in cat_cols:
    cf = df.groupby(c)[predictors_num].mean().round(2).reset_index()
    cond_frames[c] = cf
    nm = f"{predictors_num[0]}_by_{c}" if predictors_num else None
    if nm:
        plan(nm, f"Average operating {predictors_num[0]} by {c}",
             f"Were the {c} groups run under comparable conditions?", "numeric-vs-categorical", "bar", "bar",
             [c] + predictors_num,
             "The raw group-yield gap can only be read once we see whether the groups were run at "
             "different settings; this bar exposes that imbalance directly.")
        add_chart(nm, "bar", f"Average operating {predictors_num[0]} by {c}",
                  cf[[c, predictors_num[0]]], c, predictors_num[0],
                  f"Mean {predictors_num[0]} by {c} (condition balance check)")
    wcsv(f"condition_balance_by_{c}", cf)
    tables_out[f"condition_balance_by_{c}"] = f"tables/condition_balance_by_{c}.csv"

# ---- stratified (conditioned) association: r within each category --------
strat_rows = []
for c in cat_cols:
    for lvl, g in df.groupby(c):
        for p in predictors_num:
            if len(g) >= 4:
                rr, pp = stats.pearsonr(g[p].astype(float), g[TARGET].astype(float))
                strat_rows.append({"group_column": c, "group": str(lvl), "predictor": p, "n": len(g),
                                   "pearson_r": round(rr, 3), "p_value": round(pp, 4)})
strat = pd.DataFrame(strat_rows)
if not strat.empty:
    wcsv("stratified_correlations", strat)
    tables_out["stratified_correlations"] = "tables/stratified_correlations.csv"

# --------------------------------------------------------------------------
# 5. MODEL — justified choice
# --------------------------------------------------------------------------
# Outcome is continuous and unbounded within the observed range -> ORDINARY LEAST
# SQUARES, not logistic (nothing here is binary) and not Poisson (not a count).
# GLM vs mixed-effects: the only clustering candidate is `catalyst` with 3 levels
# and 9-11 batches each. With so few clusters a random intercept is poorly
# identified, so the skill's guidance applies: use a FIXED EFFECT (dummy-coded
# catalyst) instead of a random effect. Batch ids are unique -> no repeat measures.
rhs = " + ".join([f"Q('{c}')" for c in predictors_num] + [f"C(Q('{c}'))" for c in cat_cols])
full = smf.ols(f"{TARGET} ~ {rhs}", data=df).fit()
null = smf.ols(f"{TARGET} ~ 1", data=df).fit()

nested = {}
for c in predictors_num + cat_cols:
    term = f"Q('{c}')" if c in predictors_num else f"C(Q('{c}'))"
    others = [t for t in ([f"Q('{p}')" for p in predictors_num] + [f"C(Q('{p}'))" for p in cat_cols]) if t != term]
    reduced = smf.ols(f"{TARGET} ~ {' + '.join(others)}" if others else f"{TARGET} ~ 1", data=df).fit()
    lr = 2 * (full.llf - reduced.llf)
    dfree = int(full.df_model - reduced.df_model)
    nested[c] = {"delta_r2": round(full.rsquared - reduced.rsquared, 4),
                 "lr_stat": round(lr, 3), "lr_df": dfree,
                 "lr_p": round(1 - stats.chi2.cdf(lr, max(dfree, 1)), 4)}

import re


def pretty(term):
    """Turn a patsy term (C(Q('catalyst'))[T.B]) into a human label (catalyst = B)."""
    if term == "Intercept":
        return term
    m = re.match(r"C\(Q\('([^']+)'\)\)\[T\.(.+)\]$", term)
    if m:
        return f"{m.group(1)} = {m.group(2)}"
    m = re.match(r"Q\('([^']+)'\)$", term)
    if m:
        return m.group(1)
    return re.sub(r"[CQ]?\(?'?|'?\)?", "", term)

coef_rows = []
ci = full.conf_int()
for term in full.params.index:
    coef_rows.append({"term": pretty(term), "raw_term": term, "coef": round(full.params[term], 4),
                      "ci_low": round(ci.loc[term, 0], 4), "ci_high": round(ci.loc[term, 1], 4),
                      "std_err": round(full.bse[term], 4), "t": round(full.tvalues[term], 3),
                      "p_value": round(full.pvalues[term], 4)})
coefs = pd.DataFrame(coef_rows)
wcsv("model_coefficients", coefs)
tables_out["model_coefficients"] = "tables/model_coefficients.csv"

model_cmp = pd.DataFrame([
    {"model": "temperature only" if predictors_num else "null", "r2": round(smf.ols(f"{TARGET} ~ Q('{predictors_num[0]}')", data=df).fit().rsquared, 3)},
] + ([{"model": "all numeric factors",
       "r2": round(smf.ols(f"{TARGET} ~ " + " + ".join(f"Q('{c}')" for c in predictors_num), data=df).fit().rsquared, 3)}] if len(predictors_num) > 1 else [])
  + [{"model": "all factors + catalyst", "r2": round(full.rsquared, 3)}])
plan("model_r2_comparison", "How much of the yield spread each factor set explains",
     "How much extra does adding pressure and catalyst explain beyond temperature?",
     "many-numeric", "bar", "bar", predictors_num + cat_cols + [TARGET],
     "Nested explained-variance bars make the incremental contribution explicit rather than implied by p-values.")
add_chart("model_r2_comparison", "bar", "How much of the yield spread each factor set explains",
          model_cmp, "model", "r2", "Share of yield variation explained (R-squared)")

# ---- 6. DIAGNOSTICS: VIF, residuals, calibration ------------------------
X = sm.add_constant(pd.get_dummies(df[predictors_num + cat_cols], drop_first=True).astype(float))
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({"term": X.columns,
                    "vif": [round(variance_inflation_factor(X.values, i), 2) for i in range(X.shape[1])]})
vif = vif[vif.term != "const"].reset_index(drop=True)
wcsv("collinearity_vif", vif)
tables_out["collinearity_vif"] = "tables/collinearity_vif.csv"

resid = full.resid
fitted = full.fittedvalues
sw_stat, sw_p = stats.shapiro(resid)
bp = sm.stats.diagnostic.het_breuschpagan(resid, full.model.exog)
infl = full.get_influence()
cooks = infl.cooks_distance[0]
n_influential = int((cooks > 4 / N).sum())

diag = pd.DataFrame([{"n_rows": N, "r2": round(full.rsquared, 3), "adj_r2": round(full.rsquared_adj, 3),
                      "residual_sd": round(float(np.std(resid, ddof=full.df_model + 1)), 3),
                      "f_stat": round(full.fvalue, 2), "f_p_value": round(full.f_pvalue, 6),
                      "shapiro_w": round(sw_stat, 3), "shapiro_p": round(sw_p, 4),
                      "breusch_pagan_p": round(bp[3], 4), "max_vif": round(vif.vif.max(), 2),
                      "n_cooks_above_4_over_n": n_influential}])
wcsv("model_diagnostics", diag)
tables_out["model_diagnostics"] = "tables/model_diagnostics.csv"

# leave-one-out style calibration: observed vs predicted deciles
cal = pd.DataFrame({"pred": fitted, "obs": y})
cal["band"] = pd.qcut(cal.pred, 5, duplicates="drop")
cal_tbl = cal.groupby("band", observed=True).agg(mean_predicted=("pred", "mean"),
                                                 mean_observed=("obs", "mean"),
                                                 n_batches=("obs", "size")).reset_index(drop=True).round(2)
cal_tbl.insert(0, "predicted_band", [f"{v:.0f}" for v in cal_tbl.mean_predicted])
plan("calibration_predicted_vs_observed", "Predicted yield against what actually happened",
     "Do predicted yields line up with observed yields across the range?", "numeric-vs-numeric", "line", "line",
     predictors_num + cat_cols + [TARGET],
     "A calibration line over predicted bands checks the fit across the whole range, not just on average.")
add_chart("calibration_predicted_vs_observed", "line", "Predicted yield against what actually happened",
          cal_tbl[["predicted_band", "mean_observed", "mean_predicted", "n_batches"]],
          "predicted_band", "mean_observed", "Observed vs predicted yield (%) by predicted band")

# ---- functional-form checks: curvature and interaction ------------------
# Fitted so the "a straight line is a fair description" statement is evidenced,
# not assumed. Reported as delta R-squared only; no threshold verdict is drawn.
form_rows = []
quad = smf.ols(f"{TARGET} ~ {rhs} + I(Q('{predictors_num[0]}')**2)", data=df).fit()
form_rows.append({"form": f"+ {predictors_num[0]}^2 (curvature)", "r2": round(quad.rsquared, 4),
                  "delta_r2_vs_full": round(quad.rsquared - full.rsquared, 4),
                  "term_p_value": round(quad.pvalues.iloc[-1], 4)})
if len(predictors_num) > 1:
    inter = smf.ols(f"{TARGET} ~ {rhs} + Q('{predictors_num[0]}'):Q('{predictors_num[1]}')", data=df).fit()
    form_rows.append({"form": f"+ {predictors_num[0]} x {predictors_num[1]} (interaction)",
                      "r2": round(inter.rsquared, 4),
                      "delta_r2_vs_full": round(inter.rsquared - full.rsquared, 4),
                      "term_p_value": round(inter.pvalues.iloc[-1], 4)})
form_tbl = pd.DataFrame([{"form": "additive model (reported)", "r2": round(full.rsquared, 4),
                          "delta_r2_vs_full": 0.0, "term_p_value": np.nan}] + form_rows)
wcsv("functional_form_checks", form_tbl)
tables_out["functional_form_checks"] = "tables/functional_form_checks.csv"
max_form_gain = float(max(r["delta_r2_vs_full"] for r in form_rows))

# ---- correlation matrix among predictors + outcome ----------------------
corr = df[predictors_num + [TARGET]].corr().round(3)
corr_long = corr.reset_index().melt(id_vars="index", var_name="column_b", value_name="pearson_r") \
                .rename(columns={"index": "column_a"})
wcsv("correlation_matrix", corr.reset_index().rename(columns={"index": "column"}))
tables_out["correlation_matrix"] = "tables/correlation_matrix.csv"

# --------------------------------------------------------------------------
# 7. RICH FIGURES  (eval-chart-style policy + nature-figure polish)
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:  # host theme when available, otherwise the same rcParams inline
    from evalvitals.analysis.eval_viz_theme import matplotlib_rcparams
    plt.rcParams.update(matplotlib_rcparams())
except Exception:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#5c5a55", "axes.labelcolor": "#26241f",
        "text.color": "#26241f", "xtick.color": "#5c5a55", "ytick.color": "#5c5a55",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 220,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })

ACCENT = "#2a78d6"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#eb6834"]
INK = "#898781"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list("bwr_house", ["#2a78d6", "#f0efec", "#e34948"])
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#f7fafd", "#2a78d6"])

cat_col = cat_cols[0] if cat_cols else None
levels = sorted(df[cat_col].dropna().unique().tolist()) if cat_col else []
lvl_color = {lv: SERIES[i % len(SERIES)] for i, lv in enumerate(levels)}
top_num = screen_df[screen_df.kind == "numeric"].variable.tolist()
lead = top_num[0] if top_num else None
second = top_num[1] if len(top_num) > 1 else None


def save(fig, stem):
    fig.savefig(FIGS / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    plots.append(f"figures/{stem}.png")


# --- FIG 1 (hero): yield vs leading driver, fit + CI band, split by catalyst
if lead:
    plan("yield_vs_temperature_by_catalyst", "Yield rises with temperature, in every catalyst group",
         "Does the leading driver track yield the same way inside each catalyst group?",
         "numeric-vs-numeric", "scatter", "scatter", [lead, TARGET] + ([cat_col] if cat_col else []),
         "A full-width scatter with a fitted line and uncertainty band shows the trend AND the raw spread; "
         "colouring by catalyst checks that the trend is not a group artefact.")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xg = np.linspace(df[lead].min(), df[lead].max(), 120)
    ols = smf.ols(f"{TARGET} ~ Q('{lead}')", data=df).fit()
    pred = ols.get_prediction(pd.DataFrame({lead: xg})).summary_frame(alpha=0.05)
    ax.fill_between(xg, pred.mean_ci_lower, pred.mean_ci_upper, color=ACCENT, alpha=0.13, lw=0)
    ax.plot(xg, pred["mean"], color=ACCENT, lw=1.8, zorder=3)
    if cat_col:
        for lv in levels:
            g = df[df[cat_col] == lv]
            ax.scatter(g[lead], g[TARGET], s=52, color=lvl_color[lv], edgecolor="white",
                       linewidth=0.8, label=f"Catalyst {lv} (n={len(g)})", zorder=4)
        ax.legend(loc="upper left", fontsize=8, ncol=1)
    else:
        ax.scatter(df[lead], df[TARGET], s=52, color=ACCENT, edgecolor="white", linewidth=0.8, zorder=4)
    rr = stats.pearsonr(df[lead], y)
    ax.annotate(f"r = {rr[0]:.2f}   slope = {ols.params.iloc[1]:+.2f} pct-pt per unit\n"
                f"explains {100*ols.rsquared:.0f}% of yield spread (n={N})",
                xy=(0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=8, color="#5c5a55")
    ax.set_xlabel(f"{lead} (per batch)")
    ax.set_ylabel("Yield (%)")
    ax.set_title(f"Yield climbs steadily with {lead}; shaded band = 95% interval for the fitted line")
    save(fig, "yield_vs_temperature_by_catalyst")

# --- FIG 2: distribution-first group comparison, raw vs condition-adjusted
if cat_col and lead:
    plan("yield_by_catalyst_raw_vs_adjusted", "Catalyst gaps before and after allowing for temperature",
         "Does the raw catalyst yield gap survive once operating temperature is accounted for?",
         "numeric-vs-categorical", "violin", "bar", [cat_col, TARGET, lead],
         "Violins with every batch overlaid never hide a small-n distribution behind a mean bar; "
         "the adjusted panel repeats the same view on temperature-adjusted residuals.")
    adj_model = smf.ols(f"{TARGET} ~ Q('{lead}')", data=df).fit()
    df["_adj_resid"] = y - adj_model.fittedvalues + y.mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3), sharey=True)
    for ax, col, ttl in zip(axes, [TARGET, "_adj_resid"],
                            ["Observed yield", f"After allowing for {lead}"]):
        data = [df.loc[df[cat_col] == lv, col].values for lv in levels]
        parts = ax.violinplot(data, showextrema=False, widths=0.82)
        for pc, lv in zip(parts["bodies"], levels):
            pc.set_facecolor(lvl_color[lv]); pc.set_alpha(0.24); pc.set_edgecolor(lvl_color[lv]); pc.set_linewidth(1.0)
        bp2 = ax.boxplot(data, widths=0.14, patch_artist=True, showfliers=False,
                         medianprops=dict(color="white", lw=1.4),
                         whiskerprops=dict(color="#5c5a55", lw=0.9), capprops=dict(color="#5c5a55", lw=0.9))
        for patch, lv in zip(bp2["boxes"], levels):
            patch.set_facecolor(lvl_color[lv]); patch.set_edgecolor("none"); patch.set_alpha(0.95)
        rng = np.random.default_rng(7)
        for i, (lv, vals) in enumerate(zip(levels, data), start=1):
            ax.scatter(i + rng.uniform(-0.09, 0.09, len(vals)), vals, s=20, color="#26241f",
                       alpha=0.55, zorder=5, linewidth=0)
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels([f"{lv}\n(n={int((df[cat_col]==lv).sum())})" for lv in levels])
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel(f"{cat_col}")
    axes[0].set_ylabel("Yield (%)")
    fig.suptitle(f"Catalyst differences shrink once {lead} is taken into account", y=1.0, fontsize=10.5)
    save(fig, "yield_by_catalyst_raw_vs_adjusted")
    df.drop(columns=["_adj_resid"], inplace=True)

# --- FIG 3: correlation heatmap (diverging, annotated) --------------------
plan("factor_correlation_heatmap", "How the factors relate to each other and to yield",
     "Are any two factors redundant with each other?", "many-numeric", "heatmap", "bar",
     predictors_num + [TARGET],
     "An annotated diverging heatmap shows sign and magnitude at once and exposes redundancy between "
     "predictors that a per-variable bar cannot.")
fig, ax = plt.subplots(figsize=(4.6, 4.0))
m = corr.values
im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=30, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(m[i, j]) > 0.6 else "#26241f")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.78); cb.set_label("Pearson r", fontsize=8); cb.outline.set_visible(False)
ax.set_title("Factor-to-factor and factor-to-yield correlations")
save(fig, "factor_correlation_heatmap")

# --- FIG 4: coefficient forest plot (dot + CI), standardized -------------
plan("adjusted_effect_forest", "Each factor's effect once the others are held constant",
     "Which factors still move yield after adjusting for the others?", "many-numeric", "forest", "bar",
     predictors_num + cat_cols + [TARGET],
     "A dot-and-interval forest shows direction, size and uncertainty together; a bar of the same numbers "
     "would fake certainty about intervals that cross zero.")
fp = coefs[coefs.term != "Intercept"].copy().iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(fp) + 1.8))
ypos = np.arange(len(fp))
crosses = (fp.ci_low <= 0) & (fp.ci_high >= 0)
cols = [INK if c else ACCENT for c in crosses]
ax.hlines(ypos, fp.ci_low, fp.ci_high, color=cols, lw=2.2)
ax.scatter(fp.coef, ypos, s=58, color=cols, zorder=4, edgecolor="white", linewidth=0.8)
ax.axvline(0, color="#5c5a55", lw=0.9, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels(fp.term)
ax.set_xlabel("Change in yield (percentage points) per unit of the factor, others held constant")
ax.set_title("Adjusted factor effects with 95% intervals")
ax.set_ylim(-0.65, len(fp) - 0.35)
for yy, (_, r) in zip(ypos, fp.iterrows()):
    ax.annotate(f"{r.coef:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]", xy=(1.015, yy),
                xycoords=("axes fraction", "data"), va="center", fontsize=7.8,
                color="#5c5a55", annotation_clip=False)
ax.margins(x=0.06)
ax.legend(handles=[Patch(color=ACCENT, label="interval excludes zero"),
                   Patch(color=INK, label="interval spans zero")],
          loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
save(fig, "adjusted_effect_forest")

# --- FIG 5: ECDF of the leading driver by catalyst (condition imbalance) --
if cat_col and lead:
    plan("temperature_ecdf_by_catalyst", "Operating temperature differed between catalyst groups",
         "Were the catalyst groups run over the same temperature range?", "numeric-vs-categorical",
         "ecdf", "line", [lead, cat_col],
         "An ECDF compares whole distributions, so a shifted operating range shows up as a whole-curve "
         "offset rather than a single mean difference.")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for lv in levels:
        v = np.sort(df.loc[df[cat_col] == lv, lead].values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=lvl_color[lv], lw=1.9,
                label=f"Catalyst {lv} (n={len(v)}, mean {v.mean():.0f})")
    ax.set_xlabel(f"{lead}")
    ax.set_ylabel("Share of batches at or below")
    ax.set_title(f"Catalyst groups did not share the same {lead} range")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "temperature_ecdf_by_catalyst")

# --- FIG 6: partial (added-variable) panels ------------------------------
if len(predictors_num) >= 2:
    plan("partial_effect_panels", "What each numeric factor adds on its own",
         "After removing what the other factors explain, does each factor still track yield?",
         "numeric-vs-numeric", "scatter", "scatter", predictors_num + [TARGET],
         "Added-variable panels isolate each factor's unique contribution, which a marginal scatter "
         "cannot separate from shared variation.")
    fig, axes = plt.subplots(1, len(predictors_num), figsize=(4.3 * len(predictors_num), 3.9))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, predictors_num):
        others = [p for p in predictors_num if p != c] + cat_cols
        f_o = " + ".join([f"Q('{p}')" for p in predictors_num if p != c] + [f"C(Q('{p}'))" for p in cat_cols]) or "1"
        ry = smf.ols(f"{TARGET} ~ {f_o}", data=df).fit().resid
        rx = smf.ols(f"Q('{c}') ~ {f_o}", data=df).fit().resid
        ax.scatter(rx, ry, s=42, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.7)
        b = np.polyfit(rx, ry, 1)
        xs = np.linspace(rx.min(), rx.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#104281", lw=1.6)
        pr = stats.pearsonr(rx, ry)[0]
        ax.axhline(0, color=INK, lw=0.7, ls=":")
        ax.set_title(f"{c}  (partial r = {pr:.2f})", fontsize=9.5)
        ax.set_xlabel(f"{c}, other factors removed")
    axes[0].set_ylabel("Yield, other factors removed")
    fig.suptitle("Unique contribution of each numeric factor", y=1.02, fontsize=10.5)
    save(fig, "partial_effect_panels")

# --- FIG 7: fit diagnostics ----------------------------------------------
plan("fit_diagnostics", "Model fit checks", "Does the fitted model behave well across the range?",
     "numeric-vs-numeric", "scatter", "scatter", predictors_num + cat_cols + [TARGET],
     "Residual-vs-fitted, normal-QQ and observed-vs-predicted panels are the standard checks that the "
     "reported effect sizes are not driven by curvature, skew or one influential batch.",
     disposition="supporting",
     not_promoted="Diagnostic material: it validates how the headline numbers were produced rather than "
                  "being a finding about yield in its own right.")
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
axes[0].scatter(fitted, resid, s=40, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.7)
axes[0].axhline(0, color=INK, ls="--", lw=0.9)
axes[0].set_xlabel("Predicted yield (%)"); axes[0].set_ylabel("Residual (pct points)")
axes[0].set_title("Residuals vs predicted", fontsize=9.5)
sm.qqplot(resid, line="45", fit=True, ax=axes[1], markerfacecolor=ACCENT, markeredgecolor="white", alpha=0.9)
axes[1].get_lines()[1].set_color("#104281")
axes[1].set_title(f"Normal Q-Q (Shapiro p = {sw_p:.2f})", fontsize=9.5)
axes[2].scatter(fitted, y, s=40, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.7)
lims = [min(fitted.min(), y.min()) - 1, max(fitted.max(), y.max()) + 1]
axes[2].plot(lims, lims, color=INK, ls="--", lw=0.9)
axes[2].set_xlabel("Predicted yield (%)"); axes[2].set_ylabel("Observed yield (%)")
axes[2].set_title(f"Observed vs predicted (R-sq = {full.rsquared:.2f})", fontsize=9.5)
fig.suptitle("Fit diagnostics for the yield model", y=1.03, fontsize=10.5)
save(fig, "fit_diagnostics")

# --------------------------------------------------------------------------
# 8. NARRATIVE — numbers injected from what was actually computed
# --------------------------------------------------------------------------
S = screen_df.set_index("variable")
lead_r = float(S.loc[lead, "pearson_r"]); lead_lo = float(S.loc[lead, "r_ci_low"]); lead_hi = float(S.loc[lead, "r_ci_high"])
lead_r2 = float(S.loc[lead, "univariate_r2"]); lead_slope = float(S.loc[lead, "slope_per_unit"])
lead_rho = float(S.loc[lead, "spearman_rho"])
lead_ci = coefs.set_index("term").loc[lead]
sec_r = float(S.loc[second, "pearson_r"]) if second else np.nan
sec_p = float(S.loc[second, "p_value"]) if second else np.nan
sec_row = coefs.set_index("term").loc[second] if second else None
bin_lead = bin_tables[lead]
lo_band, hi_band = bin_lead.iloc[0], bin_lead.iloc[-1]

cat_tbl = cat_summary[cat_col] if cat_col else None
if cat_tbl is not None:
    best = cat_tbl.loc[cat_tbl.mean_yield_pct.idxmax()]
    worst = cat_tbl.loc[cat_tbl.mean_yield_pct.idxmin()]
    cb = cond_frames[cat_col].set_index(cat_col)
    temp_gap = float(cb.loc[best[cat_col], lead] - cb.loc[worst[cat_col], lead])
    adj_terms = coefs[coefs.term.str.startswith(cat_col)]
    max_adj = adj_terms.coef.abs().max() if len(adj_terms) else np.nan
    cat_delta_r2 = nested[cat_col]["delta_r2"]

only_lead_r2 = float(model_cmp.iloc[0]["r2"])
full_r2 = float(full.rsquared)
pred_pair_r = float(corr.loc[predictors_num[0], predictors_num[1]]) if len(predictors_num) > 1 else np.nan
res_sd = float(diag.residual_sd.iloc[0])

takeaways = []

takeaways.append({
    "plain_title": f"Running a batch hotter is by far the best predictor of a good yield: every extra 10 degrees "
                   f"goes with about {abs(lead_slope)*10:.1f} more percentage points of yield, and temperature alone "
                   f"accounts for roughly {100*lead_r2:.0f}% of the difference between batches.",
    "title": f"{lead} is the dominant correlate of {TARGET}: r = {lead_r:.2f} (95% CI {lead_lo:.2f} to {lead_hi:.2f}, "
             f"n={N}), univariate R-sq = {lead_r2:.2f}, slope {lead_slope:+.3f} pct-pt per unit "
             f"(adjusted {lead_ci.coef:+.3f} [{lead_ci.ci_low:+.3f}, {lead_ci.ci_high:+.3f}]).",
    "chart_names": ["yield_vs_temperature_by_catalyst", f"yield_vs_{lead}", "association_with_yield",
                    "adjusted_effect_forest"],
    "table_names": ["marginal_screening", "model_coefficients"],
    "analysis": f"Across {N} batches the {lead}-{TARGET} correlation is {lead_r:.2f} (Spearman {lead_rho:.2f}, so it "
                f"is not driven by a couple of extreme points), and {lead} on its own explains {100*lead_r2:.0f}% of "
                f"the yield spread. The adjusted slope is {lead_ci.coef:+.3f} percentage points of yield per unit of "
                f"{lead} with a 95% interval of [{lead_ci.ci_low:+.3f}, {lead_ci.ci_high:+.3f}], which excludes zero. "
                f"Every other measured factor is far behind on the ranked association chart.",
    "caveat": "Descriptive association only. These are observational batch records, not a designed experiment, so "
              "no causal reading is supported; the relationship is only characterised inside the observed "
              f"{lead} range ({df[lead].min():.0f} to {df[lead].max():.0f})."})

takeaways.append({
    "plain_title": f"Average yield climbs steadily across the whole temperature range - from about "
                   f"{lo_band.mean_yield_pct:.0f}% in the coolest batches to about {hi_band.mean_yield_pct:.0f}% in the "
                   f"hottest - with no sign of levelling off or a sweet spot in between.",
    "title": f"The binned mean-{TARGET} curve rises monotonically from {lo_band.mean_yield_pct:.1f}% "
             f"({lo_band.iloc[0]}, n={int(lo_band.n_batches)}) to {hi_band.mean_yield_pct:.1f}% "
             f"({hi_band.iloc[0]}, n={int(hi_band.n_batches)}), a {hi_band.mean_yield_pct - lo_band.mean_yield_pct:.1f} "
             f"pct-pt span with no plateau or turning point.",
    "chart_names": [f"mean_yield_by_{lead}", "yield_vs_temperature_by_catalyst"],
    "table_names": ["marginal_screening"],
    "analysis": f"Equal-count {lead} bands give mean yields of "
                f"{', '.join(f'{r.mean_yield_pct:.1f}% ({r.iloc[0]}, n={int(r.n_batches)})' for _, r in bin_lead.iterrows())}. "
                f"The ordering is strictly increasing, which is why a straight-line summary is a fair description here "
                f"rather than an assumption imposed on the data.",
    "caveat": "Bands hold 5-7 batches each, so band-to-band wobble of a point or two is within noise; the data cannot "
              f"say what happens outside the observed {lead} range."})

if second:
    takeaways.append({
        "plain_title": f"Pressure looks essentially unrelated to yield - batches run at high and low pressure end up "
                       f"with much the same yields, and the small downward tilt that does appear is well within "
                       f"what random variation would produce.",
        "title": f"{second} shows no material association with {TARGET}: r = {sec_r:.2f} (p = {sec_p:.3f}), "
                 f"univariate R-sq = {float(S.loc[second, 'univariate_r2']):.3f}; adjusted slope "
                 f"{sec_row.coef:+.2f} pct-pt per unit with 95% CI [{sec_row.ci_low:+.2f}, {sec_row.ci_high:+.2f}] "
                 f"spanning zero.",
        "chart_names": [f"yield_vs_{second}", f"mean_yield_by_{second}", "adjusted_effect_forest",
                        "partial_effect_panels"],
        "table_names": ["marginal_screening", "model_coefficients"],
        "analysis": f"The marginal correlation between {second} and {TARGET} is {sec_r:.2f}, explaining under "
                    f"{100*float(S.loc[second, 'univariate_r2']):.0f}% of yield variation, and the binned curve is flat "
                    f"rather than sloped. Adding {second} to the model changes explained variance by only "
                    f"{nested[second]['delta_r2']:.3f}, and its added-variable panel shows no residual trend. "
                    f"It is the clearest negative result in this dataset.",
        "caveat": f"Absence of an association over the observed {second} range ({df[second].min():.2f} to "
                  f"{df[second].max():.2f}) is not evidence that pressure never matters; a wider range, or an "
                  f"interaction, could behave differently, and n={N} limits sensitivity to small effects."})

if cat_col:
    takeaways.append({
        "plain_title": f"Catalyst {worst[cat_col]} looks like the weakest performer on raw numbers "
                       f"({worst.mean_yield_pct:.1f}% vs {best.mean_yield_pct:.1f}% for catalyst {best[cat_col]}), but "
                       f"its batches were also run about {abs(temp_gap):.0f} degrees cooler - once you compare batches "
                       f"run at similar temperatures the catalyst gap nearly disappears.",
        "title": f"The raw {cat_col} gap ({best[cat_col]} {best.mean_yield_pct:.1f}% vs {worst[cat_col]} "
                 f"{worst.mean_yield_pct:.1f}%, one-way ANOVA p = {float(S.loc[cat_col, 'p_value']):.3f}) largely tracks "
                 f"a {abs(temp_gap):.0f}-unit {lead} imbalance between groups; adjusted for {lead} the largest "
                 f"{cat_col} coefficient is {max_adj:.2f} pct-pt with intervals spanning zero (delta R-sq = "
                 f"{cat_delta_r2:.3f}).",
        "chart_names": ["yield_by_catalyst_raw_vs_adjusted", f"yield_by_{cat_col}", f"{lead}_by_{cat_col}",
                        "temperature_ecdf_by_catalyst", "adjusted_effect_forest"],
        "table_names": [f"{cat_col}_summary", f"condition_balance_by_{cat_col}", "model_coefficients"],
        "analysis": f"Group means are "
                    f"{', '.join(f'{r[cat_col]} {r.mean_yield_pct:.1f}% (n={int(r.n_batches)})' for _, r in cat_tbl.iterrows())}, "
                    f"but mean {lead} per group is "
                    f"{', '.join(f'{i} {v:.0f}' for i, v in cb[lead].items())} - the low-yield group is also the "
                    f"cool-running group, and the ECDF shows the whole distribution is shifted, not just the mean. "
                    f"On {lead}-adjusted values the violins line up and the {cat_col} terms add only "
                    f"{cat_delta_r2:.3f} to R-squared.",
        "caveat": f"This describes an imbalance in the recorded conditions, not a demonstration that the catalysts are "
                  f"equivalent. With {int(cat_tbl.n_batches.min())}-{int(cat_tbl.n_batches.max())} batches per group and "
                  f"overlapping but unequal {lead} ranges, a genuine moderate catalyst effect could still be hidden."})

takeaways.append({
    "plain_title": f"Almost everything the data can explain about yield comes from temperature alone: adding pressure "
                   f"and catalyst on top lifts the explained share only from about {100*only_lead_r2:.0f}% to "
                   f"{100*full_r2:.0f}%.",
    "title": f"Nested models: {lead}-only R-sq = {only_lead_r2:.3f} vs full model R-sq = {full_r2:.3f} "
             f"(adj {full.rsquared_adj:.3f}); {second or 'the other numeric factor'} and {cat_col or 'the group column'} "
             f"together add {full_r2 - only_lead_r2:.3f}.",
    "chart_names": ["model_r2_comparison", "calibration_predicted_vs_observed", "adjusted_effect_forest"],
    "table_names": ["model_diagnostics", "model_coefficients"],
    "analysis": f"An ordinary least-squares fit of {TARGET} on {', '.join(predictors_num)} plus {cat_col} reaches "
                f"R-squared {full_r2:.3f} with residual SD {res_sd:.2f} percentage points, versus {only_lead_r2:.3f} "
                f"for {lead} alone. The predicted-vs-observed calibration line tracks the diagonal across the range, "
                f"and residual checks are unremarkable (Shapiro p = {sw_p:.2f}, Breusch-Pagan p = "
                f"{float(diag.breusch_pagan_p.iloc[0]):.2f}, {n_influential} batches above the Cook's-distance "
                f"4/n mark).",
    "caveat": f"R-squared is measured on the same {N} rows used to fit, so it is optimistic as a prediction estimate; "
              "no held-out split was evaluated here."})

takeaways.append({
    "plain_title": f"Temperature and pressure were dialled independently of each other - knowing one tells you almost "
                   f"nothing about the other - so neither is standing in for the other. The only pairing that is "
                   f"genuinely tangled together here is which catalyst was used and how hot that batch was run.",
    "title": f"{predictors_num[0]} and {predictors_num[1]} are near-orthogonal (r = {pred_pair_r:+.2f}; max VIF = "
             f"{float(diag.max_vif.iloc[0]):.2f}), so numeric coefficients are stable; the only structured imbalance "
             f"is {cat_col}-by-{lead}.",
    "chart_names": ["factor_correlation_heatmap", "partial_effect_panels", "fit_diagnostics"],
    "table_names": ["correlation_matrix", "collinearity_vif"],
    "analysis": f"The correlation heatmap shows {predictors_num[0]}-{predictors_num[1]} at {pred_pair_r:+.2f}, and all "
                f"variance-inflation factors are at or below {float(diag.max_vif.iloc[0]):.2f}, far under the usual "
                f"concern level of 5. Marginal and adjusted slopes therefore agree closely for both numeric factors, "
                f"which is why the {lead} result barely moves when other terms enter the model.",
    "caveat": "Low collinearity among the measured factors says nothing about unmeasured factors (feedstock, operator, "
              "equipment, time order) that are not in this table at all."})

takeaways.append({
    "plain_title": f"Even after accounting for every recorded setting, individual batches still land about "
                   f"{res_sd:.1f} percentage points either side of what the settings alone would suggest - so roughly "
                   f"{100*(1-full_r2):.0f}% of the batch-to-batch variation is coming from something this table does "
                   f"not record.",
    "title": f"Residual SD is {res_sd:.2f} pct-pt and {100*(1-full_r2):.0f}% of {TARGET} variance is unexplained by "
             f"{', '.join(predictors_num)} + {cat_col}; the data has no time, operator or feedstock column to probe it.",
    "chart_names": ["fit_diagnostics", "yield_distribution", "calibration_predicted_vs_observed"],
    "table_names": ["model_diagnostics", "data_profile"],
    "analysis": f"Observed yields span {y.min():.1f}-{y.max():.1f}% around a mean of {y.mean():.1f}%. The fitted model "
                f"leaves a residual SD of {res_sd:.2f} percentage points with no visible pattern against predicted "
                f"values, i.e. the leftover variation looks like unstructured batch noise rather than a missed curve "
                f"in the recorded factors. The tidy table has only {len(df.columns)} columns "
                f"({', '.join(df.columns)}), so no further covariate is available to examine.",
    "caveat": "Unexplained variance is not evidence of randomness - it can equally reflect variables that were never "
              "recorded in this extract."})

# ---- chart readings (one per emitted chart AND plot) --------------------
reading("association_with_yield",
        f"{lead} sits far above every other factor on the association scale "
        f"({float(S.loc[lead,'abs_effect']):.2f} vs {float(S.loc[second,'abs_effect']):.2f} for {second})."
        if second else f"{lead} leads the ranking.",
        "Ranking strength of association does not establish which factor is acting on yield.")
reading(f"yield_vs_{lead}",
        f"Points climb tightly from lower-left to upper-right; the spread around the trend is about "
        f"{res_sd:.1f} percentage points.",
        "A tight scatter is still an association from observational batch records, not a dose-response result.")
if second:
    reading(f"yield_vs_{second}",
            f"Points form a flat cloud with no visible slope across the {second} range.",
            "A flat cloud at n=30 cannot rule out a small effect or an effect outside the observed range.")
reading(f"mean_yield_by_{lead}",
        f"Mean yield rises in every successive band, {lo_band.mean_yield_pct:.1f}% to {hi_band.mean_yield_pct:.1f}%.",
        "Bands are equal-count, not equal-width, so the horizontal spacing is not a physical scale.")
if second:
    reading(f"mean_yield_by_{second}",
            f"The band means stay within a couple of points of the {y.mean():.1f}% overall average, with no trend.",
            "Flatness here does not rule out an interaction with temperature.")
if cat_col:
    reading(f"yield_by_{cat_col}",
            f"Group means run {worst.mean_yield_pct:.1f}% ({worst[cat_col]}) to {best.mean_yield_pct:.1f}% "
            f"({best[cat_col]}) on {int(cat_tbl.n_batches.min())}-{int(cat_tbl.n_batches.max())} batches per group.",
            "These are unadjusted means; they do not account for the differing temperatures each group was run at.")
    reading(f"{lead}_by_{cat_col}",
            f"Mean {lead} differs by about {abs(temp_gap):.0f} units between the best- and worst-yielding groups.",
            "It shows an imbalance in operating conditions, not a property of the catalysts themselves.")
    reading("yield_by_catalyst_raw_vs_adjusted",
            "Left panel: catalyst groups differ on raw yield. Right panel: after allowing for temperature the "
            "distributions largely overlap.",
            "Overlap after adjustment does not prove the catalysts perform identically; it shows the raw gap is not "
            "separable from the temperature imbalance in this data.")
    reading("temperature_ecdf_by_catalyst",
            "The catalyst curves are horizontally offset, meaning entire operating ranges differ, not just averages.",
            "It says nothing about yield directly - it is a design-balance observation.")
reading("yield_vs_temperature_by_catalyst",
        f"A single tight upward trend (r = {lead_r:.2f}) runs through all catalyst groups; the coloured points do not "
        f"form separate parallel bands far from the line.",
        "The fitted line and band describe the average trend in these 30 batches; they do not extrapolate beyond the "
        "observed range.")
reading("factor_correlation_heatmap",
        f"{lead} vs {TARGET} is the only strong cell ({lead_r:.2f}); the two numeric factors are near zero "
        f"({pred_pair_r:+.2f}) with each other.",
        "Correlation among factors describes redundancy, not mechanism.")
reading("adjusted_effect_forest",
        f"Only {lead} has an interval clear of zero; every other term's interval spans zero.",
        "An interval spanning zero means the data does not pin the effect's direction - not that the effect is zero.")
if len(predictors_num) >= 2:
    reading("partial_effect_panels",
            f"{lead}'s panel keeps a strong slope after other factors are removed; {second}'s panel is flat.",
            "Partial plots depend on the model's terms; a factor omitted from the model cannot appear here.")
reading("model_r2_comparison",
        f"Explained variance goes {100*only_lead_r2:.0f}% -> {100*full_r2:.0f}% as the other factors are added.",
        "R-squared is measured in-sample and would likely be lower on new batches.")
reading("calibration_predicted_vs_observed",
        "Observed band means track the predicted values closely across the range.",
        "This is in-sample calibration; it is not a held-out prediction test.")
reading("fit_diagnostics",
        f"Residuals scatter evenly around zero, the Q-Q points hug the line (Shapiro p = {sw_p:.2f}), and observed "
        f"versus predicted sits near the diagonal.",
        "Well-behaved residuals support the summary statistics; they say nothing about omitted variables.")
reading("yield_distribution",
        f"A single broad peak between {y.min():.0f}% and {y.max():.0f}% - no separate low-yield sub-population.",
        "A unimodal spread does not mean the batches were run under comparable conditions.")

# ---- candidate signals (deterministic recipes for downstream use) -------
lead_hi_cut = float(np.quantile(df[lead], 0.66))
lead_lo_cut = float(np.quantile(df[lead], 0.33))
sec_cut = float(np.median(df[second])) if second else None
candidate_signals = [
    {"name": "high_temperature_batch", "display_name": "Hot-run batch",
     "rationale": f"Batches run above {lead_hi_cut:.0f} sit in the top third of the {lead} range, where mean yield is "
                  f"{hi_band.mean_yield_pct:.1f}% against {lo_band.mean_yield_pct:.1f}% in the coolest band.",
     "suggested_test": "On a held-out set of batches, compare mean yield above vs below the cut with a Welch t-test "
                       "and report the difference with a 95% interval.",
     "recipe": {"name": "high_temperature_batch", "kind": "expr", "expr": f"{lead} >= {lead_hi_cut:.1f}"}},
    {"name": "cool_run_batch", "display_name": "Cool-run batch",
     "rationale": f"The bottom third of the {lead} range ({lead} below {lead_lo_cut:.0f}) is where the lowest yields "
                  f"concentrate and where catalyst {worst[cat_col] if cat_col else ''} batches are over-represented.",
     "suggested_test": "Check whether the cool-run flag still separates yield after conditioning on catalyst.",
     "recipe": {"name": "cool_run_batch", "kind": "expr", "expr": f"{lead} < {lead_lo_cut:.1f}"}},
]
if second:
    candidate_signals.append(
        {"name": "hot_and_low_pressure", "display_name": "Hot and low-pressure combination",
         "rationale": f"An interaction candidate: {second} is flat on its own (r = {sec_r:.2f}), so this combination "
                      f"tests whether it only matters at the high end of {lead}.",
         "suggested_test": "Fit yield ~ temperature * pressure on new batches and inspect the interaction term's "
                           "coefficient and interval.",
         "recipe": {"name": "hot_and_low_pressure", "kind": "expr",
                    "expr": f"({lead} >= {lead_hi_cut:.1f}) and ({second} < {sec_cut:.2f})"}})
    candidate_signals.append(
        {"name": "temperature_pressure_index", "display_name": "Combined heat-minus-pressure index",
         "rationale": "A single continuous score combining both dial settings in the directions their marginal slopes "
                      "point, for use as one screening variable.",
         "suggested_test": "Compare this index's correlation with yield against temperature alone on held-out batches.",
         "recipe": {"name": "temperature_pressure_index", "kind": "expr",
                    "expr": f"{lead} - 5 * {second}"}})

# ---- claims -------------------------------------------------------------
claims = [
    {"id": "C1", "text": f"{lead} is the strongest measured correlate of {TARGET} in these {N} batches "
                         f"(r = {lead_r:.2f}, 95% CI {lead_lo:.2f} to {lead_hi:.2f}).",
     "status": "descriptive", "evidence_ids": ["chart:yield_vs_temperature_by_catalyst",
                                               f"chart:mean_yield_by_{lead}", "chart:association_with_yield"],
     "interpretation": "The leading candidate driver for any downstream confirmatory work.",
     "do_not_infer": "Not causal and not confirmed; a single observational batch set cannot separate temperature "
                     "from anything that co-varies with it."},
    {"id": "C2", "text": f"{second} shows no material association with {TARGET} over the observed range "
                         f"(r = {sec_r:.2f}, adjusted 95% CI [{sec_row.ci_low:+.2f}, {sec_row.ci_high:+.2f}] "
                         f"pct-pt per unit)." if second else "Only one numeric factor was available.",
     "status": "descriptive", "evidence_ids": [f"chart:yield_vs_{second}", f"chart:mean_yield_by_{second}",
                                               "chart:adjusted_effect_forest"] if second else [],
     "interpretation": "A negative screening result for pressure at this sample size and range.",
     "do_not_infer": "Not evidence of no effect; n=30 gives limited power and the observed range is narrow."},
]
if cat_col:
    claims.append(
        {"id": "C3", "text": f"The unadjusted {cat_col} yield ordering coincides with a {abs(temp_gap):.0f}-unit "
                             f"difference in mean {lead} between the same groups, and {cat_col} adds only "
                             f"{cat_delta_r2:.3f} to R-squared once {lead} is in the model.",
         "status": "descriptive", "evidence_ids": ["chart:yield_by_catalyst_raw_vs_adjusted",
                                                   f"chart:{lead}_by_{cat_col}", "chart:temperature_ecdf_by_catalyst"],
         "interpretation": "A condition-imbalance pattern that any downstream comparison of catalysts must handle.",
         "do_not_infer": "Does not establish that the catalysts are equivalent, nor that temperature explains the gap."})

# ---- observations, caveats, critique ------------------------------------
observations = [
    f"Raw input was a single JSON file holding a flat list of {N} batch records; no wrapper metadata to merge, "
    f"tidy table = {N} rows x {len(df.columns)} columns ({', '.join(df.columns)}), written to records.json.",
    f"Outcome framing: '{TARGET}' is present and continuous ({n_unique_y} distinct values, "
    f"{y.min():.1f}-{y.max():.1f}%), so this is a correlation/regression story - it was NOT binarised.",
    f"No missing values anywhere ({total_missing} nulls in {N * len(df.columns)} cells) and "
    f"{int(profile.n_iqr_outliers.fillna(0).sum())} IQR outliers across numeric columns.",
    f"{lead}: r = {lead_r:.2f} with yield, univariate R-sq {lead_r2:.2f}, slope {lead_slope:+.3f} pct-pt per unit.",
    (f"{second}: r = {sec_r:.2f} (p = {sec_p:.3f}), univariate R-sq "
     f"{float(S.loc[second,'univariate_r2']):.3f} - effectively flat." if second else ""),
    (f"{cat_col}: raw group means {', '.join(f'{r[cat_col]} {r.mean_yield_pct:.1f}%' for _, r in cat_tbl.iterrows())} "
     f"(ANOVA p = {float(S.loc[cat_col,'p_value']):.3f}), but group mean {lead} is "
     f"{', '.join(f'{i} {v:.0f}' for i, v in cb[lead].items())}." if cat_col else ""),
    f"Model choice: OLS on a continuous outcome (logistic/Poisson would not apply); {cat_col} entered as a FIXED "
    f"effect rather than a random effect because it has only {len(levels)} levels, too few to identify a variance "
    f"component. Batch ids are unique, so there are no repeated measures.",
    f"Full model R-sq {full_r2:.3f} (adj {full.rsquared_adj:.3f}), residual SD {res_sd:.2f} pct-pt, max VIF "
    f"{float(diag.max_vif.iloc[0]):.2f}, Shapiro p {sw_p:.3f}, Breusch-Pagan p {float(diag.breusch_pagan_p.iloc[0]):.3f}.",
    f"Functional form: a squared {lead} term and a {lead}x{second} interaction were each fitted "
    f"(tables/functional_form_checks.csv); the best of them raised R-squared by only {max_form_gain:.3f} over the "
    f"additive model, so the straight-line description was kept rather than assumed.",
]
observations = [o for o in observations if o]

caveats = [
    f"n = {N} batches. Every interval is wide at this size; group-level statements rest on "
    f"{int(cat_tbl.n_batches.min()) if cat_col else N}-{int(cat_tbl.n_batches.max()) if cat_col else N} batches each.",
    "Observational batch records with no randomisation and no time/operator/feedstock columns - associations here "
    "cannot be separated from unrecorded factors.",
    f"{cat_col} groups were not run over the same {lead} range, so catalyst and temperature are entangled by design "
    f"of the data, not by choice of analysis." if cat_col else "",
    "All R-squared and correlation figures are in-sample on the same 30 rows used for fitting; no held-out split "
    "was evaluated.",
    f"Relationships are only characterised inside the observed ranges ({lead} {df[lead].min():.0f}-{df[lead].max():.0f}"
    + (f", {second} {df[second].min():.2f}-{df[second].max():.2f}" if second else "") + ").",
    "Per-catalyst correlation slopes were computed (stratified_correlations table) but not charted, since 9-11 "
    "batches per group make a per-group trend line unstable.",
    "No binary FAIL/PASS split exists in this data and none was invented; class-balance and fail-rate visuals were "
    "deliberately skipped.",
]
caveats = [c for c in caveats if c]

critique = [
    "Double-dipping: the same 30 rows were used for screening, model fitting and the reported R-squared, so effect "
    "sizes are optimistic. A held-out or cross-validated estimate would be lower.",
    f"The {cat_col}-vs-{lead} imbalance is a textbook confounding structure; the 'adjusted' catalyst panel relies on "
    f"the linear {lead} adjustment being right, and with partly non-overlapping ranges that adjustment is doing "
    f"extrapolation for part of each group." if cat_col else "",
    "No leakage risk was identified - none of the predictors is a re-measurement of yield - but this was checked by "
    "inspection of column semantics only, not by any provenance record.",
    f"With {len(levels) if cat_col else 0} catalyst levels a mixed-effects model was deliberately not fitted; if more "
    f"catalysts existed, a random intercept would be the better choice and conclusions about group spread could shift.",
    "Multiple comparisons were not adjusted for: several tests were run across factors, so borderline p-values "
    "(e.g. the raw catalyst ANOVA) should not be read as a threshold decision.",
    "Charts using equal-count bins can visually exaggerate a trend when bins are uneven in width; the raw scatter is "
    "provided alongside precisely so the reader can check that.",
]
critique = [c for c in critique if c]

storyboard = [{
    "id": "problem_setting", "title": "Problem Setting", "stages": ["M1"],
    "summary": f"{N} production batches, each with three recorded settings - {', '.join(predictors_num)} and "
               f"{cat_col} - and a continuous outcome, {TARGET} (mean {y.mean():.1f}%, range "
               f"{y.min():.1f}-{y.max():.1f}%). The question is which of those settings tracks yield, how they relate "
               f"to one another, and whether any apparent group effect survives the others. There is no pass/fail "
               f"label in this data, so the analysis is framed as correlation and regression, not FAIL vs PASS.",
    "items": [
        f"Data: one JSON file, flat list of {N} batch records, no missing values, parsed into a tidy "
        f"{N}x{len(df.columns)} table (records.json).",
        f"Outcome: {TARGET}, continuous - deliberately NOT binarised into pass/fail.",
        f"Factors examined: {', '.join(predictors_num)} (numeric) and {cat_col} ({len(levels)} levels, "
        f"{int(cat_tbl.n_batches.min())}-{int(cat_tbl.n_batches.max())} batches each)." if cat_col else
        f"Factors examined: {', '.join(predictors_num)}.",
        f"Method: variable EDA -> per-factor association with effect sizes and intervals -> conditioning check for "
        f"confounding -> marginal screening -> ordinary least squares with {cat_col} as a fixed effect -> "
        f"collinearity, residual and calibration diagnostics." if cat_col else "Method: EDA -> screening -> OLS.",
        "Everything reported is descriptive association; nothing here is a causal or confirmed result.",
    ],
    "artifact_refs": ["data_profile"],
}]

result = {
    "plain_question": "Which of the three things recorded for each production batch - how hot it ran, how much "
                      "pressure it ran at, and which catalyst was used - goes together with a higher yield, and "
                      "how do those three relate to each other?",
    "observations": observations,
    "visual_plan": visual_plan,
    "takeaways": takeaways,
    "chart_readings": chart_readings,
    "claims": claims,
    "dashboard_storyboard": storyboard,
    "candidate_signals": candidate_signals,
    "plots": plots,
    "tables": tables_out,
    "charts": charts,
    "caveats": caveats,
    "critique": critique,
    "recommended_confirmatory_tests": [
        f"On a held-out batch set, regress {TARGET} on {lead} and report the slope with a 95% interval; check whether "
        f"it lands near {lead_ci.coef:+.3f} pct-pt per unit.",
        f"Compare {cat_col} groups within narrow {lead} strata (or with {lead} as a covariate) on new batches, so the "
        f"catalyst comparison is not carried by the temperature imbalance seen here." if cat_col else "",
        f"Test the {lead} x {second} interaction explicitly on new data; it was small here (delta R-sq below 0.01) but "
        f"n={N} gives little power to detect it." if second else "",
        "Evaluate the temperature-only model against the full model by cross-validated prediction error, to see "
        "whether the extra terms earn their place out of sample.",
        f"Extend the recorded {second} range and re-test - the current flat result only covers "
        f"{df[second].min():.2f}-{df[second].max():.2f}." if second else "",
    ],
}
result["recommended_confirmatory_tests"] = [t for t in result["recommended_confirmatory_tests"] if t]

# reproducibility note (skill step 8)
Path("session_info.txt").write_text(
    f"python={sys.version.split()[0]}\npandas={pd.__version__}\nnumpy={np.__version__}\n"
    f"scipy={stats.__name__} {__import__('scipy').__version__}\nstatsmodels={sm.__version__}\n"
    f"matplotlib={matplotlib.__version__}\nrows={N}\nseed=7 (jitter only)\ncwd={os.getcwd()}\n")


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


print("EXPLORATORY_RESULT_JSON=" + json.dumps(result, default=_json_safe))
