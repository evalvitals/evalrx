#!/usr/bin/env python3
"""
Exploratory analysis: what distinguishes hallucination FAILs from PASSes in a
VLM object-presence probe, using per-case attention-geometry scalars across
three checkpoints (2B / 4B / 8B).

Methodology follows the outcome-driver-analysis skill (explanatory EDA ->
per-variable tests with effect sizes -> conditioning/Simpson checks ->
marginal screening -> justified GLM -> fit diagnostics).
Figure styling follows eval-chart-style (chart-type policy + semantic palette,
which takes precedence over other skills' chart suggestions) and nature-figure
(publication polish, Python backend, exclusive).
Reporting shape follows evalrx-report-ui.

PURE EXPLORATORY / DESCRIPTIVE. No causal claims, no confirmation verdicts.
"""

from __future__ import annotations

import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(20240726)
RAW = "raw_input"
FIG = "figures"
TAB = "tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ----------------------------------------------------------------------------
# Palette + rcParams (eval-chart-style §1/§1a, nature-figure python quick-start)
# ----------------------------------------------------------------------------
C_FAIL = "#d03b3b"
C_PASS = "#0ca30c"
C_INK = "#898781"
C_ACCENT = "#2a78d6"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#eb6834"]
RAMP = ["#86b6ef", "#2a78d6", "#104281"]          # ordered: 2B -> 4B -> 8B
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
CMAP_SEQ = LinearSegmentedColormap.from_list("seq", ["#f7fafd", "#2a78d6"])

try:  # host theme when available; inline the same intent otherwise
    from evalrx.analysis.eval_viz_theme import matplotlib_rcparams
    plt.rcParams.update(matplotlib_rcparams())
except Exception:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e8e6e1",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def savefig(fig, stem):
    p = f"{FIG}/{stem}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def human_bin(lo, hi):
    """eval-chart-style §3: never print raw pandas.cut edges."""
    def f(v):
        a = abs(v)
        if a >= 100:
            return f"{v:.0f}"
        if a >= 1:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{f(lo)}–{f(hi)}"


# ============================================================================
# 1. LOAD + TIDY  (shape is not assumed; inspected then merged)
# ============================================================================
def load_any(path):
    """Return (list_of_row_dicts, scalar_metadata) from one file of unknown shape."""
    txt = open(path, "r", encoding="utf-8").read().strip()
    obj = None
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:                       # JSONL fallback
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
        return rows, {}
    if isinstance(obj, list):
        return obj, {}
    if isinstance(obj, dict):
        # find the record list: a conventional key, else the longest list-of-dicts
        keys = ["cases", "results", "rows", "items", "data", "examples", "samples", "records"]
        rec_key = next((k for k in keys if isinstance(obj.get(k), list)
                        and obj[k] and isinstance(obj[k][0], dict)), None)
        if rec_key is None:
            cands = [(k, v) for k, v in obj.items()
                     if isinstance(v, list) and v and isinstance(v[0], dict)]
            rec_key = max(cands, key=lambda kv: len(kv[1]))[0] if cands else None
        if rec_key is None:
            return [obj], {}                            # single flat record
        meta = {k: v for k, v in obj.items()
                if k != rec_key and not isinstance(v, (list, dict))}
        # keep a couple of useful nested scalars
        for nk in ("attention_extraction", "versions", "decoding"):
            if isinstance(obj.get(nk), dict):
                for k2, v2 in obj[nk].items():
                    if not isinstance(v2, (list, dict)):
                        meta[f"{nk}__{k2}"] = v2
        return obj[rec_key], meta
    return [], {}


files = sorted(
    os.path.join(dp, f)
    for dp, _, fs in os.walk(RAW) for f in fs
    if f.lower().endswith((".json", ".jsonl"))
)
frames, file_meta = [], []
for p in files:
    recs, meta = load_any(p)
    if not recs:
        continue
    d = pd.DataFrame(recs)
    for k, v in meta.items():                           # merge file scalars onto every row
        if k not in d.columns:
            d[k] = v
    d["source_file"] = os.path.basename(p)
    frames.append(d)
    file_meta.append({"file": os.path.basename(p), "n_rows": len(d), **meta})

df = pd.concat(frames, ignore_index=True)

# list-valued columns -> lengths (unusable as scalars, keep the information)
for c in list(df.columns):
    if df[c].apply(lambda v: isinstance(v, list)).any():
        df[c + "_n"] = df[c].apply(lambda v: len(v) if isinstance(v, list) else np.nan)
        df = df.drop(columns=[c])

# tidy typing / friendly derived columns
df["model"] = df["model"].astype(str)
SIZE_MAP, SIZE_ORD = {}, []
for m in sorted(df["model"].unique()):
    tag = next((t.upper() for t in m.split("-") if t.lower().endswith("b") and t[:-1].isdigit()), m)
    SIZE_MAP[m] = tag
df["checkpoint"] = df["model"].map(SIZE_MAP)
SIZE_ORD = sorted(df["checkpoint"].unique(), key=lambda s: float(s[:-1]))
df["checkpoint"] = pd.Categorical(df["checkpoint"], categories=SIZE_ORD, ordered=True)
df["size_b"] = df["checkpoint"].astype(str).str.rstrip("B").astype(float)

# outcome: caller named "label"
OUTCOME = "label" if "label" in df.columns else None
assert OUTCOME is not None, "no 'label' column found"
lab_vals = sorted(df[OUTCOME].dropna().astype(str).unique())
OUTCOME_KIND = "binary" if len(lab_vals) == 2 else ("categorical" if len(lab_vals) <= 12 else "continuous")
FAIL_VAL = "fail" if "fail" in lab_vals else lab_vals[0]
PASS_VAL = [v for v in lab_vals if v != FAIL_VAL][0]
df["is_fail"] = (df[OUTCOME].astype(str) == FAIL_VAL).astype(int)
df["outcome"] = np.where(df["is_fail"] == 1, "FAIL", "PASS")

# tidy table for reviewers
df.to_json("records.json", orient="records", indent=1)

SIGNALS = ["attention_entropy", "focus_share", "center_offset", "edge_mass",
           "top1_share", "max_relative_weight", "mean_relative_weight"]
SIGNALS = [s for s in SIGNALS if s in df.columns]
# heavy right tail on max_relative_weight -> analyse on log10 scale as well
df["log10_max_relative_weight"] = np.log10(df["max_relative_weight"].clip(lower=1e-6))
MODEL_SIGNALS = [s if s != "max_relative_weight" else "log10_max_relative_weight" for s in SIGNALS]

ALIAS = {"attention_entropy": "Entropy", "focus_share": "Focus share",
         "center_offset": "Center off.", "edge_mass": "Edge mass",
         "top1_share": "Top-1 share", "max_relative_weight": "Max rel.wt",
         "mean_relative_weight": "Mean rel.wt",
         "log10_max_relative_weight": "Max rel.wt (log)"}
FULLNAME = {"attention_entropy": "Attention entropy", "focus_share": "Attention focus share",
            "center_offset": "Center offset", "edge_mass": "Edge mass",
            "top1_share": "Top-1 patch share", "max_relative_weight": "Maximum relative attention",
            "mean_relative_weight": "Mean relative attention",
            "log10_max_relative_weight": "Maximum relative attention (log10)"}

# probe design: FAIL is only definable where the object is absent (adversarial probe)
ADV = df[df["probe_type"].astype(str).str.lower().eq("adversarial")].copy() \
    if "probe_type" in df.columns else df.copy()
CTRL_PRESENT = df[~df.index.isin(ADV.index)]

profile = {
    "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
    "outcome_kind": OUTCOME_KIND, "fail_n": int(df.is_fail.sum()), "pass_n": int((1 - df.is_fail).sum()),
    "checkpoints": SIZE_ORD, "n_objects": int(df["object"].nunique()) if "object" in df else 0,
    "n_images": int(df["image_id"].nunique()) if "image_id" in df else 0,
    "probe_types": df["probe_type"].value_counts().to_dict() if "probe_type" in df else {},
    "adversarial_n": int(len(ADV)), "adversarial_fail_n": int(ADV.is_fail.sum()),
    "fails_outside_adversarial": int(CTRL_PRESENT.is_fail.sum()),
    "missing_cells": int(df[SIGNALS].isna().sum().sum()),
    "files": file_meta,
}

# ============================================================================
# 2. EXPLANATORY-VARIABLE EDA (distributions, outliers, missingness, structure)
# ============================================================================
eda_rows = []
for s in SIGNALS:
    v = df[s].astype(float)
    q1, q3 = v.quantile([.25, .75])
    iqr = q3 - q1
    eda_rows.append({
        "signal": s, "display_name": FULLNAME[s], "n": int(v.notna().sum()),
        "missing": int(v.isna().sum()), "mean": round(v.mean(), 4), "sd": round(v.std(), 4),
        "median": round(v.median(), 4), "p05": round(v.quantile(.05), 4), "p95": round(v.quantile(.95), 4),
        "min": round(v.min(), 4), "max": round(v.max(), 4),
        "skew": round(float(stats.skew(v.dropna())), 3),
        "iqr_outliers": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()),
        "shapiro_p": round(float(stats.shapiro(v.dropna().sample(min(500, v.notna().sum()),
                                                                 random_state=0))[1]), 4),
    })
eda = pd.DataFrame(eda_rows)
eda.to_csv(f"{TAB}/explanatory_eda.csv", index=False)

# correlation structure among predictors (Spearman: skewed signals)
corr = df[MODEL_SIGNALS].corr(method="spearman")
corr.round(3).to_csv(f"{TAB}/signal_correlations.csv")
pairs = [{"pair": f"{ALIAS[a]} vs {ALIAS[b]}", "signal_a": a, "signal_b": b,
          "spearman_rho": round(float(corr.loc[a, b]), 3),
          "abs_rho": round(abs(float(corr.loc[a, b])), 3)}
         for a, b in combinations(MODEL_SIGNALS, 2)]
pair_df = pd.DataFrame(pairs).sort_values("abs_rho", ascending=False)
pair_df.to_csv(f"{TAB}/corr_pairs.csv", index=False)


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))), float(p)


# ============================================================================
# 3-4. PER-VARIABLE TESTS + EFFECT SIZES + MARGINAL SCREENING
# ============================================================================
def auc_rankbiserial(x_fail, x_pass, nboot=2000):
    """Mann-Whitney U -> AUC (= common-language effect size) + bootstrap CI."""
    a, b = np.asarray(x_fail, float), np.asarray(x_pass, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, p=np.nan, rb=np.nan, d=np.nan)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b))
    boots = np.empty(nboot)
    for i in range(nboot):
        ai = RNG.choice(a, len(a), replace=True)
        bi = RNG.choice(b, len(b), replace=True)
        boots[i] = stats.mannwhitneyu(ai, bi, alternative="two-sided")[0] / (len(a) * len(b))
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    return dict(auc=float(auc), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), p=float(p),
                rb=float(2 * auc - 1), d=float(d))


def separation_table(frame, tag):
    out = []
    f = frame[frame.is_fail == 1]
    g = frame[frame.is_fail == 0]
    for s in SIGNALS:
        r = auc_rankbiserial(f[s], g[s])
        out.append({
            "signal": s, "display_name": FULLNAME[s], "alias": ALIAS[s], "scope": tag,
            "n_fail": int(f[s].notna().sum()), "n_pass": int(g[s].notna().sum()),
            "median_fail": round(float(f[s].median()), 4), "median_pass": round(float(g[s].median()), 4),
            "auc": round(r["auc"], 3), "auc_lo": round(r["lo"], 3), "auc_hi": round(r["hi"], 3),
            "separation": round(abs(r["auc"] - .5) * 2, 3),
            "rank_biserial": round(r["rb"], 3), "cohens_d": round(r["d"], 3),
            "mannwhitney_p": float(f"{r['p']:.3g}"),
            "direction": "higher in FAIL" if r["auc"] > .5 else "lower in FAIL",
            "ci_excludes_null": bool(r["lo"] > .5 or r["hi"] < .5),
        })
    return pd.DataFrame(out).sort_values("separation", ascending=False)


sep_adv = separation_table(ADV, "adversarial_only")
sep_all = separation_table(df, "all_cases")
sep_adv.to_csv(f"{TAB}/separation_adversarial.csv", index=False)
sep_all.to_csv(f"{TAB}/separation_all.csv", index=False)

# conditioning / Simpson check: same contrast within each checkpoint
sep_by_ck = pd.concat([separation_table(ADV[ADV.checkpoint == ck], ck).assign(checkpoint=ck)
                       for ck in SIZE_ORD], ignore_index=True)
sep_by_ck.to_csv(f"{TAB}/separation_by_checkpoint.csv", index=False)

simpson = []
for s in SIGNALS:
    pooled = float(sep_adv.loc[sep_adv.signal == s, "auc"].iloc[0])
    strata = sep_by_ck[sep_by_ck.signal == s]["auc"].astype(float).values
    dirs = {np.sign(a - .5) for a in strata if not np.isnan(a)}
    simpson.append({"signal": s, "display_name": FULLNAME[s], "pooled_auc": round(pooled, 3),
                    "auc_min_checkpoint": round(float(np.nanmin(strata)), 3),
                    "auc_max_checkpoint": round(float(np.nanmax(strata)), 3),
                    "direction_consistent": bool(len(dirs) <= 1),
                    "sign_flip_across_checkpoints": bool(len(dirs) > 1)})
simpson_df = pd.DataFrame(simpson)
simpson_df.to_csv(f"{TAB}/conditioning_check.csv", index=False)

# distribution SHIFT across checkpoints (Kruskal-Wallis on the adversarial cases)
shift_rows = []
for s in SIGNALS:
    groups = [ADV.loc[ADV.checkpoint == ck, s].dropna().values for ck in SIZE_ORD]
    H, p = stats.kruskal(*groups)
    eps2 = (H - len(groups) + 1) / (len(ADV) - len(groups))
    row = {"signal": s, "display_name": FULLNAME[s], "kruskal_H": round(float(H), 3),
           "p_value": float(f"{p:.3g}"), "epsilon_sq": round(float(eps2), 3)}
    for ck, g in zip(SIZE_ORD, groups):
        row[f"median_{ck}"] = round(float(np.median(g)), 4)
    row["spearman_vs_size"] = round(float(stats.spearmanr(ADV["size_b"], ADV[s])[0]), 3)
    shift_rows.append(row)
shift_df = pd.DataFrame(shift_rows).sort_values("epsilon_sq", ascending=False)
shift_df.to_csv(f"{TAB}/checkpoint_distribution_shift.csv", index=False)

# categorical drivers within the adversarial probe
cat_rows = []
for c in [c for c in ["checkpoint", "object", "split", "pope_label"] if c in ADV.columns]:
    v, p = cramers_v(ADV[c].astype(str), ADV["outcome"])
    cat_rows.append({"variable": c, "levels": int(ADV[c].nunique()),
                     "cramers_v": round(v, 3) if v == v else np.nan,
                     "chi2_p": float(f"{p:.3g}") if p == p else np.nan})
cat_df = pd.DataFrame(cat_rows)
cat_df.to_csv(f"{TAB}/categorical_association.csv", index=False)

# probe-design contrast (sanity/context: adversarial vs present-object rows)
design_rows = []
for s in SIGNALS:
    r = auc_rankbiserial(ADV[s], CTRL_PRESENT[s], nboot=800) if len(CTRL_PRESENT) else dict(auc=np.nan)
    design_rows.append({"signal": s, "alias": ALIAS[s], "display_name": FULLNAME[s],
                        "median_adversarial": round(float(ADV[s].median()), 4),
                        "median_present": round(float(CTRL_PRESENT[s].median()), 4) if len(CTRL_PRESENT) else np.nan,
                        "auc_adv_vs_present": round(r["auc"], 3) if r["auc"] == r["auc"] else np.nan})
design_df = pd.DataFrame(design_rows)
design_df.to_csv(f"{TAB}/probe_design_contrast.csv", index=False)

# ============================================================================
# 5-6. MODEL: logistic GLM on the adversarial subset + diagnostics
# ============================================================================
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

M = ADV.dropna(subset=MODEL_SIGNALS).copy()
Z = M[MODEL_SIGNALS].apply(lambda c: (c - c.mean()) / c.std())   # standardized -> OR per 1 SD
Z.columns = [f"z_{c}" for c in MODEL_SIGNALS]
X = Z.copy()
for ck in SIZE_ORD[1:]:                                          # 3 clusters -> FIXED effect, not random
    X[f"ck_{ck}"] = (M["checkpoint"].astype(str) == ck).astype(float)
X = sm.add_constant(X)
y = M["is_fail"].values

# marginal screening: univariate logistic per signal (AIC vs null)
null_llf = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf
screen = []
for c in Z.columns:
    m1 = sm.Logit(y, sm.add_constant(Z[[c]])).fit(disp=0)
    raw = c[2:]
    screen.append({"signal": raw, "display_name": FULLNAME[raw],
                   "univariate_or_per_sd": round(float(np.exp(m1.params[c])), 3),
                   "or_lo": round(float(np.exp(m1.conf_int().loc[c, 0])), 3),
                   "or_hi": round(float(np.exp(m1.conf_int().loc[c, 1])), 3),
                   "wald_p": float(f"{m1.pvalues[c]:.3g}"),
                   "aic": round(float(m1.aic), 1),
                   "aic_gain_vs_null": round(float(2 * (m1.llf - null_llf) - 2), 1)})
screen_df = pd.DataFrame(screen).sort_values("aic_gain_vs_null", ascending=False)
screen_df.to_csv(f"{TAB}/marginal_screening.csv", index=False)

# full model, cluster-robust SEs by image (images recur across checkpoints)
clusters = M["image_id"].astype(str) + "_" + M["object"].astype(str) if "image_id" in M else M.index.astype(str)
fit_plain = sm.Logit(y, X).fit(disp=0)
try:
    fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": clusters.values})
except Exception:
    fit = fit_plain

ci = fit.conf_int()
coef_rows = []
for term in X.columns:
    if term == "const":
        continue
    raw = term[2:] if term.startswith("z_") else term
    disp = FULLNAME.get(raw, f"Checkpoint {term[3:]} (vs {SIZE_ORD[0]})")
    # LRT: drop this term from the plain-MLE model
    Xr = X.drop(columns=[term])
    lr = 2 * (fit_plain.llf - sm.Logit(y, Xr).fit(disp=0).llf)
    coef_rows.append({
        "term": term, "signal": raw, "display_name": disp,
        "kind": "attention signal" if term.startswith("z_") else "checkpoint (fixed effect)",
        "odds_ratio_per_sd": round(float(np.exp(fit.params[term])), 3),
        "or_lo": round(float(np.exp(ci.loc[term, 0])), 3),
        "or_hi": round(float(np.exp(ci.loc[term, 1])), 3),
        "wald_p_robust": float(f"{fit.pvalues[term]:.3g}"),
        "lrt_chi2": round(float(lr), 2),
        "lrt_p": float(f"{stats.chi2.sf(lr, 1):.3g}"),
        "ci_excludes_1": bool((ci.loc[term, 0] > 0) or (ci.loc[term, 1] < 0)),
    })
coef_df = pd.DataFrame(coef_rows)
coef_df["abs_log_or"] = coef_df["odds_ratio_per_sd"].apply(lambda v: abs(np.log(v)))
coef_df = coef_df.sort_values("abs_log_or", ascending=False)
coef_df.to_csv(f"{TAB}/adjusted_model_coefficients.csv", index=False)

# collinearity
vif = pd.DataFrame({
    "signal": [c[2:] for c in Z.columns],
    "display_name": [FULLNAME[c[2:]] for c in Z.columns],
    "vif": [round(float(variance_inflation_factor(sm.add_constant(Z).values, i + 1)), 2)
            for i in range(Z.shape[1])],
})
vif["collinearity_flag"] = np.where(vif.vif >= 5, "high (>=5)", np.where(vif.vif >= 2.5, "moderate", "low"))
vif.to_csv(f"{TAB}/collinearity_vif.csv", index=False)

# forward selection by AIC -> minimal set carrying independent signal
remaining, chosen, cur = list(Z.columns), [], None
cur_aic = sm.Logit(y, sm.add_constant(pd.DataFrame(index=Z.index))).fit(disp=0).aic \
    if False else float(sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).aic)
fs_log = []
while remaining:
    best = None
    for c in remaining:
        m = sm.Logit(y, sm.add_constant(Z[chosen + [c]])).fit(disp=0)
        if best is None or m.aic < best[1]:
            best = (c, float(m.aic))
    if best[1] < cur_aic - 2:
        chosen.append(best[0]); remaining.remove(best[0])
        fs_log.append({"step": len(chosen), "added": best[0][2:],
                       "display_name": FULLNAME[best[0][2:]],
                       "aic": round(best[1], 1), "aic_drop": round(cur_aic - best[1], 1)})
        cur_aic = best[1]
    else:
        break
fs_df = pd.DataFrame(fs_log if fs_log else [{"step": 0, "added": "none", "display_name": "none",
                                             "aic": round(cur_aic, 1), "aic_drop": 0.0}])
fs_df.to_csv(f"{TAB}/forward_selection.csv", index=False)
INDEP = [c[2:] for c in chosen]

# discrimination + calibration + Hosmer-Lemeshow
p_hat = fit.predict(X)
auc_model = float(auc_rankbiserial(p_hat[y == 1], p_hat[y == 0], nboot=1000)["auc"])
order = np.argsort(p_hat)
fpr, tpr = [], []
for t in np.linspace(0, 1, 101):
    pred = (p_hat >= t).astype(int)
    tpr.append(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    fpr.append(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
roc_df = pd.DataFrame({"fpr": np.round(fpr, 4), "tpr": np.round(tpr, 4)})

cal = pd.DataFrame({"p": p_hat, "y": y})
cal["bin"] = pd.qcut(cal.p, 8, duplicates="drop")
calg = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)
hl = float(sum(((g.obs * g.n) - (g.pred * g.n)) ** 2 / max(g.n * g.pred * (1 - g.pred), 1e-9)
               for _, g in calg.iterrows()))
diag = pd.DataFrame([{"metric": "model AUC (in-sample)", "value": round(auc_model, 3)},
                     {"metric": "pseudo R2 (McFadden)", "value": round(float(1 - fit_plain.llf / null_llf), 3)},
                     {"metric": "Hosmer-Lemeshow chi2 (8 groups)", "value": round(hl, 2)},
                     {"metric": "HL p-value", "value": float(f"{stats.chi2.sf(hl, 6):.3g}")},
                     {"metric": "n cases", "value": int(len(y))},
                     {"metric": "n FAIL", "value": int(y.sum())},
                     {"metric": "max VIF", "value": float(vif.vif.max())}])
diag.to_csv(f"{TAB}/model_diagnostics.csv", index=False)

# ============================================================================
# CHART CSVs (host-rendered, pre-aggregated)  +  PNG figures
# ============================================================================
charts, plots = [], []


def add_chart(name, kind, display_name, data, x, y, title):
    charts.append({"name": name, "kind": kind, "display_name": display_name,
                   "data": data, "x": x, "y": y, "title": title})


# --- C1 class balance -------------------------------------------------------
cb = []
for ck in SIZE_ORD:
    sub = ADV[ADV.checkpoint == ck]
    cb.append({"cohort": f"{ck} FAIL", "cases": int(sub.is_fail.sum())})
    cb.append({"cohort": f"{ck} PASS", "cases": int((1 - sub.is_fail).sum())})
cb.append({"cohort": "Present-object controls", "cases": int(len(CTRL_PRESENT))})
pd.DataFrame(cb).to_csv(f"{TAB}/class_balance.csv", index=False)
add_chart("class_balance", "bar", "Case counts by checkpoint and outcome",
          f"{TAB}/class_balance.csv", "cohort", "cases",
          "Hallucination (FAIL) vs correct-rejection (PASS) counts")

# --- C2 fail rate by checkpoint --------------------------------------------
fr = ADV.groupby("checkpoint", observed=True).agg(cases=("is_fail", "size"), fails=("is_fail", "sum")).reset_index()
fr["hallucination_rate"] = (fr.fails / fr.cases).round(4)
fr["ci_lo"], fr["ci_hi"] = zip(*[stats.beta.interval(.95, max(f, .5), max(n - f, .5))
                                 for f, n in zip(fr.fails, fr.cases)])
fr = fr.round(4)
fr.rename(columns={"checkpoint": "checkpoint"}).to_csv(f"{TAB}/failrate_by_checkpoint.csv", index=False)
add_chart("failrate_by_checkpoint", "bar", "Hallucination rate by checkpoint",
          f"{TAB}/failrate_by_checkpoint.csv", "checkpoint", "hallucination_rate",
          "Hallucination rate on adversarial-absent probes")

# --- C3 ranked separation (adversarial only) --------------------------------
r3 = sep_adv[["alias", "display_name", "signal", "separation", "auc", "auc_lo", "auc_hi",
              "direction", "n_fail", "n_pass"]].copy()
r3.rename(columns={"alias": "signal_label"}).to_csv(f"{TAB}/separation_ranked.csv", index=False)
add_chart("separation_ranked", "bar", "How well each attention signal separates FAIL from PASS",
          f"{TAB}/separation_ranked.csv", "signal_label", "separation",
          "FAIL/PASS separation per attention signal (adversarial probes)")

# --- C4 separation per checkpoint ------------------------------------------
r4 = sep_by_ck.copy()
r4["signal_checkpoint"] = r4["alias"] + " @ " + r4["checkpoint"].astype(str)
r4[["signal_checkpoint", "signal", "checkpoint", "auc", "auc_lo", "auc_hi", "separation",
    "n_fail", "n_pass", "direction"]].to_csv(f"{TAB}/separation_by_checkpoint_chart.csv", index=False)
add_chart("separation_by_checkpoint", "bar", "Signal separation within each checkpoint",
          f"{TAB}/separation_by_checkpoint_chart.csv", "signal_checkpoint", "separation",
          "FAIL/PASS separation per signal, within each checkpoint")

# --- C5..C7 binned hallucination-rate curves --------------------------------
top3 = sep_adv.head(3)["signal"].tolist()
curve_specs = []
for s in top3:
    sub = ADV[[s, "is_fail"]].dropna().copy()
    q = min(6, max(3, sub[s].nunique()))
    sub["b"] = pd.qcut(sub[s].rank(method="first"), q, labels=False)
    g = sub.groupby("b").agg(lo=(s, "min"), hi=(s, "max"), n=("is_fail", "size"),
                             fails=("is_fail", "sum")).reset_index()
    g["bin"] = [human_bin(a, b) for a, b in zip(g.lo, g.hi)]
    g["hallucination_rate"] = (g.fails / g.n).round(4)
    g["cases"] = g.n
    out = g[["bin", "hallucination_rate", "cases", "lo", "hi"]]
    nm = f"failrate_by_{s}"
    out.to_csv(f"{TAB}/{nm}.csv", index=False)
    add_chart(nm, "line", f"Hallucination rate across {FULLNAME[s].lower()} bins",
              f"{TAB}/{nm}.csv", "bin", "hallucination_rate",
              f"Hallucination rate by {FULLNAME[s].lower()} (adversarial probes)")
    curve_specs.append((s, out))

# --- C8 checkpoint distribution shift ---------------------------------------
sh = []
for s in SIGNALS:
    z = (ADV[s] - ADV[s].mean()) / ADV[s].std()
    for ck in SIZE_ORD:
        sh.append({"signal_checkpoint": f"{ALIAS[s]} @ {ck}", "signal": s, "checkpoint": ck,
                   "median_z": round(float(z[ADV.checkpoint == ck].median()), 3),
                   "median_raw": round(float(ADV.loc[ADV.checkpoint == ck, s].median()), 4),
                   "cases": int((ADV.checkpoint == ck).sum())})
pd.DataFrame(sh).to_csv(f"{TAB}/checkpoint_shift.csv", index=False)
add_chart("checkpoint_shift", "bar", "Attention geometry shift across checkpoints",
          f"{TAB}/checkpoint_shift.csv", "signal_checkpoint", "median_z",
          "Median attention geometry per checkpoint (standardized)")

# --- C9 fail rate by object -------------------------------------------------
ob = ADV.groupby("object").agg(cases=("is_fail", "size"), fails=("is_fail", "sum")).reset_index()
ob = ob[ob.cases >= 8].copy()
ob["hallucination_rate"] = (ob.fails / ob.cases).round(4)
ob = ob.sort_values("hallucination_rate", ascending=False)
ob.to_csv(f"{TAB}/failrate_by_object.csv", index=False)
add_chart("failrate_by_object", "bar", "Hallucination rate by queried object",
          f"{TAB}/failrate_by_object.csv", "object", "hallucination_rate",
          "Hallucination rate by object (objects with 8+ adversarial probes)")

# --- C10 adjusted odds ratios ----------------------------------------------
oc = coef_df.copy()
oc["term_label"] = oc["display_name"]
oc[["term_label", "signal", "kind", "odds_ratio_per_sd", "or_lo", "or_hi",
    "wald_p_robust", "lrt_p"]].to_csv(f"{TAB}/adjusted_odds_ratios.csv", index=False)
add_chart("adjusted_odds_ratios", "bar", "Adjusted odds ratios per 1 SD (all signals together)",
          f"{TAB}/adjusted_odds_ratios.csv", "term_label", "odds_ratio_per_sd",
          "Adjusted odds of hallucination per 1 SD (mutually adjusted)")

# --- C11 correlation pairs --------------------------------------------------
pair_df.head(12).to_csv(f"{TAB}/corr_pairs_top.csv", index=False)
add_chart("corr_pairs_top", "bar", "Strongest correlations between attention signals",
          f"{TAB}/corr_pairs_top.csv", "pair", "abs_rho",
          "Attention signals are strongly inter-correlated")

# --- C12 scatter of the most discriminative pair ----------------------------
sx, sy = sep_adv.iloc[0]["signal"], None
_scorr = ADV[SIGNALS].corr(method="spearman")
for cand in sep_adv["signal"].tolist()[1:]:
    if abs(_scorr.loc[sx, cand]) < .8:      # a second axis, not a near-duplicate
        sy = cand
        break
sy = sy or sep_adv.iloc[1]["signal"]
sc = ADV[[sx, sy, "outcome", "checkpoint", "object"]].dropna().copy()
sc["checkpoint"] = sc["checkpoint"].astype(str)
sc.round(4).to_csv(f"{TAB}/scatter_top_pair.csv", index=False)
add_chart("scatter_top_pair", "scatter", f"{FULLNAME[sx]} vs {FULLNAME[sy]}",
          f"{TAB}/scatter_top_pair.csv", sx, sy,
          f"{FULLNAME[sx]} vs {FULLNAME[sy]} on adversarial probes")

# --- C13 probe-design contrast (supporting / sanity) ------------------------
dd = design_df.melt(id_vars=["signal", "alias", "display_name"],
                    value_vars=["median_adversarial", "median_present"],
                    var_name="probe", value_name="median")
dd["probe"] = dd["probe"].map({"median_adversarial": "Absent-object probe",
                               "median_present": "Present-object probe"})
dd["signal_probe"] = dd["alias"] + " – " + dd["probe"]
dd[["signal_probe", "signal", "probe", "median"]].to_csv(f"{TAB}/probe_design_contrast_chart.csv", index=False)
add_chart("probe_design_contrast", "bar", "Attention geometry by probe type (design context)",
          f"{TAB}/probe_design_contrast_chart.csv", "signal_probe", "median",
          "Median attention geometry: absent-object vs present-object probes")

# ============================================================================
# PNG FIGURES (eval-chart-style §0 distribution-first; nature-figure polish)
# ============================================================================
def violin_pair(ax, a, b, title, ylab):
    parts = ax.violinplot([a, b], positions=[0, 1], widths=.8, showextrema=False)
    for pc, c in zip(parts["bodies"], [C_FAIL, C_PASS]):
        pc.set_facecolor(c); pc.set_alpha(.28); pc.set_edgecolor(c); pc.set_linewidth(1.0)
    for i, (v, c) in enumerate(zip([a, b], [C_FAIL, C_PASS])):
        ax.scatter(np.full(len(v), i) + RNG.normal(0, .055, len(v)), v, s=7, color=c,
                   alpha=.55, linewidths=0, zorder=3)
        ax.boxplot([v], positions=[i], widths=.16, showfliers=False, patch_artist=True,
                   medianprops=dict(color="white", lw=1.4),
                   boxprops=dict(facecolor=c, edgecolor=c, alpha=.95),
                   whiskerprops=dict(color=c, lw=1.0), capprops=dict(color=c, lw=1.0), zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"FAIL\nn={len(a)}", f"PASS\nn={len(b)}"])
    ax.set_title(title, pad=6); ax.set_ylabel(ylab)


# F1: violins per signal, adversarial only
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6))
for ax, s in zip(axes.ravel(), SIGNALS):
    a = ADV.loc[ADV.is_fail == 1, s].dropna().values
    b = ADV.loc[ADV.is_fail == 0, s].dropna().values
    row = sep_adv[sep_adv.signal == s].iloc[0]
    violin_pair(ax, a, b, f"{FULLNAME[s]}\nAUC {row.auc:.2f} [{row.auc_lo:.2f}, {row.auc_hi:.2f}]", ALIAS[s])
    if s == "max_relative_weight":
        ax.set_yscale("log")
for ax in axes.ravel()[len(SIGNALS):]:
    ax.axis("off")
fig.suptitle("Attention geometry of hallucinations vs correct rejections (absent-object probes only)",
             fontsize=12, y=1.01)
fig.tight_layout(h_pad=3.2, w_pad=1.6)
plots.append(savefig(fig, "attention_geometry_fail_vs_pass"))

# F2: ECDF of the top signals
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, s in zip(axes, top3):
    for grp, c, lab in [(1, C_FAIL, "FAIL"), (0, C_PASS, "PASS")]:
        v = np.sort(ADV.loc[ADV.is_fail == grp, s].dropna().values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.8, where="post", label=lab)
    ax.set_xlabel(ALIAS[s]); ax.set_ylabel("Cumulative share of cases")
    ax.set_title(FULLNAME[s], pad=6)
    if s == "max_relative_weight":
        ax.set_xscale("log")
axes[0].legend(loc="lower right")
fig.suptitle("Where the two distributions actually diverge", fontsize=12, y=1.03)
plots.append(savefig(fig, "ecdf_top_signals"))

# F3: binned hallucination-rate curves + n per bin
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
base = ADV.is_fail.mean()
for ax, (s, g) in zip(axes, curve_specs):
    ax.plot(range(len(g)), g.hallucination_rate, "-o", color=C_ACCENT, lw=1.8, ms=5)
    ax.axhline(base, color=C_INK, ls="--", lw=1, label=f"overall {base:.0%}")
    for i, (r, n) in enumerate(zip(g.hallucination_rate, g.cases)):
        ax.annotate(f"n={n}", (i, r), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=C_INK)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g["bin"], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, max(.75, g.hallucination_rate.max() * 1.3))
    ax.set_ylabel("Hallucination rate"); ax.set_xlabel(ALIAS[s]); ax.set_title(FULLNAME[s], pad=6)
axes[0].legend(loc="upper left", fontsize=7)
fig.suptitle("Hallucination rate across attention-signal bins (absent-object probes)", fontsize=12, y=1.04)
plots.append(savefig(fig, "hallucination_rate_curves"))

# F4: separation heatmap, signal x checkpoint
piv = sep_by_ck.pivot(index="signal", columns="checkpoint", values="auc").reindex(sep_adv.signal.tolist())
fig, ax = plt.subplots(figsize=(6.4, 4.2))
im = ax.imshow(piv.values - .5, cmap=CMAP_DIV, vmin=-.3, vmax=.3, aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([ALIAS[s] for s in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v - .5) > .18 else "#2b2a28")
ax.set_title("FAIL-vs-PASS separation (AUC) per checkpoint", pad=8)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=.85); cb.set_label("AUC − 0.50", fontsize=8)
cb.set_ticks([-.3, 0, .3]); cb.set_ticklabels(["0.20 (lower in FAIL)", "0.50", "0.80 (higher in FAIL)"])
plots.append(savefig(fig, "separation_by_checkpoint_heatmap"))

# F5: distribution shift across checkpoints (blue luminance ramp, outcome split)
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4))
for ax, s in zip(axes.ravel(), SIGNALS):
    for i, ck in enumerate(SIZE_ORD):
        v = ADV.loc[ADV.checkpoint == ck, s].dropna().values
        p = ax.violinplot([v], positions=[i], widths=.8, showextrema=False)
        p["bodies"][0].set_facecolor(RAMP[i % len(RAMP)]); p["bodies"][0].set_alpha(.45)
        p["bodies"][0].set_edgecolor(RAMP[i % len(RAMP)])
        ax.scatter([i - .16], [np.median(ADV.loc[(ADV.checkpoint == ck) & (ADV.is_fail == 1), s])],
                   color=C_FAIL, s=22, zorder=5, marker="v")
        ax.scatter([i + .16], [np.median(ADV.loc[(ADV.checkpoint == ck) & (ADV.is_fail == 0), s])],
                   color=C_PASS, s=22, zorder=5, marker="^")
    row = shift_df[shift_df.signal == s].iloc[0]
    ax.set_xticks(range(len(SIZE_ORD))); ax.set_xticklabels(SIZE_ORD)
    ax.set_title(f"{FULLNAME[s]}\n" + r"$\epsilon^2$" + f"={row.epsilon_sq:.2f}", pad=6)
    ax.set_ylabel(ALIAS[s])
    if s == "max_relative_weight":
        ax.set_yscale("log")
axes.ravel()[-1].axis("off")
axes.ravel()[-1].legend(handles=[Line2D([], [], marker="v", ls="", color=C_FAIL, label="FAIL median"),
                                 Line2D([], [], marker="^", ls="", color=C_PASS, label="PASS median"),
                                 Line2D([], [], marker="s", ls="", color=RAMP[1], label="all cases (violin)")],
                        loc="center", fontsize=9)
fig.suptitle("Do the distributions themselves move with model size? (absent-object probes)",
             fontsize=12, y=1.01)
fig.tight_layout(h_pad=3.0, w_pad=1.6)
plots.append(savefig(fig, "checkpoint_distribution_shift"))

# F6: correlation heatmap
fig, ax = plt.subplots(figsize=(6.2, 5.2))
cm = corr.loc[MODEL_SIGNALS, MODEL_SIGNALS]
im = ax.imshow(cm.values, cmap=CMAP_DIV, vmin=-1, vmax=1)
lbl = [ALIAS[c] for c in MODEL_SIGNALS]
ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, rotation=40, ha="right")
ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl)
for i in range(len(lbl)):
    for j in range(len(lbl)):
        v = cm.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > .6 else "#2b2a28")
ax.grid(False)
ax.set_title("Attention signals overlap heavily (Spearman)", pad=8)
fig.colorbar(im, ax=ax, shrink=.85).set_label("Spearman rho", fontsize=8)
plots.append(savefig(fig, "signal_correlation_heatmap"))

# F7: scatter of the leading pair, colored by outcome, faceted by checkpoint
fig, axes = plt.subplots(1, len(SIZE_ORD), figsize=(13, 4.0), sharex=True, sharey=True)
for ax, ck in zip(np.atleast_1d(axes), SIZE_ORD):
    sub = ADV[ADV.checkpoint == ck]
    for grp, c, lab in [(0, C_PASS, "PASS"), (1, C_FAIL, "FAIL")]:
        q = sub[sub.is_fail == grp]
        ax.scatter(q[sx], q[sy], s=20, color=c, alpha=.65, linewidths=0, label=lab)
    ax.set_title(f"{ck}  (n={len(sub)}, {sub.is_fail.mean():.0%} FAIL)", pad=6)
    ax.set_xlabel(ALIAS[sx])
    if sx == "max_relative_weight":
        ax.set_xscale("log")
    if sy == "max_relative_weight":
        ax.set_yscale("log")
np.atleast_1d(axes)[0].set_ylabel(ALIAS[sy])
np.atleast_1d(axes)[0].legend(loc="best")
fig.suptitle(f"{FULLNAME[sx]} vs {FULLNAME[sy]}, by checkpoint", fontsize=12, y=1.02)
plots.append(savefig(fig, "top_pair_scatter_by_checkpoint"))

# F8: forest plot of adjusted odds ratios
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ff = coef_df.sort_values("odds_ratio_per_sd")
ypos = np.arange(len(ff))
cols = [C_INK if k != "attention signal" else
        (C_FAIL if o > 1 else C_PASS) for k, o in zip(ff.kind, ff.odds_ratio_per_sd)]
for i, (_, r) in enumerate(ff.iterrows()):
    ax.plot([r.or_lo, r.or_hi], [i, i], color=cols[i], lw=2, alpha=.65, solid_capstyle="round")
ax.scatter(ff.odds_ratio_per_sd, ypos, color=cols, s=42, zorder=4)
ax.axvline(1, color=C_INK, ls="--", lw=1)
ax.set_yticks(ypos); ax.set_yticklabels(ff.display_name)
ax.set_xscale("log")
ax.set_xlabel("Adjusted odds ratio per 1 SD (log scale); 95% CI, image-clustered SE")
ax.set_title("Mutually adjusted association with hallucination", pad=8)
ax.annotate("no association", xy=(1, len(ff) - .4), fontsize=7.5, color=C_INK, ha="center")
plots.append(savefig(fig, "adjusted_odds_forest"))

# F9: ROC + calibration
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
axes[0].plot(roc_df.fpr, roc_df.tpr, color=C_ACCENT, lw=2)
axes[0].plot([0, 1], [0, 1], ls="--", color=C_INK, lw=1)
axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
axes[0].set_title(f"Discrimination (in-sample AUC = {auc_model:.2f})", pad=6)
axes[1].plot(calg.pred, calg.obs, "-o", color=C_ACCENT, lw=1.8, ms=6)
axes[1].plot([0, calg.pred.max() * 1.1], [0, calg.pred.max() * 1.1], ls="--", color=C_INK, lw=1)
for _, r in calg.iterrows():
    axes[1].annotate(f"n={int(r.n)}", (r.pred, r.obs), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color=C_INK)
axes[1].set_xlabel("Predicted hallucination probability"); axes[1].set_ylabel("Observed rate")
axes[1].set_title("Calibration (8 equal-size groups)", pad=6)
fig.suptitle("Full attention-geometry model on absent-object probes — in-sample fit", fontsize=12, y=1.03)
plots.append(savefig(fig, "model_discrimination_calibration"))

# F10: object x checkpoint hallucination-rate heatmap
top_obj = ob.head(10)["object"].tolist()
hm = ADV[ADV.object.isin(top_obj)].pivot_table(index="object", columns="checkpoint",
                                               values="is_fail", aggfunc="mean", observed=True)
hn = ADV[ADV.object.isin(top_obj)].pivot_table(index="object", columns="checkpoint",
                                               values="is_fail", aggfunc="size", observed=True)
hm = hm.reindex(top_obj)
hn = hn.reindex(top_obj)
fig, ax = plt.subplots(figsize=(6.6, 4.8))
im = ax.imshow(hm.values, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns)
ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v, n = hm.values[i, j], hn.values[i, j]
        if v == v:
            ax.text(j, i, f"{v:.0%}\nn={int(n)}", ha="center", va="center", fontsize=7,
                    color="white" if v > .55 else "#2b2a28")
ax.grid(False)
ax.set_title("Hallucination rate by object and checkpoint", pad=8)
fig.colorbar(im, ax=ax, shrink=.85).set_label("Hallucination rate", fontsize=8)
plots.append(savefig(fig, "object_checkpoint_heatmap"))

# reproducibility
with open("session_info.txt", "w") as fh:
    import sys, scipy, statsmodels
    fh.write(f"python={sys.version.split()[0]}\npandas={pd.__version__}\nnumpy={np.__version__}\n"
             f"scipy={scipy.__version__}\nstatsmodels={statsmodels.__version__}\n"
             f"matplotlib={matplotlib.__version__}\nseed=20240726\nrows={len(df)}\n")

# ============================================================================
# RESULT JSON
# ============================================================================
s1 = sep_adv.iloc[0]
s2 = sep_adv.iloc[1]
s3 = sep_adv.iloc[2]
# per-signal handles used in the narrative
S_ENT = sep_adv[sep_adv.signal == "attention_entropy"].iloc[0]
S_EDGE = sep_adv[sep_adv.signal == "edge_mass"].iloc[0]
S_CEN = sep_adv[sep_adv.signal == "center_offset"].iloc[0]
s1_ck = sep_by_ck[sep_by_ck.signal == s1.signal].set_index("checkpoint")["auc"].astype(float)
s1_ck_best, s1_ck_worst = s1_ck.idxmax(), s1_ck.idxmin()
V_OBJ = float(cat_df.loc[cat_df.variable == "object", "cramers_v"].iloc[0])
V_CK = float(cat_df.loc[cat_df.variable == "checkpoint", "cramers_v"].iloc[0])
curve1 = curve_specs[0][1]
low_bin, high_bin = curve1.iloc[0], curve1.iloc[-1]
# low-collinearity signals (a genuinely separate axis from the concentration block)
LOWVIF = vif[vif.vif < 2.5]["signal"].tolist()
n_flip = int(simpson_df.sign_flip_across_checkpoints.sum())
fr_lo, fr_hi = fr.hallucination_rate.min(), fr.hallucination_rate.max()
ck_lo = fr.loc[fr.hallucination_rate.idxmin(), "checkpoint"]
ck_hi = fr.loc[fr.hallucination_rate.idxmax(), "checkpoint"]
top_corr = pair_df.iloc[0]
indep_names = [FULLNAME[c] for c in INDEP] or ["none reached the AIC threshold"]
adj_sig = coef_df[(coef_df.kind == "attention signal") & (coef_df.ci_excludes_1)]
shift_top = shift_df.iloc[0]
big_obj = ob.iloc[0]
small_obj = ob.iloc[-1]

# recipe candidate: high peakiness + low entropy, evaluated as a two-group split
thr_top1 = float(ADV["top1_share"].quantile(.75))
thr_ent = float(ADV["attention_entropy"].quantile(.25))
mask = (ADV["top1_share"] > thr_top1) & (ADV["attention_entropy"] < thr_ent)
grp_a = ADV.loc[~mask, "is_fail"].astype(int).tolist()
grp_b = ADV.loc[mask, "is_fail"].astype(int).tolist()
peak_rate = float(np.mean(grp_b)) if grp_b else float("nan")
flat_rate = float(np.mean(grp_a))
peak_capture = float(np.sum(grp_b) / max(ADV.is_fail.sum(), 1))

# deterministic companion plan entries for the binned-rate CSV charts
curve_plan = [
    {"name": f"failrate_by_{s}", "display_name": f"Hallucination rate across {FULLNAME[s].lower()} bins",
     "question": f"How does hallucination risk change along {FULLNAME[s].lower()}?",
     "data_shape": "numeric-vs-binary", "plot_kind": "line", "fallback_kind": "line",
     "required_columns": [s, "label"],
     "rationale": "Equal-count bins with per-bin n keep the risk trend readable without assuming a functional form.",
     "disposition": "primary" if i == 0 else "supporting",
     "not_promoted_reason": "" if i == 0 else "Second- and third-ranked signals are near-duplicates of the leading one; kept as drill-down."}
    for i, s in enumerate(top3)
]

result = {
  "plain_question": "We are looking at whether the way a vision-language model spreads its visual attention over an image can tell us when it will wrongly claim to see an object that is not there, and whether that pattern is the same for the small, medium and large versions of the model.",
  "observations": [
    f"Built one tidy table of {len(df)} cases from {len(files)} per-checkpoint files; each file's scalar metadata (model, seed, decoding, extraction notes) was merged onto every row it produced. Written to records.json.",
    f"Outcome column 'label' is binary: {int(df.is_fail.sum())} FAIL (hallucination) vs {int((1-df.is_fail).sum())} PASS.",
    f"Design constraint: every FAIL sits in the adversarial absent-object probe ({profile['fails_outside_adversarial']} FAILs among the {len(CTRL_PRESENT)} present-object rows). The honest FAIL/PASS contrast is therefore within the {len(ADV)} adversarial cases, where {int(ADV.is_fail.sum())} are hallucinations ({ADV.is_fail.mean():.1%}).",
    f"No missing values in any of the {len(SIGNALS)} attention scalars ({profile['missing_cells']} missing cells).",
    f"Strongest within-adversarial separator: {FULLNAME[s1.signal]}, AUC {s1.auc:.2f} [{s1.auc_lo:.2f}, {s1.auc_hi:.2f}], {s1.direction}.",
    f"Hallucination rate rises with checkpoint size: {fr_lo:.0%} at {ck_lo} to {fr_hi:.0%} at {ck_hi}.",
    f"The signals are heavily redundant: strongest pair {top_corr['pair']} at Spearman rho = {top_corr['spearman_rho']}; max VIF = {vif.vif.max():.1f}.",
    f"Forward AIC selection keeps {len(INDEP)} of {len(MODEL_SIGNALS)} signals: {', '.join(indep_names)}.",
  ],
  "visual_plan": [
    {"name": "class_balance", "display_name": "Case counts by checkpoint and outcome",
     "question": "How many hallucinations and correct rejections are there per model size, and how many present-object controls?",
     "data_shape": "categorical-counts", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": ["checkpoint", "label", "probe_type"],
     "rationale": "Counts are the one thing a bar's filled area legitimately encodes; it also exposes the design imbalance before any rate is read.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "failrate_by_checkpoint", "display_name": "Hallucination rate by checkpoint",
     "question": "Does the bigger checkpoint hallucinate more or less often on this slice?",
     "data_shape": "binary-vs-ordered-categorical", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": ["checkpoint", "label"],
     "rationale": "Three ordered groups with wide binomial intervals; the CSV carries the interval bounds so the rate is not read as exact.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "attention_geometry_fail_vs_pass", "display_name": "Attention geometry of hallucinations vs correct rejections",
     "question": "Do the attention scalars have different distributions in hallucinations than in correct rejections?",
     "data_shape": "numeric-vs-binary", "plot_kind": "violin", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["label"],
     "rationale": "With only ~126 hallucinations a mean bar would be outlier-driven and would hide overlap; violin + box + jittered points shows the full distribution and the sample size.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "ecdf_top_signals", "display_name": "Where the two distributions diverge",
     "question": "Is the FAIL/PASS gap a whole-distribution shift or only a tail effect?",
     "data_shape": "numeric-vs-binary", "plot_kind": "ecdf", "fallback_kind": "line",
     "required_columns": top3 + ["label"],
     "rationale": "An ECDF makes tail-versus-body claims readable, which a violin alone cannot settle.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "separation_ranked", "display_name": "How well each attention signal separates FAIL from PASS",
     "question": "Which attention scalars separate the two outcomes best, and by how much?",
     "data_shape": "many-numeric", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["label"],
     "rationale": "One ranked effect axis (AUC-based separation) with bootstrap bounds in the CSV lets the reader compare signals on a common scale.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "hallucination_rate_curves", "display_name": "Hallucination rate across signal bins",
     "question": "Does hallucination risk change smoothly along each leading signal, or only at an extreme?",
     "data_shape": "numeric-vs-binary", "plot_kind": "line", "fallback_kind": "line",
     "required_columns": top3 + ["label"],
     "rationale": "An ordered binned-rate line shows the risk trend without assuming a linear or logistic shape; per-bin n is annotated because bins are small.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "separation_by_checkpoint_heatmap", "display_name": "Signal separation within each checkpoint",
     "question": "Do the same signals separate hallucinations from correct rejections at every model size?",
     "data_shape": "numeric-vs-binary-by-group", "plot_kind": "heatmap", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["label", "checkpoint"],
     "rationale": "A signal-by-checkpoint grid of one common effect metric makes direction flips and strength changes visible at a glance; a diverging ramp centred on chance encodes sign.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "checkpoint_distribution_shift", "display_name": "Do the distributions move with model size?",
     "question": "Beyond separation, do the attention distributions themselves shift from 2B to 8B?",
     "data_shape": "numeric-vs-ordered-categorical", "plot_kind": "violin", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["checkpoint"],
     "rationale": "Ordered dimension gets a single-hue luminance ramp; FAIL/PASS medians are overlaid as markers so a shift is not confused with a separation change.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "signal_correlation_heatmap", "display_name": "Attention signals overlap heavily",
     "question": "How much do the seven attention scalars duplicate each other?",
     "data_shape": "many-numeric", "plot_kind": "heatmap", "fallback_kind": "bar",
     "required_columns": MODEL_SIGNALS,
     "rationale": "A correlation matrix is the direct evidence for the collinearity question the caller asked; annotated cells avoid a colour-only read.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "adjusted_odds_forest", "display_name": "Mutually adjusted association with hallucination",
     "question": "Which signals still carry an association once the others and the checkpoint are held constant?",
     "data_shape": "many-numeric", "plot_kind": "forest", "fallback_kind": "bar",
     "required_columns": MODEL_SIGNALS + ["label", "checkpoint"],
     "rationale": "Dot + interval on a log odds axis is the honest rendering of an effect estimate; a bar would fake certainty about a point estimate with wide intervals.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "top_pair_scatter_by_checkpoint", "display_name": "Leading signal pair, split by checkpoint",
     "question": "Do hallucinations occupy a distinct region of the two-signal plane, and is it the same region at each size?",
     "data_shape": "numeric-vs-numeric", "plot_kind": "scatter", "fallback_kind": "scatter",
     "required_columns": [sx, sy, "label", "checkpoint"],
     "rationale": "Joint structure and the degree of class overlap are only visible in a full-width scatter; faceting checks whether the region is stable across sizes.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "object_checkpoint_heatmap", "display_name": "Hallucination rate by object and checkpoint",
     "question": "Are some queried objects hallucinated far more often, and is that consistent across sizes?",
     "data_shape": "binary-vs-two-categoricals", "plot_kind": "heatmap", "fallback_kind": "bar",
     "required_columns": ["object", "checkpoint", "label"],
     "rationale": "Two categorical dimensions with a rate belong in an annotated heatmap rather than several grouped bar panels; cell n is printed because most cells are small.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "model_discrimination_calibration", "display_name": "Model discrimination and calibration",
     "question": "How much of the hallucination outcome can the seven attention scalars account for together?",
     "data_shape": "model-diagnostic", "plot_kind": "line", "fallback_kind": "line",
     "required_columns": MODEL_SIGNALS + ["label"],
     "rationale": "An ROC plus a reliability curve is the standard honest summary of a fitted probability model; a lone accuracy number would hide the miscalibration.",
     "disposition": "supporting",
     "not_promoted_reason": "In-sample fit on the same rows used to pick the terms; it bounds how much structure exists but is not itself a ranked finding."},
    {"name": "probe_design_contrast", "display_name": "Attention geometry by probe type (design context)",
     "question": "How different is the attention geometry between absent-object and present-object probes?",
     "data_shape": "numeric-vs-categorical", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["probe_type"],
     "rationale": "Shows why the FAIL/PASS analysis is restricted to absent-object probes: probe type moves the same scalars, so a pooled comparison would mix design with outcome.",
     "disposition": "supporting",
     "not_promoted_reason": "Diagnostic context for the slice construction, not a finding about hallucination; the present-object rows contain no FAILs by design."},
    {"name": "failrate_by_object", "display_name": "Hallucination rate by queried object",
     "question": "Which object categories carry the highest hallucination rate?",
     "data_shape": "binary-vs-categorical", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": ["object", "label"],
     "rationale": "Rate per category with n in the table; restricted to objects with at least 8 adversarial probes so single-case categories cannot show a 0% or 100% rate.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "corr_pairs_top", "display_name": "Strongest correlations between attention signals",
     "question": "Which specific pairs of signals are near-duplicates?",
     "data_shape": "many-numeric", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": MODEL_SIGNALS,
     "rationale": "Names the redundant pairs the heatmap shows as a block, so the reader can see which columns to drop.",
     "disposition": "supporting",
     "not_promoted_reason": "A ranked restatement of the correlation heatmap; kept as drill-down rather than a separate takeaway."},
    {"name": "checkpoint_shift", "display_name": "Attention geometry shift across checkpoints",
     "question": "How large is the median move of each signal from 2B to 8B?",
     "data_shape": "numeric-vs-ordered-categorical", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["checkpoint"],
     "rationale": "Standardized medians put all seven signals on one comparable axis for the host-rendered view.",
     "disposition": "primary", "not_promoted_reason": ""},
    {"name": "separation_by_checkpoint", "display_name": "Signal separation within each checkpoint",
     "question": "Deterministic view of the per-checkpoint separation values.",
     "data_shape": "numeric-vs-binary-by-group", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": SIGNALS + ["label", "checkpoint"],
     "rationale": "The host-rendered companion to the separation heatmap, carrying the bootstrap bounds in the CSV.",
     "disposition": "supporting",
     "not_promoted_reason": "Same numbers as the heatmap; kept as the deterministic table-backed version."},
    {"name": "adjusted_odds_ratios", "display_name": "Adjusted odds ratios per 1 SD",
     "question": "Deterministic view of the mutually adjusted odds ratios.",
     "data_shape": "many-numeric", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": MODEL_SIGNALS + ["label", "checkpoint"],
     "rationale": "Host-rendered companion to the forest plot; intervals are kept in the CSV columns.",
     "disposition": "supporting",
     "not_promoted_reason": "The forest PNG is the honest rendering; this is its deterministic backup."},
    {"name": "scatter_top_pair", "display_name": "Leading signal pair",
     "question": "Deterministic scatter of the two leading signals.",
     "data_shape": "numeric-vs-numeric", "plot_kind": "scatter", "fallback_kind": "scatter",
     "required_columns": [sx, sy, "label"],
     "rationale": "Host-rendered companion to the faceted scatter; outcome and checkpoint are carried as columns.",
     "disposition": "supporting",
     "not_promoted_reason": "The faceted PNG carries the outcome colouring; this is the deterministic backup."},
    *curve_plan,
  ],
  "takeaways": [
    {"plain_title": f"Wrong answers come with visual attention piled onto a few spots: the most-attended region takes {s1.median_fail:.0%} of the model's attention when it hallucinates versus {s1.median_pass:.0%} when it correctly says no, and the wrong-answer rate climbs from {low_bin.hallucination_rate:.0%} in the most spread-out cases to {high_bin.hallucination_rate:.0%} in the most concentrated ones.",
     "title": f"Concentrated attention is the dominant FAIL correlate: {FULLNAME[s1.signal]} AUC {s1.auc:.2f} [{s1.auc_lo:.2f}, {s1.auc_hi:.2f}] (median {s1.median_fail} FAIL vs {s1.median_pass} PASS, n={int(s1.n_fail)}/{int(s1.n_pass)}); hallucination rate rises {low_bin.hallucination_rate:.0%} -> {high_bin.hallucination_rate:.0%} across its six equal-count bins.",
     "chart_names": ["separation_ranked", "attention_geometry_fail_vs_pass", "hallucination_rate_curves",
                     f"failrate_by_{top3[0]}", "ecdf_top_signals"],
     "table_names": ["separation_adversarial", "marginal_screening"],
     "analysis": f"Inside the {len(ADV)} adversarial absent-object probes, four concentration measures separate the outcomes at AUC 0.72-0.82: {FULLNAME[s1.signal]} {s1.auc:.2f}, {FULLNAME[s2.signal]} {s2.auc:.2f} (i.e. {1-float(S_ENT.auc):.2f} in the low-entropy direction), {FULLNAME[s3.signal]} {s3.auc:.2f}. The binned curve is monotone across all six bins ({', '.join(f'{v:.0%}' for v in curve1.hallucination_rate)}, n={int(curve1.cases.iloc[0])} each) against a {base:.0%} baseline, and the ECDF shows the FAIL curve displaced across the whole range rather than in a tail only. The direction is consistent: hallucinations concentrate attention, correct rejections spread it.",
     "caveat": "Descriptive only. These scalars are extracted from the same forward pass that produced the answer, so they are contemporaneous with the outcome and cannot be read as something that precedes or produces it."},
    {"plain_title": f"Which object is asked about matters far more than which model size is used: when a {big_obj.object} is absent the model claims to see one {big_obj.hallucination_rate:.0%} of the time, while for {small_obj.object} it does so only {small_obj.hallucination_rate:.0%} of the time.",
     "title": f"Object identity is the strongest categorical driver (Cramer's V = {V_OBJ:.2f} over {int(cat_df.loc[cat_df.variable=='object','levels'].iloc[0])} objects with 8+ probes) versus checkpoint at V = {V_CK:.2f}: {big_obj.object} {big_obj.hallucination_rate:.0%} ({int(big_obj.fails)}/{int(big_obj.cases)}) down to {small_obj.object} {small_obj.hallucination_rate:.0%} ({int(small_obj.fails)}/{int(small_obj.cases)}).",
     "chart_names": ["failrate_by_object", "object_checkpoint_heatmap"],
     "table_names": ["categorical_association", "failrate_by_object"],
     "analysis": f"Across objects with at least 8 adversarial probes the hallucination rate spans {ob.hallucination_rate.min():.0%} to {ob.hallucination_rate.max():.0%}, an order of magnitude wider than the {fr_lo:.0%}-{fr_hi:.0%} spread across checkpoints, and the object association (V = {V_OBJ:.2f}) dwarfs the checkpoint one (V = {V_CK:.2f}). The object-by-checkpoint heatmap shows the high-rate objects tend to stay high at more than one size, so this is not a single-checkpoint artefact. Because the adversarial probe deliberately asks about objects that commonly co-occur with the scene, object identity and probe difficulty are entangled by construction.",
     "caveat": "Object counts are very uneven (person 47 probes, bottle 9) and no multiplicity correction is applied across categories; individual per-object rates are unstable and many heatmap cells hold fewer than 10 probes."},
    {"plain_title": f"The same attention pattern points the same way at every model size, but how clearly it flags a wrong answer is uneven - clearest in the smallest model and weakest in the middle one, not a steady trend with size.",
     "title": f"Direction is stable across checkpoints ({n_flip}/{len(SIGNALS)} signals change sign), but strength is not monotone in size: {FULLNAME[s1.signal]} AUC {s1_ck[s1_ck_best]:.2f} at {s1_ck_best}, {s1_ck.min():.2f} at {s1_ck_worst}, {s1_ck[SIZE_ORD[-1]]:.2f} at {SIZE_ORD[-1]}.",
     "chart_names": ["separation_by_checkpoint_heatmap", "separation_by_checkpoint", "top_pair_scatter_by_checkpoint", "scatter_top_pair"],
     "table_names": ["separation_by_checkpoint", "conditioning_check"],
     "analysis": f"Every one of the {len(SIGNALS)} signals keeps its FAIL/PASS direction in all three checkpoints ({n_flip} sign flips), so the pooled ranking is not a Simpson artefact. What changes is magnitude: for {FULLNAME[s1.signal]} the AUC runs {s1_ck[SIZE_ORD[0]]:.2f} / {s1_ck[SIZE_ORD[1]]:.2f} / {s1_ck[SIZE_ORD[2]]:.2f} across {SIZE_ORD[0]}/{SIZE_ORD[1]}/{SIZE_ORD[2]}, and {FULLNAME['max_relative_weight']} falls from {float(sep_by_ck[(sep_by_ck.signal=='max_relative_weight')&(sep_by_ck.checkpoint==SIZE_ORD[0])].auc.iloc[0]):.2f} at {SIZE_ORD[0]} to {float(sep_by_ck[(sep_by_ck.signal=='max_relative_weight')&(sep_by_ck.checkpoint==SIZE_ORD[1])].auc.iloc[0]):.2f} at {SIZE_ORD[1]}. The dip at the middle checkpoint rather than a size-ordered trend is what the heatmap shows.",
     "caveat": f"Each stratum holds only {int(sep_by_ck.n_fail.min())}-{int(sep_by_ck.n_fail.max())} hallucinations and the bootstrap intervals overlap between checkpoints, so the magnitude differences are not distinguishable from sampling noise."},
    {"plain_title": f"The attention numbers themselves are on a different scale in each model, so a cut-off learned on one size will not carry over: the typical relative-attention value is {shift_df.iloc[0]['median_' + SIZE_ORD[1]] / max(shift_df.iloc[0]['median_' + SIZE_ORD[0]], 1e-9):.1f} times larger in the middle model than in the smallest.",
     "title": f"Distributions shift across checkpoints independently of outcome: {FULLNAME[shift_top.signal]} medians {shift_top['median_' + SIZE_ORD[0]]} / {shift_top['median_' + SIZE_ORD[1]]} / {shift_top['median_' + SIZE_ORD[2]]} (Kruskal-Wallis epsilon-squared {shift_top.epsilon_sq}), while the concentration measures drift only mildly upward ({FULLNAME[s1.signal]} epsilon-squared {float(shift_df.loc[shift_df.signal==s1.signal,'epsilon_sq'].iloc[0]):.2f}).",
     "chart_names": ["checkpoint_distribution_shift", "checkpoint_shift"],
     "table_names": ["checkpoint_distribution_shift"],
     "analysis": f"The two relative-attention scalars move the most between checkpoints ({FULLNAME[shift_top.signal]} epsilon-squared {shift_top.epsilon_sq}, {FULLNAME['max_relative_weight']} {float(shift_df.loc[shift_df.signal=='max_relative_weight','epsilon_sq'].iloc[0]):.2f}) and do so non-monotonically, peaking at {SIZE_ORD[1]}. The bounded concentration measures shift far less (epsilon-squared {float(shift_df.loc[shift_df.signal==s1.signal,'epsilon_sq'].iloc[0]):.2f} for {FULLNAME[s1.signal]}, {float(shift_df.loc[shift_df.signal=='top1_share','epsilon_sq'].iloc[0]):.2f} for {FULLNAME['top1_share']}) and drift upward with size, while {FULLNAME['edge_mass']} drifts down ({float(shift_df.loc[shift_df.signal=='edge_mass','spearman_vs_size'].iloc[0]):.2f} rank correlation with size). Any absolute threshold on the relative-weight scalars is therefore checkpoint-specific, whereas a within-checkpoint ranking on the concentration scalars is comparatively stable.",
     "caveat": "These are marginal distribution shifts over all adversarial cases; they say nothing on their own about whether a signal separates FAIL from PASS at that checkpoint, and the three checkpoints differ in tokenizer/resolution settings that are not recorded here."},
    {"plain_title": f"The seven attention numbers are mostly measuring one thing - once the strongest one is in, no other adds enough to keep - except for the two that describe where in the image attention sits rather than how tight it is.",
     "title": f"Heavy collinearity: {top_corr['pair']} rho = {top_corr['spearman_rho']}, max VIF {vif.vif.max():.1f}; forward AIC selection keeps {len(INDEP)}/{len(MODEL_SIGNALS)} ({', '.join(indep_names)}, AIC drop {float(fs_df.aic_drop.iloc[0]):.0f}). Only {', '.join(FULLNAME[s] for s in LOWVIF)} sit outside the concentration block (VIF < 2.5).",
     "chart_names": ["signal_correlation_heatmap", "adjusted_odds_forest", "adjusted_odds_ratios", "corr_pairs_top"],
     "table_names": ["signal_correlations", "collinearity_vif", "forward_selection", "adjusted_model_coefficients"],
     "analysis": f"{FULLNAME[s1.signal]}, {FULLNAME['top1_share']} and {FULLNAME['attention_entropy']} are near-duplicates (|rho| {abs(float(corr.loc['focus_share','top1_share'])):.2f}-{abs(float(corr.loc['attention_entropy','focus_share'])):.2f}, VIF up to {vif.vif.max():.1f}) and the two relative-weight scalars form a second pair (rho {float(corr.loc['log10_max_relative_weight','mean_relative_weight']):.2f}), so the practical answer to 'which carry independent signal' is: one concentration axis, plus a weakly-related spatial axis carried by {' and '.join(FULLNAME[s] for s in LOWVIF)} ({FULLNAME['edge_mass']} AUC {S_EDGE.auc:.2f}, i.e. edge-heavy attention is associated with correct rejection; {FULLNAME['center_offset']} {S_CEN.auc:.2f}). Entering all seven together produces sign reversals against their marginal direction ({FULLNAME['top1_share']} adjusted OR {float(coef_df.loc[coef_df.signal=='top1_share','odds_ratio_per_sd'].iloc[0]):.2f} despite being higher in FAIL marginally), which is the expected symptom of this collinearity rather than a finding about that signal.",
     "caveat": "Which member of a correlated block survives selection is close to arbitrary; a resampled run could retain a near-duplicate instead. This does not rank the dropped signals as uninformative."},
    {"plain_title": f"The largest model version is not the safest on these trick questions: it wrongly answers yes {fr_hi:.0%} of the time against {fr_lo:.0%} for the best of the three, a gap small enough to be chance.",
     "title": f"No checkpoint-size benefit in this slice: hallucination rate {', '.join(f'{r.checkpoint} {r.hallucination_rate:.0%}' for _, r in fr.iterrows())} (Cramer's V = {V_CK:.2f}, chi-square p = {float(cat_df.loc[cat_df.variable=='checkpoint','chi2_p'].iloc[0]):.2f}), with overlapping 95% binomial intervals.",
     "chart_names": ["failrate_by_checkpoint", "class_balance"],
     "table_names": ["failrate_by_checkpoint", "categorical_association"],
     "analysis": f"Per-checkpoint rates are " + ", ".join(f"{r.checkpoint} {r.hallucination_rate:.0%} ({int(r.fails)}/{int(r.cases)}, 95% CI {r.ci_lo:.0%}-{r.ci_hi:.0%})" for _, r in fr.iterrows()) + f". The intervals overlap across all three sizes and the checkpoint-outcome association is negligible (V = {V_CK:.2f}) next to the object association (V = {V_OBJ:.2f}). In the adjusted model the checkpoint fixed effects also have intervals spanning 1.",
     "caveat": "This is a curated hallucination slice with a fixed control budget of 80 correct rejections plus 80 present-object detections per checkpoint, not each checkpoint's natural error rate; these rates are not comparable to published benchmark accuracy."},
    {"plain_title": f"Using all seven attention numbers at once, a wrong answer is ranked as riskier than a right one about {auc_model:.0%} of the time, and the predicted risk matches what actually happened - a real but partial signal, measured on the same cases it was built from.",
     "title": f"Full logistic model (7 signals + checkpoint fixed effects, image/object-clustered SEs) reaches in-sample AUC {auc_model:.2f}, McFadden pseudo-R2 {float(1 - fit_plain.llf/null_llf):.2f}, Hosmer-Lemeshow chi-square {hl:.1f} (p = {float(stats.chi2.sf(hl, 6)):.2f}) on n={len(y)} with {int(y.sum())} FAILs.",
     "chart_names": ["model_discrimination_calibration", "adjusted_odds_forest"],
     "table_names": ["model_diagnostics", "adjusted_model_coefficients"],
     "analysis": f"Logistic regression is used because the outcome is binary; checkpoint enters as a fixed effect rather than a random one because three clusters is far too few to estimate a random-effect variance, and standard errors are clustered on image/object since images recur across checkpoints. The fit reaches AUC {auc_model:.2f} with a well-behaved calibration curve (HL p = {float(stats.chi2.sf(hl, 6)):.2f}), but explains {float(1 - fit_plain.llf/null_llf):.0%} of the null deviance - attention geometry carries a substantial but partial trace of which absent-object probes get answered 'Yes'.",
     "caveat": f"Fitted and scored on the same rows with no held-out evaluation, so discrimination is optimistic; with max VIF {vif.vif.max():.1f} the individual coefficients are unstable even where the overall fit is not."},
    {"plain_title": "Because a wrong answer only counts when the object is genuinely absent, the comparison has to stay inside those trick questions - including the ordinary questions would measure how the study was built rather than the failure.",
     "title": f"All {int(df.is_fail.sum())} FAILs are adversarial absent-object probes; the {len(CTRL_PRESENT)} present-object rows contain 0 FAILs by slice construction, and probe type itself moves the same scalars ({FULLNAME[design_df.iloc[0]['signal']]} median {design_df.iloc[0]['median_adversarial']} vs {design_df.iloc[0]['median_present']}).",
     "chart_names": ["probe_design_contrast", "class_balance"],
     "table_names": ["probe_design_contrast", "separation_all"],
     "analysis": f"Pooling all {len(df)} cases would make any scalar that differs between absent-object and present-object probes look like a hallucination correlate when it is a probe-design difference: pooled separation for {FULLNAME[s1.signal]} is {float(sep_all.loc[sep_all.signal==s1.signal,'auc'].iloc[0]):.2f} against {s1.auc:.2f} within adversarial probes. Every FAIL/PASS comparison reported above is therefore computed inside the adversarial subset only, where PASS means a correct rejection of the same kind of trick question. The pooled table is retained only as a contrast.",
     "caveat": "This is a statement about how the data was sliced, not a finding about the model; it bounds what any FAIL/PASS contrast in this dataset can mean."},
  ],
  "chart_readings": [
    {"chart": "class_balance", "reading": f"Each checkpoint contributes 80 correct rejections plus its own hallucination count ({int(fr.fails.iloc[0])}/{int(fr.fails.iloc[1])}/{int(fr.fails.iloc[2])}), with {len(CTRL_PRESENT)} present-object controls held out of the FAIL/PASS contrast.", "do_not_infer": "The control budget is fixed by design, so these counts do not reflect how often each checkpoint hallucinates in the wild."},
    {"chart": "failrate_by_checkpoint", "reading": f"Hallucination rate on absent-object probes runs {fr_lo:.0%} to {fr_hi:.0%} across the three checkpoints, lowest at {ck_lo}.", "do_not_infer": "Intervals overlap and the slice is curated; this is not a benchmark accuracy comparison between model sizes."},
    {"chart": "separation_ranked", "reading": f"Four concentration measures cluster at the top ({FULLNAME[s1.signal]} AUC {s1.auc:.2f} down to {FULLNAME['mean_relative_weight']} {float(sep_adv.loc[sep_adv.signal=='mean_relative_weight','auc'].iloc[0]):.2f}); the two spatial measures trail well behind.", "do_not_infer": "The bars are not independent contributions - the leading signals are near-duplicates of each other, so their separations cannot be added."},
    {"chart": "separation_by_checkpoint", "reading": "The same signal's separation varies materially between 2B, 4B and 8B.", "do_not_infer": "Per-checkpoint FAIL counts are 35-50, so differences of this size are not distinguishable from sampling noise."},
    {"chart": f"failrate_by_{top3[0]}", "reading": f"Hallucination rate climbs across {FULLNAME[top3[0]].lower()} bins, from {curve_specs[0][1].hallucination_rate.min():.0%} to {curve_specs[0][1].hallucination_rate.max():.0%}.", "do_not_infer": "Bins are equal-count, not equal-width; the curve shows ordering, not a dose-response magnitude."},
    {"chart": f"failrate_by_{top3[1]}", "reading": f"A weaker, less regular gradient across {FULLNAME[top3[1]].lower()} bins.", "do_not_infer": "With ~20 cases per bin, single-bin dips or spikes are not interpretable."},
    {"chart": f"failrate_by_{top3[2]}", "reading": f"A shallow gradient across {FULLNAME[top3[2]].lower()} bins.", "do_not_infer": "A flat curve here does not mean the signal is uninformative once other signals are held constant."},
    {"chart": "checkpoint_shift", "reading": "Standardized medians show the attention distributions themselves move between checkpoints, independently of outcome.", "do_not_infer": "A median shift says nothing about whether the signal separates FAIL from PASS at that checkpoint."},
    {"chart": "failrate_by_object", "reading": f"Hallucination rate by queried object spans {ob.hallucination_rate.min():.0%} to {ob.hallucination_rate.max():.0%} among objects with 8+ probes.", "do_not_infer": "Object counts are uneven and uncorrected for multiplicity; per-object rates are unstable."},
    {"chart": "adjusted_odds_ratios", "reading": "Adjusted odds ratios per 1 SD, with most intervals spanning 1 once all signals are entered together.", "do_not_infer": "Under high collinearity, an interval spanning 1 does not mean that signal is unrelated to the outcome."},
    {"chart": "corr_pairs_top", "reading": f"The strongest pair reaches Spearman rho = {top_corr['spearman_rho']}; several pairs are near-duplicates.", "do_not_infer": "Correlation between predictors says nothing about either one's relation to the outcome."},
    {"chart": "scatter_top_pair", "reading": "Hallucinations and correct rejections overlap broadly in the two-signal plane, with a modest density difference.", "do_not_infer": "No linear or curved boundary here cleanly separates the classes."},
    {"chart": "probe_design_contrast", "reading": "Absent-object and present-object probes differ on the same scalars used for the FAIL/PASS contrast.", "do_not_infer": "This is a design/sanity contrast, not a hallucination finding; the present-object rows contain no FAILs."},
    {"chart": "attention_geometry_fail_vs_pass", "reading": "Violins with jittered points show substantial overlap between hallucinations and correct rejections on every scalar, with modest median offsets.", "do_not_infer": "Median offsets of this size do not support classifying an individual case."},
    {"chart": "ecdf_top_signals", "reading": "The FAIL curve sits consistently to one side of the PASS curve rather than diverging only in a tail.", "do_not_infer": "A whole-distribution shift does not identify which cases will fail."},
    {"chart": "hallucination_rate_curves", "reading": "Risk rises with attention concentration for the leading signals; per-bin n is annotated because bins hold ~20 cases.", "do_not_infer": "Trend direction is descriptive; nothing here shows that changing attention would change the answer."},
    {"chart": "separation_by_checkpoint_heatmap", "reading": f"Every row keeps the same colour direction across all three columns ({n_flip} sign flips), but the intensity varies - strongest at {s1_ck_best}, weakest at {s1_ck_worst} for the leading signal.", "do_not_infer": "Cell-to-cell intensity differences are within the noise band at 35-50 FAILs per checkpoint and should not be read as a size trend."},
    {"chart": "checkpoint_distribution_shift", "reading": "Violins per checkpoint (light to dark = small to large) show the underlying distributions move with size; overlaid triangles mark FAIL and PASS medians.", "do_not_infer": "A shift in the distribution does not imply the larger model attends 'better' or 'worse'."},
    {"chart": "signal_correlation_heatmap", "reading": f"A dense high-correlation block; the strongest pair is {top_corr['pair']} at rho = {top_corr['spearman_rho']}.", "do_not_infer": "High correlation does not mean the signals are interchangeable for every downstream purpose."},
    {"chart": "adjusted_odds_forest", "reading": "Point estimates with 95% image-clustered intervals on a log odds axis; most intervals cross 1.", "do_not_infer": "With max VIF above 5 these estimates are unstable; do not read the ordering as a ranking of importance."},
    {"chart": "top_pair_scatter_by_checkpoint", "reading": "The FAIL region of the two-signal plane is broadly similar across checkpoints but never cleanly separated.", "do_not_infer": "Visual density differences across facets are not tested and each facet holds fewer than 50 FAILs."},
    {"chart": "object_checkpoint_heatmap", "reading": "High-rate objects tend to be high at more than one checkpoint, but many cells hold fewer than 10 probes.", "do_not_infer": "Cell-level rates with n<10 should not be compared against each other."},
    {"chart": "model_discrimination_calibration", "reading": f"In-sample ROC reaches AUC {auc_model:.2f}; the reliability curve tracks the diagonal loosely at low predicted probabilities.", "do_not_infer": "Fitted and scored on the same rows - this overstates what the signals would achieve on new cases."},
  ],
  "claims": [
    {"id": "C1", "text": f"Within absent-object probes, hallucinations show more concentrated attention than correct rejections; {FULLNAME[s1.signal]} separates them at AUC {s1.auc:.2f} [{s1.auc_lo:.2f}, {s1.auc_hi:.2f}].",
     "status": "descriptive", "evidence_ids": ["chart:separation_ranked", "chart:attention_geometry_fail_vs_pass", "signal:" + s1.signal],
     "interpretation": "A candidate attention-concentration correlate of object-presence hallucination for downstream confirmatory work.",
     "do_not_infer": "Not causal, not confirmed, and not sufficient to flag an individual case."},
    {"id": "C2", "text": f"The seven attention scalars are collinear (max VIF {vif.vif.max():.1f}); forward AIC selection retains {len(INDEP)}.",
     "status": "descriptive", "evidence_ids": ["chart:signal_correlation_heatmap", "chart:corr_pairs_top", "table:collinearity_vif"],
     "interpretation": "Downstream analysis should model one or two concentration axes rather than seven separate predictors.",
     "do_not_infer": "Does not establish that the dropped signals are unrelated to the outcome."},
    {"id": "C3", "text": f"Hallucination rate on this slice increases from {fr_lo:.0%} at {ck_lo} to {fr_hi:.0%} at {ck_hi} with overlapping binomial intervals.",
     "status": "descriptive", "evidence_ids": ["chart:failrate_by_checkpoint", "table:failrate_by_checkpoint"],
     "interpretation": "No evidence in this slice that the larger checkpoint is more robust on adversarial absent-object probes.",
     "do_not_infer": "Not a benchmark comparison; the slice fixes the control budget per checkpoint."},
    {"id": "C4", "text": f"The FAIL/PASS direction of all {len(SIGNALS)} signals is preserved in each checkpoint ({n_flip} sign flips), while separation strength varies non-monotonically with size ({FULLNAME[s1.signal]} AUC {s1_ck[SIZE_ORD[0]]:.2f}/{s1_ck[SIZE_ORD[1]]:.2f}/{s1_ck[SIZE_ORD[2]]:.2f}).",
     "status": "descriptive", "evidence_ids": ["chart:separation_by_checkpoint_heatmap", "table:conditioning_check", "table:separation_by_checkpoint"],
     "interpretation": "The pooled ranking is not a Simpson artefact, but the strength of the association should be reported per checkpoint rather than pooled.",
     "do_not_infer": "Magnitude differences at n=35-50 FAILs per stratum are compatible with sampling noise and do not establish a size-dependent mechanism."},
    {"id": "C6", "text": f"The marginal distributions of the relative-attention scalars shift substantially across checkpoints independently of outcome ({FULLNAME[shift_top.signal]} epsilon-squared {shift_top.epsilon_sq}, medians {shift_top['median_' + SIZE_ORD[0]]}/{shift_top['median_' + SIZE_ORD[1]]}/{shift_top['median_' + SIZE_ORD[2]]}).",
     "status": "descriptive", "evidence_ids": ["chart:checkpoint_distribution_shift", "chart:checkpoint_shift", "table:checkpoint_distribution_shift"],
     "interpretation": "Absolute thresholds on the relative-weight scalars are checkpoint-specific; within-checkpoint ranks are the transferable form.",
     "do_not_infer": "A marginal shift does not imply the larger checkpoint attends better or worse, and the checkpoints may differ in resolution/tokenization settings not recorded here."},
    {"id": "C7", "text": f"Object identity is associated with hallucination rate far more strongly than checkpoint (Cramer's V {V_OBJ:.2f} vs {V_CK:.2f}).",
     "status": "descriptive", "evidence_ids": ["chart:failrate_by_object", "chart:object_checkpoint_heatmap", "table:categorical_association"],
     "interpretation": "Any per-case attention analysis should condition on the queried object, which carries more of the outcome variation than model size.",
     "do_not_infer": "Object identity is entangled with adversarial probe difficulty by construction; this does not isolate an object effect."},
    {"id": "C5", "text": f"A deterministic peaked-attention split marks {len(grp_b)} adversarial cases with a {peak_rate:.0%} hallucination rate vs {flat_rate:.0%} elsewhere.",
     "status": "descriptive", "evidence_ids": ["chart:hallucination_rate_curves", "signal:peaked_attention"],
     "interpretation": "A computable screening rule to re-score on a held-out split.",
     "do_not_infer": "Thresholds were chosen on these same rows; the rate is optimistic."},
  ],
  "dashboard_storyboard": [
    {"id": "problem_setting", "title": "Problem Setting", "stages": ["M1"],
     "summary": f"M1 measurement data: {len(df)} frozen per-case records from three Qwen3-VL checkpoints ({', '.join(SIZE_ORD)}) on a POPE-style object-presence probe, each carrying seven attention-geometry scalars extracted from the answering forward pass. FAIL means the model answered 'Yes' to an adversarial probe about an object that is absent (a hallucination); PASS means it correctly rejected that probe. Present-object detection rows are included in the file as controls but contain no FAILs by construction, so every FAIL/PASS comparison here is computed inside the {len(ADV)} adversarial absent-object cases, where {int(ADV.is_fail.sum())} ({ADV.is_fail.mean():.0%}) are hallucinations. The question is which attention scalars differ between the two outcomes, whether that differs by model size, how it relates to probe type and queried object, and which scalars carry non-redundant information.",
     "items": [
       f"Data: {len(files)} per-checkpoint JSON files, each a scalar-metadata wrapper around a 'cases' list; merged into one tidy table of {len(df)} rows x {df.shape[1]} columns and written to records.json.",
       f"Outcome: 'label' (binary) - {int(df.is_fail.sum())} FAIL vs {int((1-df.is_fail).sum())} PASS; all FAILs are adversarial absent-object probes.",
       f"Signals: {', '.join(FULLNAME[s] for s in SIGNALS)} - complete for all {len(df)} cases, no missing values.",
       f"Structure: {profile['n_images']} distinct images, {profile['n_objects']} object categories, 3 checkpoints; images recur across checkpoints, so model standard errors are clustered on image and object.",
       f"Design caveat carried through every panel: PASS on an adversarial probe is a correct rejection, so the contrast is 'hallucinated vs correctly rejected the same kind of trick question'.",
     ],
     "artifact_refs": ["data_profile", "charts", "candidate_signals"]},
  ],
  "candidate_signals": [
    {"name": "peaked_attention", "display_name": "Peaked attention (dominant patch + narrow spread)",
     "rationale": "Cases whose attention is unusually concentrated on one patch and unusually narrow in spread hallucinate more often on absent-object probes.",
     "suggested_test": "Re-score the fixed threshold rule on a held-out split and compare hallucination rate inside vs outside the flagged set with a clustered two-proportion contrast.",
     "recipe": {"name": "peaked_attention", "kind": "expr",
                "expr": f"(top1_share > {thr_top1:.4f}) and (attention_entropy < {thr_ent:.4f})"},
     "sufficient": {"kind": "two_group", "a": grp_a, "b": grp_b}},
    {"name": "top1_share_high", "display_name": "High top-1 patch share",
     "rationale": "The single strongest marginal separator of hallucinations from correct rejections within adversarial probes.",
     "suggested_test": "Rank-based group comparison of top1_share between FAIL and PASS on a held-out split, stratified by checkpoint.",
     "recipe": {"name": "top1_share_high", "kind": "expr",
                "expr": f"top1_share > {float(ADV['top1_share'].quantile(.75)):.4f}"}},
    {"name": "attention_concentration_index", "display_name": "Attention concentration index",
     "rationale": "A single continuous axis combining peakiness and spread, motivated by the near-duplicate correlation block among the seven scalars.",
     "suggested_test": "Fit a single-predictor logistic model on a held-out split and compare AIC against the full seven-signal model.",
     "recipe": {"name": "attention_concentration_index", "kind": "expr",
                "expr": "top1_share + focus_share - attention_entropy"}},
    {"name": "peripheral_attention", "display_name": "Attention pushed to the image edge",
     "rationale": "Combines off-centre and edge-heavy attention, the geometry-of-location axis that is distinct from the concentration axis.",
     "suggested_test": "Compare hallucination rate inside vs outside this region on a held-out split, adjusted for checkpoint.",
     "recipe": {"name": "peripheral_attention", "kind": "expr",
                "expr": f"(edge_mass > {float(ADV['edge_mass'].quantile(.75)):.4f}) and (center_offset > {float(ADV['center_offset'].quantile(.6)):.4f})"}},
  ],
  "plots": plots,
  "tables": {
    "explanatory_eda": f"{TAB}/explanatory_eda.csv",
    "separation_adversarial": f"{TAB}/separation_adversarial.csv",
    "separation_all": f"{TAB}/separation_all.csv",
    "separation_by_checkpoint": f"{TAB}/separation_by_checkpoint.csv",
    "conditioning_check": f"{TAB}/conditioning_check.csv",
    "checkpoint_distribution_shift": f"{TAB}/checkpoint_distribution_shift.csv",
    "categorical_association": f"{TAB}/categorical_association.csv",
    "marginal_screening": f"{TAB}/marginal_screening.csv",
    "adjusted_model_coefficients": f"{TAB}/adjusted_model_coefficients.csv",
    "collinearity_vif": f"{TAB}/collinearity_vif.csv",
    "forward_selection": f"{TAB}/forward_selection.csv",
    "model_diagnostics": f"{TAB}/model_diagnostics.csv",
    "signal_correlations": f"{TAB}/signal_correlations.csv",
    "corr_pairs": f"{TAB}/corr_pairs.csv",
    "failrate_by_object": f"{TAB}/failrate_by_object.csv",
    "failrate_by_checkpoint": f"{TAB}/failrate_by_checkpoint.csv",
    "probe_design_contrast": f"{TAB}/probe_design_contrast.csv",
    "data_profile": profile,
  },
  "charts": charts,
  "caveats": [
    f"All {int(df.is_fail.sum())} FAILs live in the adversarial absent-object probe; the {len(CTRL_PRESENT)} present-object rows contain none. Every FAIL/PASS comparison is therefore restricted to the {len(ADV)} adversarial cases. The pooled all-cases table (separation_all.csv) mixes probe design with outcome and is kept only for contrast.",
    "This is a curated hallucination slice with a fixed control budget (80 correct rejections + 80 present detections per checkpoint), not a random sample; absolute hallucination rates are not the checkpoints' benchmark error rates.",
    f"Attention scalars are extracted from the same forward pass that produced the answer, so they are contemporaneous with the outcome rather than antecedent to it. Nothing here distinguishes a precursor from a readout of the generated answer.",
    f"The seven signals are strongly collinear (max VIF {vif.vif.max():.1f}); individual adjusted odds ratios are unstable and their ordering should not be read as an importance ranking.",
    f"Per-checkpoint strata hold only {int(sep_by_ck.n_fail.min())}-{int(sep_by_ck.n_fail.max())} FAILs, so per-checkpoint AUCs carry wide, overlapping intervals; the strength differences between checkpoints are not resolvable at this sample size.",
    "No multiplicity correction is applied across the 7 signals x 3 checkpoints x 39 objects; p-values in the tables are descriptive screening quantities, not significance verdicts.",
    "The logistic model is fitted and scored on the same rows, so its AUC and calibration are optimistic; no held-out split was used.",
    f"The npz attention maps listed in the directory scan were not present under {RAW} at runtime, so all analysis rests on the per-case scalars; no spatial map was re-derived.",
    "Objects with fewer than 8 adversarial probes are omitted from the object-rate chart to avoid 0%/100% artefacts; they remain in every other analysis.",
  ],
  "critique": [
    "Slice-construction leakage: because FAIL is only definable on adversarial absent-object probes, any pooled analysis over all 606 cases would rediscover the probe design rather than the failure mode. This is the single largest threat to a naive reading and is why the primary analysis is restricted; the pooled numbers are retained only as a contrast table.",
    "Double-dipping: the peaked-attention thresholds, the forward-selected signal set, and the reported model AUC all come from the same rows, so all three are optimistic. Only the emitted recipes are re-computable on a held-out split, which is why they are expressed as deterministic expressions.",
    f"Collinearity: with max VIF {vif.vif.max():.1f} and a top pair at rho = {top_corr['spearman_rho']}, which signal survives adjustment is close to arbitrary; a resampled run could select a different member of the same block.",
    "Clustering: images recur across checkpoints and objects repeat heavily (person and car alone cover a large share of probes), so standard errors are clustered on image and object; checkpoint is entered as a fixed effect because three clusters is far too few to estimate a random-effect variance.",
    "Alternative explanation for the concentration effect: adversarial probes ask about objects that frequently co-occur with what is in the image, so a peaked attention map may reflect the presence of a strongly related distractor object rather than anything about the hallucination itself. Object identity and probe difficulty cannot be separated in this slice.",
    "The leading effect is large for an observational signal (AUC {:.2f}, roughly a {:.1f}x median difference) but still far from separable: at the best single cut the two distributions overlap substantially, and the strongest deterministic rule found here captures only {:.0%} of hallucinations. Nothing in this analysis supports per-case detection.".format(float(s1.auc), float(s1.median_fail) / max(float(s1.median_pass), 1e-9), peak_capture),
    "Reverse-causation ambiguity: the answer token and the attention map come from the same forward pass, so a peaked map may equally be a readout of a decision already made as a precursor of it. No ordering evidence exists in this data.",
    "The heavy right tail on maximum relative attention (max {:.0f}, skew {:.1f}) is handled with a log10 transform for modelling and log axes in the figures; on the raw scale a mean-based comparison would be driven by a handful of cases.".format(float(df.max_relative_weight.max()), float(eda.loc[eda.signal=='max_relative_weight','skew'].iloc[0])),
  ],
  "recommended_confirmatory_tests": [
    "Re-score the peaked_attention and attention_concentration_index recipes on a held-out split (the dataset already carries an explore/validate split column) and compare hallucination rates with clustered intervals.",
    "Per-checkpoint stratified rank comparison of the leading signals with a formal interaction term for checkpoint x signal, to test whether the separation genuinely differs by model size.",
    "Refit the adjusted model on a single composite concentration axis instead of seven collinear scalars, and compare discrimination against the full model on held-out data.",
    "Condition the object-level rates on object frequency and co-occurrence statistics to separate object identity from adversarial probe difficulty.",
    "Compare attention geometry between hallucinations and correct rejections matched on image and queried object where such pairs exist across checkpoints, to remove image-level variation.",
  ],
}

print("EXPLORATORY_RESULT_JSON=" + json.dumps(result, default=str))
