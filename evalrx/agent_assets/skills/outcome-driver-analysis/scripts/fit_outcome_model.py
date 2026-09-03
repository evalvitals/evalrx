"""
Steps 5-7 starting template: fit a justified model for a binary outcome, test
per-variable significance, run goodness-of-fit diagnostics, and visualize results.

Given a dataframe, a binary outcome column, a list of predictor columns, and an
optional cluster column (for repeated/clustered observations):
- fits logistic regression (statsmodels Logit) when there's no clustering
- for clustered data, python's mixed-effects binary GLM support is thin -- this script
  fits a cluster-robust-SE logistic regression as an approximation and prints a note
  recommending scripts/fit_outcome_model.R (lme4::glmer / glmmTMB) for a true GLMM
- reports per-variable Wald and likelihood-ratio significance with odds ratios
- runs Hosmer-Lemeshow, VIF, ROC/AUC, and a calibration check
- produces a coefficient/odds-ratio plot, a predicted-probability curve, an ROC curve,
  and a calibration plot

This is a template to adapt to the real column names and data -- not a black box.

Usage:
    python fit_outcome_model.py data.csv --outcome success --vars age,region,score \
        [--cluster cohort] --out output/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fit_model(df: pd.DataFrame, outcome: str, predictors: list[str], cluster: str | None):
    formula = f"{outcome} ~ " + " + ".join(predictors)
    if cluster is None:
        model = smf.logit(formula, data=df).fit(disp=0)
        print("Fit plain logistic regression (observations treated as independent).")
    else:
        model = smf.logit(formula, data=df).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": df[cluster]}
        )
        print(f"Observations are clustered by '{cluster}': fit logistic regression with "
              f"cluster-robust standard errors as an approximation.")
        print("For a full mixed-effects logistic regression (random intercept/slope for "
              f"'{cluster}'), use fit_outcome_model.R instead (lme4::glmer / glmmTMB) -- "
              "python's support for mixed binary GLMs is limited.")
    return model


def per_variable_significance(df: pd.DataFrame, outcome: str, predictors: list[str],
                                full_model) -> pd.DataFrame:
    """Wald p-value/odds ratio per model coefficient (one row per dummy level for a
    categorical predictor), plus a likelihood-ratio test dropping each *variable* (all
    of its levels at once) so marginal-screening signal (steps 3-4) can be compared
    against adjusted significance."""
    conf_int = full_model.conf_int()
    rows = []
    for v in predictors:
        # a categorical predictor expands to one coefficient per non-reference level,
        # named like "region[T.south]" by patsy -- match all of them, not just "v" itself
        coef_names = [c for c in full_model.params.index if c == v or c.startswith(f"{v}[")]

        reduced_predictors = [p for p in predictors if p != v]
        reduced_formula = f"{outcome} ~ " + (" + ".join(reduced_predictors) if reduced_predictors else "1")
        reduced_model = smf.logit(reduced_formula, data=df).fit(disp=0)
        lrt_stat = 2 * (full_model.llf - reduced_model.llf)
        lrt_p = chi2.sf(lrt_stat, df=len(coef_names))

        for coef_name in coef_names:
            rows.append({
                "variable": coef_name,
                "odds_ratio": np.exp(full_model.params[coef_name]),
                "ci_low": np.exp(conf_int.loc[coef_name, 0]),
                "ci_high": np.exp(conf_int.loc[coef_name, 1]),
                "wald_p": full_model.pvalues[coef_name],
                "lrt_p": lrt_p,  # shared across all levels of this variable
            })
    return pd.DataFrame(rows)


def hosmer_lemeshow(y_true, y_pred, n_bins: int = 10) -> tuple[float, float]:
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    grouped = df.groupby("bin")
    obs = grouped["y"].sum()
    exp = grouped["p"].sum()
    n = grouped["y"].count()
    stat = (((obs - exp) ** 2) / (exp * (1 - exp / n))).sum()
    p_value = chi2.sf(stat, df=n_bins - 2)
    return float(stat), float(p_value)


def vif_table(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    numeric_predictors = [p for p in predictors if pd.api.types.is_numeric_dtype(df[p])]
    X = sm.add_constant(df[numeric_predictors].dropna())
    return pd.DataFrame({
        "variable": X.columns,
        "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).query("variable != 'const'")


def plot_coefficients(sig_table: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 0.5 * len(sig_table) + 1))
    y = np.arange(len(sig_table))
    ax.errorbar(sig_table["odds_ratio"], y,
                xerr=[sig_table["odds_ratio"] - sig_table["ci_low"], sig_table["ci_high"] - sig_table["odds_ratio"]],
                fmt="o", color="steelblue")
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sig_table["variable"])
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coefficient_plot.pdf"))
    plt.close(fig)


def plot_roc(y_true, y_pred, out_dir: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, color="steelblue", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.pdf"))
    plt.close(fig)
    return float(auc)


def plot_calibration(y_true, y_pred, out_dir: str, n_bins: int = 10) -> None:
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    grouped = df.groupby("bin").agg(observed=("y", "mean"), predicted=("p", "mean"))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(grouped["predicted"], grouped["observed"], "o-", color="steelblue")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_plot.pdf"))
    plt.close(fig)


def run(csv_path: str, outcome: str, predictors: list[str], out_dir: str, cluster: str | None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=[outcome] + predictors)

    model = fit_model(df, outcome, predictors, cluster)
    print(model.summary())

    sig_table = per_variable_significance(df, outcome, predictors, model)
    sig_table.to_csv(os.path.join(out_dir, "per_variable_significance.csv"), index=False)
    print("\nPer-variable significance (adjusted for other predictors):\n", sig_table)

    y_pred = model.predict(df)
    hl_stat, hl_p = hosmer_lemeshow(df[outcome], y_pred)
    print(f"\nHosmer-Lemeshow: chi2={hl_stat:.3f}, p={hl_p:.4g}")

    vif = vif_table(df, predictors)
    vif.to_csv(os.path.join(out_dir, "vif.csv"), index=False)
    print("\nVIF:\n", vif)

    auc = plot_roc(df[outcome], y_pred, out_dir)
    plot_calibration(df[outcome], y_pred, out_dir)
    plot_coefficients(sig_table, out_dir)
    print(f"\nAUC: {auc:.3f}")
    print(f"Plots written to {out_dir}: coefficient_plot.pdf, roc_curve.pdf, calibration_plot.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--outcome", required=True, help="binary outcome column name")
    parser.add_argument("--vars", required=True, help="comma-separated predictor column names")
    parser.add_argument("--cluster", default=None, help="optional clustering/grouping column")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    run(args.csv_path, args.outcome, args.vars.split(","), args.out, args.cluster)
