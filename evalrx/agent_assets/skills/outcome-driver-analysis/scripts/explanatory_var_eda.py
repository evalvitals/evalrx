"""
Step 2 starting template: explanatory-variable EDA.

Given a dataframe and a list of explanatory variable columns, produce:
- a per-variable distribution plot (histogram for continuous, bar chart for categorical)
- an outlier / missingness summary table
- a mixed-type correlation / association matrix among the explanatory variables

This is a template to adapt to the real column names and data -- not a black box.
Edit CONTINUOUS_VARS / CATEGORICAL_VARS below, or pass them in when calling the
functions from another script / notebook.

Usage:
    python explanatory_var_eda.py data.csv --vars age,region,score --out output/
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_variable_types(df: pd.DataFrame, variables: list[str]) -> dict[str, str]:
    """Heuristic: numeric dtype with many unique values -> continuous, else categorical."""
    types = {}
    for v in variables:
        if pd.api.types.is_numeric_dtype(df[v]) and df[v].nunique() > 10:
            types[v] = "continuous"
        else:
            types[v] = "categorical"
    return types


def outlier_summary(df: pd.DataFrame, var: str) -> dict:
    """IQR rule for continuous variables."""
    q1, q3 = df[var].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[var] < lower) | (df[var] > upper)).sum()
    return {"variable": var, "n_outliers": int(n_outliers), "lower_bound": lower, "upper_bound": upper}


def rare_category_summary(df: pd.DataFrame, var: str, threshold: float = 0.01) -> dict:
    """Flag categories making up less than `threshold` fraction of non-missing rows."""
    counts = df[var].value_counts(normalize=True)
    rare = counts[counts < threshold]
    return {"variable": var, "n_categories": len(counts), "rare_categories": list(rare.index)}


def missingness_summary(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        n_missing = df[v].isna().sum()
        rows.append({"variable": v, "n_missing": int(n_missing), "pct_missing": round(100 * n_missing / len(df), 1)})
    return pd.DataFrame(rows)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Association between two categorical variables, bias-corrected."""
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))


def correlation_ratio(categorical: pd.Series, continuous: pd.Series) -> float:
    """Eta: association between a categorical and a continuous variable."""
    df = pd.DataFrame({"cat": categorical, "cont": continuous}).dropna()
    overall_mean = df["cont"].mean()
    groups = df.groupby("cat")["cont"]
    ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for _, g in groups)
    ss_total = ((df["cont"] - overall_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def association_matrix(df: pd.DataFrame, var_types: dict[str, str]) -> pd.DataFrame:
    """Mixed-type association matrix: Pearson r (cont-cont), Cramer's V (cat-cat), eta (mixed)."""
    variables = list(var_types.keys())
    mat = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for a, b in itertools.combinations_with_replacement(variables, 2):
        if a == b:
            mat.loc[a, b] = 1.0
            continue
        ta, tb = var_types[a], var_types[b]
        if ta == "continuous" and tb == "continuous":
            val = df[[a, b]].corr().iloc[0, 1]
        elif ta == "categorical" and tb == "categorical":
            val = cramers_v(df[a], df[b])
        else:
            cat_var, cont_var = (a, b) if ta == "categorical" else (b, a)
            val = correlation_ratio(df[cat_var], df[cont_var])
        mat.loc[a, b] = mat.loc[b, a] = val
    return mat


def plot_distribution(df: pd.DataFrame, var: str, var_type: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if var_type == "continuous":
        ax.hist(df[var].dropna(), bins=30, color="steelblue", edgecolor="white")
    else:
        df[var].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(var)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dist_{var}.pdf"))
    plt.close(fig)


def run(csv_path: str, variables: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    var_types = infer_variable_types(df, variables)

    for v in variables:
        plot_distribution(df, v, var_types[v], out_dir)

    outliers = [outlier_summary(df, v) for v, t in var_types.items() if t == "continuous"]
    rares = [rare_category_summary(df, v) for v, t in var_types.items() if t == "categorical"]
    missing = missingness_summary(df, variables)
    assoc = association_matrix(df, var_types)

    pd.DataFrame(outliers).to_csv(os.path.join(out_dir, "outlier_summary.csv"), index=False)
    pd.DataFrame(rares).to_csv(os.path.join(out_dir, "rare_category_summary.csv"), index=False)
    missing.to_csv(os.path.join(out_dir, "missingness_summary.csv"), index=False)
    assoc.to_csv(os.path.join(out_dir, "association_matrix.csv"))

    print("Variable types:", var_types)
    print("\nMissingness:\n", missing)
    print("\nOutliers (continuous):\n", pd.DataFrame(outliers))
    print("\nRare categories (categorical):\n", pd.DataFrame(rares))
    print("\nAssociation matrix:\n", assoc.round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--vars", required=True, help="comma-separated explanatory variable column names")
    parser.add_argument("--out", default="output/figures")
    args = parser.parse_args()
    run(args.csv_path, args.vars.split(","), args.out)
