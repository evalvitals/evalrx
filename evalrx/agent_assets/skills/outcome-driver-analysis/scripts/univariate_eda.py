"""
Steps 3-4 starting template: per-variable exploration vs. outcome, plus marginal
variable screening.

Given a dataframe, a binary outcome column, and a list of explanatory variables:
- categorical variable -> contingency table vs. outcome + chi-square/Fisher's exact
- continuous variable -> side-by-side boxplot vs. outcome + Welch's t-test / Mann-Whitney
- optional conditioning variable -> repeat the above within each stratum
- marginal screening -> univariate logistic regression per variable, ranked by p-value

This is a template to adapt to the real column names and data -- not a black box.

Usage:
    python univariate_eda.py data.csv --outcome success --vars age,region,score \
        --condition_on cohort --out output/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def infer_variable_types(df: pd.DataFrame, variables: list[str]) -> dict[str, str]:
    types = {}
    for v in variables:
        if pd.api.types.is_numeric_dtype(df[v]) and df[v].nunique() > 10:
            types[v] = "continuous"
        else:
            types[v] = "categorical"
    return types


def categorical_vs_outcome(df: pd.DataFrame, var: str, outcome: str) -> dict:
    ct = pd.crosstab(df[var], df[outcome])
    expected = stats.contingency.expected_freq(ct)
    if (expected < 5).any():
        _, p, _ = stats.fisher_exact(ct) if ct.shape == (2, 2) else (None, np.nan, None)
        test_used = "fisher_exact"
    else:
        chi2, p, _, _ = stats.chi2_contingency(ct)
        test_used = "chi_square"
    n = ct.sum().sum()
    phi2 = stats.chi2_contingency(ct)[0] / n
    r, k = ct.shape
    cramers_v = float(np.sqrt(phi2 / min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else np.nan
    return {"variable": var, "test": test_used, "p_value": p, "cramers_v": cramers_v,
            "contingency_table": ct}


def continuous_vs_outcome(df: pd.DataFrame, var: str, outcome: str) -> dict:
    groups = df.groupby(outcome)[var]
    labels = list(groups.groups.keys())
    a, b = groups.get_group(labels[0]).dropna(), groups.get_group(labels[1]).dropna()
    normal_a = stats.shapiro(a.sample(min(len(a), 5000)))[1] > 0.05
    normal_b = stats.shapiro(b.sample(min(len(b), 5000)))[1] > 0.05
    if normal_a and normal_b:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        test_used = "welch_t"
    else:
        stat, p = stats.mannwhitneyu(a, b)
        test_used = "mann_whitney_u"
    pooled_sd = np.sqrt(((a.std() ** 2) + (b.std() ** 2)) / 2)
    cohens_d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    return {"variable": var, "test": test_used, "statistic": stat, "p_value": p, "cohens_d": cohens_d}


def boxplot_vs_outcome(df: pd.DataFrame, var: str, outcome: str, out_dir: str, facet_on: str | None = None) -> None:
    if facet_on is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        df.boxplot(column=var, by=outcome, ax=ax)
        ax.set_title(var)
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"box_{var}_vs_{outcome}.pdf"))
        plt.close(fig)
    else:
        strata = df[facet_on].dropna().unique()
        fig, axes = plt.subplots(1, len(strata), figsize=(4 * len(strata), 3.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, strata):
            sub = df[df[facet_on] == s]
            sub.boxplot(column=var, by=outcome, ax=ax)
            ax.set_title(f"{facet_on} = {s}")
        plt.suptitle(var)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"box_{var}_vs_{outcome}_by_{facet_on}.pdf"))
        plt.close(fig)


def conditioned_analysis(df: pd.DataFrame, var: str, outcome: str, var_type: str,
                          condition_on: str, out_dir: str) -> pd.DataFrame:
    """Repeat the marginal test within each stratum of `condition_on` to check for
    confounding/interaction (does the relationship appear, vanish, or reverse?)."""
    rows = []
    for stratum, sub in df.groupby(condition_on):
        if var_type == "categorical":
            result = categorical_vs_outcome(sub, var, outcome)
            rows.append({"stratum": stratum, "test": result["test"],
                         "p_value": result["p_value"], "effect_size": result["cramers_v"]})
        else:
            result = continuous_vs_outcome(sub, var, outcome)
            rows.append({"stratum": stratum, "test": result["test"],
                         "p_value": result["p_value"], "effect_size": result["cohens_d"]})
    if var_type == "continuous":
        boxplot_vs_outcome(df, var, outcome, out_dir, facet_on=condition_on)
    return pd.DataFrame(rows)


def marginal_screen(df: pd.DataFrame, outcome: str, variables: list[str]) -> pd.DataFrame:
    """Univariate logistic regression per variable, ranked by likelihood-ratio p-value
    (a joint test across all dummy levels for a categorical variable, not just one
    coefficient) and AIC improvement over the null model.

    The null model is refit on each variable's own non-missing subset, since comparing
    a null model fit on all rows against a univariate model that silently drops rows
    with a missing predictor (mismatched sample sizes) invalidates the LRT."""
    rows = []
    for v in variables:
        try:
            sub = df[[outcome, v]].dropna()
            null_model = smf.logit(f"{outcome} ~ 1", data=sub).fit(disp=0)
            model = smf.logit(f"{outcome} ~ Q('{v}')", data=sub).fit(disp=0)
            df_diff = model.df_model - null_model.df_model
            lrt_stat = 2 * (model.llf - null_model.llf)
            p_value = stats.chi2.sf(lrt_stat, df=df_diff) if df_diff > 0 else np.nan
            aic_improvement = null_model.aic - model.aic
        except Exception:
            p_value, aic_improvement = np.nan, np.nan
        rows.append({"variable": v, "p_value": p_value, "aic_improvement": aic_improvement})
    return pd.DataFrame(rows).sort_values("p_value")


def run(csv_path: str, outcome: str, variables: list[str], out_dir: str, condition_on: str | None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    var_types = infer_variable_types(df, variables)

    univariate_results = []
    for v in variables:
        if var_types[v] == "categorical":
            result = categorical_vs_outcome(df, v, outcome)
            print(f"\n{v} (categorical) vs {outcome}:\n{result['contingency_table']}")
            print(f"  {result['test']}: p={result['p_value']:.4g}, Cramer's V={result['cramers_v']:.3f}")
            univariate_results.append({k: v2 for k, v2 in result.items() if k != "contingency_table"})
        else:
            boxplot_vs_outcome(df, v, outcome, out_dir)
            result = continuous_vs_outcome(df, v, outcome)
            print(f"\n{v} (continuous) vs {outcome}:")
            print(f"  {result['test']}: p={result['p_value']:.4g}, Cohen's d={result['cohens_d']:.3f}")
            univariate_results.append(result)

        if condition_on:
            cond = conditioned_analysis(df, v, outcome, var_types[v], condition_on, out_dir)
            print(f"  conditioned on {condition_on}:\n{cond}")
            cond.to_csv(os.path.join(out_dir, f"conditioned_{v}_on_{condition_on}.csv"), index=False)

    pd.DataFrame(univariate_results).to_csv(os.path.join(out_dir, "univariate_vs_outcome.csv"), index=False)

    screen = marginal_screen(df, outcome, variables)
    screen.to_csv(os.path.join(out_dir, "marginal_screening.csv"), index=False)
    print("\nMarginal screening (ranked by p-value):\n", screen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--outcome", required=True, help="binary outcome column name")
    parser.add_argument("--vars", required=True, help="comma-separated explanatory variable column names")
    parser.add_argument("--condition_on", default=None, help="optional column to stratify/facet on")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    run(args.csv_path, args.outcome, args.vars.split(","), args.out, args.condition_on)
