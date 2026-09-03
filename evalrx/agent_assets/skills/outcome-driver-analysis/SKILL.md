---
name: outcome-driver-analysis
description: >
  Run a full, disciplined statistical analysis to find what differentiates a binary
  outcome (success vs. fail, pass vs. fail, correct vs. incorrect) using a set of
  explanatory variables the user specifies. Covers exploring the explanatory variables
  themselves (distributions, outliers, missingness, correlations), exploring each
  variable against the outcome (contingency tables for categorical variables,
  distribution comparisons for continuous variables, conditioning on other variables),
  marginal screening, fitting a justified regression model (logistic vs. linear vs.
  mixed-effects, with explicit reasoning), goodness-of-fit diagnostics, result
  visualization, and a plain-language written report. Use this whenever the user has
  examples labeled by a binary outcome plus candidate explanatory variables and wants
  to know what drives the difference -- e.g. analyzing model error cases ("why do
  these examples fail"), pass/fail experiment results, or any dataset framed as
  success/failure with covariates -- even if they don't use the word "statistics."
  Do NOT use this for pure simulation studies with no real data, generating
  paper-ready LaTeX tables for an already-written paper, or benchmarking many methods
  against each other across datasets -- those are out of scope.
version: 0.2.0
license: MIT
compatibility: Claude Code project-scoped skill. Assumes R and/or Python available on PATH; pick whichever is appropriate per task rather than requiring both.
metadata:
  tags: statistics research-workflow eda logistic-regression mixed-effects reproducibility R Python
  agentskills_spec: "1.0"
---

# Outcome driver analysis

You are running a statistical analysis to explain a binary outcome (success/fail,
pass/fail, correct/incorrect) using explanatory variables the user provides. Work
through the pipeline below in order -- each step's findings feed the next one. Do not
skip straight to modeling: the exploratory steps surface issues (outliers, missing
data, collinearity, confounding) that change how the model should be built and read.

## Scope

This skill is for analyzing a real dataset with a binary outcome and candidate
explanatory variables -- most often model error analysis (why did these examples
fail?), but equally applicable to any pass/fail or success/failure outcome with
covariates. It is not for simulation studies with no real data, generating
publication-ready LaTeX tables for a paper that's already written, or benchmarking
many methods against each other across datasets.

## Chart types and palettes: defer to a chart-style skill when present

This skill decides WHAT to visualize at each step (the statistical intent);
it does not own HOW. When a dedicated chart-style skill (e.g.
`eval-chart-style`) is installed alongside this one, that skill's chart-type
policy and palette GOVERN every figure called for below — read the concrete
figure prescriptions in steps 2, 3 and 7 as the standalone fallback for when
no chart-style skill is present, never as an override of one.

## Choosing R or Python

Pick per task, in this order:
1. If the user is continuing an existing project, match whatever language that
   project's code already uses.
2. Otherwise, pick based on the modeling need identified in step 5: mixed-effects /
   hierarchical models are frequently more ergonomic in R (`lme4`, `glmmTMB`), while
   plain logistic regression, screening, and diagnostics are equally well supported in
   both R and Python (`statsmodels`, `scikit-learn`).
3. If genuinely ambiguous, ask the user.

Bundled helper scripts exist in both languages under `scripts/` -- use the one that
matches your choice; they are templates to adapt to the actual variable names and
data, not black boxes to run unmodified.

## 1. Intake -- clarify variables and structure

Before writing any code, confirm (from what the user already said, or by asking):
- Which column is the binary outcome, and which columns are the explanatory variables
  to investigate.
- The research question or problem behind the analysis -- what decision or
  understanding this is meant to support. Record it; the final report must be framed
  around it, not written as a generic statistical summary.
- Each explanatory variable's type: categorical, continuous, or count.
- Whether observations are clustered or repeated -- e.g. multiple examples from the
  same underlying model, prompt template, dataset, or subject. This is needed later to
  decide whether a mixed-effects model is warranted.

## 2. Explanatory-variable EDA

Before relating anything to the outcome, characterize the explanatory variables on
their own terms:
- **Distribution**: histogram (continuous) or bar chart (categorical) per variable.
- **Outliers**: flag with an IQR or z-score rule for continuous variables; flag rare
  categories for categorical variables.
- **Missingness**: count and pattern per variable. Spot-check whether missingness
  itself is associated with the outcome (missing values are often not random).
- **Inter-variable structure**: correlation matrix for continuous-continuous pairs,
  Cramer's V for categorical-categorical pairs, correlation ratio (eta-squared) or
  ANOVA for mixed pairs. This surfaces redundant or collinear explanatory variables
  early, before they cause multicollinearity problems at the modeling stage.

Use `scripts/explanatory_var_eda.R` or `.py` as a starting template.

## 3. Per-variable exploration vs. outcome

For each explanatory variable, relate it to the outcome:
- **Categorical variable** -> contingency table (variable x outcome) with row/column
  proportions; chi-square test of independence, or Fisher's exact test when any
  expected cell count is small (below ~5).
- **Continuous variable** -> a per-group distribution-comparison figure
  (standalone fallback: side-by-side boxplot; an installed chart-style skill's
  distribution-first policy — e.g. violin + jittered points — takes
  precedence), plus a group-comparison test: Welch's t-test if roughly normal,
  Mann-Whitney U as the robust default when normality is doubtful.
- Always report an effect size next to the test, not just a p-value: Cramer's V or
  odds ratio for categorical variables, Cohen's d or rank-biserial correlation for
  continuous variables.
- **Conditioning**: repeat the above stratified by, or faceted on, a plausible
  confounding or interacting variable. Flag relationships that appear, vanish, or
  reverse within strata (a Simpson's-paradox pattern) -- these are candidates for an
  interaction term or control variable in the model stage, not things to quietly drop.

Use `scripts/univariate_eda.R` or `.py` as a starting template.

## 4. Marginal variable screening

Before committing to a multivariable model, screen each explanatory variable for a
marginal (unadjusted) signal against the outcome: fit a univariate logistic regression
per variable (or reuse the chi-square/t-test/Mann-Whitney results from step 3), and
rank by p-value or by AIC improvement over the null model.

This matters most when there are many candidate variables -- it narrows the field to a
manageable candidate set for the full model. Do not silently drop a variable just
because its bivariate test wasn't significant; a real effect can be masked by a
confounder and only emerge once other variables are adjusted for in step 5. Screening
informs the candidate list, it does not replace the full model.

Screening metrics and thresholds are summarized in `references/model_selection.md`.

## 5. Model selection and fitting, with explicit justification

Consult `references/model_selection.md` for the full decision table. The core logic:

- **Binary outcome -> logistic regression, not linear regression.** State explicitly
  why: a linear model can predict outside [0, 1], and its error/variance structure
  doesn't match a 0/1 target (violates linearity and homoscedasticity assumptions that
  linear regression relies on). Continuous outcomes would call for linear regression
  instead, and count outcomes for Poisson or negative binomial -- this decision logic
  generalizes even though the immediate case here is binary.
- **Plain GLM vs. mixed-effects (GLMM)**: use a random effect for the clustering
  variable identified in step 1 when observations are not independent (repeated
  examples from the same model, prompt, or dataset). Use a plain GLM when observations
  are reasonably independent. State this decision and the reasoning tied to the actual
  data structure -- don't pick silently.
- **Candidate predictors**: start from the variables that passed step 4 screening, plus
  any interaction flagged by the step 3 conditioning check.
- **Per-variable significance in the fitted model**: after fitting, test each
  variable's significance with a Wald test or a likelihood-ratio test (comparing the
  model with and without that term). Report this alongside the bivariate/screening
  results from steps 3-4, so the reader can see whether a variable's marginal signal
  holds up after adjusting for the others, or was actually explained by a confounder.

Use `scripts/fit_outcome_model.R` or `.py` as a starting template.

## 6. Goodness-of-fit and diagnostics

- Hosmer-Lemeshow test (or a suitable alternative when there are many continuous
  predictors, since Hosmer-Lemeshow can be unreliable there).
- Deviance or Pearson residuals, binned residual plots.
- VIF for multicollinearity among the final model's predictors.
- Influence diagnostics (e.g. Cook's-distance analogs for GLMs).
- ROC curve and AUC for discrimination; a calibration plot for calibration.
- For GLMMs specifically: also check random-effect variance estimates, intraclass
  correlation (ICC), and convergence warnings.

## 7. Visualize results

- Coefficient / odds-ratio forest plot with confidence intervals.
- Predicted-probability curves for the key continuous predictors (holding other
  variables at a reference value or mean).
- ROC curve and calibration plot (carried over from step 6, presented as final results
  rather than diagnostics here).

## 8. Conclusion and report

State which variables matter, in which direction, with what size and uncertainty, and
tie the finding back to the steps 2-5 evidence (distribution/outlier caveats, bivariate
and screening signal, adjusted-model significance) as corroboration or explanation.

Frame the conclusion around the specific research question recorded in step 1 -- not
as a generic statistical summary. Write it in plain language: the audience is
scientific researchers who understand research methodology but are not necessarily
statisticians, so translate statistical results into substantive meaning (for example,
"cases where X exceeded N were far more likely to fail" rather than reporting only an
odds ratio and a p-value), while still surfacing the effect size, uncertainty, and any
caveats a careful reader would need (small samples, assumption violations, correlated
predictors).

Figure styling conventions come from the installed chart-style/polish skills
(e.g. `eval-chart-style` for chart types and palette, `nature-figure` for
publication polish) -- do not duplicate those conventions here.

Save reproducibility information -- environment/package versions and any random seeds
used -- to `session_info.txt` alongside the analysis.

Start each new analysis under `projects/<analysis-name>/`, with the raw dataset copied
(read-only) into `data/`, outputs written to `output/figures/` and `output/tables/`,
and the writeup as `report.md` following `assets/analysis_report_template.md`.

## Reference files

- `references/model_selection.md` -- decision tables for outcome type x clustering x
  assumption status -> model family, the univariate-test decision table from step 3,
  and the marginal-screening metrics from step 4. Read this when deciding on a test or
  model.
- `scripts/explanatory_var_eda.R` / `.py` -- step 2 starting template.
- `scripts/univariate_eda.R` / `.py` -- steps 3-4 starting template.
- `scripts/fit_outcome_model.R` / `.py` -- steps 5-7 starting template.
- `assets/analysis_report_template.md` -- report skeleton for step 8.
