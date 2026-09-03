<!--
Report skeleton for step 8 of the outcome-driver-analysis skill. Copy this into
projects/<analysis-name>/report.md and fill it in.

Figure/table formatting and prose style are covered by the clear-technical-writing
skill -- apply its conventions when writing the actual content, don't duplicate them
here.
-->

# [Analysis title]

## Research question

What decision or understanding is this analysis meant to support (from intake, step 1)?

## Data

Source, size, outcome definition, explanatory variables and their types, any known
data-quality caveats.

## Explanatory-variable summary

Distributions, outliers, missingness, notable correlations among explanatory variables
(step 2).

## Univariate findings

How each explanatory variable relates to the outcome on its own, and what changes when
conditioning on other variables (step 3). Marginal screening results (step 4).

## Model and justification

Which model was fit and why (outcome type, independence/clustering structure, step 5).
Per-variable significance after adjusting for other predictors.

## Diagnostics

Goodness-of-fit, residuals, multicollinearity, discrimination and calibration (step 6).

## Results

Coefficient/odds-ratio plot, predicted-probability curves, ROC and calibration plots
(step 7).

## Conclusions

Plain-language answer to the research question above: which variables matter, in what
direction, how strongly, and with what caveats. Written for a scientific-researcher
audience, not a statistics audience (step 8).

## Reproducibility

Environment/package versions and random seeds used (see `session_info.txt`).
