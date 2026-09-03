# Steps 5-7 starting template: fit a justified model for a binary outcome, test
# per-variable significance, run goodness-of-fit diagnostics, and visualize results.
#
# Given a dataframe, a binary outcome column, a list of predictor columns, and an
# optional cluster column (for repeated/clustered observations):
# - fits a plain logistic regression (glm, family = binomial) when there's no clustering
# - fits a mixed-effects logistic regression (lme4::glmer, random intercept for the
#   cluster) when clustering is present -- this is the language where GLMMs are most
#   mature, per references/model_selection.md
# - reports per-variable Wald and likelihood-ratio significance with odds ratios
# - runs Hosmer-Lemeshow (fixed-effects case), VIF, ROC/AUC, and a calibration check,
#   plus ICC/convergence checks for the mixed-effects case
# - produces a coefficient/odds-ratio plot, an ROC curve, and a calibration plot
#
# This is a template to adapt to the real column names and data -- not a black box.
#
# Usage:
#   Rscript fit_outcome_model.R data.csv outcome "age,region,score" output/ [cluster]

suppressMessages({
  library(dplyr)
  library(ggplot2)
  library(car)      # vif()
  library(pROC)     # roc()
})

fit_model <- function(df, outcome, predictors, cluster = NULL) {
  formula_str <- paste(outcome, "~", paste(predictors, collapse = " + "))
  if (is.null(cluster)) {
    message("Fit plain logistic regression (observations treated as independent).")
    glm(as.formula(formula_str), data = df, family = binomial)
  } else {
    suppressMessages(library(lme4))
    message(sprintf("Observations are clustered by '%s': fit mixed-effects logistic regression",
                     cluster))
    mixed_formula <- paste(formula_str, "+ (1 |", cluster, ")")
    lme4::glmer(as.formula(mixed_formula), data = df, family = binomial,
                control = lme4::glmerControl(optimizer = "bobyqa"))
  }
}

is_mixed <- function(model) inherits(model, "merMod")

per_variable_significance <- function(df, outcome, predictors, full_model, cluster = NULL) {
  # a categorical predictor expands to one coefficient per non-reference level (e.g.
  # "regionnorth", "regionsouth") -- report one row per level, but share a single
  # likelihood-ratio p-value (a joint test dropping the whole variable) across them
  formula_base <- if (is_mixed(full_model)) paste("+ (1 |", cluster, ")") else ""
  coef_table <- summary(full_model)$coefficients
  pr_col <- grep("Pr", colnames(coef_table), value = TRUE)[1]
  rows <- lapply(predictors, function(v) {
    reduced_predictors <- setdiff(predictors, v)
    reduced_rhs <- if (length(reduced_predictors) > 0) paste(reduced_predictors, collapse = " + ") else "1"
    reduced_formula <- paste(outcome, "~", reduced_rhs, formula_base)
    reduced_model <- if (is_mixed(full_model)) {
      lme4::glmer(as.formula(reduced_formula), data = df, family = binomial)
    } else {
      glm(as.formula(reduced_formula), data = df, family = binomial)
    }
    lrt <- anova(reduced_model, full_model, test = "Chisq")
    # stats::anova.glm names this column "Pr(>Chi)"; lme4:::anova.merMod names it
    # "Pr(>Chisq)" -- look it up rather than hardcoding either spelling
    lrt_p <- lrt[[grep("^Pr", colnames(lrt), value = TRUE)]][2]

    coef_names <- grep(paste0("^", v), names(fixef_or_coef(full_model)), value = TRUE)
    do.call(rbind, lapply(coef_names, function(coef_name) {
      est <- fixef_or_coef(full_model)[[coef_name]]
      se <- coef_table[coef_name, "Std. Error"]
      wald_p <- coef_table[coef_name, pr_col]
      data.frame(
        variable = coef_name,
        odds_ratio = exp(est),
        ci_low = exp(est - 1.96 * se),
        ci_high = exp(est + 1.96 * se),
        wald_p = wald_p,
        lrt_p = lrt_p
      )
    }))
  })
  do.call(rbind, rows)
}

fixef_or_coef <- function(model) if (is_mixed(model)) lme4::fixef(model) else coef(model)

hosmer_lemeshow <- function(y_true, y_pred, n_bins = 10) {
  bins <- cut(y_pred, breaks = quantile(y_pred, probs = seq(0, 1, length.out = n_bins + 1)),
              include.lowest = TRUE)
  obs <- tapply(y_true, bins, sum)
  exp <- tapply(y_pred, bins, sum)
  n <- tapply(y_true, bins, length)
  stat <- sum((obs - exp)^2 / (exp * (1 - exp / n)), na.rm = TRUE)
  p_value <- pchisq(stat, df = n_bins - 2, lower.tail = FALSE)
  list(statistic = stat, p_value = p_value)
}

plot_coefficients <- function(sig_table, out_dir) {
  p <- ggplot(sig_table, aes(x = odds_ratio, y = reorder(variable, odds_ratio))) +
    geom_point(color = "steelblue") +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.2, color = "steelblue") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    labs(x = "Odds ratio (95% CI)", y = NULL) +
    theme_minimal()
  ggsave(file.path(out_dir, "coefficient_plot.pdf"), p, width = 5, height = 0.5 * nrow(sig_table) + 1)
}

plot_roc <- function(y_true, y_pred, out_dir) {
  roc_obj <- pROC::roc(y_true, y_pred, quiet = TRUE)
  pdf(file.path(out_dir, "roc_curve.pdf"), width = 4, height = 4)
  plot(roc_obj, col = "steelblue", main = sprintf("AUC = %.3f", pROC::auc(roc_obj)))
  dev.off()
  as.numeric(pROC::auc(roc_obj))
}

plot_calibration <- function(y_true, y_pred, out_dir, n_bins = 10) {
  bins <- cut(y_pred, breaks = quantile(y_pred, probs = seq(0, 1, length.out = n_bins + 1)),
              include.lowest = TRUE)
  observed <- tapply(y_true, bins, mean)
  predicted <- tapply(y_pred, bins, mean)
  df <- data.frame(observed = observed, predicted = predicted)
  p <- ggplot(df, aes(predicted, observed)) +
    geom_point(color = "steelblue") + geom_line(color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    labs(x = "Mean predicted probability", y = "Observed proportion") +
    theme_minimal()
  ggsave(file.path(out_dir, "calibration_plot.pdf"), p, width = 4, height = 4)
}

run <- function(csv_path, outcome, predictors, out_dir, cluster = NULL) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  df <- read.csv(csv_path)
  df <- df[complete.cases(df[, c(outcome, predictors)]), ]

  model <- fit_model(df, outcome, predictors, cluster)
  print(summary(model))

  sig_table <- per_variable_significance(df, outcome, predictors, model, cluster)
  write.csv(sig_table, file.path(out_dir, "per_variable_significance.csv"), row.names = FALSE)
  cat("\nPer-variable significance (adjusted for other predictors):\n"); print(sig_table)

  y_pred <- predict(model, type = "response")

  if (is_mixed(model)) {
    cat("\nMixed-effects model: checking random-effect variance, ICC, and convergence.\n")
    print(lme4::VarCorr(model))
    if (!is.null(model@optinfo$conv$lme4$messages)) {
      cat("Convergence warnings:\n"); print(model@optinfo$conv$lme4$messages)
    } else {
      cat("No convergence warnings.\n")
    }
  } else {
    hl <- hosmer_lemeshow(df[[outcome]], y_pred)
    cat(sprintf("\nHosmer-Lemeshow: chi2=%.3f, p=%.4g\n", hl$statistic, hl$p_value))
    vif_vals <- car::vif(model)
    # car::vif() returns a plain named vector when every predictor has 1 df, but a
    # GVIF/Df/GVIF^(1/(2*Df)) matrix as soon as any factor predictor has >2 levels --
    # use the adjusted GVIF^(1/(2*Df))^2 (comparable to a plain VIF) in that case
    vif_df <- if (is.matrix(vif_vals)) {
      data.frame(variable = rownames(vif_vals), vif = vif_vals[, "GVIF^(1/(2*Df))"]^2)
    } else {
      data.frame(variable = names(vif_vals), vif = vif_vals)
    }
    write.csv(vif_df, file.path(out_dir, "vif.csv"), row.names = FALSE)
    cat("\nVIF:\n"); print(vif_df)
  }

  auc <- plot_roc(df[[outcome]], y_pred, out_dir)
  plot_calibration(df[[outcome]], y_pred, out_dir)
  plot_coefficients(sig_table, out_dir)
  cat(sprintf("\nAUC: %.3f\n", auc))
  cat(sprintf("Plots written to %s: coefficient_plot.pdf, roc_curve.pdf, calibration_plot.pdf\n", out_dir))
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  csv_path <- args[1]
  outcome <- args[2]
  predictors <- strsplit(args[3], ",")[[1]]
  out_dir <- ifelse(length(args) >= 4, args[4], "output")
  cluster <- if (length(args) >= 5) args[5] else NULL
  run(csv_path, outcome, predictors, out_dir, cluster)
}
