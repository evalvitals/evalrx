# Steps 3-4 starting template: per-variable exploration vs. outcome, plus marginal
# variable screening.
#
# Given a dataframe, a binary outcome column, and a list of explanatory variables:
# - categorical variable -> contingency table vs. outcome + chi-square/Fisher's exact
# - continuous variable -> side-by-side boxplot vs. outcome + Welch's t-test / Mann-Whitney
# - optional conditioning variable -> repeat the above within each stratum
# - marginal screening -> univariate logistic regression per variable, ranked by p-value
#
# This is a template to adapt to the real column names and data -- not a black box.
#
# Usage:
#   Rscript univariate_eda.R data.csv outcome "age,region,score" output/ cohort

suppressMessages({
  library(dplyr)
  library(ggplot2)
})

infer_variable_types <- function(df, variables) {
  sapply(variables, function(v) {
    if (is.numeric(df[[v]]) && length(unique(df[[v]])) > 10) "continuous" else "categorical"
  })
}

categorical_vs_outcome <- function(df, var, outcome) {
  ct <- table(df[[var]], df[[outcome]])
  expected <- suppressWarnings(chisq.test(ct)$expected)
  if (any(expected < 5) && all(dim(ct) == 2)) {
    test_used <- "fisher_exact"
    p_value <- fisher.test(ct)$p.value
  } else {
    test_used <- "chi_square"
    p_value <- suppressWarnings(chisq.test(ct)$p.value)
  }
  chi2 <- suppressWarnings(chisq.test(ct)$statistic)
  n <- sum(ct)
  r <- nrow(ct); k <- ncol(ct)
  cramers_v <- sqrt((chi2 / n) / min(k - 1, r - 1))
  list(variable = var, test = test_used, p_value = p_value, cramers_v = as.numeric(cramers_v),
       contingency_table = ct)
}

continuous_vs_outcome <- function(df, var, outcome) {
  groups <- split(df[[var]], df[[outcome]])
  a <- na.omit(groups[[1]]); b <- na.omit(groups[[2]])
  normal_a <- shapiro.test(sample(a, min(length(a), 5000)))$p.value > 0.05
  normal_b <- shapiro.test(sample(b, min(length(b), 5000)))$p.value > 0.05
  if (normal_a && normal_b) {
    test_used <- "welch_t"
    test_result <- t.test(a, b)
  } else {
    test_used <- "mann_whitney_u"
    test_result <- wilcox.test(a, b)
  }
  pooled_sd <- sqrt((sd(a)^2 + sd(b)^2) / 2)
  cohens_d <- (mean(a) - mean(b)) / pooled_sd
  list(variable = var, test = test_used, statistic = unname(test_result$statistic),
       p_value = test_result$p.value, cohens_d = cohens_d)
}

boxplot_vs_outcome <- function(df, var, outcome, out_dir, facet_on = NULL) {
  p <- ggplot(df, aes(.data[[outcome]], .data[[var]])) + geom_boxplot(fill = "steelblue", alpha = 0.6) +
    labs(title = var) + theme_minimal()
  fname <- paste0("box_", var, "_vs_", outcome)
  if (!is.null(facet_on)) {
    p <- p + facet_wrap(vars(.data[[facet_on]]))
    fname <- paste0(fname, "_by_", facet_on)
  }
  ggsave(file.path(out_dir, paste0(fname, ".pdf")), p, width = 5, height = 3.5)
}

conditioned_analysis <- function(df, var, outcome, var_type, condition_on, out_dir) {
  strata <- unique(na.omit(df[[condition_on]]))
  rows <- lapply(strata, function(s) {
    sub <- df[df[[condition_on]] == s, ]
    if (var_type == "categorical") {
      r <- categorical_vs_outcome(sub, var, outcome)
      data.frame(stratum = s, test = r$test, p_value = r$p_value, effect_size = r$cramers_v)
    } else {
      r <- continuous_vs_outcome(sub, var, outcome)
      data.frame(stratum = s, test = r$test, p_value = r$p_value, effect_size = r$cohens_d)
    }
  })
  if (var_type == "continuous") boxplot_vs_outcome(df, var, outcome, out_dir, facet_on = condition_on)
  do.call(rbind, rows)
}

marginal_screen <- function(df, outcome, variables) {
  # the null model is refit on each variable's own non-missing subset below, since
  # comparing a null model fit on all rows against a univariate model that drops rows
  # with a missing predictor (mismatched sample sizes) invalidates the LRT
  rows <- lapply(variables, function(v) {
    tryCatch({
      sub <- df[stats::complete.cases(df[, c(outcome, v)]), ]
      null_model <- glm(as.formula(paste(outcome, "~ 1")), data = sub, family = binomial)
      model <- glm(as.formula(paste(outcome, "~", v)), data = sub, family = binomial)
      # likelihood-ratio test: a joint test across all dummy levels for a categorical
      # variable, not just one coefficient's Wald p-value
      lrt <- anova(null_model, model, test = "Chisq")
      p_value <- lrt[["Pr(>Chi)"]][2]
      aic_improvement <- AIC(null_model) - AIC(model)
      data.frame(variable = v, p_value = p_value, aic_improvement = aic_improvement)
    }, error = function(e) data.frame(variable = v, p_value = NA, aic_improvement = NA))
  })
  do.call(rbind, rows) %>% arrange(p_value)
}

run <- function(csv_path, outcome, variables, out_dir, condition_on = NULL) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  df <- read.csv(csv_path)
  var_types <- infer_variable_types(df, variables)

  univariate_results <- list()
  for (v in variables) {
    if (var_types[[v]] == "categorical") {
      result <- categorical_vs_outcome(df, v, outcome)
      cat("\n", v, "(categorical) vs", outcome, ":\n")
      print(result$contingency_table)
      cat(sprintf("  %s: p=%.4g, Cramer's V=%.3f\n", result$test, result$p_value, result$cramers_v))
      univariate_results[[v]] <- result[c("variable", "test", "p_value", "cramers_v")]
    } else {
      boxplot_vs_outcome(df, v, outcome, out_dir)
      result <- continuous_vs_outcome(df, v, outcome)
      cat("\n", v, "(continuous) vs", outcome, ":\n")
      cat(sprintf("  %s: p=%.4g, Cohen's d=%.3f\n", result$test, result$p_value, result$cohens_d))
      univariate_results[[v]] <- result
    }
    if (!is.null(condition_on)) {
      cond <- conditioned_analysis(df, v, outcome, var_types[[v]], condition_on, out_dir)
      cat("  conditioned on", condition_on, ":\n"); print(cond)
      write.csv(cond, file.path(out_dir, paste0("conditioned_", v, "_on_", condition_on, ".csv")), row.names = FALSE)
    }
  }

  screen <- marginal_screen(df, outcome, variables)
  write.csv(screen, file.path(out_dir, "marginal_screening.csv"), row.names = FALSE)
  cat("\nMarginal screening (ranked by p-value):\n"); print(screen)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  csv_path <- args[1]
  outcome <- args[2]
  variables <- strsplit(args[3], ",")[[1]]
  out_dir <- ifelse(length(args) >= 4, args[4], "output")
  condition_on <- if (length(args) >= 5) args[5] else NULL
  run(csv_path, outcome, variables, out_dir, condition_on)
}
