# Step 2 starting template: explanatory-variable EDA.
#
# Given a dataframe and a list of explanatory variable columns, produce:
# - a per-variable distribution plot (histogram for continuous, bar chart for categorical)
# - an outlier / missingness summary table
# - a mixed-type correlation / association matrix among the explanatory variables
#
# This is a template to adapt to the real column names and data -- not a black box.
#
# Usage:
#   Rscript explanatory_var_eda.R data.csv "age,region,score" output/

suppressMessages({
  library(dplyr)
  library(ggplot2)
})

infer_variable_types <- function(df, variables) {
  sapply(variables, function(v) {
    if (is.numeric(df[[v]]) && length(unique(df[[v]])) > 10) "continuous" else "categorical"
  })
}

outlier_summary <- function(df, var) {
  q <- quantile(df[[var]], c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  lower <- q[1] - 1.5 * iqr
  upper <- q[2] + 1.5 * iqr
  n_outliers <- sum(df[[var]] < lower | df[[var]] > upper, na.rm = TRUE)
  data.frame(variable = var, n_outliers = n_outliers, lower_bound = lower, upper_bound = upper)
}

rare_category_summary <- function(df, var, threshold = 0.01) {
  counts <- prop.table(table(df[[var]]))
  rare <- names(counts[counts < threshold])
  data.frame(variable = var, n_categories = length(counts), rare_categories = paste(rare, collapse = ";"))
}

missingness_summary <- function(df, variables) {
  do.call(rbind, lapply(variables, function(v) {
    n_missing <- sum(is.na(df[[v]]))
    data.frame(variable = v, n_missing = n_missing, pct_missing = round(100 * n_missing / nrow(df), 1))
  }))
}

cramers_v <- function(x, y) {
  ct <- table(x, y)
  chi2 <- suppressWarnings(chisq.test(ct)$statistic)
  n <- sum(ct)
  phi2 <- chi2 / n
  r <- nrow(ct); k <- ncol(ct)
  phi2_corr <- max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
  r_corr <- r - (r - 1)^2 / (n - 1)
  k_corr <- k - (k - 1)^2 / (n - 1)
  sqrt(phi2_corr / min(k_corr - 1, r_corr - 1))
}

correlation_ratio <- function(categorical, continuous) {
  d <- na.omit(data.frame(cat = categorical, cont = continuous))
  overall_mean <- mean(d$cont)
  agg <- aggregate(cont ~ cat, d, function(g) c(n = length(g), mean = mean(g)))
  ss_between <- sum(sapply(split(d$cont, d$cat), function(g) length(g) * (mean(g) - overall_mean)^2))
  ss_total <- sum((d$cont - overall_mean)^2)
  if (ss_total > 0) sqrt(ss_between / ss_total) else 0
}

association_matrix <- function(df, var_types) {
  variables <- names(var_types)
  mat <- matrix(NA, length(variables), length(variables), dimnames = list(variables, variables))
  for (a in variables) {
    for (b in variables) {
      if (a == b) { mat[a, b] <- 1; next }
      ta <- var_types[[a]]; tb <- var_types[[b]]
      if (ta == "continuous" && tb == "continuous") {
        mat[a, b] <- cor(df[[a]], df[[b]], use = "pairwise.complete.obs")
      } else if (ta == "categorical" && tb == "categorical") {
        mat[a, b] <- cramers_v(df[[a]], df[[b]])
      } else {
        if (ta == "categorical") mat[a, b] <- correlation_ratio(df[[a]], df[[b]])
        else mat[a, b] <- correlation_ratio(df[[b]], df[[a]])
      }
    }
  }
  mat
}

plot_distribution <- function(df, var, var_type, out_dir) {
  p <- if (var_type == "continuous") {
    ggplot(df, aes(.data[[var]])) + geom_histogram(bins = 30, fill = "steelblue", color = "white")
  } else {
    ggplot(df, aes(.data[[var]])) + geom_bar(fill = "steelblue")
  }
  p <- p + labs(title = var, x = NULL) + theme_minimal() +
    theme(panel.grid.minor = element_blank())
  ggsave(file.path(out_dir, paste0("dist_", var, ".pdf")), p, width = 5, height = 3.5)
}

run <- function(csv_path, variables, out_dir) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  df <- read.csv(csv_path)
  var_types <- infer_variable_types(df, variables)

  for (v in variables) plot_distribution(df, v, var_types[[v]], out_dir)

  cont_vars <- variables[var_types == "continuous"]
  cat_vars <- variables[var_types == "categorical"]
  outliers <- do.call(rbind, lapply(cont_vars, outlier_summary, df = df))
  rares <- do.call(rbind, lapply(cat_vars, rare_category_summary, df = df))
  missing <- missingness_summary(df, variables)
  assoc <- association_matrix(df, var_types)

  write.csv(outliers, file.path(out_dir, "outlier_summary.csv"), row.names = FALSE)
  write.csv(rares, file.path(out_dir, "rare_category_summary.csv"), row.names = FALSE)
  write.csv(missing, file.path(out_dir, "missingness_summary.csv"), row.names = FALSE)
  write.csv(assoc, file.path(out_dir, "association_matrix.csv"))

  cat("Variable types:\n"); print(var_types)
  cat("\nMissingness:\n"); print(missing)
  cat("\nOutliers (continuous):\n"); print(outliers)
  cat("\nRare categories (categorical):\n"); print(rares)
  cat("\nAssociation matrix:\n"); print(round(assoc, 2))
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  csv_path <- args[1]
  variables <- strsplit(args[2], ",")[[1]]
  out_dir <- ifelse(length(args) >= 3, args[3], "output/figures")
  run(csv_path, variables, out_dir)
}
