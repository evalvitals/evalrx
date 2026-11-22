"""Exact test for a difference of two proportions (rows independent).

The percentile bootstrap degenerates on small, cleanly separated samples: four
failures against four passes resample to the same difference every time, so
its p-value is as small as the number of replicates allows however little data
there is. Conditioning on the margins gives the exact answer instead (2/70 for
that table), so a two-group comparison of a binary outcome with independent
rows uses this test for its p-value and keeps the bootstrap for its interval.

Fisher's exact test:
  "On the Interpretation of χ² from Contingency Tables, and the Calculation of P"
  Fisher (1922), J. Royal Statistical Society 85(1):87-94
  https://doi.org/10.2307/2340521
"""

from __future__ import annotations

import math


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def two_group_exact_p(
    fail_a: int, n_a: int, fail_b: int, n_b: int, alternative: str = "two-sided"
) -> float:
    """Exact p-value for ``rate_b - rate_a`` given both group sizes and the total.

    ``alternative`` is ``"two-sided"`` (a difference at least as large in
    either direction), ``"greater"`` (``rate_b > rate_a``) or ``"less"``.
    Group ``a`` has ``fail_a`` failures in ``n_a`` rows; likewise ``b``.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative {alternative!r}")
    if n_a <= 0 or n_b <= 0:
        return 1.0
    n, total = n_a + n_b, fail_a + fail_b
    observed = fail_b / n_b - fail_a / n_a
    log_denom = _log_comb(n, n_b)
    eps = 1e-12
    prob = 0.0
    for k in range(max(0, n_b - (n - total)), min(n_b, total) + 1):
        diff = k / n_b - (total - k) / n_a
        if alternative == "two-sided":
            extreme = abs(diff) + eps >= abs(observed)
        elif alternative == "greater":
            extreme = diff + eps >= observed
        else:
            extreme = diff - eps <= observed
        if extreme:
            prob += math.exp(_log_comb(total, k) + _log_comb(n - total, n_b - k) - log_denom)
    return min(1.0, max(0.0, prob))
