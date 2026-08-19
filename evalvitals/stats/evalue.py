"""E-values — anytime-valid evidence (safe under optional stopping / peeking).

The closed loop keeps adding hypotheses and peeking at results; p-values break
under that, e-values don't.  ``evalue_bernoulli`` is the mixture (Bayes-factor
with a uniform prior) e-value for a Bernoulli mean vs ``p0`` — a valid e-value:
under H0, E[e] <= 1, so rejecting when ``e >= 1/alpha`` controls type-I error at
any stopping time.  For a paired A/B test, feed the discordant pairs that favour
B as ``successes`` out of ``n_discordant`` with ``p0=0.5`` (the McNemar null).

E-values and safe testing:
  "Safe Testing" — Grünwald, de Heide & Koolen (2022)
  J. Royal Statistical Society B — https://arxiv.org/abs/1906.07801

Mixture / Bayes-factor e-value for the Bernoulli mean:
  "Estimating means of bounded random variables by betting"
  Waudby-Smith & Ramdas (2023), JRSSB — https://arxiv.org/abs/2010.09686
"""

from __future__ import annotations

import math


def evalue_bernoulli(successes: int, n: int, p0: float = 0.5) -> float:
    """Mixture e-value for Bernoulli(p) vs H0: p = p0 (uniform prior over p)."""
    if n <= 0:
        return 1.0
    s = int(successes)
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0, 1)")
    # numerator: log Beta(s+1, n-s+1) = mixture marginal (binomial coeff cancels with f0)
    log_num = math.lgamma(s + 1) + math.lgamma(n - s + 1) - math.lgamma(n + 2)
    log_den = s * math.log(p0) + (n - s) * math.log(1 - p0)
    return math.exp(log_num - log_den)


def e_value_test(successes: int, n: int, p0: float = 0.5, alpha: float = 0.05) -> dict:
    """Convenience wrapper: e-value + reject decision at level *alpha*."""
    e = evalue_bernoulli(successes, n, p0)
    return {"e_value": e, "reject": e >= 1.0 / alpha, "threshold": 1.0 / alpha, "alpha": alpha}


#: Fixed betting fractions the bounded-mean e-value mixes over (a convex mixture
#: of e-values is an e-value). Spread on a log-ish scale so both a few large
#: differences and many small ones can accumulate evidence.
_LAMBDA_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)


def evalue_bounded_mean(
    diffs,
    *,
    low: float = -1.0,
    high: float = 1.0,
    null_mean: float = 0.0,
    lambdas=None,
) -> float:
    """Betting (capital-process) e-value for H0: E[d] <= ``null_mean``, d in [low, high].

    For each betting fraction λ the capital ``Π_i (1 + λ·(d_i − null_mean)/(null_mean − low))``
    is a non-negative supermartingale under H0 (each factor is ≥ 0 and has
    expectation ≤ 1), so it is an e-value; the returned value is the average
    over a fixed λ grid, itself an e-value. One-sided: it accumulates evidence
    that the mean is ABOVE ``null_mean``; pass ``-d`` to test the other side.

    Use for PAIRED per-case rate differences (candidate rate − baseline rate,
    each estimated from k samples): unlike McNemar on one sample per arm, a
    case whose baseline passes 2/5 of the time and which the candidate gets
    right contributes +0.6, not +1 — sampling-unstable cases are weighed, not
    dropped and not mistaken for repairs.

    Betting e-values for bounded means:
      Waudby-Smith & Ramdas (2023), JRSSB — https://arxiv.org/abs/2010.09686
    """
    xs = [float(d) for d in diffs]
    if not xs:
        return 1.0
    if not (low < null_mean < high):
        raise ValueError("need low < null_mean < high")
    scale = null_mean - low  # largest step down; keeps every factor >= 0 for λ <= 1
    grid = tuple(lambdas) if lambdas is not None else _LAMBDA_GRID
    log_caps = []
    for lam in grid:
        if not (0.0 < lam <= 1.0):
            raise ValueError("betting fractions must lie in (0, 1]")
        total = 0.0
        for x in xs:
            x = min(max(x, low), high)
            factor = 1.0 + lam * (x - null_mean) / scale
            if factor <= 0.0:
                total = -math.inf
                break
            total += math.log(factor)
        log_caps.append(total)
    finite = [v for v in log_caps if v > -math.inf]
    if not finite:
        return 0.0
    m = max(finite)
    return math.exp(m) * sum(math.exp(v - m) for v in finite) / len(grid)
