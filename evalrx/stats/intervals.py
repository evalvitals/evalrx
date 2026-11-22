"""Closed-form intervals and quantiles, in the standard library only.

For code that needs an interval at a level other than 95% (a family-wise adjusted one)
without resampling: the two-sided normal and Student-t quantiles, the Wilson interval of
a rate, Newcombe's interval of a difference of two rates, and a rank correlation taken
within groups.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
         2.045, 2.042)


def _t975(df: float) -> float:
    """Two-sided 95% Student-t quantile at ``floor(df)`` (conservative), normal beyond 120."""
    k = max(1, int(math.floor(df)))
    if k <= 30:
        return _T975[k - 1]
    return 2.021 if k < 60 else 2.000 if k < 120 else 1.960


def z_quantile(level: float) -> float:
    """Two-sided normal quantile: ``z`` with ``P(|Z| <= z) = level``."""
    return NormalDist().inv_cdf(0.5 + level / 2)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _t_two_sided(t: float, k: int) -> float:
    """``P(|T| <= t)`` for Student's t with *k* degrees of freedom: ``1 - I_{k/(k+t^2)}(k/2, 1/2)``."""
    x = k / (k + t * t)
    a, b = k / 2.0, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    ibeta = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - front * _betacf(b, a, 1 - x) / b
    return 1.0 - ibeta


def t_quantile(level: float, df: float) -> float:
    """Two-sided Student-t quantile at ``floor(df)`` degrees of freedom (conservative): the table
    above at 95%, else the CDF (incomplete beta) inverted by bisection, e.g. for a family-wise
    adjusted interval."""
    if abs(level - 0.95) < 1e-12:
        return _t975(df)
    k = max(1, int(math.floor(df)))
    lo, hi = 0.0, 1.0
    while _t_two_sided(hi, k) < level:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_two_sided(mid, k) < level else (lo, mid)
        if hi - lo < 1e-10:
            break
    return hi


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of the rate ``k/n`` (95% at the default *z*)."""
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_interval(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> "list[float] | None":
    """Interval of ``k1/n1 - k2/n2`` (Newcombe's hybrid score method; 95% at the default *z*),
    ``None`` if a side is empty."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson_interval(k1, n1, z), wilson_interval(k2, n2, z)
    d = p1 - p2
    return [round(d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 4),
            round(d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 4)]


def average_ranks(xs: list[float]) -> list[float]:
    """Ranks with ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def within_group_spearman(a: dict[str, float], b: dict[str, float], groups: list[list[str]], *,
                          min_pairs: int = 5) -> "float | None":
    """Spearman correlation of two per-case readings inside each group of cases, pooled: ranks are
    taken and centred within a group, so a shift between the groups does not count. ``None`` when no
    group has *min_pairs* cases with both readings, or a reading is constant within every group."""
    sxy = sxx = syy = 0.0
    for ids in groups:
        both = [c for c in ids if c in a and c in b]
        if len(both) < min_pairs:
            continue
        ra, rb = average_ranks([a[c] for c in both]), average_ranks([b[c] for c in both])
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        sxy += sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        sxx += sum((x - ma) ** 2 for x in ra)
        syy += sum((y - mb) ** 2 for y in rb)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)
