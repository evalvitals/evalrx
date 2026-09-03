"""Friedman omnibus test + Nemenyi post-hoc — comparing >2 strategies.

The right protocol for ranking 3+ prompting strategies across a shared benchmark:
Friedman tests whether the strategies differ at all (on within-example ranks),
then Nemenyi's critical difference (CD) says which pairs differ.  No scipy — the
chi-square survival function is computed via the regularized incomplete gamma.

Paper: "Statistical Comparisons of Classifiers over Multiple Data Sets"
       Demšar (2006), JMLR 7(1):1-30
       https://jmlr.org/papers/v7/demsarar06a.html

Nemenyi critical differences table from the same paper.
"""

from __future__ import annotations

import math
from typing import Sequence

# Nemenyi q_alpha (already /sqrt(2)) for alpha=0.05, k = number of strategies.
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes gser/gcf)."""
    if x < 0 or a <= 0:
        raise ValueError("invalid args to _gammq")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:  # series for P, return 1 - P
        ap, s, term = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    # continued fraction for Q
    b, c = x + 1.0 - a, 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P(X > x) for X ~ chi2(df)."""
    if x <= 0:
        return 1.0
    return _gammq(df / 2.0, x / 2.0)


def _avg_ranks_desc(values: Sequence[float]) -> list[float]:
    """Rank values with 1 = best (highest), averaging ranks for ties."""
    n = len(values)
    order = sorted(range(n), key=lambda i: -values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 1-based positions i+1..j+1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def friedman_test(success_by_strategy: dict) -> dict:
    """Friedman test over k strategies × n paired examples.

    ``success_by_strategy``: ``{name: [per-example metric]}`` (equal-length lists).
    """
    names = list(success_by_strategy)
    data = [list(success_by_strategy[n]) for n in names]
    k = len(data)
    if k < 2:
        raise ValueError("Friedman needs >= 2 strategies")
    n = len(data[0])
    if any(len(d) != n for d in data):
        raise ValueError("all strategies need the same number of (paired) examples")
    rank_sums = [0.0] * k
    for j in range(n):
        col = [data[i][j] for i in range(k)]
        ranks = _avg_ranks_desc(col)
        for i in range(k):
            rank_sums[i] += ranks[i]
    chi2 = 12.0 / (n * k * (k + 1)) * sum(r * r for r in rank_sums) - 3.0 * n * (k + 1)
    return {
        "names": names,
        "k": k,
        "n": n,
        "avg_ranks": [r / n for r in rank_sums],
        "statistic": chi2,
        "df": k - 1,
        "p_value": chi2_sf(chi2, k - 1),
    }


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for average ranks."""
    if alpha != 0.05:
        raise ValueError("only alpha=0.05 is tabulated here")
    q = _NEMENYI_Q05.get(k)
    if q is None:
        raise ValueError(f"Nemenyi q not tabulated for k={k} (supported 2..10)")
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi_pairs(avg_ranks_by_name: dict, cd: float) -> list[dict]:
    """Pairs whose average-rank difference >= CD (significantly different)."""
    names = list(avg_ranks_by_name)
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            diff = abs(avg_ranks_by_name[na] - avg_ranks_by_name[nb])
            if diff >= cd:
                better = na if avg_ranks_by_name[na] < avg_ranks_by_name[nb] else nb
                out.append({"a": na, "b": nb, "rank_diff": round(diff, 4), "better": better})
    return out
