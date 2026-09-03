"""McNemar's test — the right test for PAIRED binary outcomes (same examples, A vs B).

Conditions on the discordant pairs (where A and B disagree), so it's far more
powerful than a two-proportion test when A/B are run on the *same* items.
Exact binomial p (no scipy); chi-square-with-continuity returned for reference.

Paper: "Note on the sampling error of the difference between correlated
       proportions or percentages" — McNemar (1947), Psychometrika 12(2):153-157
       https://doi.org/10.1007/BF02295996
"""

from __future__ import annotations

import math
from typing import Sequence


def _exact_two_sided_p(b: int, c: int) -> float:
    """Exact binomial two-sided p for the discordant counts under p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(success_a: Sequence[bool], success_b: Sequence[bool]) -> dict:
    """Paired binary comparison.

    ``b`` = A wrong & B right (favours B); ``c`` = A right & B wrong (favours A).
    Returns counts, the exact two-sided p, and the continuity-corrected statistic.
    """
    if len(success_a) != len(success_b):
        raise ValueError("paired test needs equal-length success vectors")
    b = sum(1 for x, y in zip(success_a, success_b) if (not x) and y)
    c = sum(1 for x, y in zip(success_a, success_b) if x and (not y))
    n_disc = b + c
    stat = ((abs(b - c) - 1) ** 2) / n_disc if n_disc > 0 else 0.0
    return {"b": b, "c": c, "n_discordant": n_disc, "statistic": stat, "p_value": _exact_two_sided_p(b, c)}
