"""Confidence-preserving test-subset sampling (tinyBenchmarks / anchor-points idea).

Stratified proportional sampling keeps each failure-relevant stratum represented;
``kendall_tau`` lets you VALIDATE that a subset preserves the model ranking before
trusting it (don't claim a coreset is faithful without checking τ vs the full set).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, Sequence


def stratified_subset(items: Iterable, key: Callable, n: int, seed: int = 0) -> list:
    """Proportionally sample ~*n* items across strata defined by ``key(item)``."""
    rng = random.Random(seed)
    items = list(items)
    total = len(items)
    if total == 0 or n <= 0:
        return []
    strata: dict = defaultdict(list)
    for it in items:
        strata[key(it)].append(it)
    chosen: list = []
    for group in strata.values():
        g = group[:]
        rng.shuffle(g)
        take = max(1, round(n * len(group) / total))
        chosen.extend(g[:take])
    rng.shuffle(chosen)
    return chosen[:n]


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's τ-a rank correlation in [-1, 1] (O(n^2), no scipy)."""
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must be equal length")
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else 1.0


def sample_subset(items, key, n, seed: int = 0):  # back-compat alias
    return stratified_subset(items, key, n, seed)
