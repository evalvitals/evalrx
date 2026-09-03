"""Permutation-sampling Shapley values — shared by MM-SHAP and VL-SHAP.

Estimates each player's Shapley value (mean marginal contribution to a coalition
value) by sampling random permutations.  ``value_fn(kept: set) -> float`` is the
coalition value (e.g. the model's logprob/confidence with only ``kept`` players
present).  Results are memoised by coalition so expensive model calls aren't
repeated.  For an additive game the estimate equals each player's exact weight.

References:
- Shapley value: 'A Value for n-Person Games', Shapley (1953), Contributions to the Theory of Games II.
- Sampling estimator: 'Polynomial calculation of the Shapley value based on sampling',
  Castro, Gomez & Tejada (2009), Computers & Operations Research 36(5).
- ML attribution: 'A Unified Approach to Interpreting Model Predictions' (SHAP),
  Lundberg & Lee, NeurIPS 2017 — arXiv:1705.07874
"""

from __future__ import annotations

import random
from typing import Callable, Iterable


def shapley_values(
    players: Iterable,
    value_fn: Callable[[set], float],
    n_samples: int = 64,
    seed: int = 0,
) -> dict:
    """Return ``{player: shapley_value}`` via permutation sampling."""
    players = list(players)
    phi = {p: 0.0 for p in players}
    if not players:
        return phi
    rng = random.Random(seed)
    memo: dict = {}

    def val(kept: set) -> float:
        key = frozenset(kept)
        if key not in memo:
            memo[key] = float(value_fn(set(kept)))
        return memo[key]

    base = val(set())
    for _ in range(n_samples):
        perm = players[:]
        rng.shuffle(perm)
        kept: set = set()
        prev = base
        for p in perm:
            kept.add(p)
            cur = val(kept)
            phi[p] += cur - prev
            prev = cur
    return {p: phi[p] / n_samples for p in players}
