"""Clustered bootstrap CI for an effect (mean difference) — examples are NOT i.i.d.

Resamples CLUSTERS (default: each example its own cluster; pass task ids to cluster
multiple samples per task) with replacement.  Reporting effect size + CI, not just
a p-value, is the point: a significant test with a CI hugging 0 is noise.

Bootstrap method:
  "Bootstrap Methods: Another Look at the Jackknife"
  Efron (1979), The Annals of Statistics 7(1):1-26
  https://doi.org/10.1214/aos/1176344552

Cluster-robust bootstrap for correlated errors:
  "Bootstrap-Based Improvements for Inference with Clustered Errors"
  Cameron, Gelbach & Miller (2008), Review of Economics and Statistics 90(3):414-427
  https://doi.org/10.1162/rest.90.3.414
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


def clustered_bootstrap_diff(
    success_a: Sequence[float],
    success_b: Sequence[float],
    clusters: Optional[Sequence] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    paired: bool = True,
) -> dict:
    """Bootstrap CI for ``mean(B) - mean(A)``.

    paired=True:  A and B are per-example aligned vectors of equal length;
                  resamples CLUSTERS of example indices (default: each example
                  its own cluster) so the pairing is preserved.
    paired=False: A and B are two independent groups (lengths may differ);
                  each group is resampled independently (two-sample bootstrap).
                  ``clusters`` is ignored — a single cluster sequence cannot
                  align to two groups of different sizes.
    """
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)

    if not paired:
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return {"effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci": ci, "n": 0}
        rng = np.random.default_rng(seed)
        point = float(b.mean() - a.mean())
        ia = rng.integers(0, na, size=(n_boot, na))
        ib = rng.integers(0, nb, size=(n_boot, nb))
        diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
        lo = float(np.quantile(diffs, (1 - ci) / 2))
        hi = float(np.quantile(diffs, 1 - (1 - ci) / 2))
        return {"effect": point, "ci_low": lo, "ci_high": hi, "ci": ci,
                "n": na + nb, "n_a": na, "n_b": nb}

    n = len(a)
    if n != len(b):
        raise ValueError(
            f"paired bootstrap needs equal-length vectors (got {n} vs {len(b)}); "
            "use paired=False for independent groups"
        )
    if n == 0:
        return {"effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci": ci, "n": 0}
    ids = list(range(n)) if clusters is None else list(clusters)
    groups: dict = defaultdict(list)
    for i, cl in enumerate(ids):
        groups[cl].append(i)
    cluster_keys = list(groups.keys())
    g = len(cluster_keys)

    rng = np.random.default_rng(seed)
    point = float(b.mean() - a.mean())
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        pick = rng.integers(0, g, size=g)
        idx = np.concatenate([np.asarray(groups[cluster_keys[k]]) for k in pick])
        diffs[t] = b[idx].mean() - a[idx].mean()
    lo = float(np.quantile(diffs, (1 - ci) / 2))
    hi = float(np.quantile(diffs, 1 - (1 - ci) / 2))
    return {"effect": point, "ci_low": lo, "ci_high": hi, "ci": ci, "n": n, "n_clusters": g}
