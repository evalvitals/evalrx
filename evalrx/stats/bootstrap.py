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
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

#: Relations a bootstrap p-value can be asked about; see :func:`bootstrap_p_value`.
RELATIONS = ("nonzero", ">", "<", "abs<")


def _cluster_codes(clusters: Optional[Sequence], n: int) -> "tuple[np.ndarray, int]":
    """Integer code per row and the number of clusters.

    ``None`` makes every row its own cluster. A missing cluster value (``None``
    or an empty string) is also its own cluster: an unknown group must not be
    merged with other unknown groups.
    """
    if clusters is None:
        return np.arange(n), n
    values = list(clusters)
    if len(values) != n:
        raise ValueError(f"clusters has {len(values)} entries for {n} rows")
    index: dict = {}
    codes = np.empty(n, dtype=np.int64)
    for i, value in enumerate(values):
        key: Any = ("__row__", i) if value is None or value == "" else ("v", value)
        codes[i] = index.setdefault(key, len(index))
    return codes, len(index)


def _weight_draws(
    codes: np.ndarray, n_clusters: int, n_boot: int, seed: int, chunk: int = 64
) -> "Iterator[np.ndarray]":
    """Yield one row-weight vector per replicate.

    Each replicate draws ``n_clusters`` clusters with replacement; a row's
    weight is how often its cluster was drawn, so rows of one cluster always
    enter or leave a replicate together.
    """
    rng = np.random.default_rng(seed)
    probs = np.full(n_clusters, 1.0 / n_clusters)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        counts = rng.multinomial(n_clusters, probs, size=size)
        for row in counts:
            yield row[codes].astype(float)
        done += size


def weighted_bootstrap(
    statistic: "Callable[[np.ndarray], float]",
    n: int,
    *,
    clusters: Optional[Sequence] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Cluster bootstrap of any statistic written as a function of row weights.

    ``statistic(w)`` receives one non-negative weight per row and returns a
    float, or NaN when the statistic is undefined for that replicate (for
    example one arm received no weight). The point estimate is
    ``statistic(ones)``. Writing the statistic over weights lets one routine
    serve a difference of means, a stratified difference, a difference in
    differences and a trend slope, with the same resampling of whole clusters.

    Returns ``effect``, ``ci_low``, ``ci_high``, ``draws`` (finite replicates
    only), ``n_clusters`` and ``n_undefined``.
    """
    if n <= 0:
        return {"effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci": ci, "n": 0,
                "n_clusters": 0, "n_undefined": 0, "draws": np.empty(0)}
    codes, n_clusters = _cluster_codes(clusters, n)
    point = float(statistic(np.ones(n)))
    values = np.fromiter(
        (statistic(w) for w in _weight_draws(codes, n_clusters, n_boot, seed)),
        dtype=float, count=n_boot,
    )
    draws = values[np.isfinite(values)]
    if draws.size == 0 or not np.isfinite(point):
        return {"effect": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "ci": ci, "n": n, "n_clusters": n_clusters,
                "n_undefined": int(n_boot - draws.size), "draws": draws}
    lo = float(np.quantile(draws, (1 - ci) / 2))
    hi = float(np.quantile(draws, 1 - (1 - ci) / 2))
    return {"effect": point, "ci_low": lo, "ci_high": hi, "ci": ci, "n": n,
            "n_clusters": n_clusters, "n_undefined": int(n_boot - draws.size),
            "draws": draws}


def bootstrap_p_value(draws: "Sequence[float] | np.ndarray", relation: str = "nonzero",
                      value: float = 0.0) -> float:
    """Percentile-bootstrap p-value for the claim ``effect <relation> value``.

    The p-value is the share of replicates on the wrong side of the claim,
    with one added to numerator and denominator so it is never zero:

    - ``"nonzero"``: two-sided, against an effect of exactly zero.
    - ``">"`` / ``"<"``: one-sided, against ``effect <= value`` / ``>= value``.
    - ``"abs<"``: against ``|effect| >= value`` (the effect is at most that
      large in either direction).

    This inverts the percentile interval, so it is approximate in the same way
    the interval is, and it cannot go below ``1 / (len(draws) + 1)``.
    """
    d = np.asarray(draws, dtype=float)
    b = d.size
    if b == 0:
        return 1.0
    if relation == "nonzero":
        below = (1 + int(np.sum(d <= 0.0))) / (b + 1)
        above = (1 + int(np.sum(d >= 0.0))) / (b + 1)
        return float(min(1.0, 2.0 * min(below, above)))
    if relation == ">":
        return float((1 + int(np.sum(d <= value))) / (b + 1))
    if relation == "<":
        return float((1 + int(np.sum(d >= value))) / (b + 1))
    if relation == "abs<":
        return float((1 + int(np.sum(np.abs(d) >= abs(value)))) / (b + 1))
    raise ValueError(f"unknown relation {relation!r}; expected one of {RELATIONS}")


def mean_difference_statistic(
    y: np.ndarray, arm: np.ndarray
) -> "Callable[[np.ndarray], float]":
    """Weighted ``mean(y | arm == 1) - mean(y | arm == 0)`` as a weight function."""
    y = np.asarray(y, dtype=float)
    in_b = np.asarray(arm, dtype=float)
    in_a = 1.0 - in_b
    y_b, y_a = y * in_b, y * in_a

    def statistic(w: np.ndarray) -> float:
        w_b, w_a = float(w @ in_b), float(w @ in_a)
        if w_b <= 0.0 or w_a <= 0.0:
            return float("nan")
        return float(w @ y_b) / w_b - float(w @ y_a) / w_a

    return statistic


def _two_group_clusters(clusters: Any, na: int, nb: int) -> "list | None":
    """``(clusters_a, clusters_b)`` flattened to one list, or None if not that shape."""
    if clusters is None or isinstance(clusters, (str, bytes)):
        return None
    try:
        if len(clusters) != 2:
            return None
        first, second = clusters
        if isinstance(first, (str, bytes)) or isinstance(second, (str, bytes)):
            return None
        if len(first) != na or len(second) != nb:
            return None
    except TypeError:
        return None
    return list(first) + list(second)


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
    paired=False: A and B are two groups (lengths may differ). Without
                  clusters each group is resampled independently (two-sample
                  bootstrap). Pass ``clusters=(clusters_a, clusters_b)``, one
                  cluster id per row of each group, to resample whole clusters
                  instead; a cluster may have rows in both groups, and they
                  then move together. A single flat sequence cannot be aligned
                  to two groups and is ignored, as before.

    Unpaired results also carry ``p_value``, the two-sided percentile-bootstrap
    p-value against a difference of zero (see :func:`bootstrap_p_value`).
    """
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)

    if not paired:
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return {"effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci": ci, "n": 0}
        joint = _two_group_clusters(clusters, na, nb)
        if joint is not None:
            arm = np.concatenate([np.zeros(na), np.ones(nb)])
            out = weighted_bootstrap(
                mean_difference_statistic(np.concatenate([a, b]), arm), na + nb,
                clusters=joint, n_boot=n_boot, ci=ci, seed=seed,
            )
            draws = out.pop("draws")
            out.update(n_a=na, n_b=nb, p_value=bootstrap_p_value(draws))
            return out
        rng = np.random.default_rng(seed)
        point = float(b.mean() - a.mean())
        ia = rng.integers(0, na, size=(n_boot, na))
        ib = rng.integers(0, nb, size=(n_boot, nb))
        diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
        lo = float(np.quantile(diffs, (1 - ci) / 2))
        hi = float(np.quantile(diffs, 1 - (1 - ci) / 2))
        return {"effect": point, "ci_low": lo, "ci_high": hi, "ci": ci,
                "n": na + nb, "n_a": na, "n_b": nb,
                "p_value": bootstrap_p_value(diffs)}

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
