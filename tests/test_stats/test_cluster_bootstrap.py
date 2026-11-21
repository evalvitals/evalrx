"""Cluster bootstrap over row weights, bootstrap p-values and the exact test."""

from __future__ import annotations

import random

import numpy as np
import pytest

from evalrx.stats import (
    bootstrap_p_value,
    clustered_bootstrap_diff,
    compare,
    mean_difference_statistic,
    two_group_exact_p,
    weighted_bootstrap,
)


def test_weighted_bootstrap_point_estimate_uses_unit_weights():
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    arm = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    out = weighted_bootstrap(mean_difference_statistic(y, arm), 5, n_boot=200)
    assert out["effect"] == pytest.approx(1.0 - 1.0 / 3.0)
    assert out["n_clusters"] == 5
    assert out["ci_low"] <= out["effect"] <= out["ci_high"]


def test_rows_of_one_cluster_are_resampled_together():
    # Two clusters, each constant: every replicate holds whole clusters, so
    # the weighted mean can only be 0, 0.5 or 1.
    y = np.array([0.0] * 5 + [1.0] * 5)
    clusters = ["a"] * 5 + ["b"] * 5

    def mean(w: np.ndarray) -> float:
        return float(w @ y) / float(w.sum())

    out = weighted_bootstrap(mean, 10, clusters=clusters, n_boot=300, seed=1)
    assert out["n_clusters"] == 2
    assert set(np.round(out["draws"], 6)) <= {0.0, 0.5, 1.0}


def test_missing_cluster_ids_are_not_merged_into_one_cluster():
    out = weighted_bootstrap(lambda w: float(w.sum()), 4,
                             clusters=[None, "", "a", "a"], n_boot=10)
    assert out["n_clusters"] == 3


def test_undefined_replicates_are_counted_not_averaged_in():
    y = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    arm = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # a single row in arm 0
    out = weighted_bootstrap(mean_difference_statistic(y, arm), 6, n_boot=400, seed=0)
    assert out["n_undefined"] > 0
    assert len(out["draws"]) == 400 - out["n_undefined"]


def test_bootstrap_p_value_relations():
    draws = np.linspace(0.1, 0.3, 99)             # every replicate is positive
    assert bootstrap_p_value(draws, "nonzero") == pytest.approx(2 / 100)
    assert bootstrap_p_value(draws, ">", 0.0) == pytest.approx(1 / 100)
    assert bootstrap_p_value(draws, "<", 0.0) == 1.0
    assert bootstrap_p_value(draws, "abs<", 0.5) == pytest.approx(1 / 100)
    assert bootstrap_p_value(draws, "abs<", 0.2) > 0.4
    assert bootstrap_p_value([], "nonzero") == 1.0
    with pytest.raises(ValueError, match="unknown relation"):
        bootstrap_p_value(draws, "==")


def test_unpaired_compare_accepts_a_pair_of_cluster_sequences():
    # Forty rows a side, cut from five source items a side, and rows of an
    # item agree with each other: the evidence is ten items, not 80 rows.
    a = [0] * 32 + [1] * 8
    b = [1] * 24 + [0] * 16
    ids_a = [f"a{i // 8}" for i in range(40)]
    ids_b = [f"b{i // 8}" for i in range(40)]
    rows = compare(a, b, paired=False)
    groups = compare(a, b, paired=False, cluster_by=(ids_a, ids_b))
    assert rows.method == "two-sample bootstrap (unpaired)"
    assert groups.method == "clustered bootstrap (unpaired)"
    assert groups.details["n_clusters"] == 10
    assert rows.effect == pytest.approx(groups.effect) == pytest.approx(0.4)
    assert rows.reject is True            # looks like a finding row by row
    assert groups.reject is False         # and is not one, item by item
    assert (groups.ci[1] - groups.ci[0]) > 1.5 * (rows.ci[1] - rows.ci[0])
    assert groups.details["p_value"] > 0.05 > rows.details["p_value"]


def test_a_flat_cluster_sequence_is_still_ignored_for_unpaired():
    a, b = [0, 0, 1, 0], [1, 1, 0, 1, 1]
    plain = clustered_bootstrap_diff(a, b, paired=False, seed=3)
    flat = clustered_bootstrap_diff(a, b, clusters=list(range(9)), paired=False, seed=3)
    assert "n_clusters" not in flat
    assert (flat["ci_low"], flat["ci_high"]) == (plain["ci_low"], plain["ci_high"])


def test_exact_p_matches_hand_counts_and_one_sided_halves():
    # 4 failures against 4 passes, cleanly split: 2 of the C(8, 4) = 70 tables.
    assert two_group_exact_p(0, 4, 4, 4) == pytest.approx(2 / 70)
    assert two_group_exact_p(0, 4, 4, 4, "greater") == pytest.approx(1 / 70)
    assert two_group_exact_p(0, 4, 4, 4, "less") == pytest.approx(1.0)
    assert two_group_exact_p(2, 10, 2, 10) == pytest.approx(1.0)
    assert two_group_exact_p(0, 0, 3, 5) == 1.0
    with pytest.raises(ValueError):
        two_group_exact_p(1, 2, 1, 2, "sideways")


def _null_rows(seed: int, n_groups: int = 80):
    """No effect of the flag; rows of a group share an outcome tendency, and
    the flag is a property of the whole group."""
    rng = random.Random(seed)
    y, arm, clusters = [], [], []
    for g in range(n_groups):
        p = 0.9 if rng.random() < 0.5 else 0.1
        flagged = rng.random() < 0.4
        for _ in range(rng.randint(3, 9)):
            y.append(1.0 if rng.random() < p else 0.0)
            arm.append(1.0 if flagged else 0.0)
            clusters.append(g)
    return np.array(y), np.array(arm), clusters


def test_ignoring_groups_inflates_false_positives_and_clustering_restores_them():
    # The reason the group column matters: under no effect at all, treating
    # correlated rows as independent rejects far more often than 5%.
    naive = clustered = 0
    runs = 120
    for seed in range(runs):
        y, arm, clusters = _null_rows(seed)
        stat = mean_difference_statistic(y, arm)
        independent = weighted_bootstrap(stat, len(y), n_boot=300, seed=seed)
        grouped = weighted_bootstrap(stat, len(y), clusters=clusters, n_boot=300, seed=seed)
        naive += bootstrap_p_value(independent["draws"]) <= 0.05
        clustered += bootstrap_p_value(grouped["draws"]) <= 0.05
    assert naive / runs > 0.25
    assert clustered / runs < 0.12
