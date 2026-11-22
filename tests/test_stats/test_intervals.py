"""Closed-form quantiles and intervals (``evalrx.stats.intervals``) against known values."""

from __future__ import annotations

import pytest

from evalrx.stats.intervals import (
    average_ranks,
    newcombe_interval,
    t_quantile,
    wilson_interval,
    within_group_spearman,
    z_quantile,
)


def test_normal_and_t_quantiles_match_the_tables():
    assert z_quantile(0.95) == pytest.approx(1.95996, abs=1e-4)
    assert z_quantile(0.99) == pytest.approx(2.57583, abs=1e-4)
    assert t_quantile(0.95, 10) == pytest.approx(2.228, abs=1e-3)      # the 95% table
    assert t_quantile(0.99, 10) == pytest.approx(3.1693, abs=1e-3)     # the inverted CDF
    assert t_quantile(0.999, 4) == pytest.approx(8.6103, abs=1e-3)
    assert t_quantile(0.99, 10.9) == t_quantile(0.99, 10)              # floor of the degrees of freedom
    assert t_quantile(0.99, 5) > t_quantile(0.99, 50) > z_quantile(0.99)


def test_wilson_and_newcombe_intervals():
    lo, hi = wilson_interval(5, 52)
    assert (lo, hi) == pytest.approx((0.0418, 0.2062), abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == pytest.approx(1.0)
    lo, hi = newcombe_interval(3, 8, 0, 8)
    assert lo < 0 < hi and newcombe_interval(13, 32, 1, 32)[0] > 0
    assert newcombe_interval(1, 0, 0, 4) is None                       # an empty side has no rate
    wide, narrow = newcombe_interval(13, 32, 1, 32, z=z_quantile(0.99)), newcombe_interval(13, 32, 1, 32)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


def test_ranks_average_ties_and_the_correlation_ignores_a_shift_between_groups():
    assert average_ranks([3.0, 1.0, 3.0, 2.0]) == [2.5, 0.0, 2.5, 1.0]
    a = {f"f{i}": float(i) for i in range(6)} | {f"c{i}": 100.0 + i for i in range(6)}
    same = {k: v * 2 for k, v in a.items()}
    flipped = {k: -v for k, v in a.items()}
    groups = [[f"f{i}" for i in range(6)], [f"c{i}" for i in range(6)]]
    assert within_group_spearman(a, same, groups) == pytest.approx(1.0)
    assert within_group_spearman(a, flipped, groups) == pytest.approx(-1.0)
    # the two groups sit 100 apart in a and at the same level in b: pooled ranks would correlate, these do not
    b = {**{f"f{i}": float(5 - i) for i in range(6)}, **{f"c{i}": float(5 - i) for i in range(6)}}
    assert within_group_spearman(a, b, groups) == pytest.approx(-1.0)
    assert within_group_spearman(a, same, groups, min_pairs=7) is None          # no group has seven pairs
    assert within_group_spearman(a, {k: 1.0 for k in a}, groups) is None        # a constant reading
