"""Friedman omnibus + Nemenyi post-hoc (>2 strategies) + chi-square SF."""

from __future__ import annotations

import pytest

from evalrx.stats import (
    chi2_sf,
    compare_multiple,
    friedman_test,
    nemenyi_cd,
)


def test_chi2_sf_known_values():
    assert abs(chi2_sf(3.841, 1) - 0.05) < 0.003   # 1-df 95% critical value
    assert abs(chi2_sf(5.991, 2) - 0.05) < 0.003   # 2-df 95% critical value
    assert chi2_sf(0.0, 2) == 1.0
    assert chi2_sf(20.0, 2) < 1e-3


def test_friedman_clear_ordering():
    fr = friedman_test({"A": [1.0] * 10, "B": [0.5] * 10, "C": [0.0] * 10})
    assert fr["avg_ranks"] == [1.0, 2.0, 3.0]      # A best -> rank 1
    assert abs(fr["statistic"] - 20.0) < 1e-6 and fr["df"] == 2
    assert fr["p_value"] < 1e-3


def test_nemenyi_cd_value():
    assert abs(nemenyi_cd(3, 10, 0.05) - 1.047) < 0.01


def test_compare_multiple_posthoc():
    r = compare_multiple({"A": [1.0] * 10, "B": [0.5] * 10, "C": [0.0] * 10}, alpha=0.05)
    assert r.reject_global is True
    assert r.avg_ranks["A"] < r.avg_ranks["B"] < r.avg_ranks["C"]
    pairs = {(p["a"], p["b"]) for p in r.significant_pairs}
    assert ("A", "C") in pairs            # rank diff 2 >= CD(~1.05)
    assert ("A", "B") not in pairs        # rank diff 1 < CD
    assert "differ" in r.summary()


def test_compare_multiple_rejects_two_strategies():
    with pytest.raises(ValueError):
        compare_multiple({"A": [1, 0], "B": [0, 1]})
