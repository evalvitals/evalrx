"""Statistics layer — known-value checks for each primitive + the compare() entry."""

from __future__ import annotations

import pytest

from evalrx.stats import (
    StatResult,
    clustered_bootstrap_diff,
    compare,
    ebh,
    evalue_bernoulli,
    kendall_tau,
    mcnemar,
    stratified_subset,
)


# ---------------- McNemar ----------------
def test_mcnemar_discordant_counts_and_p():
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 1, 1, 1]
    mc = mcnemar(a, b)
    assert mc["b"] == 2 and mc["c"] == 1 and mc["n_discordant"] == 3
    assert mc["p_value"] == 1.0  # exact two-sided, b=2 c=1


def test_mcnemar_strong_effect_small_p():
    mc = mcnemar([0] * 10, [1] * 10)  # B right everywhere A wrong
    assert mc["b"] == 10 and mc["c"] == 0
    assert mc["p_value"] < 0.01


# ---------------- bootstrap ----------------
def test_bootstrap_effect_and_ci():
    out = clustered_bootstrap_diff([0] * 50, [1] * 50, n_boot=500, seed=0)
    assert out["effect"] == 1.0 and out["ci_low"] > 0.9


def test_bootstrap_zero_effect_ci_contains_zero():
    out = clustered_bootstrap_diff([0, 1, 0, 1], [0, 1, 0, 1], n_boot=500, seed=0)
    assert out["effect"] == 0.0 and out["ci_low"] <= 0 <= out["ci_high"]


# ---------------- e-values ----------------
def test_evalue_monotone_and_thresholds():
    e_null = evalue_bernoulli(5, 10, 0.5)    # exactly p0 -> no evidence
    e_strong = evalue_bernoulli(10, 10, 0.5)  # all favour one side
    assert e_null < 1.0
    assert e_strong > 20.0  # >> 1/0.05 -> reject
    assert e_strong > e_null


def test_ebh_rejections():
    # m=4, alpha=0.05 -> threshold m/(alpha*k) = 80/k
    assert ebh([100.0, 0.5, 30.0, 1.0], alpha=0.05) == [0]
    # two large e-values clear the 60/k threshold (m=3)
    assert ebh([100.0, 90.0, 0.1], alpha=0.05) == [0, 1]
    assert ebh([], alpha=0.05) == []


# ---------------- subset sampling ----------------
def test_kendall_tau_perfect_and_reversed():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_stratified_subset_covers_strata():
    items = list(range(20))
    sub = stratified_subset(items, key=lambda x: x % 2, n=6, seed=0)
    assert len(sub) <= 6
    assert any(x % 2 == 0 for x in sub) and any(x % 2 == 1 for x in sub)


# ---------------- compare() entry ----------------
def test_compare_strong_effect_rejects():
    r = compare([0] * 20, [1] * 20, paired=True, alpha=0.05)
    assert isinstance(r, StatResult)
    assert r.reject is True and r.effect == 1.0
    assert r.e_value is not None and r.e_value >= 1 / 0.05
    assert "B>A" in r.summary()


def test_compare_no_effect_inconclusive():
    r = compare([0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0], paired=True, alpha=0.05)
    assert r.reject is False and r.effect == 0.0


def test_compare_never_exposes_bare_p_as_decision():
    # raw p lives in details; the decision is the corrected (e-value) one.
    r = compare([0, 0, 1], [1, 1, 1], paired=True)
    assert "p_value" in r.details
    assert isinstance(r.reject, bool) and isinstance(r.underpowered, bool)


# ---------------- unpaired (two-sample) bootstrap ----------------

def test_unpaired_bootstrap_unequal_groups():
    """Regression: unpaired groups of different sizes must not crash or mis-index.

    Before the fix, clustered_bootstrap_diff resampled one shared index set,
    crashing when len(a) > len(b) and silently mis-sampling when len(a) < len(b).
    """
    from evalrx.stats import compare

    # control: 15 cases, 20% fail; signal: 3 cases, 100% fail (the VQA-RAD shape)
    control = [1, 0, 0, 0, 0] * 3
    signal = [1, 1, 1]
    r = compare(control, signal, paired=False)
    assert abs(r.effect - 0.8) < 1e-9
    assert r.ci[0] > 0  # CI excludes 0 → reject
    assert r.reject is True
    assert "unpaired" in r.method

    # reversed sizes (len(a) < len(b)) must give the mirrored effect
    r2 = compare(signal, control, paired=False)
    assert abs(r2.effect + 0.8) < 1e-9


def test_paired_bootstrap_rejects_unequal_lengths():
    from evalrx.stats.bootstrap import clustered_bootstrap_diff

    with pytest.raises(ValueError):
        clustered_bootstrap_diff([1, 0, 1], [1, 0], paired=True)
