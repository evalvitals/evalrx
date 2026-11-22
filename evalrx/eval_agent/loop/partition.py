"""How a batch is divided into EXPLORE / CONFIRM / TEST partitions.

The single source of truth: every stage re-derives the identical partition
from the same input batch, so M1-M3 never see the cases M4 and the fix stage
are measured on.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch


def split_cases(
    data: "CaseBatch", *, confirm_split: float, test_split: float = 0.0, seed: int = 0,
) -> "tuple[CaseBatch, CaseBatch | None, CaseBatch | None]":
    """Deterministic, stratified (explore, confirm, test) partition.

    The single source of truth for how a batch is divided; every stage
    re-derives the identical partition from the same input batch. When
    ``confirm_split <= 0`` (or the batch is too small to split), returns
    ``(data, None, None)`` — a no-op, so existing runs are byte-for-byte
    unchanged. When ``test_split <= 0`` the third element is ``None`` (the
    two-way explore/confirm behaviour). With both fractions set, CONFIRM is
    drawn first (same seed as the two-way split), then TEST from the
    remainder (seed+1); EXPLORE is what is left.
    """
    from evalrx.core.case import CaseBatch
    from evalrx.stats.subset_sampling import stratified_subset

    cases = list(data)
    frac = confirm_split
    if frac <= 0.0 or len(cases) < 4:
        return data, None, None
    n_confirm = round(len(cases) * frac)
    if n_confirm <= 0 or n_confirm >= len(cases):
        return data, None, None
    three_way = test_split > 0.0
    confirm = stratified_subset(cases, _strat_key, n_confirm,
                                seed=seed)
    if three_way:
        # ``stratified_subset`` guarantees at MOST n (per-stratum rounding
        # can under-allocate by one or two); the train/val/test mode
        # promises exact counts, so top the draw up deterministically.
        # The two-way path is left untouched — byte-for-byte reproducible.
        confirm = _exact_size(confirm, cases, n_confirm,
                              seed)
    confirm_ids = {id(c) for c in confirm}
    rest = [c for c in cases if id(c) not in confirm_ids]
    if not three_way:
        return CaseBatch(rest), CaseBatch(confirm), None
    n_test = round(len(cases) * test_split)
    if n_test <= 0 or n_test >= len(rest):
        return CaseBatch(rest), CaseBatch(confirm), None
    test = stratified_subset(rest, _strat_key, n_test,
                             seed=seed + 1)
    test = _exact_size(test, rest, n_test, seed + 1)
    test_ids = {id(c) for c in test}
    explore = [c for c in rest if id(c) not in test_ids]
    return CaseBatch(explore), CaseBatch(confirm), CaseBatch(test)


def _strat_key(case: "Any") -> "tuple":
    """Stratify the explore/confirm split by label + probe_type when present
    (keeps the no-free-lunch control mix, e.g. present-detections, in both
    partitions). Falls back to label alone for generic batches."""
    label = getattr(getattr(case, "label", None), "value", "?")
    probe = (getattr(case, "metadata", {}) or {}).get("probe_type")
    return (label, probe)


def _exact_size(picked: "list", pool: "list", n: int, seed: int) -> "list":
    """Top a stratified draw up to exactly *n* items, deterministically.

    :func:`~evalrx.stats.subset_sampling.stratified_subset` returns at most
    *n* items but per-stratum rounding can under-allocate by one or two; the
    train/val/test split promises exact 1:1:1 counts, so the shortfall is
    filled from a seeded shuffle of the untaken remainder of *pool*.
    """
    if len(picked) >= n:
        return picked[:n]
    picked_ids = {id(c) for c in picked}
    leftover = [c for c in pool if id(c) not in picked_ids]
    random.Random(seed + 7919).shuffle(leftover)
    return picked + leftover[: n - len(picked)]
