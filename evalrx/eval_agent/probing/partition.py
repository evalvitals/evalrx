"""The EXPLORE/CONFIRM partition, derived by the diagnosis loop's own code.

A triage run must see exactly the EXPLORE cases a judge-driven run on the same
batch sees, or the two policies cannot be compared case for case. Rather than
re-implementing the stratified split (and drifting from it), this calls
``VLDiagnoseLoop._split_partitions`` with only the attributes it reads.

``by="media"`` is the triage protocol's own variant: the cases are grouped by
their media content and whole groups go to one side, so DISCOVERY and CONFIRM
never share an image and :func:`~evalrx.eval_agent.probing.validation.partition_bundle`
leaves nothing out (under the per-case split 207 of ChartQA's 500 CONFIRM cases
shared a chart with DISCOVERY and were dropped). A judge-driven run still splits
by case, so the two policies are compared case for case under ``by="case"`` only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

#: What the explore/confirm draw is over: single cases (the loop's split) or media groups.
SPLIT_BY = ("case", "media")


def explore_confirm_split(data: "CaseBatch", confirm_split: float, seed: int, *,
                          by: str = "case") -> "tuple[CaseBatch, CaseBatch | None]":
    """``(explore, confirm)`` exactly as ``VLDiagnoseLoop(confirm_split=..., confirm_split_seed=seed)``;
    with ``by="media"`` the same stratified draw over media groups (:func:`split_by_media`)."""
    if by not in SPLIT_BY:
        raise ValueError(f"by must be one of {SPLIT_BY}, not {by!r}")
    if by == "media":
        return split_by_media(data, float(confirm_split), int(seed))
    from evalrx.eval_agent.loop import VLDiagnoseLoop

    view = SimpleNamespace(
        confirm_split=float(confirm_split), test_split=0.0,
        confirm_split_seed=int(seed), _strat_key=VLDiagnoseLoop._strat_key,
    )
    explore, confirm, _test = VLDiagnoseLoop._split_partitions(view, data)
    return explore, confirm


def split_by_media(data: "CaseBatch", confirm_split: float, seed: int) -> "tuple[CaseBatch, CaseBatch | None]":
    """Whole media groups (cases on the same image, by content) go to one side. The groups are drawn
    stratified by the labels they hold (a chart with one FAIL and one PASS question is its own
    stratum), so each side keeps the batch's FAIL share. The loop's thresholds hold: no split under
    4 cases or when a side would be empty (then ``(data, None)``)."""
    from evalrx.core.case import CaseBatch
    from evalrx.eval_agent.loop import VLDiagnoseLoop
    from evalrx.eval_agent.probing.validation import source_key
    from evalrx.stats.subset_sampling import stratified_subset

    cases = list(data)
    n_confirm = round(len(cases) * confirm_split)
    if confirm_split <= 0.0 or len(cases) < 4 or n_confirm <= 0 or n_confirm >= len(cases):
        return data, None
    groups: dict[str, list] = {}
    for c in cases:
        groups.setdefault(source_key(c), []).append(c)
    members = list(groups.values())                      # first-appearance order: deterministic
    n_groups = max(1, min(len(members) - 1, round(len(members) * confirm_split)))

    def stratum(group: list) -> tuple:
        return tuple(sorted(repr(VLDiagnoseLoop._strat_key(c)) for c in group))

    chosen = {id(c) for g in stratified_subset(members, stratum, n_groups, seed=seed) for c in g}
    confirm = [c for c in cases if id(c) in chosen]
    explore = [c for c in cases if id(c) not in chosen]
    if not confirm or not explore:
        return data, None
    return CaseBatch(explore), CaseBatch(confirm)
