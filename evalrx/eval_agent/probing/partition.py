"""The EXPLORE/CONFIRM partition, derived by the diagnosis loop's own code.

A triage run must see exactly the EXPLORE cases a judge-driven run on the same
batch sees, or the two policies cannot be compared case for case. Rather than
re-implementing the stratified split (and drifting from it), this calls
``VLDiagnoseLoop._split_partitions`` with only the attributes it reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch


def explore_confirm_split(data: "CaseBatch", confirm_split: float,
                          seed: int) -> "tuple[CaseBatch, CaseBatch | None]":
    """``(explore, confirm)`` exactly as ``VLDiagnoseLoop(confirm_split=..., confirm_split_seed=seed)``."""
    from evalrx.eval_agent.loop import VLDiagnoseLoop

    view = SimpleNamespace(
        confirm_split=float(confirm_split), test_split=0.0,
        confirm_split_seed=int(seed), _strat_key=VLDiagnoseLoop._strat_key,
    )
    explore, confirm, _test = VLDiagnoseLoop._split_partitions(view, data)
    return explore, confirm
