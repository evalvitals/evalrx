"""Matched, nested FAIL/PASS samples for triage actions.

``ProbeAgent._cap_cases`` subsamples each label proportionally by even stride:
fine as a cost backstop, but a larger cap does not contain a smaller one and
the PASS cases are not matched to the FAILs. Triage needs both:

* **matched** — each FAIL is paired with a PASS from the same stratum (e.g.
  the same gold answer on a yes/no task), so a FAIL-vs-PASS contrast is not a
  gold-answer contrast in disguise;
* **nested** — every case gets a fixed rank from ``sha1(seed:id)``; a draw
  takes rank prefixes, so the 8-FAIL sample is contained in the 16-FAIL one
  and an extension reuses every earlier observation.

A stratum with too few PASS cases is reported as a shortfall; nothing is
borrowed from another stratum to fake balance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch


def gold_stratum(case: Any) -> str:
    """Stratum = the normalised gold answer (for yes/no and letter tasks)."""
    meta = getattr(case, "metadata", None) or {}
    gold = meta.get("gold", getattr(case, "expected", None))
    if isinstance(gold, (list, tuple)):
        gold = "|".join(str(g) for g in gold)
    return str(gold).strip().lower()


def single_stratum(case: Any) -> str:
    return "all"


STRATA: dict[str, Callable[[Any], str]] = {"gold": gold_stratum, "none": single_stratum}


def rank(seed: int, case_id: str) -> str:
    """The case's fixed place in every draw with this seed (draws take rank prefixes)."""
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


_rank = rank


@dataclass(frozen=True)
class SampleDraw:
    """Which cases a draw took and why; ``case_ids`` keeps the batch order."""

    case_ids: tuple[str, ...]
    fail_ids: tuple[str, ...]
    pass_ids: tuple[str, ...]
    strata: dict[str, dict[str, int]]
    shortfall: dict[str, int]
    seed: int
    stratum_rule: str
    n_fail_requested: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_ids": list(self.case_ids), "fail_ids": list(self.fail_ids),
            "pass_ids": list(self.pass_ids), "strata": self.strata,
            "shortfall": self.shortfall, "seed": self.seed,
            "stratum_rule": self.stratum_rule, "n_fail_requested": self.n_fail_requested,
            "rule": ("FAIL: first n by sha1(seed:id); PASS: per stratum, as many as the "
                     "chosen FAILs in that stratum, first by the same rank; no cross-stratum fill"),
        }


def matched_sample(cases: "CaseBatch", n_fail: int, *, seed: int,
                   stratum: str = "gold") -> "tuple[CaseBatch, SampleDraw]":
    """Draw up to *n_fail* FAIL cases and their stratum-matched PASS controls."""
    from evalrx.core.case import CaseBatch, Label

    key = STRATA[stratum]
    pool = list(cases)
    fails = sorted((c for c in pool if c.label == Label.FAIL), key=lambda c: _rank(seed, str(c.id)))
    passes = sorted((c for c in pool if c.label == Label.PASS), key=lambda c: _rank(seed, str(c.id)))
    chosen_fail = fails[:max(0, int(n_fail))]

    need: dict[str, int] = {}
    for c in chosen_fail:
        need[key(c)] = need.get(key(c), 0) + 1
    by_stratum: dict[str, list] = {}
    for c in passes:
        by_stratum.setdefault(key(c), []).append(c)

    chosen_pass: list = []
    strata: dict[str, dict[str, int]] = {}
    shortfall: dict[str, int] = {}
    for s in sorted(need):
        available = by_stratum.get(s, [])
        take = available[:need[s]]
        chosen_pass.extend(take)
        strata[s] = {"fail": need[s], "pass": len(take), "pass_available": len(available)}
        if len(take) < need[s]:
            shortfall[s] = need[s] - len(take)

    chosen = {id(c) for c in chosen_fail} | {id(c) for c in chosen_pass}
    ordered = [c for c in pool if id(c) in chosen]  # batch order, not rank order
    draw = SampleDraw(
        case_ids=tuple(str(c.id) for c in ordered),
        fail_ids=tuple(str(c.id) for c in chosen_fail),
        pass_ids=tuple(str(c.id) for c in chosen_pass),
        strata=strata, shortfall=shortfall, seed=int(seed),
        stratum_rule=stratum, n_fail_requested=int(n_fail),
    )
    return CaseBatch(ordered), draw
