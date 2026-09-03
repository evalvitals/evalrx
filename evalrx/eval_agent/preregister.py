"""Pre-registration + data splitting — the selective-inference safety machine.

The closed loop (mine results → propose hypothesis → test) is textbook data
dredging if the hypothesis is tested on the same data that suggested it.  This
module enforces the discipline that makes the loop's verdicts trustworthy:

  * :class:`DataSplit` deterministically partitions cases into **explore /
    validate / confirm** (by hash of case id) — mine freely on explore, test a
    pre-registered hypothesis once on validate, lock confirm for the final report.
  * :class:`PreregisteredHypothesis` is a falsifiable, operationalised contract
    {predicate, metric, direction, test, alpha, min_effect, split} fixed BEFORE
    unblinding the test split.
  * :class:`PreregistrationLog` hashes + timestamps the contract so a report can
    prove the hypothesis preceded the test (and counts the denominator for FDR).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Optional


class Split(str, Enum):
    EXPLORE = "explore"
    VALIDATE = "validate"
    CONFIRM = "confirm"


@dataclass
class DataSplit:
    """Deterministic explore/validate/confirm assignment by hash of case id."""

    explore_frac: float = 0.5
    validate_frac: float = 0.3
    seed: int = 0

    def assign(self, case_id: str) -> Split:
        h = int(hashlib.sha1(f"{self.seed}:{case_id}".encode()).hexdigest(), 16)
        u = (h % 1_000_000) / 1_000_000
        if u < self.explore_frac:
            return Split.EXPLORE
        if u < self.explore_frac + self.validate_frac:
            return Split.VALIDATE
        return Split.CONFIRM

    def partition(self, cases: Iterable) -> dict:
        out: dict = {Split.EXPLORE: [], Split.VALIDATE: [], Split.CONFIRM: []}
        for c in cases:
            cid = getattr(c, "id", str(c))
            out[self.assign(cid)].append(c)
        return out


@dataclass(frozen=True)
class PreregisteredHypothesis:
    """A falsifiable contract fixed before unblinding the test split."""

    predicate: str                     # operationalised filter over cases (what subset)
    statement: str = ""                # human-readable claim
    metric: str = "success"
    direction: str = "B>A"             # expected effect sign
    test: str = "mcnemar+evalue"
    alpha: float = 0.05
    min_effect: float = 0.0
    split: str = Split.VALIDATE.value

    def contract_hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class PreregistrationLog:
    """Append-only proof that hypotheses were registered before testing."""

    entries: list = field(default_factory=list)

    def register(self, hyp: PreregisteredHypothesis, timestamp: Optional[str] = None) -> str:
        if timestamp is None:
            from datetime import datetime

            timestamp = datetime.now().isoformat(timespec="seconds")
        h = hyp.contract_hash()
        self.entries.append({"hash": h, "timestamp": timestamp, "contract": asdict(hyp)})
        return h

    def is_registered(self, hyp: PreregisteredHypothesis) -> bool:
        return any(e["hash"] == hyp.contract_hash() for e in self.entries)

    def __len__(self) -> int:
        return len(self.entries)
