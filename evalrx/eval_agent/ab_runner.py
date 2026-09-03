"""A/B execution across two strategies, with statistics done right.

A *strategy* is a callable ``FailureCase -> bool`` (did this strategy succeed on
this case) — typically wrapping a model + a prompt transform + a verifier.  The
runner produces paired per-example success vectors and hands them to
``stats.compare`` (McNemar + e-value + clustered bootstrap CI), so you get an
effect size + corrected decision, never a bare p.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from evalrx.stats import StatResult, compare


@dataclass
class ABResult:
    success_a: list
    success_b: list
    stat: StatResult
    n: int


class ABRunner:
    """Run two strategies over a CaseBatch and compare them statistically."""

    def __init__(
        self,
        strategy_a: Callable,
        strategy_b: Callable,
        *,
        cluster_fn: Optional[Callable] = None,  # case -> cluster id (e.g. task); default per-example
        label_a: str = "A",
        label_b: str = "B",
    ) -> None:
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.cluster_fn = cluster_fn
        self.label_a = label_a
        self.label_b = label_b

    def run(self, cases, *, paired: bool = True, alpha: float = 0.05, min_effect: float = 0.0) -> ABResult:
        cases = list(cases)
        sa = [bool(self.strategy_a(c)) for c in cases]
        sb = [bool(self.strategy_b(c)) for c in cases]
        clusters = [self.cluster_fn(c) for c in cases] if self.cluster_fn else None
        stat = compare(sa, sb, paired=paired, alpha=alpha, min_effect=min_effect, cluster_by=clusters)
        return ABResult(success_a=sa, success_b=sb, stat=stat, n=len(cases))
