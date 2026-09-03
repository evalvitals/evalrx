"""Closed-loop orchestrator: define → split → pre-register → test → report.

Implements the selective-inference-safe path:
  1. partition cases into explore/validate/confirm (DataSplit),
  2. register the hypothesis (hash + timestamp) BEFORE unblinding its test split,
  3. run the A/B on that split,
  4. emit a DiagnosticReport carrying the pre-registration proof.

The exploratory generation that *produces* hypotheses (LLM-backed) lives in the
SelfEvolveLoop; this orchestrator enforces the discipline around testing them.
"""

from __future__ import annotations

from typing import Callable, Optional

from evalrx.core.experiment import ExperimentRunner
from evalrx.eval_agent.ab_runner import ABRunner
from evalrx.eval_agent.legacy import SelfEvolveLoop
from evalrx.eval_agent.preregister import (
    DataSplit,
    PreregisteredHypothesis,
    PreregistrationLog,
    Split,
)
from evalrx.eval_agent.report import DiagnosticReport


class EvalOrchestrator:
    """Drive a pre-registered A/B comparison end-to-end."""

    def __init__(self, loop: Optional[SelfEvolveLoop] = None, split: Optional[DataSplit] = None) -> None:
        self.loop = loop or SelfEvolveLoop()
        self.runner: ExperimentRunner = self.loop.runner
        self.split = split or DataSplit()
        self.prereg_log = PreregistrationLog()

    def run(
        self,
        cases,
        hypothesis: PreregisteredHypothesis,
        strategy_a: Callable,
        strategy_b: Callable,
        *,
        cluster_fn: Optional[Callable] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """Register *hypothesis*, test it on its split, return a DiagnosticReport dict."""
        parts = self.split.partition(cases)
        prereg_hash = self.prereg_log.register(hypothesis, timestamp)  # BEFORE unblinding
        test_cases = parts[Split(hypothesis.split)]
        ab = ABRunner(strategy_a, strategy_b, cluster_fn=cluster_fn).run(
            test_cases, alpha=hypothesis.alpha, min_effect=hypothesis.min_effect
        )
        return DiagnosticReport.from_comparison(
            ab.stat,
            title=hypothesis.statement or hypothesis.predicate,
            hypothesis=hypothesis,
            prereg_hash=prereg_hash,
            split=hypothesis.split,
        )
