"""Report dataclasses returned by the diagnosis loops.

Split out of ``loop.py`` so the loops (``loop.VLDiagnoseLoop`` and
``agentic.AgenticDiagnoseLoop``) can share them without a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalrx.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalrx.analysis.stats_agent import StatsAnalysisReport
    from evalrx.eval_agent.hypothesis import Hypothesis
    from evalrx.eval_agent.stages.hypothesis_tester import HypothesisTestResult


@dataclass
class AutoDiagnoseReport:
    """Summary returned by every diagnosis loop:
    :class:`~evalrx.eval_agent.loop.VLDiagnoseLoop.run` and
    :class:`~evalrx.eval_agent.agentic.AgenticDiagnoseLoop.run` both return
    this one class. Every field below is populated by both loops except
    where noted "M4-only", which stay at their default (``None`` / empty)
    when the producing loop has no equivalent concept.

    Attributes:
        cycles:               Number of M1→M4/M5 cycles executed (or agentic
                              decision steps taken).
        resolved:             ``True`` once the diagnosis is considered
                              closed — M4 found at least one supported,
                              protocol-consistent hypothesis
                              (``bool(verified_hypotheses)``).
        stopped_by:           M4-path only. Why the loop stopped: ``"criteria_met"``,
                              ``"max_cycles"``, ``"budget"``, ``"no_hypotheses"``,
                              ``"no_probe_results"``, ``"analysis_complete"``
                              (from ``run_analysis``, which proposes hypotheses
                              without confirming them) — or, for the agentic
                              loop, ``"agent_stop"`` / ``"max_actions"`` /
                              ``"time_budget"`` / ``"invalid_actions"``. ``"diagnosis_error"``
                              means M3 failed after bounded retries; it is not an empty finding.
        final_hypotheses:      All M3 proposals across every cycle.
        verified_hypotheses:  M4-only. Statistically supported, protocol-consistent
                              test results — sorted highest confidence first.
                              Feed into ``run_m5``.
        all_test_results:     M4-only. All M4 test results across every cycle.
        final_stats_report:   M2 report from the last cycle.
        fix_proposal:         Populated by ``run_m5`` when called after ``run``.
        m4_holdout:           Held-out M4 confirmation status (see field note).
        fix_outcome:          Populated by ``run_fix`` — tiered fix attempts +
                              escalation recommendation.
        store:                Accumulated results and hypotheses.
    """

    cycles: int
    resolved: bool = False
    stopped_by: "str | None" = None
    final_hypotheses: "list[Hypothesis]" = field(default_factory=list)
    verified_hypotheses: "list[HypothesisTestResult]" = field(default_factory=list)
    all_test_results: "list[HypothesisTestResult]" = field(default_factory=list)
    final_stats_report: "StatsAnalysisReport | None" = None
    fix_proposal: "Any | None" = None
    fix_outcome: "Any | None" = None
    store: Store = field(default_factory=InMemoryStore)
    #: How the held-out M4 pass resolved (``VLDiagnoseLoop`` only):
    #: ``"confirmed"`` — M4 ran on the held-out confirm split (the only M4
    #: this run: in-cycle testing is skipped when a confirm split is in play);
    #: ``"failed"`` — the confirm-split re-probe produced nothing, so no
    #: hypothesis could be verified (flagged loudly in the log);
    #: ``None`` — no confirm split / holdout disabled.
    m4_holdout: "str | None" = None
    # Internal — the run id the loop ran under
    _run_id: str = field(default="", repr=False)
