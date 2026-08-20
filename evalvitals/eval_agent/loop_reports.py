"""Report dataclasses returned by the diagnosis loops.

Split out of ``loop.py`` so the loop/legacy modules can share them without a
circular import (``legacy.AutoDiagnoseLoop`` and ``loop.VLDiagnoseLoop`` both
import from here; neither imports the other).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.analysis.analysis_module import AnalysisReport
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.hypothesis import Hypothesis
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult


@dataclass
class AutoDiagnoseReport:
    """Summary returned by every diagnosis loop: :class:`~evalvitals.eval_agent.legacy.AutoDiagnoseLoop.run`,
    :class:`~evalvitals.eval_agent.loop.VLDiagnoseLoop.run`, and
    :class:`~evalvitals.eval_agent.agentic.AgenticDiagnoseLoop.run` all return
    this one class — ``VLDiagnoseReport`` below is a back-compat alias for
    it, not a separate type, so ``isinstance(report, VLDiagnoseReport)``
    holds for a report from *any* of the three loops.

    Unified 2026-08 from two previously-separate dataclasses
    (``AutoDiagnoseReport`` / ``VLDiagnoseReport``) that carried the same
    information under different field names. Every field below is populated
    by all three loops except where noted "Auto-only" / "M5-only", which stay
    at their default (``None`` / empty) when the producing loop has no
    equivalent concept.

    Attributes:
        cycles:               Number of M1→M4/M5 cycles executed (or agentic
                              decision steps taken).
        resolved:             ``True`` once the diagnosis is considered
                              closed — M4 surgery confirmed a fix (legacy
                              loop), or M5 found at least one supported,
                              protocol-consistent hypothesis (``bool(verified_hypotheses)``,
                              current/agentic loops).
        stopped_by:           M5-path only. Why the loop stopped: ``"criteria_met"``,
                              ``"max_cycles"``, ``"budget"``, ``"no_hypotheses"``,
                              ``"no_probe_results"``, ``"analysis_complete"``
                              (from ``run_analysis``, which proposes hypotheses
                              without confirming them) — or, for the agentic
                              loop, ``"agent_stop"`` / ``"max_actions"`` /
                              ``"time_budget"`` / ``"invalid_actions"``. ``None``
                              for the legacy M4-per-cycle loop, which has no
                              single stopping-reason concept.
        final_hypotheses:      All M3 proposals across every cycle. (Formerly
                              ``all_hypotheses`` on the VL-shaped report —
                              ``all_hypotheses`` is now a read-only alias
                              for this field, kept for existing callers.)
        verified_hypotheses:  M5-only. Statistically supported, protocol-consistent
                              test results — sorted highest confidence first.
                              Feed into ``run_m4``.
        all_test_results:     M5-only. All M5 test results across every cycle.
        final_results:        Auto-only. Raw analyzer results from the last M1 probe.
        final_analysis:       Structured M2 report from the last cycle. Mirrors
                              ``final_stats_report`` when only that was set
                              (``StatsAnalysisReport`` is an ``AnalysisReport``
                              subclass), so this field works for every loop.
        final_stats_report:   M2 report from the last cycle (M5-path shape;
                              same object as ``final_analysis`` when set).
        fix_proposal:         Populated by ``run_m4`` when called after ``run``.
        m5_holdout:           Held-out M5 confirmation status (see field note).
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
    final_results: "dict[str, Result]" = field(default_factory=dict)
    final_analysis: "AnalysisReport | None" = None
    final_stats_report: "StatsAnalysisReport | None" = None
    fix_proposal: "Any | None" = None
    fix_outcome: "Any | None" = None
    store: Store = field(default_factory=InMemoryStore)
    #: How M5's held-out confirmation resolved (``VLDiagnoseLoop`` only):
    #: ``"confirmed"`` — verified_hypotheses passed on the held-out split;
    #: ``"nothing_screened"`` — no hypothesis screened SUPPORTED in-sample;
    #: ``"failed"`` — the confirm-split re-probe produced nothing, so the
    #: verified list is IN-SAMPLE ONLY (flagged loudly in the log);
    #: ``None`` — no confirm split / holdout disabled / legacy loop.
    m5_holdout: "str | None" = None
    # Internal — set by the loops for evolution/git integration
    _run_id: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        # StatsAnalysisReport IS-A AnalysisReport, so a report built from the
        # M5 path can satisfy readers that only know the Auto-shaped field.
        if self.final_analysis is None and self.final_stats_report is not None:
            self.final_analysis = self.final_stats_report

    @property
    def all_hypotheses(self) -> "list[Any]":
        """Read-only back-compat alias for :attr:`final_hypotheses`."""
        return self.final_hypotheses


# Back-compat alias — the two report shapes were merged into one class (see
# AutoDiagnoseReport's docstring). Keep the name importable for existing
# code; it is literally the same class, so isinstance checks against either
# name succeed for a report from any loop.
VLDiagnoseReport = AutoDiagnoseReport
