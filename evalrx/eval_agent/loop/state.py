"""What a diagnosis run carries between steps, whichever policy drives it.

:class:`Policy` is the contract every next-step policy meets
(:class:`~evalrx.eval_agent.loop.fixed.FixedOrderPolicy`,
:class:`~evalrx.eval_agent.loop.judge.policy.JudgePolicy`); :class:`RunState`
is the bookkeeping they share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.eval_agent.loop.diagnose import VLDiagnoseLoop
    from evalrx.eval_agent.loop.report import AutoDiagnoseReport


class Policy(Protocol):
    def run(self, loop: "VLDiagnoseLoop", data: "CaseBatch",
            timings: dict[str, float]) -> "AutoDiagnoseReport": ...


@dataclass
class RunState:
    """What one run accumulates, whichever policy drives it."""

    original_data: Any
    data: Any
    timings: dict[str, float] = field(default_factory=dict)
    cycle: int = 0
    all_hypotheses: list[Any] = field(default_factory=list)
    all_test_results: list[Any] = field(default_factory=list)
    #: Each finished cycle's summary, M3's context for the next one.
    prior_cycles: list[dict[str, Any]] = field(default_factory=list)
    stats_report: Any = None
    diagnosis_failed: bool = False
    #: The last M1's analyzers — what the held-out M4 pass re-runs.
    analyzer_names: list[str] = field(default_factory=list)
    # Judge-decided runs only: the decision judge and the tools' working state.
    judge: Any = None
    pending_hypotheses: list[Any] = field(default_factory=list)
    probe_results: dict[str, Any] = field(default_factory=dict)
    artifact_pngs: list[Any] = field(default_factory=list)
    failure_modes: Any = None
    probe_search_result: Any = None
    surgery_outcome: Any = None
    fix_outcome: Any = None


def cycle_summary(cycle: int, stats_report: Any, hypotheses: list) -> dict[str, Any]:
    """One finished cycle as the next M3 call sees it."""
    return {
        "cycle": cycle,
        "severity": getattr(stats_report, "severity", "none"),
        "hypotheses": [
            {
                "statement": h.statement,
                "failure_mode": h.predicted_failure_mode,
                "status": h.status.value if h.status else "pending",
            }
            for h in hypotheses
        ],
    }
