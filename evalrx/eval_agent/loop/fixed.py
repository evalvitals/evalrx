"""The default next-step policy: M1→M2→M3(→M4) cycles in a fixed order.

The loop (:class:`~evalrx.eval_agent.loop.diagnose.VLDiagnoseLoop`) owns the
stages, the run log and the confirm split; a policy only decides the order.
The alternative is :class:`~evalrx.eval_agent.loop.judge.policy.JudgePolicy`,
which asks a judge one tool call at a time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from evalrx.eval_agent.loop.report import (
    STOPPED_BY_BUDGET,
    STOPPED_BY_CRITERIA,
    STOPPED_BY_DIAG_ERROR,
    STOPPED_BY_MAX,
    STOPPED_BY_NO_HYPS,
    STOPPED_BY_NO_PROBE,
    AutoDiagnoseReport,
)
from evalrx.eval_agent.loop.state import RunState, cycle_summary

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.eval_agent.loop.diagnose import VLDiagnoseLoop

logger = logging.getLogger(__name__)


class FixedOrderPolicy:
    """M1→M2→M3(→M4) cycles until a stopping rule fires (the default policy).

    ``max_cycles``, ``token_budget``, ``analysis_only`` and ``m4_holdout`` are
    read from the loop.
    """

    def run(self, loop: "VLDiagnoseLoop", data: "CaseBatch",
            timings: dict[str, float]) -> AutoDiagnoseReport:
        # Held-out CONFIRM split (leak #3): M1-M4 see only EXPLORE; the post-loop
        # fix/surgery validate on the frozen CONFIRM partition (run_m5/run_fix
        # re-derive the same deterministic split from the same input batch).
        # With test_split > 0 (train/val/test mode) a third partition is also
        # reserved here: CONFIRM plays validation (M4 holdout + fix candidate
        # development) and TEST is only ever touched by the final frozen-fix
        # gate in run_fix.
        explore, confirm, test = loop._split_partitions(data)
        original_data = data
        if confirm is not None:
            loop._log_partitions(explore, confirm, test)
            data = explore

        # With held-out data (and m4_holdout on), M4 runs ONCE after the loop,
        # on that data — the cycles only mine (M1→M3). Without any, M4 stays
        # in-cycle.
        holdout_target = loop._holdout_target(confirm)
        loop._start_run_log(data, "VLDiagnoseLoop", confirm=confirm, test=test, val=loop.val_data)

        state = RunState(original_data=original_data, data=data, stages=loop.stages(),
                         timings=timings)
        stopped_by = STOPPED_BY_MAX
        cycles = 0
        for cycle in range(loop.max_cycles):
            cycles = cycle + 1
            stop = self._cycle(loop, cycle, state, holdout=holdout_target is not None)
            if stop is not None:
                stopped_by = stop
                break

        # The M4 pass: with a confirm split in play the hypotheses are tested
        # ONCE, on the held-out split — the same discipline run_fix already
        # applies to candidates. Without one, the in-cycle results stand.
        m4_holdout_status: "str | None" = None
        if holdout_target is not None and state.all_hypotheses:
            state.all_test_results, m4_holdout_status = state.stages.test_on_holdout(
                state.all_hypotheses, holdout_target, state.analyzer_names, timings,
            )
        verified = loop.hypothesis_tester.best_hypotheses(state.all_test_results)

        return AutoDiagnoseReport(
            cycles=cycles,
            resolved=bool(verified),
            stopped_by=stopped_by,
            verified_hypotheses=verified,
            final_hypotheses=state.all_hypotheses,
            all_test_results=state.all_test_results,
            final_stats_report=state.stats_report,
            store=loop.store,
            m4_holdout=m4_holdout_status,
            _run_id=loop._run_id,
        )

    @staticmethod
    def _cycle(loop: "VLDiagnoseLoop", cycle: int, state: RunState, *, holdout: bool) -> "str | None":
        """One M1→M2→M3(→M4) cycle. Returns why the loop stops, or None to go on.

        With *holdout*, M4 is deferred to the single held-out pass after the
        loop: testing here would re-use the explore data the hypotheses were
        mined from, and its verdict must not steer the loop (no early stop /
        no extra cycles keyed on SUPPORTED-or-not).
        """
        data, timings = state.data, state.timings
        state.cycle = cycle
        stages = state.stages
        if loop.token_budget > 0 and stages.memory.tokens_used >= loop.token_budget:
            logger.warning("Token budget %d exhausted after %d cycles", loop.token_budget, cycle)
            return STOPPED_BY_BUDGET
        if loop.run_logger is not None:
            loop.run_logger.current_cycle = cycle

        # ── M1: protocol-guided probing (+ signal bridge) ─────────
        probe_results, artifact_pngs = stages.probe(cycle, data, state.all_hypotheses, timings)
        if not probe_results:
            logger.info("M1 produced no probe results — stopping.")
            return STOPPED_BY_NO_PROBE
        state.analyzer_names = list(probe_results)

        # ── explore (optional): free-form EDA on M1's table → M3 notes ──
        stages.explore(cycle, probe_results, data, timings)

        # ── M2: protocol-aware stats analysis ────────────────────
        stats_report = stages.analyze(cycle, probe_results, data, artifact_pngs, timings)
        state.stats_report = stats_report
        if loop.analysis_only:
            return STOPPED_BY_NO_HYPS

        # ── M3: hypothesis generation ("AI scientist") ────────────
        diag = stages.diagnose(cycle, stats_report, state.prior_cycles, timings, cases=data)
        if diag is None:
            state.diagnosis_failed = True
            return STOPPED_BY_DIAG_ERROR
        if not diag.hypotheses:
            logger.info("M3 produced no hypotheses at cycle %d.", cycle)
            return STOPPED_BY_NO_HYPS
        for h in diag.hypotheses:
            loop.store.add_hypothesis(h)
        state.all_hypotheses.extend(diag.hypotheses)

        # ── M4: hypothesis testing (stats + protocol consistency) ─
        if not holdout:
            test_results = stages.test_hypotheses(cycle, diag.hypotheses, stats_report, data, timings)
            state.all_test_results.extend(test_results)
            if loop.hypothesis_tester.stopping_criteria_met(test_results, loop.protocol):
                logger.info(
                    "Stopping criteria met at cycle %d: verified, protocol-consistent "
                    "hypothesis found.",
                    cycle,
                )
                return STOPPED_BY_CRITERIA

        state.prior_cycles.append(cycle_summary(cycle, stats_report, diag.hypotheses))
        return None
