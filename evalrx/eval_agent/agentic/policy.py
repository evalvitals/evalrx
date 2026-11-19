"""JudgePolicy — a judge picks each next stage of a VLDiagnoseLoop run.

``VLDiagnoseLoop`` runs M1->M2->M3->M4 in a fixed cycle. Given
``policy=JudgePolicy(judge)``, the same loop instead asks the judge, one tool
call at a time, what to do next — probe, run stats, explore the raw data,
propose hypotheses, test one, fix, or stop. The stages, the confirm split and
the post-loop run_m5/run_fix discipline stay the loop's own. The host, not the
judge, enforces call caps, tool preconditions, and the pre-registered stopping
discipline (no declaring success without a tested, supported,
protocol-consistent hypothesis).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evalrx.eval_agent.agentic.actions import decide
from evalrx.eval_agent.agentic.board import BudgetState, EvidenceBoard
from evalrx.eval_agent.agentic.tools import ToolRegistry, _RunState, build_default_registry
from evalrx.eval_agent.loop_reports import AutoDiagnoseReport
from evalrx.eval_agent.run_metadata import _data_provenance

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.eval_agent.loop import VLDiagnoseLoop

logger = logging.getLogger(__name__)

_MAX_INVALID_STREAK = 3


def _protocol_summary(protocol: "Any | None") -> str:
    if protocol is None:
        return ""
    bits = [f"description: {getattr(protocol, 'description', '')}"]
    domain = getattr(protocol, "task_domain", None)
    if domain:
        bits.append(f"task_domain: {domain}")
    patterns = getattr(protocol, "failure_patterns", None)
    if patterns:
        bits.append(f"failure_patterns: {patterns}")
    return "\n".join(bits)


class JudgePolicy:
    """Next-step policy for :class:`~evalrx.eval_agent.loop.VLDiagnoseLoop`.

    Args:
        judge:            Judge that decides the next action each turn (e.g.
                          ``ClaudeModel()`` / ``AgyModel()``).
        max_actions:      Hard cap on decision turns (default 12).
        time_budget_sec:  Wall-clock cap for the whole run (0 = unlimited).
        registry_factory: ``(loop, run_state) -> ToolRegistry`` in place of
                          :func:`build_default_registry` (e.g. to withhold
                          the repair tools).

    The loop's ``explorer`` (when set) becomes the ``explore_data`` tool
    instead of running in every cycle.
    """

    def __init__(
        self,
        judge: Any,
        *,
        max_actions: int = 12,
        time_budget_sec: float = 0.0,
        registry_factory: "Any | None" = None,
    ) -> None:
        if judge is None:
            raise ValueError("JudgePolicy needs a decision judge, e.g. ClaudeModel() or AgyModel().")
        self.judge = judge
        self.max_actions = max_actions
        self.time_budget_sec = time_budget_sec
        self.registry_factory = registry_factory

    def run(self, loop: "VLDiagnoseLoop", data: "CaseBatch") -> AutoDiagnoseReport:
        """Drive the judge-decided action loop to a stop or a budget limit."""
        loop._tokens_used = 0
        timings: dict[str, float] = {}

        explore, confirm = loop._split_explore_confirm(data)
        original_data = data
        if confirm is not None:
            data = explore

        # run_start's judge is the M3 *stage* judge; the decision judge is
        # recorded separately so the report shows who is actually deciding.
        loop._start_run_log(
            data, "JudgePolicy", confirm=confirm,
            extra={"decision_judge": repr(self.judge), "max_actions": self.max_actions},
        )

        board = EvidenceBoard(
            protocol_summary=_protocol_summary(loop.protocol),
            data_summary=_data_provenance(data),
            budget=BudgetState(
                max_actions=self.max_actions,
                token_budget=loop.token_budget,
                time_budget_sec=self.time_budget_sec,
            ),
        )
        board.budget.start()

        state = _RunState(original_data=original_data, data=data, timings=timings,
                          judge=self.judge)
        registry: ToolRegistry = (
            self.registry_factory(loop, state)
            if self.registry_factory is not None
            else build_default_registry(loop, state)
        )

        stopped_by = "max_actions"
        invalid_streak = 0
        step = 0
        while True:
            reason = board.budget.exhausted()
            if reason:
                stopped_by = reason
                break

            action = decide(self.judge, board, registry, run_logger=loop.run_logger, step=step)
            board.budget.actions_taken += 1
            invalid_streak = 0 if action.valid else invalid_streak + 1

            outcome = registry.dispatch(action, board)
            board.action_log.append({
                "step": step, "tool": action.tool, "params": action.params,
                "ok": outcome.ok, "summary": outcome.summary,
            })
            if loop.run_logger is not None:
                loop.run_logger.log_agent_tool(
                    step, tool=action.tool, ok=outcome.ok,
                    summary=outcome.summary, error=outcome.error,
                )

            if action.tool == "stop" and outcome.ok:
                stopped_by = "agent_stop"
                step += 1
                break

            if invalid_streak >= _MAX_INVALID_STREAK:
                logger.warning(
                    "JudgePolicy: %d consecutive invalid actions — stopping.",
                    invalid_streak,
                )
                stopped_by = "invalid_actions"
                step += 1
                break

            step += 1

        if state.diagnosis_failed:
            stopped_by = "diagnosis_error"
        verified = loop.hypothesis_tester.best_hypotheses(state.all_test_results)
        report = AutoDiagnoseReport(
            cycles=step,
            resolved=bool(verified),
            stopped_by=stopped_by,
            verified_hypotheses=verified,
            final_hypotheses=state.all_hypotheses,
            all_test_results=state.all_test_results,
            final_stats_report=state.stats_report,
            store=loop.store,
            _run_id=loop._run_id,
        )
        return loop._end_run(report, timings)
