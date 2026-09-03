"""AgenticDiagnoseLoop — an LLM-driven decision loop over the M1-M5 stages.

``VLDiagnoseLoop`` runs M1->M2->M3->M5 in a hardcoded sequence every cycle.
This loop keeps every stage (and the confirm-split / post-loop run_m4/run_fix
discipline) exactly as-is, but replaces the fixed sequence with a CLI judge
that decides, one tool call at a time, what to do next — probe, run stats,
explore the raw data, propose hypotheses, test one, fix, or stop. The host,
not the judge, enforces call caps, tool preconditions, and the pre-registered
stopping discipline (no declaring success without a tested, supported,
protocol-consistent hypothesis).

``VLDiagnoseLoop`` itself is untouched — this is a new mode, not a
replacement (see the module docstring in ``eval_agent/loop.py``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evalrx.eval_agent.agentic.actions import decide
from evalrx.eval_agent.agentic.board import BudgetState, EvidenceBoard
from evalrx.eval_agent.agentic.tools import ToolRegistry, _RunState, build_default_registry
from evalrx.eval_agent.loop import VLDiagnoseLoop
from evalrx.eval_agent.loop_reports import VLDiagnoseReport
from evalrx.eval_agent.run_metadata import _attach_run_logger, _data_provenance, _run_config

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

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


class AgenticDiagnoseLoop(VLDiagnoseLoop):
    """M1-M5 investigation driven by judge decisions instead of a fixed cycle.

    Reuses :class:`~evalrx.eval_agent.loop.VLDiagnoseLoop`'s constructor,
    stage helpers (``_do_m1``..``_do_m5``), confirm-split, and post-loop
    ``run_m4``/``run_fix`` — only :meth:`run` differs.

    Args:
        judge:            CLI-backed judge that decides the next action each
                           turn (e.g. ``ClaudeModel()`` / ``AgyModel()``).
                           Required — this is a genuinely new capability, not
                           a Gemini-fallback stage judge.
        explorer:         Optional ``ExploratoryAnalysisAgent`` — when given,
                           registers the ``explore_data`` tool.
        max_actions:      Hard cap on judge decision turns (default 12).
        time_budget_sec:  Wall-clock cap for the whole run (0 = unlimited).
        registry_factory: Advanced override — ``(loop, run_state) -> ToolRegistry``
                          instead of :func:`build_default_registry`.
        **kwargs:         Forwarded to ``VLDiagnoseLoop.__init__`` (model,
                          protocol, probe_agent, stats_agent, hypothesis_tester,
                          surgery_agent, fix_agent, store, token_budget,
                          confirm_split, run_logger, ...).
    """

    def __init__(
        self,
        model: "Any",
        protocol: "Any",
        *,
        judge: "Any",
        explorer: "Any | None" = None,
        max_actions: int = 12,
        time_budget_sec: float = 0.0,
        registry_factory: "Any | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, protocol, **kwargs)
        if judge is None:
            raise ValueError(
                "AgenticDiagnoseLoop requires a decision judge, e.g. "
                "judge=ClaudeModel() or judge=AgyModel()."
            )
        self.judge = judge
        self.explorer = explorer
        self.max_actions = max_actions
        self.time_budget_sec = time_budget_sec
        self._registry_factory = registry_factory

    def run(self, data: "CaseBatch") -> VLDiagnoseReport:
        """Drive the judge-decided action loop to a stop or a budget limit."""
        self._tokens_used = 0
        timings: dict[str, float] = {}

        explore, confirm = self._split_explore_confirm(data)
        original_data = data
        if confirm is not None:
            data = explore

        _attach_run_logger(self.run_logger, self.probe_agent, self.stats_agent)
        if self.run_logger is not None:
            # _run_config records the M3 *stage* judge (loop.diagnosis_agent.judge),
            # not the decision judge driving this loop's own tool-calling — add it
            # explicitly so the dashboard header can show who is actually deciding.
            cfg = _run_config(self, data, loop_name="AgenticDiagnoseLoop")
            cfg["decision_judge"] = repr(self.judge)
            cfg["max_actions"] = self.max_actions
            self.run_logger.log_run_start(cfg)
            self.run_logger.log_cases(data)

        board = EvidenceBoard(
            protocol_summary=_protocol_summary(self.protocol),
            data_summary=_data_provenance(data),
            budget=BudgetState(
                max_actions=self.max_actions,
                token_budget=self.token_budget,
                time_budget_sec=self.time_budget_sec,
            ),
        )
        board.budget.start()

        state = _RunState(original_data=original_data, data=data, timings=timings)
        registry: ToolRegistry = (
            self._registry_factory(self, state)
            if self._registry_factory is not None
            else build_default_registry(self, state)
        )

        stopped_by = "max_actions"
        invalid_streak = 0
        step = 0
        while True:
            reason = board.budget.exhausted()
            if reason:
                stopped_by = reason
                break

            action = decide(self.judge, board, registry, run_logger=self.run_logger, step=step)
            board.budget.actions_taken += 1
            invalid_streak = 0 if action.valid else invalid_streak + 1

            outcome = registry.dispatch(action, board)
            board.action_log.append({
                "step": step, "tool": action.tool, "params": action.params,
                "ok": outcome.ok, "summary": outcome.summary,
            })
            if self.run_logger is not None:
                self.run_logger.log_agent_tool(
                    step, tool=action.tool, ok=outcome.ok,
                    summary=outcome.summary, error=outcome.error,
                )

            if action.tool == "stop" and outcome.ok:
                stopped_by = "agent_stop"
                step += 1
                break

            if invalid_streak >= _MAX_INVALID_STREAK:
                logger.warning(
                    "AgenticDiagnoseLoop: %d consecutive invalid actions — stopping.",
                    invalid_streak,
                )
                stopped_by = "invalid_actions"
                step += 1
                break

            step += 1

        verified = self.hypothesis_tester.best_hypotheses(state.all_test_results)
        report = VLDiagnoseReport(
            cycles=step,
            resolved=bool(verified),
            stopped_by=stopped_by,
            verified_hypotheses=verified,
            final_hypotheses=state.all_hypotheses,
            all_test_results=state.all_test_results,
            final_stats_report=state.stats_report,
            store=self.store,
            _run_id=self._run_id,
        )
        if self.run_logger:
            self.run_logger.log_loop_end(report, tokens_used=self._tokens_used, timings=timings)
        return report
