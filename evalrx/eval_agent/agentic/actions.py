"""Judge-decided actions for the agentic diagnosis loop.

Each decision turn, the judge picks the next tool. :func:`decide` asks for it
through :func:`~evalrx.agent_runtime.judge_call.ask_judge` with an output
model whose ``tool`` field only admits registered tool names, so an unknown
tool, a missing rationale or a non-JSON reply goes back to the judge with the
validation errors. When no valid answer comes back, the deterministic
next-unmet-stage heuristic runs instead and the decision is logged with
``valid=False``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, create_model

from evalrx.agent_runtime.judge_call import JudgeError, ask_judge

if TYPE_CHECKING:
    from evalrx.eval_agent.agentic.board import EvidenceBoard
    from evalrx.eval_agent.agentic.tools import ToolRegistry
    from evalrx.eval_agent.run_logger import RunLogger

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """One judge-decided tool call."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    raw: str = ""
    repair_attempts: int = 0
    valid: bool = True


def action_model(registry: "ToolRegistry") -> type[BaseModel]:
    """The decision's output model: ``tool`` is one of *registry*'s names."""
    tool_name: Any = Literal[tuple(sorted(registry.tool_names()))]
    return create_model(
        "NextAction",
        tool=(tool_name, Field(description="a tool name from AVAILABLE TOOLS")),
        params=(dict[str, Any], Field(default_factory=dict, description="the tool's parameters")),
        rationale=(str, Field(min_length=1, description="one sentence: why this tool, why now")),
    )


def _fallback_action(board: "EvidenceBoard") -> Action:
    """Deterministic next-step heuristic: the first unmet M1->M4 stage.

    Used when the judge fails or gives no valid answer within its retries, so
    a stuck judge doesn't stall the run — the action is marked ``valid=False``
    so the loop can track a consecutive-failure streak and give up cleanly.
    """
    if not board.probe_findings:
        tool = "run_probe"
    elif not board.stats_findings:
        tool = "run_stats"
    elif not board.hypotheses:
        tool = "propose_hypotheses"
    elif not board.has_supported_hypothesis():
        tool = "test_hypothesis"
    else:
        tool = "stop"
    params = {"resolved": True, "reason": "fallback heuristic"} if tool == "stop" else {}
    return Action(
        tool=tool, params=params,
        rationale="fallback heuristic: judge output unusable", valid=False,
    )


def decide(
    judge: Any,
    board: "EvidenceBoard",
    registry: "ToolRegistry",
    *,
    run_logger: "RunLogger | None" = None,
    step: int = 0,
    retries: int = 1,
) -> Action:
    """Ask the judge for the next action; never raises.

    An answer that fails validation is sent back up to *retries* times. When
    the judge fails or no answer is valid, returns :func:`_fallback_action`
    with ``valid=False``.
    """
    from evalrx.eval_agent.prompts.agentic import DECISION_PROMPT

    prompt = DECISION_PROMPT.format(evidence=board.to_prompt(registry))
    try:
        reply = ask_judge(
            judge, prompt, action_model(registry), role="agent_decision", retries=retries,
        )
    except JudgeError as exc:
        logger.warning("decide: %s; using the next-unmet-stage heuristic", exc)
        exchanges = exc.exchanges
        action = _fallback_action(board)
    else:
        exchanges = reply.exchanges
        chosen = reply.output
        action = Action(
            tool=str(chosen.tool), params=dict(chosen.params), rationale=chosen.rationale,  # type: ignore[attr-defined]
            repair_attempts=len(exchanges) - 1,
        )
    action.raw = (exchanges[-1].output or "") if exchanges else ""

    if run_logger is not None:
        try:
            run_logger.log_agent_decision(
                step=step,
                action=action.tool,
                params=action.params,
                rationale=action.rationale,
                valid=action.valid,
                repair_attempts=action.repair_attempts,
                fallback_used=not action.valid,
                judge_prompt=prompt,
                judge_raw=action.raw,
                judge_calls=[e.record() for e in exchanges],
            )
        except Exception:  # noqa: BLE001 — logging must never break the loop
            logger.warning("decide: log_agent_decision failed", exc_info=True)

    return action
