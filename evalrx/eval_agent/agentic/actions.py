"""Judge-decided actions for the agentic diagnosis loop.

Each decision turn, the CLI judge returns one strict JSON object describing
the next tool to call. ``parse_action`` turns that raw text into a validated
:class:`Action`, using the shared :func:`~evalrx.agent_runtime.json_shape.validate_json_shape`
(re-exported here for convenience — also used by M3's hypothesis parser and
failure-mode cluster naming).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalrx.agent_runtime.json_shape import validate_json_shape

if TYPE_CHECKING:
    from evalrx.eval_agent.agentic.board import EvidenceBoard
    from evalrx.eval_agent.agentic.tools import ToolRegistry
    from evalrx.eval_agent.run_logger import RunLogger

logger = logging.getLogger(__name__)

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tool", "rationale"],
    "properties": {
        "tool": {"type": "string", "minLength": 1},
        "params": {"type": "object"},
        "rationale": {"type": "string", "minLength": 1},
    },
}


class ActionParseError(Exception):
    """Raised when the judge's raw output cannot be turned into a valid Action."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


@dataclass
class Action:
    """One judge-decided tool call."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    raw: str = ""
    repair_attempts: int = 0
    valid: bool = True


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_action(raw: str, registry: "ToolRegistry") -> Action:
    """Parse the judge's raw text into a validated :class:`Action`.

    Raises :class:`ActionParseError` when the text is not JSON, fails the
    action schema, or names a tool the registry doesn't know about.
    """
    text = _strip_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"not valid JSON: {exc}") from exc

    errors = validate_json_shape(data, ACTION_SCHEMA)
    if errors:
        raise ActionParseError("action schema validation failed", errors)

    tool = str(data["tool"])
    if tool not in registry.tool_names():
        raise ActionParseError(
            f"unknown tool {tool!r}", [f"must be one of {sorted(registry.tool_names())}"]
        )

    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ActionParseError("params must be an object", [f"$.params: got {type(params).__name__}"])

    return Action(tool=tool, params=params, rationale=str(data["rationale"]), raw=raw)


def _fallback_action(board: "EvidenceBoard") -> Action:
    """Deterministic next-step heuristic: the first unmet M1->M4 stage.

    Used when the judge's output is unusable even after one repair prompt, so
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
    max_repairs: int = 1,
) -> Action:
    """Ask the judge for the next action, repairing/falling back as needed.

    Always returns an :class:`Action` — never raises. When the judge's
    response can't be salvaged after *max_repairs* repair prompts, returns the
    deterministic :func:`_fallback_action` with ``valid=False``.
    """
    from evalrx.eval_agent.prompts.agentic import ACTION_REPAIR_PROMPT, DECISION_PROMPT

    evidence = board.to_prompt(registry)
    prompt = DECISION_PROMPT.format(evidence=evidence)
    raw = judge.generate(prompt)

    action: Action | None = None
    try:
        action = parse_action(raw, registry)
    except ActionParseError as exc:
        last_raw, last_exc = raw, exc
        for attempt in range(1, max_repairs + 1):
            repair_prompt = ACTION_REPAIR_PROMPT.format(
                raw=last_raw,
                errors="\n".join(last_exc.errors) or str(last_exc),
                evidence=evidence,
            )
            last_raw = judge.generate(repair_prompt)
            try:
                action = parse_action(last_raw, registry)
                action.repair_attempts = attempt
                break
            except ActionParseError as exc2:
                last_exc = exc2
                action = None
        if action is None:
            action = _fallback_action(board)
            action.raw = raw

    if run_logger is not None:
        try:
            run_logger.log_agent_decision(
                step,
                action=action.tool,
                params=action.params,
                rationale=action.rationale,
                valid=action.valid,
                repair_attempts=action.repair_attempts,
                fallback_used=not action.valid,
                judge_prompt=prompt,
                judge_raw=raw,
            )
        except Exception:  # noqa: BLE001 — logging must never break the loop
            logger.warning("decide: log_agent_decision failed", exc_info=True)

    return action
