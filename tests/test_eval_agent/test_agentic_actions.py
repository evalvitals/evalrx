"""Pure-function tests for the agentic loop's action parsing/repair/fallback.

No subprocess, no real judge — a stub judge returns canned text so these run
fast and deterministically.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalrx.eval_agent.agentic.actions import (
    Action,
    _fallback_action,
    action_model,
    decide,
)
from evalrx.eval_agent.agentic.board import EvidenceBoard
from evalrx.eval_agent.agentic.tools import ToolOutcome, ToolRegistry, ToolSpec


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="run_probe", description="run M1",
        params_schema={"type": "object", "properties": {}},
        handler=lambda board, params: ToolOutcome(True, "ok"),
    ))
    registry.register(ToolSpec(
        name="stop", description="stop",
        params_schema={"type": "object", "properties": {}},
        handler=lambda board, params: ToolOutcome(True, "stopped"),
    ))
    return registry


# ---------------------------------------------------------------------------
# action_model
# ---------------------------------------------------------------------------

def test_action_model_accepts_a_registered_tool():
    choice = action_model(_registry()).model_validate(
        {"tool": "run_probe", "params": {"k": 1}, "rationale": "start with M1"}
    )
    assert (choice.tool, choice.params, choice.rationale) == ("run_probe", {"k": 1}, "start with M1")


@pytest.mark.parametrize("data", [
    {"tool": "nonexistent", "rationale": "x"},
    {"tool": "run_probe"},
    {"tool": "run_probe", "rationale": ""},
    {"tool": "run_probe", "rationale": "x", "params": []},
])
def test_action_model_rejects_unknown_tools_and_missing_fields(data):
    with pytest.raises(ValidationError):
        action_model(_registry()).model_validate(data)


# ---------------------------------------------------------------------------
# _fallback_action
# ---------------------------------------------------------------------------

def test_fallback_action_picks_first_unmet_stage():
    board = EvidenceBoard()
    action = _fallback_action(board)
    assert action.tool == "run_probe"
    assert action.valid is False

    board.probe_findings = [{"analyzer": "x"}]
    assert _fallback_action(board).tool == "run_stats"

    board.stats_findings = [{"tool": "y"}]
    assert _fallback_action(board).tool == "propose_hypotheses"

    board.hypotheses = [{"statement": "s", "status": "proposed"}]
    assert _fallback_action(board).tool == "test_hypothesis"

    board.hypotheses = [{"statement": "s", "status": "supported", "is_consistent_with_protocol": True}]
    fallback = _fallback_action(board)
    assert fallback.tool == "stop"
    assert fallback.params["resolved"] is True


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

class _StubJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


class _StubRunLogger:
    def __init__(self) -> None:
        self.decisions: list[dict] = []

    def log_agent_decision(self, step, **kwargs):
        self.decisions.append({"step": step, **kwargs})


def test_decide_returns_action_on_first_valid_response():
    registry = _registry()
    board = EvidenceBoard()
    judge = _StubJudge(['{"tool": "run_probe", "params": {}, "rationale": "start"}'])
    run_logger = _StubRunLogger()

    action = decide(judge, board, registry, run_logger=run_logger, step=0)

    assert action.tool == "run_probe"
    assert action.valid is True
    assert action.repair_attempts == 0
    assert len(judge.calls) == 1
    assert run_logger.decisions[0]["valid"] is True
    assert run_logger.decisions[0]["fallback_used"] is False


def test_decide_repairs_once_then_succeeds():
    registry = _registry()
    board = EvidenceBoard()
    judge = _StubJudge([
        '```json\n{"tool": "nonexistent", "rationale": "x"}\n```',
        '{"tool": "run_probe", "params": {}, "rationale": "retry worked"}',
    ])

    action = decide(judge, board, registry, step=0)

    assert action.tool == "run_probe"
    assert action.valid is True
    assert action.repair_attempts == 1
    assert len(judge.calls) == 2
    assert "nonexistent" in judge.calls[1] and "run_probe" in judge.calls[1]


def test_decide_falls_back_when_the_judge_raises():
    class _Broken:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("limit reached")

    run_logger = _StubRunLogger()
    action = decide(_Broken(), EvidenceBoard(), _registry(), run_logger=run_logger, step=0)
    assert (action.tool, action.valid) == ("run_probe", False)
    (call,) = run_logger.decisions[0]["judge_calls"]
    assert "limit reached" in call["error"]


def test_decide_falls_back_after_exhausting_repairs():
    registry = _registry()
    board = EvidenceBoard()
    judge = _StubJudge(["garbage 1", "garbage 2"])
    run_logger = _StubRunLogger()

    action = decide(judge, board, registry, run_logger=run_logger, step=3)

    assert action.tool == "run_probe"  # fallback heuristic: no probe findings yet
    assert action.valid is False
    assert len(judge.calls) == 2
    assert run_logger.decisions[0]["valid"] is False
    assert run_logger.decisions[0]["fallback_used"] is True


def test_decide_never_raises_when_run_logger_is_broken():
    registry = _registry()
    board = EvidenceBoard()
    judge = _StubJudge(['{"tool": "stop", "params": {}, "rationale": "done"}'])

    class _BrokenLogger:
        def log_agent_decision(self, *a, **k):
            raise RuntimeError("boom")

    action = decide(judge, board, registry, run_logger=_BrokenLogger(), step=0)
    assert action.tool == "stop"


def test_action_dataclass_defaults():
    action = Action(tool="stop")
    assert action.params == {}
    assert action.rationale == ""
    assert action.valid is True
    assert action.repair_attempts == 0
