"""Pure-function tests for the agentic loop's action parsing/repair/fallback.

No subprocess, no real judge — a stub judge returns canned text so these run
fast and deterministically.
"""

from __future__ import annotations

from evalrx.eval_agent.agentic.actions import (
    Action,
    ActionParseError,
    _fallback_action,
    decide,
    parse_action,
    validate_json_shape,
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
# validate_json_shape
# ---------------------------------------------------------------------------

def test_validate_json_shape_passes_valid_object():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
    assert validate_json_shape({"a": "x"}, schema) == []


def test_validate_json_shape_flags_missing_required():
    schema = {"type": "object", "required": ["a"], "properties": {}}
    errors = validate_json_shape({}, schema)
    assert any("missing required field 'a'" in e for e in errors)


def test_validate_json_shape_flags_wrong_type():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    errors = validate_json_shape({"a": 5}, schema)
    assert any("expected string" in e for e in errors)


def test_validate_json_shape_flags_short_string():
    schema = {"type": "string", "minLength": 3}
    errors = validate_json_shape("ab", schema)
    assert any("too short" in e for e in errors)


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------

def test_parse_action_accepts_valid_json():
    registry = _registry()
    raw = '{"tool": "run_probe", "params": {}, "rationale": "need M1 first"}'
    action = parse_action(raw, registry)
    assert action.tool == "run_probe"
    assert action.rationale == "need M1 first"
    assert action.valid is True


def test_parse_action_strips_markdown_fences():
    registry = _registry()
    raw = '```json\n{"tool": "stop", "params": {}, "rationale": "done"}\n```'
    action = parse_action(raw, registry)
    assert action.tool == "stop"


def test_parse_action_rejects_non_json():
    registry = _registry()
    try:
        parse_action("not json at all", registry)
        raise AssertionError("expected ActionParseError")
    except ActionParseError as exc:
        assert "not valid JSON" in str(exc)


def test_parse_action_rejects_missing_rationale():
    registry = _registry()
    try:
        parse_action('{"tool": "run_probe", "params": {}}', registry)
        raise AssertionError("expected ActionParseError")
    except ActionParseError as exc:
        assert any("rationale" in e for e in exc.errors)


def test_parse_action_rejects_unknown_tool():
    registry = _registry()
    try:
        parse_action('{"tool": "nonexistent", "rationale": "x"}', registry)
        raise AssertionError("expected ActionParseError")
    except ActionParseError as exc:
        assert "unknown tool" in str(exc)


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
        "garbage, not json",
        '{"tool": "run_probe", "params": {}, "rationale": "retry worked"}',
    ])

    action = decide(judge, board, registry, step=0)

    assert action.tool == "run_probe"
    assert action.valid is True
    assert action.repair_attempts == 1
    assert len(judge.calls) == 2


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
