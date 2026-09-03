"""The agentic loop's `search_probes` tool: wires ProbeSearchAgent into the
ToolRegistry, reusing the loop's own decision judge + model + protocol.
Exercised directly against the tool handler with a minimal fake loop (not the
full AgenticDiagnoseLoop machinery) and a fake model/judge — no GPU."""

from __future__ import annotations

from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.eval_agent.agentic.board import EvidenceBoard
from evalrx.eval_agent.agentic.tools import _RunState, build_default_registry


class _ScriptedJudge:
    """Handles both search_probes' internal calls: paraphrasing (PARAPHRASE_PROMPT
    contains "Rewrite it as") and PASS/FAIL scoring (case_discovery's _JUDGE_PROMPT
    contains "Return a JSON object"). Scoring is driven by whether the model's
    observed answer contains WRONG."""

    def __init__(self) -> None:
        self.n = 0

    def generate(self, prompt: str, **kwargs) -> str:
        if "Rewrite it as" in prompt:
            self.n += 1
            return f"Paraphrase variant {self.n}?"
        if "WRONG" in prompt:
            return '{"label": "FAIL", "reason": "mismatch"}'
        return '{"label": "PASS", "reason": "match"}'


class _AlwaysWrongModel:
    def generate(self, inputs, **kwargs) -> str:
        return "WRONG"


class _FakeLoop:
    """Duck-typed stand-in for AgenticDiagnoseLoop — search_probes only reads
    .judge/.model/.protocol/.run_logger off the loop."""

    def __init__(self, judge, model) -> None:
        self.judge = judge
        self.model = model
        self.protocol = None
        self.run_logger = None


def _seed_pool() -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt="What color is the car?", image="car.png"), expected="red"),
    ])


def test_search_probes_registered_in_default_registry():
    loop = _FakeLoop(_ScriptedJudge(), _AlwaysWrongModel())
    state = _RunState(original_data=_seed_pool(), data=_seed_pool())
    registry = build_default_registry(loop, state)
    assert "search_probes" in registry.tool_names()


def test_search_probes_runs_and_populates_board_and_state():
    loop = _FakeLoop(_ScriptedJudge(), _AlwaysWrongModel())
    state = _RunState(original_data=_seed_pool(), data=_seed_pool())
    registry = build_default_registry(loop, state)
    board = EvidenceBoard()

    spec = registry.get("search_probes")
    outcome = spec.handler(board, {"budget": 4})

    assert outcome.ok is True
    assert state.probe_search_result is not None
    assert state.probe_search_result.n_simulations == 4
    assert len(board.probe_search_findings) == 4  # every case fails -> all 4 surfaced
    assert "4 simulation" in outcome.summary


def test_search_probes_empty_seed_pool_returns_failure_without_crashing():
    loop = _FakeLoop(_ScriptedJudge(), _AlwaysWrongModel())
    state = _RunState(original_data=CaseBatch([]), data=CaseBatch([]))
    registry = build_default_registry(loop, state)
    board = EvidenceBoard()

    outcome = registry.get("search_probes").handler(board, {})
    assert outcome.ok is False
    assert outcome.error == "empty"


def test_search_probes_respects_call_cap_via_registry_dispatch():
    from evalrx.eval_agent.agentic.actions import Action

    loop = _FakeLoop(_ScriptedJudge(), _AlwaysWrongModel())
    state = _RunState(original_data=_seed_pool(), data=_seed_pool())
    registry = build_default_registry(loop, state)
    board = EvidenceBoard()
    action = Action(tool="search_probes", params={"budget": 1}, rationale="t")

    for _ in range(2):
        outcome = registry.dispatch(action, board)
        assert outcome.ok is True
        board.action_log.append({
            "step": 0, "tool": "search_probes", "params": {}, "ok": True, "summary": "",
        })

    third = registry.dispatch(action, board)
    assert third.ok is False
    assert third.error == "max_calls"
