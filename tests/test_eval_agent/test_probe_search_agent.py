"""ProbeSearchAgent (eval_agent.stages.probe_search_agent): wires
analysis.probe_search's MCTS to CaseDiscoveryAgent (verifier) +
VLMProbeCandidateGenerator (Macro/Micro), exercised end-to-end with a fake
model and a fake composite judge (no real model, no GPU)."""

from __future__ import annotations

import pytest

from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.eval_agent.stages.probe_search_agent import ProbeSearchAgent


def _seed_pool() -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt="What color is the car?", image="car.png"), expected="red"),
        FailureCase(inputs=Inputs(prompt="How many apples are on the table?", image="apples.png"), expected="3"),
    ])


class _ScriptedJudge:
    """Handles both roles the same judge instance plays: paraphrasing Macro/Micro
    candidates (PARAPHRASE_PROMPT contains "Rewrite it as") and PASS/FAIL
    scoring (case_discovery's _JUDGE_PROMPT contains "Return a JSON object").
    Scoring decision is driven by whether the observed answer contains WRONG.
    """

    def __init__(self) -> None:
        self.n = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "Rewrite it as" in prompt:
            self.n += 1
            return f"Paraphrase variant {self.n} of the question?"
        if "WRONG" in prompt:
            return '{"label": "FAIL", "reason": "observed does not match expected"}'
        return '{"label": "PASS", "reason": "observed matches expected"}'


class _AlwaysWrongModel:
    def generate(self, inputs, **kwargs) -> str:
        return "WRONG"


class _AlwaysRightModel:
    def generate(self, inputs, **kwargs) -> str:
        return "definitely correct"


def test_probe_search_agent_requires_a_judge():
    with pytest.raises(ValueError):
        ProbeSearchAgent(judge=None)


def test_probe_search_agent_all_failures_populate_failure_cases():
    agent = ProbeSearchAgent(judge=_ScriptedJudge(), budget=6, w_max=2)
    result = agent.run(_AlwaysWrongModel(), _seed_pool())
    assert result.n_simulations == 6
    assert len(result.failure_cases) == 6
    assert result.error_rate == 1.0
    assert result.n_macro + result.n_micro == 6


def test_probe_search_agent_all_passes_yield_no_failures():
    agent = ProbeSearchAgent(judge=_ScriptedJudge(), budget=6, w_max=2)
    result = agent.run(_AlwaysRightModel(), _seed_pool())
    assert result.n_simulations == 6
    assert len(result.failure_cases) == 0
    assert result.error_rate == 0.0


def test_probe_search_agent_failure_cases_carry_seed_expected_and_image():
    agent = ProbeSearchAgent(judge=_ScriptedJudge(), budget=3, w_max=2)
    result = agent.run(_AlwaysWrongModel(), _seed_pool())
    for case in result.failure_cases:
        assert case.expected in {"red", "3"}
        assert case.inputs.image in {"car.png", "apples.png"}
