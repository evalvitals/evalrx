"""VLM probe candidate generator (eval_agent.stages.probe_candidate_generator):
Macro/Micro question paraphrasing over a fixed seed pool, exercised with a
fake text-only judge (no model, no GPU, no vision)."""

from __future__ import annotations

from evalrx.analysis.probe_search import ProbeNode
from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.eval_agent.stages.probe_candidate_generator import (
    VLMProbeCandidateGenerator,
    _jaccard_distance,
)


class _FakeJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, **kwargs) -> str:
        self.prompts.append(prompt)
        return next(self._responses)


def _seed_pool() -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt="What color is the car?", image="car.png"), expected="red"),
        FailureCase(inputs=Inputs(prompt="How many apples are on the table?", image="apples.png"), expected="3"),
        FailureCase(inputs=Inputs(prompt="Is the dog running?", image="dog.png"), expected="yes"),
    ])


def _node(prompt: str, image: str = "car.png", expected: str = "red") -> ProbeNode:
    case = FailureCase(inputs=Inputs(prompt=prompt, image=image), expected=expected)
    return ProbeNode(case=case, regime="macro")


# ---------------------------------------------------------------------------
# _jaccard_distance
# ---------------------------------------------------------------------------

def test_jaccard_distance_identical_is_zero():
    assert _jaccard_distance("what color is the car", "what color is the car") == 0.0


def test_jaccard_distance_disjoint_is_one():
    assert _jaccard_distance("apples oranges", "dog cat") == 1.0


# ---------------------------------------------------------------------------
# availability gating
# ---------------------------------------------------------------------------

def test_unavailable_without_judge():
    gen = VLMProbeCandidateGenerator(seed_pool=_seed_pool(), judge=None)
    assert not gen.available
    assert gen.macro(_node("q"), []) is None
    assert gen.micro(_node("q")) is None


def test_unavailable_with_empty_seed_pool():
    gen = VLMProbeCandidateGenerator(seed_pool=CaseBatch([]), judge=_FakeJudge(["x"]))
    assert not gen.available


# ---------------------------------------------------------------------------
# macro: picks the least-explored seed, paraphrases it
# ---------------------------------------------------------------------------

def test_macro_picks_most_dissimilar_seed_and_paraphrases():
    judge = _FakeJudge(["What is the total count of fruit on the table?"])
    gen = VLMProbeCandidateGenerator(seed_pool=_seed_pool(), judge=judge)
    explored = [_node("What color is the car?", image="car.png")]
    candidate = gen.macro(explored[0], explored)
    assert candidate is not None
    # "How many apples are on the table?" shares zero (non-stopword) tokens
    # with the explored car question and out-scores the dog question on raw
    # token-set Jaccard distance -> should be the one paraphrased.
    assert candidate.expected == "3"
    assert candidate.inputs.image == "apples.png"
    assert candidate.inputs.prompt == "What is the total count of fruit on the table?"
    assert candidate.metadata["seed_case_id"]


def test_macro_returns_none_when_judge_only_echoes():
    judge = _FakeJudge([
        "How many apples are on the table?", "How many apples are on the table?",
    ])
    gen = VLMProbeCandidateGenerator(seed_pool=_seed_pool(), judge=judge, max_repairs=1)
    explored = [_node("What color is the car?", image="car.png")]
    assert gen.macro(explored[0], explored) is None


# ---------------------------------------------------------------------------
# micro: paraphrases the current node's own case
# ---------------------------------------------------------------------------

def test_micro_paraphrases_current_node_preserving_image_and_expected():
    judge = _FakeJudge(["What is the shade of the vehicle?"])
    gen = VLMProbeCandidateGenerator(seed_pool=_seed_pool(), judge=judge)
    node = _node("What color is the car?", image="car.png", expected="red")
    candidate = gen.micro(node)
    assert candidate is not None
    assert candidate.inputs.prompt == "What is the shade of the vehicle?"
    assert candidate.inputs.image == "car.png"
    assert candidate.expected == "red"


def test_micro_retries_once_then_gives_up_on_repeated_echo():
    judge = _FakeJudge(["What color is the car?", "What color is the car?"])
    gen = VLMProbeCandidateGenerator(seed_pool=_seed_pool(), judge=judge, max_repairs=1)
    node = _node("What color is the car?")
    assert gen.micro(node) is None
    assert len(judge.prompts) == 2


def test_judge_generation_exception_returns_none_without_crashing():
    class _BrokenJudge:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("boom")

    gen = VLMProbeCandidateGenerator(seed_pool=_seed_pool(), judge=_BrokenJudge())
    assert gen.micro(_node("q")) is None
