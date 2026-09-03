"""Tests for M1 feeding observed PASS/FAIL cases into the selection prompt."""

from __future__ import annotations

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.eval_agent import ProbeAgent
from evalrx.eval_agent.stages.probe_agent import _summarize_cases
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel


def _labeled_cases() -> CaseBatch:
    cases = [
        FailureCase(id="c0", inputs=Inputs(prompt="How many boxes?"),
                    expected="3", label=Label.FAIL),
        FailureCase(id="c1", inputs=Inputs(prompt="What colour is the car?"),
                    expected="red", label=Label.FAIL),
        FailureCase(id="c2", inputs=Inputs(prompt="Is the sky visible?"),
                    expected="yes", label=Label.PASS),
    ]
    cases[0].metadata["discovery_observed"] = "I think there are five boxes."
    cases[1].metadata["discovery_observed"] = "The car is blue."
    cases[2].metadata["discovery_observed"] = "Yes, the sky is visible."
    return CaseBatch(cases)


# ── _summarize_cases ────────────────────────────────────────────────────────


def test_summarize_includes_counts_prompts_and_answers():
    s = _summarize_cases(_labeled_cases(), max_fail=4, max_pass=2)
    assert "OBSERVED CASES (PASS=1 FAIL=2 UNKNOWN=0)" in s
    assert "How many boxes?" in s          # a failing prompt
    assert "I think there are five boxes." in s  # the model's actual answer
    assert "expected: 3" in s
    assert "[FAIL]" in s and "[PASS]" in s


def test_summarize_respects_caps():
    s = _summarize_cases(_labeled_cases(), max_fail=1, max_pass=0)
    assert s.count("[FAIL]") == 1
    assert "[PASS]" not in s


def test_summarize_disabled_or_empty():
    assert _summarize_cases(_labeled_cases(), max_fail=0, max_pass=0) == ""
    assert _summarize_cases(None) == ""
    # All-unknown → no labeled cases → empty digest
    unknown = CaseBatch([FailureCase(id="u", inputs=Inputs(prompt="x"), label=Label.UNKNOWN)])
    assert _summarize_cases(unknown) == ""


def test_summarize_clips_long_text():
    case = FailureCase(id="c", inputs=Inputs(prompt="P" * 500), label=Label.FAIL)
    case.metadata["discovery_observed"] = "O" * 500
    s = _summarize_cases(CaseBatch([case]), max_chars=50)
    assert "…" in s
    # No single digest line should carry the full 500-char blob.
    assert "P" * 200 not in s


# ── integration: digest reaches the judge's selection prompt ────────────────


class CapturingJudge(FakeModel):
    """Records the last prompt; returns a fixed valid analyzer selection."""

    def __init__(self) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.last_prompt = ""

    def generate(self, inputs, **kwargs) -> str:
        self.last_prompt = str(inputs)
        return '{"analyzers": ["self_consistency"], "rationale": "grounded in observed failures"}'


def test_llm_selection_prompt_contains_case_digest():
    judge = CapturingJudge()
    model = FakeModel(capabilities={Capability.GENERATE})
    agent = ProbeAgent(judge=judge, max_analyzers=2)
    protocol = ExperimentProtocol(description="Counting and colour QA.", task_domain="vqa")
    agent.probe(model, _labeled_cases(), protocol=protocol)
    assert "OBSERVED CASES" in judge.last_prompt
    assert "How many boxes?" in judge.last_prompt
    assert "I think there are five boxes." in judge.last_prompt


def test_case_examples_zero_disables_digest_in_prompt():
    judge = CapturingJudge()
    model = FakeModel(capabilities={Capability.GENERATE})
    agent = ProbeAgent(judge=judge, max_analyzers=2, case_examples=(0, 0))
    protocol = ExperimentProtocol(description="Counting QA.", task_domain="vqa")
    agent.probe(model, _labeled_cases(), protocol=protocol)
    assert "OBSERVED CASES" not in judge.last_prompt
