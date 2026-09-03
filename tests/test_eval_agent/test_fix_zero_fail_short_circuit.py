"""A validation subset with zero repairable failure mass skips the search.

Live finding (2026-08-25 gemma sweep, bbh_tracking7 at baseline 0.996): with 0
FAIL cases in the validation split no candidate can ever validate (n_fixed
stays 0, the e-value ceiling is 1), yet the fix stage still spent 67 minutes
executing 15 EXPLORE candidates. The gate must use the FRESH baseline the
paired test uses, not the stale case labels.
"""

from __future__ import annotations

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent import FixAgent
from evalvitals.eval_agent.hypothesis import Hypothesis


class ScriptedJudge(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        self.prompts.append(str(inputs))
        return self._reply

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class AlwaysRightModel(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return "Yes."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class HopelessModel(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _hyp(statement: str) -> Hypothesis:
    return Hypothesis(
        statement=statement, target_model="m", predicted_failure_mode="", test_design=""
    )


def _gold_yes_batch(n: int = 8) -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    return CaseBatch(
        [
            FailureCase(
                id=f"c{i}",
                inputs=Inputs(prompt=f"Is there a lesion {i}?"),
                expected=yes,
                # stale labels say FAIL on purpose: the gate must re-measure
                label=Label.FAIL,
            )
            for i in range(n)
        ]
    )


_JUDGE_REPLY = '[{"name": "careful", "prompt_template": "Look carefully. {prompt}"}]'


def test_zero_fail_baseline_skips_the_candidate_search():
    judge = ScriptedJudge(_JUDGE_REPLY)
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(
        AlwaysRightModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.fixed is False
    assert out.attempted == []
    assert judge.prompts == []  # the candidate budget was never spent
    assert out.recommendation is not None
    assert out.recommendation["action"] == "gather_more_failures"
    assert "pass the fresh baseline" in out.recommendation["reason"]


def test_zero_fail_rates_skip_in_repeats_mode():
    judge = ScriptedJudge(_JUDGE_REPLY)
    agent = FixAgent(judge=judge, max_tier="L1", baseline_repeats=2)
    out = agent.propose_and_validate(
        AlwaysRightModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.attempted == [] and judge.prompts == []
    assert out.recommendation["action"] == "gather_more_failures"


def test_failing_baseline_still_reaches_the_judge():
    judge = ScriptedJudge(_JUDGE_REPLY)
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(
        HopelessModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert len(judge.prompts) == 1
    assert out.attempted  # the candidate was proposed and validated
