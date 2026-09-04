"""Winner-as-new-baseline: a deployed L2 spec becomes the baseline pipeline.

A pure L1 template winner deploys as data (pre-rendered prompts), but an L2
spec winner (n_samples / strategy / generation_kwargs) has to RUN — the live
finding (flash-lite word_sorting claude-r2b, 2026-09-04) is that second-round
winners tend to BE specs (vote + budget + template), and the old
template_only freeze measured flat because it dropped exactly those parts.
``SpecPipelineModel`` wraps the deployed spec as a Model handle so Stage-0 and
the fix baseline arm measure the fixed pipeline, while
``FixAgent(candidate_model=...)`` runs candidates on the raw model as full
REPLACEMENT pipelines paired against it.
"""

from __future__ import annotations

import json

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent import FixAgent
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.stages.fix_tools import SpecPipelineModel


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


class SequenceModel(Model):
    """Cycles through scripted outputs; records every prompt it was given."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self._i = 0
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        self.prompts.append(str(getattr(inputs, "prompt", inputs)))
        out = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return out

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
                inputs=Inputs(prompt=f"Is item {i} in order?"),
                expected=yes,
                label=Label.FAIL,
            )
            for i in range(n)
        ]
    )


def test_spec_pipeline_model_votes_and_strips_override_gate():
    inner = SequenceModel(["Answer: A", "Answer: B", "Answer: A"])
    # baseline_override_min_support=3 would defer to the (empty) recorded
    # baseline on a bare generate() call — deploying must strip it.
    wrapped = SpecPipelineModel(
        inner,
        {
            "name": "vote3",
            "prompt_template": "Scaffolded. {prompt}",
            "n_samples": 3,
            "baseline_override_min_support": 3,
        },
    )
    out = wrapped.generate(Inputs(prompt="sort these"))
    assert out == "Answer: A"  # 2-of-3 majority, not the empty incumbent
    assert wrapped.spec.baseline_override_min_support == 0
    assert inner.prompts == ["Scaffolded. sort these"] * 3


def test_candidates_replace_the_deployed_pipeline_end_to_end():
    inner = SequenceModel(["No."])  # the deployed pipeline fails every case
    wrapped = SpecPipelineModel(
        inner,
        {"name": "bad_scaffold", "prompt_template": "Use the scaffold. {prompt}", "n_samples": 1},
    )
    raw = SequenceModel(["Yes."])
    judge = ScriptedJudge(
        json.dumps(
            [{"name": "plain_answer", "prompt_template": "Answer plainly. {prompt}"}]
        )
    )
    agent = FixAgent(
        judge=judge,
        max_tier="L1",
        candidate_model=raw,
        deployed_spec={"name": "bad_scaffold", "prompt_template": "Use the scaffold. {prompt}"},
    )
    out = agent.propose_and_validate(
        wrapped, _gold_yes_batch(), [_hyp("the deployed scaffold breaks every answer")]
    )
    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "plain_answer"
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    # baseline arm went through the deployed pipeline (inner model, scaffolded
    # prompts); the candidate arm ran on the RAW handle with only its own
    # template — replacement, not composition.
    assert all(p.startswith("Use the scaffold.") for p in inner.prompts)
    assert raw.prompts and all(p.startswith("Answer plainly.") for p in raw.prompts)
    assert all("Use the scaffold." not in p for p in raw.prompts)


def test_deployed_spec_is_shown_to_the_proposer():
    judge = ScriptedJudge("[]")
    spec = {"name": "vote3", "prompt_template": "Scaffolded. {prompt}", "n_samples": 3}
    agent = FixAgent(judge=judge, max_tier="L1", deployed_spec=spec)
    note = agent._edit_note(_gold_yes_batch())
    assert "DEPLOYED PIPELINE" in note and '"vote3"' in note
    candidates = agent._l1_candidates("h", "e", context_block=note)
    assert "DEPLOYED PIPELINE" in judge.prompts[0]
    assert candidates  # falls back to the conservative default on "[]"
