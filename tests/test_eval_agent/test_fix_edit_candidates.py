"""Recursive rounds may EDIT the deployed template, not just wrap it.

Live finding (2026-09-04, flash-lite word_sorting claude-r2b rerun): the best
second-round candidate was the judge hand-rolling an edit inside the append
mechanism — an OVERRIDE paragraph replacing the deployed template's PART C —
and it reached e=15.93 (17F/4B, one discordant pair short of the gate).  The
append form pays an instruction-conflict tax: the model must parse "ignore the
text above".  When a batch carries a recursive-round manifest's metadata
(``original_prompt`` + ``recursive_stack[-1]["template"]``), the L1/L2
proposers are now shown the deployed template and invited to write EDIT
candidates against ``{original_prompt}``, which REPLACE it cleanly.  Default
runs (no such metadata) are byte-identical to before.
"""

from __future__ import annotations

import json

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent import FixAgent
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

_DEPLOYED = "{prompt}\n\nWork through this in three parts before answering."


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


class StackHostileModel(Model):
    """Fails exactly on prompts still carrying the deployed scaffold text."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", inputs))
        self.prompts.append(prompt)
        return "No." if "three parts" in prompt else "Yes."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _hyp(statement: str) -> Hypothesis:
    return Hypothesis(
        statement=statement, target_model="m", predicted_failure_mode="", test_design=""
    )


def _batch(n: int = 8, recursive: bool = True) -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    cases = []
    for i in range(n):
        original = f"Is item {i} sorted correctly?"
        stacked = _DEPLOYED.replace("{prompt}", original)
        metadata = (
            {
                "original_prompt": original,
                "recursive_stack": [
                    {"round": 2, "winner": "02_L1_three_parts", "template": _DEPLOYED}
                ],
            }
            if recursive
            else {}
        )
        cases.append(
            FailureCase(
                id=f"c{i}",
                inputs=Inputs(prompt=stacked),
                expected=yes,
                label=Label.FAIL,
                metadata=metadata,
            )
        )
    return CaseBatch(cases)


def test_edit_note_shows_deployed_template_and_parser_accepts_edit():
    judge = ScriptedJudge(
        json.dumps(
            [
                {
                    "name": "drop_three_parts",
                    "prompt_template": "Answer directly. {original_prompt}",
                },
                {"name": "wrap_more", "prompt_template": "Be careful. {prompt}"},
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    note = agent._edit_note(_batch())
    assert "DEPLOYED TEMPLATE" in note and _DEPLOYED.rstrip() in note
    candidates = agent._l1_candidates("h", "e", context_block=note)
    by_name = {c.name: c for c in candidates}
    assert by_name["drop_three_parts"].payload == {
        "prompt_template": "Answer directly. {original_prompt}"
    }
    assert "wrap_more" in by_name  # append candidates keep working alongside
    assert "DEPLOYED TEMPLATE" in judge.prompts[0]


def test_default_batches_are_unchanged():
    agent = FixAgent(judge=ScriptedJudge("[]"), max_tier="L1")
    assert agent._edit_note(_batch(recursive=False)) == ""
    # a mixed batch (one case missing the stack) has no coherent template to edit
    mixed = CaseBatch(list(_batch(2)) + list(_batch(2, recursive=False)))
    assert agent._edit_note(mixed) == ""


def test_edit_candidate_replaces_deployed_template_end_to_end():
    judge = ScriptedJudge(
        json.dumps(
            [
                {
                    "name": "drop_three_parts",
                    "prompt_template": "Answer with Yes or No only. {original_prompt}",
                }
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    model = StackHostileModel()
    out = agent.propose_and_validate(
        model, _batch(), [_hyp("the deployed scaffold itself breaks the answers")]
    )
    assert out.fixed is True
    assert out.best is not None
    assert out.best.candidate.name == "drop_three_parts"
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    fixed_prompts = [p for p in model.prompts if p.startswith("Answer with Yes or No only.")]
    assert fixed_prompts and all("three parts" not in p for p in fixed_prompts)


def test_spec_pipeline_fills_original_prompt():
    class CaptureModel(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text"})

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, inputs, **kwargs) -> str:
            self.prompts.append(str(getattr(inputs, "prompt", inputs)))
            return "yes"

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    case = FailureCase(
        id="c0",
        inputs=Inputs(prompt="STACKED TEXT"),
        expected={"all_of": ["yes"]},
        label=Label.FAIL,
        metadata={"original_prompt": "ORIGINAL TEXT"},
    )
    model = CaptureModel()
    spec = PipelineSpec(name="edit", prompt_template="Do it plainly. {original_prompt}")
    result = run_pipeline(model, case, spec, score_fn=lambda c, o: "yes" in o)
    assert result is True
    assert model.prompts == ["Do it plainly. ORIGINAL TEXT"]
