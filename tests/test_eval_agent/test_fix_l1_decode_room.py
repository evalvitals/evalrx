"""L1 templates may carry decode ROOM (max_tokens only), floor-enforced.

Live finding (2026-08-28 vlm 12-cell run): at a 64-token baseline budget every
"write down X then answer" L1 template truncated mid-work and scored as a pure
regression (0 repairs / 32 breaks on three models) — a decoding artefact, not
a verdict on the prompt idea. L1 payloads now accept ``generation_kwargs``
with ``max_tokens`` alone; sampler controls stay L2-only and are dropped.
"""

from __future__ import annotations

import json

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent import FixAgent, FixTier
from evalvitals.eval_agent.hypothesis import Hypothesis
from evalvitals.eval_agent.stages.fix_agent import FixCandidate


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


class DecodeRoomModel(Model):
    """Completes the answer only when given more room than the 64-token default."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return "Yes." if int(kwargs.get("max_tokens", 64)) >= 128 else "No."

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
                label=Label.FAIL,
            )
            for i in range(n)
        ]
    )


def test_l1_proposal_keeps_max_tokens_and_drops_sampler_controls():
    judge = ScriptedJudge(
        json.dumps(
            [
                {
                    "name": "roomy",
                    "prompt_template": "Show your work. {prompt}",
                    "generation_kwargs": {"max_tokens": 256, "temperature": 0.9},
                },
                {
                    "name": "pipelineish",
                    "prompt_template": "Vote. {prompt}",
                    "n_samples": 5,
                },
                {"name": "plain", "prompt_template": "Read twice. {prompt}"},
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    candidates = agent._l1_candidates("h", "e")
    by_name = {c.name: c for c in candidates}
    # an L2-shaped proposal (n_samples) is still rejected at L1
    assert "pipelineish" not in by_name
    assert by_name["roomy"].payload["generation_kwargs"] == {"max_tokens": 256}
    assert by_name["plain"].payload == {"prompt_template": "Read twice. {prompt}"}


def test_generation_floor_now_covers_template_candidates():
    agent = FixAgent(judge=ScriptedJudge("[]"), max_tier="L1")
    shy = FixCandidate(
        tier=FixTier.L1_PROMPT,
        name="shy",
        kind="template",
        payload={
            "prompt_template": "Think first. {prompt}",
            "generation_kwargs": {"max_tokens": 16},
        },
    )
    pure = FixCandidate(
        tier=FixTier.L1_PROMPT,
        name="pure",
        kind="template",
        payload={"prompt_template": "Think first. {prompt}"},
    )
    agent._max_tokens_floor = 128
    agent._enforce_generation_floor([shy, pure])
    assert shy.payload["generation_kwargs"]["max_tokens"] == 128
    assert shy.payload["generation_kwargs_proposed"] == {"max_tokens": 16}
    # a payload without generation_kwargs is left alone (baseline budget applies)
    assert pure.payload == {"prompt_template": "Think first. {prompt}"}


def test_template_candidate_generates_with_its_decode_room():
    judge = ScriptedJudge(
        json.dumps(
            [
                {
                    "name": "roomy_workspace",
                    "prompt_template": "Show your work, then answer. {prompt}",
                    "generation_kwargs": {"max_tokens": 256},
                }
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(
        DecodeRoomModel(),
        _gold_yes_batch(),
        [_hyp("answers are cut off before the final answer appears")],
    )
    assert out.fixed is True
    assert out.best is not None
    assert out.best.candidate.payload["generation_kwargs"] == {"max_tokens": 256}
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
