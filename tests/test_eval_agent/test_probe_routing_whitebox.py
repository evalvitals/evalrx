"""Routing of the white-box intervention analyzers (knockout, causal trace, gradients, lenses).

Two guarantees:

* every analyzer a routing table names is actually registered — a misspelt
  name is silently dropped by ``select`` (the video list carried
  ``"relative_attn"`` for months), so the table must be checked, not trusted;
* a white-box VLM/ALM gets the new analyzers in its priority list and when an
  M3 hypothesis names the mechanism they test, while a black-box model never
  sees them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # the attention analyzers register only with torch

import evalrx.analyzers  # noqa: F401,E402 - populate the registry
from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs  # noqa: E402
from evalrx.core.registry import registry  # noqa: E402
from evalrx.eval_agent.stages.probe import (  # noqa: E402
    _FAILURE_MODE_TO_ANALYZERS,
    _PRIORITY,
    _SLOT_PRIORITY,
    StrategyProbe,
)
from tests.conftest import FakeModel  # noqa: E402

WRITE_SIDE = {"attention_knockout", "attention_sensitivity", "causal_trace", "attribution_patching",
              "gradcam", "generic_attention", "head_attribution", "tuned_lens"}
_WHITE_BOX = {Capability.GENERATE, Capability.LOGITS, Capability.LOGPROBS,
              Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.GRADIENTS}


def _image_batch() -> CaseBatch:
    return CaseBatch([FailureCase(inputs=Inputs(prompt="Is there a dog?", image="dog.png"))])


def _audio_batch() -> CaseBatch:
    return CaseBatch([FailureCase(inputs=Inputs(prompt="What do you hear?", audio="a.wav"))])


def test_every_routed_analyzer_is_registered():
    registered = set(registry.analyzers.list())
    tables = {
        **{f"failure_mode:{k}": v for k, v in _FAILURE_MODE_TO_ANALYZERS.items()},
        **{f"slot:{k}": v for k, v in _SLOT_PRIORITY.items()},
        **{f"kind:{k}": v for k, v in _PRIORITY.items()},
    }
    unknown = {where: sorted(set(names) - registered) for where, names in tables.items()
               if set(names) - registered}
    assert not unknown, f"routing tables name unregistered analyzers: {unknown}"


def test_every_write_side_analyzer_is_routed_somewhere():
    routed = {name for names in _FAILURE_MODE_TO_ANALYZERS.values() for name in names}
    routed |= {name for names in _SLOT_PRIORITY.values() for name in names}
    assert WRITE_SIDE <= routed


def test_white_box_vlm_ranks_knockout_right_after_modality_ablation():
    vlm = FakeModel(capabilities=_WHITE_BOX, modalities={"text", "image"})
    ranked = StrategyProbe().select(vlm, data=_image_batch())
    assert ranked.index("attention_knockout") == ranked.index("modality_ablation") + 1
    assert WRITE_SIDE - {"tuned_lens", "head_attribution"} <= set(ranked[: len(_SLOT_PRIORITY["image"]) + 2])


def test_white_box_alm_gets_the_audio_interventions():
    alm = FakeModel(capabilities=_WHITE_BOX, modalities={"text", "audio"})
    ranked = StrategyProbe().select(alm, data=_audio_batch())
    assert ranked.index("attention_knockout") == ranked.index("modality_ablation") + 1
    assert "gradcam" not in ranked  # image-only: starved on an audio batch


@pytest.mark.parametrize("mode, first", [
    ("visual_neglect", "attention_knockout"),
    ("language_prior_bias", "prompt_contrast"),
    ("insufficient_visual_attention", "attention_sensitivity"),
    ("visual_information_loss", "causal_trace"),
    ("mislocalized_attention", "gradcam"),
    ("late_suppression", "tuned_lens"),
])
def test_m3_failure_modes_promote_the_analyzer_that_tests_them(mode, first):
    vlm = FakeModel(capabilities=_WHITE_BOX, modalities={"text", "image"})
    ranked = StrategyProbe().select(vlm, data=_image_batch(), hint_failure_modes=[mode])
    assert ranked[0] == first
    assert set(_FAILURE_MODE_TO_ANALYZERS[mode]) & set(ranked) <= set(ranked[: len(_FAILURE_MODE_TO_ANALYZERS[mode])])


def test_language_prior_bias_now_reaches_the_media_interventions():
    assert {"attention_knockout", "attention_sensitivity"} <= set(_FAILURE_MODE_TO_ANALYZERS["language_prior_bias"])


def test_black_box_model_is_never_offered_write_side_analyzers():
    api = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS},
                    modalities={"text", "image"})
    ranked = StrategyProbe().select(api, data=_image_batch(), hint_failure_modes=["visual_neglect"])
    assert not WRITE_SIDE & set(ranked)


def test_m3_prompt_lists_the_white_box_tags():
    from evalrx.eval_agent.prompts import diagnosis

    text = " ".join(str(v) for v in vars(diagnosis).values() if isinstance(v, str))
    for tag in ("visual_neglect", "insufficient_visual_attention", "visual_information_loss",
                "mislocalized_attention", "late_suppression"):
        assert tag in text and tag in _FAILURE_MODE_TO_ANALYZERS
