"""Analyzer routing follows the modality slots the BATCH fills.

Before this, routing branched on a three-member ModelKind read off the model's
declaration. Two consequences, both silent: an audio model was indistinguishable
from a text-only one (identical analyzer list, audio slot never examined), and an
omni model evaluated on audio was ranked as a VLM because it declares image.
"""

from __future__ import annotations

import pytest

import evalvitals.analyzers  # noqa: F401  (populate the registry)
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Step, StepRole, Trajectory
from evalvitals.core.model import Model
from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe


def _model(*modalities: str, tools: bool = False) -> Model:
    caps = {Capability.GENERATE, Capability.LOGPROBS}
    if tools:
        caps.add(Capability.TOOL_CALLS)
    # Bound outside the class body: a class-level `modalities = ...` shadows the
    # parameter name and the frozenset() call would look it up in class scope.
    slots = frozenset(modalities or ("text",))

    class _M(Model):
        capabilities = frozenset(caps)
        modalities = slots

        def generate(self, inputs, **kw):
            return ""

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    return _M()


def _batch(**slots) -> CaseBatch:
    return CaseBatch([FailureCase(inputs=Inputs(prompt="q", **slots))])


def _traj_batch() -> CaseBatch:
    traj = Trajectory(sample_id="s", goal="g", steps=[Step(idx=0, role=StepRole.PLANNER)])
    return CaseBatch([FailureCase(id="s", inputs=Inputs(prompt="g"), trajectory=traj)])


# ── the regression this exists to prevent ────────────────────────────────────

def test_an_audio_run_is_not_ranked_the_same_as_a_text_run():
    probe = StrategyProbe()
    text = probe.select(_model("text"), max_analyzers=6, data=_batch())
    audio = probe.select(_model("text", "audio"), max_analyzers=6,
                         data=_batch(audio="clip.wav"))
    assert text != audio, "an ALM run produced the identical analyzer list to a text-only run"
    assert "modality_ablation" in audio, "nothing checked whether the model was listening"


def test_an_omni_model_is_routed_on_the_benchmark_not_its_declaration():
    """The load-bearing case: gemma-4 declares image, an ALM benchmark fills audio."""
    omni = _model("text", "image", "audio", "video")
    probe = StrategyProbe()

    on_audio = probe.select(omni, data=_batch(audio="clip.wav"))
    on_image = probe.select(omni, data=_batch(image="scene.png"))

    assert probe.routed_slots(omni, _batch(audio="clip.wav")) == {"text", "audio"}
    assert probe.routed_slots(omni, _batch(image="scene.png")) == {"text", "image"}
    # Image hallucination metrics must not be offered a batch with no images.
    assert not ({"pope", "chair"} & set(on_audio))
    assert {"pope", "chair"} <= set(on_image)


def test_a_batch_with_no_media_falls_back_to_the_declaration():
    """No media persisted is no evidence, not evidence of absence.

    A text-only sample of a vision benchmark must not be read as "images are not
    under test" — that would silently drop every image analyzer.
    """
    vlm = _model("text", "image")
    assert StrategyProbe().routed_slots(vlm, _batch()) == {"text", "image"}


@pytest.mark.parametrize("data", [None, 42, "not a batch"])
def test_unreadable_data_never_narrows_routing(data):
    vlm = _model("text", "image")
    assert StrategyProbe().routed_slots(vlm, data) == {"text", "image"}


# ── the labels, which are display only ───────────────────────────────────────

@pytest.mark.parametrize("modalities,kind", [
    (("text",),                            ModelKind.LLM),
    (("text", "image"),                    ModelKind.VLM),
    (("text", "audio"),                    ModelKind.ALM),
    (("text", "image", "audio"),           ModelKind.AVLM),
    (("text", "audio", "video"),           ModelKind.AVLM),
])
def test_detect_kind_covers_every_modality_combination(modalities, kind):
    assert StrategyProbe().detect_kind(_model(*modalities)) is kind


def test_trajectories_still_win_outright():
    assert StrategyProbe().detect_kind(_model("text", "image"), _traj_batch()) is ModelKind.AGENT
    ranked = StrategyProbe().select(_model("text", "image"), data=_traj_batch())
    assert ranked[0] == "loop_detect"


def test_declared_tool_support_does_not_outrank_media():
    """A VLM that merely supports tools is answering a vision question."""
    ranked = StrategyProbe().select(_model("text", "image", tools=True),
                                    data=_batch(image="scene.png"))
    assert ranked[0] != "loop_detect"


# ── the gate ─────────────────────────────────────────────────────────────────

def test_slot_starved_drops_analyzers_whose_slot_is_empty():
    starved = StrategyProbe.slot_starved({"pope", "chair", "attention"},
                                         _batch(audio="clip.wav"))
    assert starved == {"pope", "chair"}


def test_slot_starved_drops_nothing_without_a_media_batch():
    assert StrategyProbe.slot_starved({"pope", "chair"}, _batch()) == set()
    assert StrategyProbe.slot_starved({"pope", "chair"}, None) == set()


def test_priority_override_is_still_looked_up_by_kind():
    """The documented escape hatch keeps its exact old behaviour."""
    model, data = _model("text", "image"), _batch(image="s.png")
    # Reversed against the built-in image ordering (pope, then chair), so the
    # assertion can only pass if the override actually decided the ranking.
    probe = StrategyProbe(priority_override={ModelKind.VLM: ["chair", "pope"]})
    assert StrategyProbe().select(model, data=data)[:2] == ["pope", "chair"]
    assert probe.select(model, data=data) == ["chair", "pope"]


def test_priority_override_does_not_fill_an_incompatible_pinned_slot():
    model, data = _model("text", "image"), _batch(image="s.png")
    probe = StrategyProbe(priority_override={
        ModelKind.VLM: ["chair", "relative_attention", "pope"],
    })
    # This fake model has no ATTENTION. Pinned mode keeps the two compatible
    # entries and does not substitute an unrelated analyzer to reach three.
    assert probe.select(model, max_analyzers=3, data=data) == ["chair", "pope"]


def test_a_text_only_model_is_not_offered_media_analyzers():
    """The no-media fallback is about unpersisted media, not about text models.

    A text-only batch from a VLM may simply be a sample whose images did not
    make it to disk, so the gate stays off. A text-only MODEL removes that
    ambiguity — it could not have had an image — and leaving the gate off there
    offered a plain LLM run every media analyzer in the registry.
    """
    llm, data = _model("text"), _batch()
    assert StrategyProbe.slot_starved({"modality_ablation", "mm_shap"}, data, llm) == {
        "modality_ablation", "mm_shap",
    }
    assert "modality_ablation" not in StrategyProbe().select(llm, data=data)
    # Same batch, a model that CAN see: ambiguous, so nothing is dropped.
    assert StrategyProbe.slot_starved({"modality_ablation"}, data, _model("text", "image")) == set()
