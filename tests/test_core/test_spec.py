"""ModelSpec layer: backend-orthogonal specs + registry."""

from __future__ import annotations

import pytest

from evalvitals.core.spec import AttnSemantics, AudioSpec, ModelSpec, VisionSpec
from evalvitals.specs import REGISTRY, get_spec, list_specs


def test_llm_spec_has_no_vision():
    spec = get_spec("qwen3-8b")
    assert spec.vision is None
    assert spec.is_vlm is False
    assert spec.is_omni is False
    assert spec.modalities == frozenset({"text"})
    assert spec.api_only is False


def test_vlm_spec_has_vision():
    spec = get_spec("qwen3-vl-8b-instruct")
    assert isinstance(spec.vision, VisionSpec)
    assert spec.is_vlm is True
    assert spec.is_omni is False
    assert spec.modalities == frozenset({"text", "image"})
    # token id is read from config by attribute NAME, never baked as a value
    assert spec.vision.image_token_id_attr == "image_token_id"


def test_omni_instruct_spec_has_all_modalities():
    spec = get_spec("qwen3-omni-30b-a3b-instruct")
    assert spec.is_omni is True
    assert spec.is_vlm is True  # carries a vision tower too
    assert isinstance(spec.vision, VisionSpec) and isinstance(spec.audio, AudioSpec)
    assert spec.video is True
    assert spec.modalities == frozenset({"text", "image", "audio", "video"})
    assert spec.is_moe is True


def test_omni_captioner_is_audio_only():
    # audio-in / text-out: no vision tower, not a VLM, still an omni model.
    spec = get_spec("qwen3-omni-30b-a3b-captioner")
    assert spec.is_omni is True and spec.is_vlm is False
    assert spec.modalities == frozenset({"text", "audio"})


def test_omni_thinking_spec_reasons():
    spec = get_spec("qwen3-omni-30b-a3b-thinking")
    assert spec.is_omni is True and spec.is_reasoning is True
    assert spec.modalities == frozenset({"text", "image", "audio", "video"})


def test_api_only_spec():
    spec = get_spec("step-1o-vision")
    assert spec.api_only is True
    assert spec.module_paths is None
    assert spec.attn_semantics is AttnSemantics.NONE


def test_mla_models_flagged():
    for key in ("deepseek-v3", "kimi-vl-a3b-thinking"):
        assert get_spec(key).attn_semantics is AttnSemantics.MLA_LATENT


def test_registry_listing_and_unknown():
    assert "qwen3-vl-8b-instruct" in list_specs()
    assert set(list_specs()) == set(REGISTRY)
    with pytest.raises(KeyError):
        get_spec("does-not-exist")


def test_spec_is_frozen_dataclass():
    spec = ModelSpec(key="t", family="t", model_type="t")
    with pytest.raises(Exception):
        spec.key = "other"  # frozen


def test_qwen2_5_omni_spec_has_all_modalities():
    spec = get_spec("qwen2.5-omni-7b")
    assert spec.is_omni is True and spec.is_vlm is True
    assert isinstance(spec.vision, VisionSpec) and isinstance(spec.audio, AudioSpec)
    assert spec.modalities == frozenset({"text", "image", "audio", "video"})
    # nested under thinker_config on this family — not a bare top-level attr,
    # since Qwen2_5OmniConfig has no attribute_map forwarding it itself.
    assert spec.vision.image_token_id_attr == "thinker_config.image_token_id"
    assert spec.audio.audio_token_id_attr == "thinker_config.audio_token_id"
    assert spec.needs_multimodal_encode is True


def test_qwen2_audio_spec_is_audio_only():
    spec = get_spec("qwen2-audio-7b-instruct")
    assert spec.is_omni is True and spec.is_vlm is False
    assert spec.modalities == frozenset({"text", "audio"})
    # top-level config here, unlike the Omni families' thinker_config nesting.
    assert spec.audio.audio_token_id_attr == "audio_token_id"
    assert spec.audio.audio_tower == "model.audio_tower"
    assert spec.needs_multimodal_encode is True


def test_needs_multimodal_encode_false_for_plain_llm():
    spec = get_spec("qwen3-8b")
    assert spec.needs_multimodal_encode is False
