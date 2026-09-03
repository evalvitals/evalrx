"""Gemma 4 / Nemotron 3 Nano specs registered for examples/benchmark (2026-08-21)."""

from __future__ import annotations

import pytest

from evalrx.core.spec import AttnSemantics
from evalrx.specs import get_spec

GEMMA = ("gemma-4-e2b-it", "gemma-4-e4b-it", "gemma-4-12b-it")


@pytest.mark.parametrize("key", GEMMA)
def test_gemma4_specs_are_omni_with_thinking_off(key):
    spec = get_spec(key)
    assert spec.family == "gemma4" and spec.hf_repo.startswith("google/gemma-4-")
    assert spec.modalities == frozenset({"text", "image", "audio"})
    assert spec.auto_class == "AutoModelForImageTextToText" and spec.processor_class == "AutoProcessor"
    assert spec.chat_template_kwargs == {"enable_thinking": False} and spec.is_reasoning
    assert spec.attn_semantics is AttnSemantics.STANDARD
    assert spec.vision.image_token_id_attr == "image_token_id" and spec.vision.merge_size_attr is None
    assert spec.audio.audio_token_id_attr == "audio_token_id"
    assert spec.module_paths.decoder_layers == "model.language_model.layers"
    assert spec.min_transformers == "5.15.0"


def test_gemma4_12b_is_the_encoder_free_unified_variant():
    small, unified = get_spec("gemma-4-e4b-it"), get_spec("gemma-4-12b-it")
    assert small.model_type == "gemma4" and unified.model_type == "gemma4_unified"
    assert small.module_paths.vision_tower == "model.vision_tower" and small.audio.audio_tower == "model.audio_tower"
    assert unified.module_paths.vision_tower is None and unified.audio.audio_tower is None


def test_nemotron_4b_hf_local_spec_is_the_bf16_remote_code_hybrid():
    """Remote code on purpose: the native nemotron_h module generated only newline
    tokens on this checkpoint (2026-08-21); the caveat records it."""
    spec = get_spec("nemotron-3-nano-4b")
    assert spec.hf_repo == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert spec.model_type == "nemotron_h" and spec.trust_remote_code
    assert any("REMOTE CODE" in c and "mamba_ssm" in c for c in spec.caveats)
    assert spec.attn_semantics is AttnSemantics.HYBRID_SPARSE
    assert spec.module_paths.decoder_layers == "backbone.layers"
    assert spec.chat_template_kwargs == {"enable_thinking": False} and spec.is_reasoning
    assert not spec.is_vlm and spec.audio is None


@pytest.mark.parametrize("key,bf16", [
    ("nemotron-3-nano-4b-fp8", "nemotron-3-nano-4b"),
    ("nemotron-3-nano-omni-30b-a3b-reasoning-fp8", "nemotron-3-nano-omni-30b-a3b-reasoning"),
])
def test_fp8_exports_are_flagged_endpoint_only_and_mirror_their_bf16_sibling(key, bf16):
    fp8, local = get_spec(key), get_spec(bf16)
    assert fp8.hf_repo.endswith("-FP8") and local.hf_repo.endswith("-BF16")
    assert fp8.caveats[0].startswith("ENDPOINT ONLY") and bf16 in fp8.caveats[0]
    assert fp8.modalities == local.modalities and fp8.family == local.family
    assert fp8.chat_template_kwargs == local.chat_template_kwargs == {"enable_thinking": False}


def test_nemotron_omni_spec_declares_every_modality_through_remote_code():
    spec = get_spec("nemotron-3-nano-omni-30b-a3b-reasoning")
    assert spec.trust_remote_code and spec.is_moe and spec.is_reasoning
    assert spec.modalities == frozenset({"text", "image", "audio", "video"})
    assert spec.auto_class == "AutoModelForCausalLM" and spec.processor_class == "AutoProcessor"
    assert spec.vision.image_token_id_attr == "img_context_token_id"
    assert spec.vision.grid_source == "fixed" and spec.vision.fixed_tokens_per_tile == 256
    assert spec.audio.audio_token_id_attr == "sound_context_token_id"
    assert spec.module_paths.decoder_layers == "language_model.backbone.layers"
    assert spec.attn_semantics is AttnSemantics.HYBRID_SPARSE
    assert any("62 GB" in c for c in spec.caveats)
