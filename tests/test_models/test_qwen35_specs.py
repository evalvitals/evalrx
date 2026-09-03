"""Qwen3.5 specs: thinking is OFF on every chat-template render.

The two released checkpoints disagree on the template default when
``enable_thinking`` is absent (Qwen3.5-2B: off, Qwen3.5-9B: on), so the specs
send the kwarg explicitly; hf_local / vllm_offline forward
``spec.chat_template_kwargs`` into every ``apply_chat_template`` call.
"""

from __future__ import annotations

import pytest

from evalrx.specs import get_spec

TEXT_KEYS = ("qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b")


@pytest.mark.parametrize("key", TEXT_KEYS)
def test_text_specs_disable_thinking_explicitly(key):
    spec = get_spec(key)
    assert spec.chat_template_kwargs == {"enable_thinking": False}
    assert spec.is_reasoning and not spec.is_vlm
    assert spec.auto_class == "AutoModelForCausalLM"


@pytest.mark.parametrize("key", TEXT_KEYS)
def test_vl_specs_load_the_same_checkpoint_with_its_vision_tower(key):
    """``<key>-vl``: same repo, vision tower on, image modality offered, thinking
    still off. Module hints follow the Qwen3-VL layout verified on transformers
    5.15 (model.language_model.layers / model.visual.blocks)."""
    text, vl = get_spec(key), get_spec(f"{key}-vl")
    assert vl.hf_repo == text.hf_repo and vl.family == text.family == "qwen3_5"
    assert vl.is_vlm and "image" in vl.modalities and "image" not in text.modalities
    assert vl.auto_class == "AutoModelForImageTextToText"
    assert vl.processor_class == "AutoProcessor"
    assert vl.chat_template_kwargs == {"enable_thinking": False}
    assert vl.attn_semantics is text.attn_semantics
    assert vl.module_paths.decoder_layers == "model.language_model.layers"
    assert vl.module_paths.vision_blocks == "model.visual.blocks"
    assert vl.vision.image_token_id_attr == "image_token_id"
    assert vl.vision.merge_size_attr == "vision_config.spatial_merge_size"
    assert vl.vision.grid_source == "grid_thw"
    assert vl.min_transformers == "5.15.0"
