"""Helpers shared by :class:`~evalrx.models.backends.hf.model.HFLocalModel` and its
repair mixins: processor-output bookkeeping for image / audio inputs, new-token
slicing, and reading nested config attributes. torch-free at import."""

from __future__ import annotations

import logging
from typing import Any

from evalrx.models._media import AUDIO_SAMPLE_RATE
from evalrx.models._media import resolve_image as _resolve_image

logger = logging.getLogger(__name__)


def _read_nested_attr(obj: Any, attr_path: "str | None", *, default: Any) -> Any:
    """Walk a dotted attribute path on *obj*, returning *default* if any step is missing."""
    if not attr_path:
        return default
    for part in attr_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


def _populate_vision_extras(
    extras: dict,
    input_ids: Any,  # CPU torch.Tensor
    proc_in: dict,
    model_config: Any,
    vision: Any,  # VisionSpec — avoid circular import; duck-typed
) -> None:
    """Fill *extras* with image-token mask and spatial layout for analyzers.

    Called from ``HFLocalModel._vlm_forward`` after the processor runs.
    Writes:
      ``image_token_mask``    — bool tensor (seq_len,) marking image-pad positions.
      ``image_spatial_shape`` — (H, W) patch grid after spatial merge, for reshaping.
      ``image_grid_thw``      — raw (T, H, W) tensor if grid_source=="grid_thw".
    """
    # A dotted attr (e.g. Omni's "thinker_config.image_token_id") needs the
    # nested walker; a bare name behaves identically to getattr(), so no VLM
    # spec's resolution changes.
    image_token_id = _read_nested_attr(model_config, vision.image_token_id_attr, default=None)
    if image_token_id is not None:
        extras["image_token_mask"] = input_ids == image_token_id

    merge = int(_read_nested_attr(model_config, vision.merge_size_attr, default=1) or 1)

    if vision.grid_source == "grid_thw":
        grid_t = proc_in.get("image_grid_thw")
        if grid_t is not None:
            grid = grid_t.cpu() if hasattr(grid_t, "cpu") else grid_t
            extras["image_grid_thw"] = grid
            _, h, w = int(grid[0, 0]), int(grid[0, 1]), int(grid[0, 2])
            extras["image_spatial_shape"] = (h // merge, w // merge)
    elif vision.grid_source == "grid_hw":
        grid_t = proc_in.get("image_grid_hw")
        if grid_t is not None:
            grid = grid_t.cpu() if hasattr(grid_t, "cpu") else grid_t
            extras["image_grid_hw"] = grid
            h, w = int(grid[0, 0]), int(grid[0, 1])
            extras["image_spatial_shape"] = (h // merge, w // merge)


def _populate_audio_extras(
    extras: dict,
    input_ids: Any,  # CPU torch.Tensor
    model_config: Any,
    audio: Any,  # AudioSpec — avoid circular import; duck-typed
) -> None:
    """Fill *extras* with the audio-token mask (the audio TokenTypeMap analog).

    Symmetric to :func:`_populate_vision_extras`, minus the spatial-grid part —
    audio placeholders are a flat run of one token per encoded frame-group, no
    2D reshape. Writes ``audio_token_mask`` — bool tensor (seq_len,) marking
    audio-placeholder positions — which downstream paper methods (e.g. a
    contrastive-decoding audio-reliance signal) read to isolate how much of the
    decoder's attention lands on audio vs. text/image tokens.
    """
    audio_token_id = _read_nested_attr(model_config, audio.audio_token_id_attr, default=None)
    if audio_token_id is not None:
        extras["audio_token_mask"] = input_ids == audio_token_id


def _check_audio_duration(audios: list, processor: Any, model_key: str) -> None:
    """Raise before the processor silently truncates audio past its encoder window.

    ``WhisperFeatureExtractor.chunk_length`` (seconds) differs per checkpoint —
    Qwen2-Audio's is 30s, Qwen2.5-Omni's is 300s — and padding beyond it is
    silently dropped rather than erroring (verified empirically: token count
    and ``feature_attention_mask`` both cap at the window with no warning).
    Read live from the processor, never baked, per this module's convention.
    """
    feature_extractor = getattr(processor, "feature_extractor", None)
    chunk_length = getattr(feature_extractor, "chunk_length", None)
    if chunk_length is None:
        return  # nothing to check this checkpoint's contract against
    # ``audios`` are already resolved to AUDIO_SAMPLE_RATE by _resolve_audio, but
    # the limit itself is read from the feature extractor's own rate rather than
    # the module constant, so this stays correct if that ever diverges.
    sampling_rate = getattr(feature_extractor, "sampling_rate", AUDIO_SAMPLE_RATE)
    limit_samples = int(chunk_length) * int(sampling_rate)
    for i, wav in enumerate(audios):
        if len(wav) > limit_samples:
            duration_sec = len(wav) / sampling_rate
            logger.warning(
                "%s: audio[%d] is %.1fs, longer than the %ds encoder window — refusing rather "
                "than letting the processor silently truncate it",
                model_key, i, duration_sec, chunk_length,
            )
            raise ValueError(
                f"{model_key}: audio[{i}] is {duration_sec:.1f}s, longer than "
                f"this checkpoint's {chunk_length}s encoder window — it would be silently "
                "truncated rather than raising inside the processor. Chunk the audio yourself "
                "before calling, or accept a documented context window in the caller."
            )


def _new_tokens(out: Any, input_ids: Any) -> Any:
    """The newly generated token ids from a ``generate()`` output sequence.

    ``out`` is normally the prompt concatenated with the continuation, so the
    new tokens start after ``len(input_ids)``. A model whose own
    ``generate()`` converts the input to ``inputs_embeds`` internally
    (Qwen3-Omni and other omni/audio architectures whose thinker merges
    audio/vision embeddings before the LM) never hands ``input_ids`` back to
    ``GenerationMixin`` — which then cannot prepend prompt token ids it was
    never given, so it returns ONLY the new tokens (transformers itself warns
    about this: "... without input_ids to generate ..."). Slicing at
    ``len(input_ids)`` in that case cuts past the end of a sequence shorter
    than the prompt and silently decodes to "" — seen live on
    Qwen3-Omni-30B-A3B-Instruct/MMAU, every one of 128 cases, 2026-08-27.

    A length check alone (``out.shape[0] > len(input_ids)``) is ambiguous at
    the edges — a concatenated output with zero new tokens, or an
    inputs_embeds-only output that happens to be exactly as long as the
    prompt — so this compares the actual leading token ids: only a real
    prefix match means ``out`` truly contains the prompt to cut off.
    """
    import torch

    input_len = len(input_ids)
    if out.shape[0] >= input_len and torch.equal(out[:input_len], torch.as_tensor(input_ids)):
        return out[input_len:]
    return out


def _collect_message_images(messages: list) -> list:
    """Extract images from chat messages in appearance order.

    Messages follow the transformers content-block convention: ``content`` is a
    plain string OR a list of blocks, where an image block is
    ``{"type": "image", "image": <PIL | path>}``.  The block order across the
    whole conversation must match the ``images=`` list handed to the processor.
    """
    images: list = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                img = block.get("image")
                if img is not None:
                    images.append(_resolve_image(img))
    return images
