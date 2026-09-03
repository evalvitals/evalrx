"""Qwen3-Omni models (white-box, local) — per-version convenience factories.

Same pattern as :mod:`evalrx.models.whitebox.qwen` /
:mod:`~evalrx.models.whitebox.qwen_vl`: thin wrappers over
``compose(spec, "hf_local")`` (via :func:`evalrx.load`); identity lives in
:mod:`evalrx.specs`.  An "omni" model is **not** a new class — it is a spec
that carries vision *and* audio (+ video), so its modality set is
``{"text", "image", "audio", "video"}`` and analyzers match on it::

    from evalrx.models.whitebox.qwen_omni import qwen3_omni_30b_a3b_instruct
    model = qwen3_omni_30b_a3b_instruct(device="cuda", dtype="bfloat16")
    model.modalities   # frozenset({'text', 'image', 'audio', 'video'})

NOTE: text generation works via the thinker; full multimodal generate
(image/audio/video in) and white-box forward-capture over the vision/audio towers
are Stage-2 (needs ``transformers>=5.2.0`` and ``qwen_omni_utils.process_mm_info``;
pass ``use_audio_in_video`` consistently).  The Captioner is audio-in / text-out
(modalities ``{"text", "audio"}``).

Reference: https://github.com/QwenLM/Qwen3-Omni
"""

from __future__ import annotations

from typing import Any

# Spec keys exposed as factories (each must exist in evalrx.specs).
_OMNI_KEYS = [
    "qwen3-omni-30b-a3b-instruct",
    "qwen3-omni-30b-a3b-thinking",
    "qwen3-omni-30b-a3b-captioner",
]


def _version_factory(key: str):
    def build(**runtime: Any):
        from evalrx.models import load
        return load(key, backend="hf_local", **runtime)

    name = key.replace("-", "_").replace(".", "_")
    build.__name__ = build.__qualname__ = name
    build.__doc__ = (
        f"Build {key!r} on the hf_local (white-box) backend. "
        f"Thin wrapper over compose('{key}', 'hf_local')."
    )
    return build


for _k in _OMNI_KEYS:
    globals()[_k.replace("-", "_").replace(".", "_")] = _version_factory(_k)


__all__ = [k.replace("-", "_").replace(".", "_") for k in _OMNI_KEYS]
