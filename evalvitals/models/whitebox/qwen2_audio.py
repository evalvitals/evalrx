"""Qwen2-Audio-Instruct (white-box, local) — convenience factory.

Audio-in / text-out only (no vision tower) — the simplest audio-capable spec
in the registry. Same thin-wrapper pattern as the other whitebox factories::

    from evalvitals.models.whitebox.qwen2_audio import qwen2_audio_7b_instruct
    model = qwen2_audio_7b_instruct(device="cuda", dtype="bfloat16")
    model.modalities   # frozenset({'text', 'audio'})

Reference: https://github.com/QwenLM/Qwen2-Audio
"""

from __future__ import annotations

from typing import Any


def qwen2_audio_7b_instruct(**runtime: Any):
    """Build 'qwen2-audio-7b-instruct' on the hf_local (white-box) backend."""
    from evalvitals.models import load

    return load("qwen2-audio-7b-instruct", backend="hf_local", **runtime)


__all__ = ["qwen2_audio_7b_instruct"]
