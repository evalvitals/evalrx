"""Qwen2.5-Omni (white-box, local) — convenience factory.

Same pattern as :mod:`evalrx.models.whitebox.qwen_omni`: a thin wrapper over
``compose(spec, "hf_local")`` (via :func:`evalrx.load`); identity lives in
:mod:`evalrx.specs`. Registered separately from the Qwen3-Omni family
because it is the checkpoint published paper methods (e.g. audio contrastive
decoding) tend to report hyperparameters against — see the spec's caveats for
its 300s audio-window truncation guard::

    from evalrx.models.whitebox.qwen2_5_omni import qwen2_5_omni_7b
    model = qwen2_5_omni_7b(device="cuda", dtype="bfloat16")
    model.modalities   # frozenset({'text', 'image', 'audio', 'video'})

Reference: https://github.com/QwenLM/Qwen2.5-Omni
"""

from __future__ import annotations

from typing import Any


def qwen2_5_omni_7b(**runtime: Any):
    """Build 'qwen2.5-omni-7b' on the hf_local (white-box) backend."""
    from evalrx.models import load

    return load("qwen2.5-omni-7b", backend="hf_local", **runtime)


__all__ = ["qwen2_5_omni_7b"]
