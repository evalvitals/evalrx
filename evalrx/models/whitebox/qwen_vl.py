"""Qwen vision-language models (white-box, local) — per-version convenience factories.

Same pattern as :mod:`evalrx.models.whitebox.qwen`: thin wrappers over
``compose(spec, "hf_local")`` (via :func:`evalrx.load`); identity lives in
:mod:`evalrx.specs`.  Each ``qwen…vl…(**runtime)`` returns an hf_local VLM
(modalities ``{"text", "image"}``)::

    from evalrx.models.whitebox.qwen_vl import qwen3_vl_8b_instruct
    model = qwen3_vl_8b_instruct(device="cuda", dtype="bfloat16")

NOTE: text generation + logprobs work today; white-box forward-capture over image
tokens (attention/hidden) uses the hf_local VLM path + ``core.tokentype.TokenTypeMap``.
``QwenVL(...)`` remains as a deprecated back-compat shim.
"""

from __future__ import annotations

import warnings
from typing import Any

# Spec keys exposed as factories (each must exist in evalrx.specs).
_VL_KEYS = [
    "qwen2-vl-7b-instruct",
    "qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct", "qwen2.5-vl-72b-instruct",
    "qwen3-vl-2b-instruct", "qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct",
    "qwen3-vl-30b-a3b-instruct", "qwen3-vl-235b-a22b-instruct",
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


for _k in _VL_KEYS:
    globals()[_k.replace("-", "_").replace(".", "_")] = _version_factory(_k)


def QwenVL(checkpoint: str | None = None, device: str = "auto", dtype: str = "bfloat16", **runtime: Any):
    """Deprecated. Use a version factory (e.g. ``qwen3_vl_8b_instruct()``) or
    ``evalrx.load('qwen2.5-vl-7b-instruct')``."""
    warnings.warn(
        "QwenVL is deprecated; use a version factory (e.g. qwen3_vl_8b_instruct()) "
        "or evalrx.load('qwen2.5-vl-7b-instruct').",
        DeprecationWarning, stacklevel=2,
    )
    from evalrx.models import load
    return load("qwen2.5-vl-7b-instruct", backend="hf_local",
                checkpoint=checkpoint, device=device, dtype=dtype, **runtime)


__all__ = ["QwenVL"] + [k.replace("-", "_").replace(".", "_") for k in _VL_KEYS]
