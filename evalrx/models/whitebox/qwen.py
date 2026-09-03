"""Qwen text LLMs (white-box, local) — per-version convenience factories.

Model identity lives in :mod:`evalrx.specs`; these factories are **thin
wrappers** over ``compose(spec, "hf_local")`` (via :func:`evalrx.load`), so
there is NO per-version ``load``/``forward`` duplication — the single generic
``hf_local`` backend does the work.  Each ``qwen…(**runtime)`` returns an
hf_local model (caps: GENERATE, LOGITS, LOGPROBS, HIDDEN_STATES, ATTENTION; plus
TOOL_CALLS when the model's chat template supports tools)::

    from evalrx.models.whitebox.qwen import qwen3_8b
    model = qwen3_8b(device="cuda", dtype="bfloat16")
    # equivalent to: evalrx.load("qwen3-8b", backend="hf_local", ...)

``QwenLLM(...)`` remains as a deprecated back-compat shim.
"""

from __future__ import annotations

import warnings
from typing import Any

# Spec keys exposed as factories (each must exist in evalrx.specs).
_TEXT_KEYS = [
    "qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "qwen2.5-32b-instruct", "qwen2.5-72b-instruct",
    "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b", "qwen3-235b-a22b",
]


def _version_factory(key: str):
    """A thin ``(**runtime) -> hf_local model`` builder for a spec *key*."""
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


for _k in _TEXT_KEYS:
    globals()[_k.replace("-", "_").replace(".", "_")] = _version_factory(_k)


def QwenLLM(checkpoint: str | None = None, device: str = "auto", dtype: str = "bfloat16", **runtime: Any):
    """Deprecated. Use a version factory (e.g. ``qwen2_5_7b_instruct()``) or
    ``evalrx.load('qwen2.5-7b-instruct')``."""
    warnings.warn(
        "QwenLLM is deprecated; use a version factory (e.g. qwen2_5_7b_instruct()) "
        "or evalrx.load('qwen2.5-7b-instruct').",
        DeprecationWarning, stacklevel=2,
    )
    from evalrx.models import load
    return load("qwen2.5-7b-instruct", backend="hf_local",
                checkpoint=checkpoint, device=device, dtype=dtype, **runtime)


__all__ = ["QwenLLM"] + [k.replace("-", "_").replace(".", "_") for k in _TEXT_KEYS]
