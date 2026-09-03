"""Model construction — one engine: ``compose(spec, backend)``.

``evalrx.specs`` is the single source of truth for *which* models exist;
``models/backends`` provides the runtimes.  Everything funnels through
:func:`load` (friendly, key-based) and :func:`load_model` (``ModelConfig``-based),
both of which build via ``compose`` — so there is exactly one way a model is
constructed, and the capability set always reflects the chosen backend.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from evalrx.config import ModelConfig
from evalrx.core.capability import Capability
from evalrx.core.model import Model

# Backend layer + agent loop (torch-free at import; heavy deps are lazy in build/load).
from evalrx.core.tool import ChatTurn, Tool, ToolCall
from evalrx.models.agent import Agent, APIToolHandlerExecutor, ToolExecutor
from evalrx.models.backends import BACKENDS, RuntimeConfig, call_vision_api_chat_fn
from evalrx.models.base import BaseAgent
from evalrx.models.compose import compose
from evalrx.models.toolcodec import OpenAIToolCodec, QwenToolCodec, codec_for

# Legacy concrete white-box model (now a thin deprecated alias) — optional on the
# light, pure-API install (its module imports nothing heavy at top level).
try:
    from evalrx.models import whitebox as _whitebox  # noqa: F401
except ImportError:  # pragma: no cover - light install
    _whitebox = None

__all__ = [
    "Model",
    "BaseAgent",
    "load",
    "load_model",
    "wrap",
    "compose",
    "RuntimeConfig",
    "BACKENDS",
    "resolve_spec_key",
    "Agent",
    "ToolExecutor",
    "APIToolHandlerExecutor",
    "Tool",
    "ToolCall",
    "ChatTurn",
    "OpenAIToolCodec",
    "QwenToolCodec",
    "codec_for",
    "call_vision_api_chat_fn",
]

# Legacy name → canonical spec key.  Keeps old configs/code working.
_SPEC_ALIASES: dict[str, str] = {
    "qwen": "qwen2.5-7b-instruct",
    "qwen2.5-7b": "qwen2.5-7b-instruct",
}


def resolve_spec_key(name: str) -> str:
    """Resolve a spec key or legacy alias to a canonical :mod:`evalrx.specs` key."""
    from evalrx.specs import REGISTRY

    key = name.lower()
    if key in REGISTRY:
        return key
    if key in _SPEC_ALIASES:
        return _SPEC_ALIASES[key]
    raise ValueError(
        f"Unknown model {name!r}. Known specs: {sorted(REGISTRY)}; "
        f"aliases: {sorted(_SPEC_ALIASES)}."
    )


def load(
    key: str,
    *,
    backend: str = "hf_local",
    want: "list[str] | set[Capability] | tuple" = (),
    checkpoint: str | None = None,
    **runtime: Any,
) -> Model:
    """Build a model by spec key — the friendly one-liner.

    ``evalrx.load("qwen2.5-7b-instruct")`` is the modern replacement for the
    old ``QwenLLM()``.  It wraps ``compose(get_spec(key), backend, RuntimeConfig(**runtime))``
    and negotiates capabilities up front (``want``).

    Args:
        key:        Spec key or legacy alias (see :func:`resolve_spec_key`).
        backend:    ``"hf_local"`` (internals) / ``"api"`` / ``"vllm_offline"``.
                    Forced to ``"api"`` for ``api_only`` (closed-weight) specs.
        want:       Capabilities the backend must provide (names or ``Capability``).
        checkpoint: Optional override of the spec's ``hf_repo``.
        **runtime:  Forwarded to :class:`RuntimeConfig` (``device``, ``dtype``, …).
    """
    from evalrx.specs import get_spec

    spec = get_spec(resolve_spec_key(key))
    if checkpoint:
        spec = replace(spec, hf_repo=checkpoint)
    chosen = "api" if spec.api_only else backend
    caps = {w if isinstance(w, Capability) else Capability(w) for w in want}
    return compose(spec, chosen, RuntimeConfig(**runtime), caps)


def load_model(cfg: ModelConfig) -> Model:
    """Instantiate a model from a :class:`~evalrx.config.ModelConfig` via :func:`load`."""
    return load(
        cfg.name,
        backend=cfg.backend,
        want=cfg.want,
        checkpoint=cfg.checkpoint,
        device=cfg.device,
        dtype=cfg.dtype,
    )


def wrap(
    model: Any,
    tokenizer: Any = None,
    *,
    want: "list[str] | set[Capability] | tuple" = (),
    **runtime: Any,
) -> Model:
    """Wrap an ALREADY-LOADED HF causal LM + tokenizer into an analyzable model.

    The captum-style public on-ramp: the user brings their own model, no registry
    key needed.  The result is the same :class:`~evalrx.models.backends.hf_local.HFLocalModel`
    that ``evalrx.load("qwen...")`` produces, so every capability-compatible
    analyzer works on it::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained("my-org/my-llama")
        tok = AutoTokenizer.from_pretrained("my-org/my-llama")

        wrapped = evalrx.wrap(m, tok)
        LogitLensAnalyzer().run(wrapped, "The capital of France is")

    Capabilities are inferred from the live model; attention capture needs eager
    attention, which ``wrap`` enables when it can (see ``HFLocalModel.from_loaded``).

    Args:
        model:     an instantiated ``transformers`` decoder-only model.
        tokenizer: its tokenizer or processor (positional, recommended).
        want:      capabilities to assert up front (names or ``Capability``); raises
                   :class:`~evalrx.core.capability.CapabilityError` if unavailable.
        **runtime: forwarded to :class:`RuntimeConfig` (``max_new_tokens``, ``device``…).

    Raises:
        NotImplementedError: if *model* looks like a VLM (Stage 2; text-only for now).
    """
    from evalrx.models.backends.hf_local import HFLocalModel
    from evalrx.models.compose import negotiate

    handle = HFLocalModel.from_loaded(model, tokenizer, runtime=RuntimeConfig(**runtime))
    caps = {w if isinstance(w, Capability) else Capability(w) for w in want}
    return negotiate(handle, caps, model_id=handle.spec.key, where="wrap")
