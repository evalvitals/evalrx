"""Infer a :class:`~evalrx.core.spec.ModelSpec` from a live, already-loaded model.

The curated path (``evalrx.load("qwen...")``) looks a spec up by registry key.
The public on-ramp (``evalrx.wrap(model, tokenizer)``) has no key — the user
brings their own loaded HF model — so we *infer* the same ``ModelSpec`` fields from
the live ``model.config`` and tokenizer instead.  Both paths then feed the identical
:class:`~evalrx.models.backends.hf_local.HFLocalModel`, so every analyzer sees one
contract.

This module is torch-free: it only reads attributes off objects the caller already
constructed.
"""

from __future__ import annotations

from typing import Any

from evalrx.core.spec import AttnSemantics, ModelSpec


def _is_vlm(config: Any) -> bool:
    """Best-effort detection of a vision-language config.

    A text decoder-only config has no separate vision tower; VLM configs carry a
    nested ``vision_config`` (Qwen-VL, LLaVA, GLM-V, …) or expose a distinct text
    sub-config via ``get_text_config()``.
    """
    if getattr(config, "vision_config", None) is not None:
        return True
    if hasattr(config, "get_text_config"):
        try:
            text_cfg = config.get_text_config()
        except Exception:  # pragma: no cover - defensive
            text_cfg = None
        if text_cfg is not None and text_cfg is not config:
            return True
    return False


def _chat_template(tokenizer: Any) -> str:
    """Return the tokenizer's chat template string (``""`` if none)."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # processor -> inner tokenizer
    return getattr(tok, "chat_template", None) or ""


def infer_spec(model: Any, tokenizer: Any) -> ModelSpec:
    """Build a :class:`ModelSpec` describing an already-loaded HF causal LM.

    Args:
        model:     an instantiated ``transformers`` decoder-only model.
        tokenizer: its tokenizer (or processor).

    Returns:
        A ``ModelSpec`` carrying only the fields ``HFLocalModel`` reads at analysis
        time (identity, ``tool_calling``, ``attn_semantics``).  ``hf_repo`` is left
        empty because the weights are already in memory — ``HFLocalModel`` never
        calls ``load()`` on a wrapped model.

    Raises:
        NotImplementedError: if *model* looks like a VLM.  ``wrap()`` supports text
            decoder-only models in this pass; VLM forward-capture (image tokens +
            TokenTypeMap) is Stage 2 — consistent with ``HFLocalModel.forward``.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            "wrap() expects a transformers model with a .config; "
            f"got {type(model).__name__} with none."
        )
    if _is_vlm(config):
        raise NotImplementedError(
            "wrap() supports text decoder-only models; VLM internals capture "
            "(image tokens + TokenTypeMap) is Stage 2."
        )

    model_type = getattr(config, "model_type", None) or "unknown"
    key = getattr(config, "_name_or_path", "") or model_type or "wrapped-model"
    tool_calling = "tools" in _chat_template(tokenizer)

    return ModelSpec(
        key=key,
        family=model_type,
        model_type=model_type,
        hf_repo="",  # already in memory — never reloaded
        tool_calling=tool_calling,
        eager_required_for_attn=True,
        attn_semantics=AttnSemantics.STANDARD,
    )
