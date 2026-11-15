"""Runtime module-path discovery for white-box hooking.

Hardcoded dotted paths (``model.model.language_model.layers``) are the #1 source
of silent white-box bugs: they differ per model AND per transformers release (the
doubled-``.model.`` / Llama4-no-``.model`` / v5 fused-experts traps).  So instead
of trusting ``ModelSpec.module_paths``, we DISCOVER the decoder-layer
``ModuleList`` from the live module tree and derive everything relative to it.

``torch`` is imported lazily inside the functions, so importing this module on the
light (pure-API) install does not pull torch.
"""

from __future__ import annotations

from typing import Any


def find_decoder_layers(model: Any, num_hidden_layers: int | None = None) -> tuple[Any, str]:
    """Find the decoder-layer ``ModuleList`` and its dotted path.

    Strategy: among all ``nn.ModuleList`` modules, prefer the one whose length
    equals ``num_hidden_layers`` (read from config when not given) and whose
    children look like decoder layers (have a ``self_attn`` attr or a
    ``*DecoderLayer`` class name).  Ties broken by the *shallowest* path so we
    pick the language tower, not a vision encoder's block list.
    """
    import torch.nn as nn

    if num_hidden_layers is None:
        cfg = getattr(model, "config", None)
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        num_hidden_layers = getattr(text_cfg, "num_hidden_layers", None)

    candidates: list[tuple[str, Any]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList) or len(mod) == 0:
            continue
        looks_like_layer = hasattr(mod[0], "self_attn") or "DecoderLayer" in type(mod[0]).__name__
        if not looks_like_layer:
            continue
        if num_hidden_layers is not None and len(mod) != num_hidden_layers:
            continue
        candidates.append((name, mod))

    if not candidates:
        raise RuntimeError(
            "Could not discover the decoder-layer ModuleList"
            + (f" (expected length {num_hidden_layers})" if num_hidden_layers else "")
            + f". Tree top-level: {[n for n, _ in model.named_children()]}"
        )
    name, layers = min(candidates, key=lambda kv: kv[0].count("."))
    return layers, name


def get_unembed(model: Any) -> Any:
    """Return the unembedding (lm_head), handling tied embeddings.

    Uses transformers' own ``get_output_embeddings()`` so we never hardcode the
    ``lm_head`` path and the tied-weights case is handled by the model itself.
    """
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None:  # tied / unusual — fall back to input embeddings' weight
        emb = model.get_input_embeddings()
        return emb
    return head


def get_final_norm(model: Any) -> Any:
    """Return the final normalization module applied after the last decoder layer.

    Found relative to the discovered decoder-layer list: its parent module's
    norm child (``norm`` / ``final_layernorm`` / ``ln_f``, else any direct
    child whose class name contains "Norm").  RMSNorm-family models (Llama,
    Qwen, ...) need this applied before unembedding or logit-lens trajectories
    distort.  Returns ``None`` when nothing can be discovered (lens callers
    must degrade gracefully).
    """
    try:
        _, layers_path = find_decoder_layers(model)
    except Exception:
        return None
    parent = model if "." not in layers_path else resolve(model, layers_path.rsplit(".", 1)[0])
    layers_attr = layers_path.rsplit(".", 1)[-1]
    for attr in ("norm", "final_layernorm", "ln_f"):
        mod = getattr(parent, attr, None)
        if mod is not None and "norm" in type(mod).__name__.lower():
            return mod
    for name, mod in parent.named_children():
        if name != layers_attr and "norm" in type(mod).__name__.lower():
            return mod
    return None


def resolve(model: Any, dotted: str) -> Any:
    """Resolve a dotted attribute path (``a.b.c``) against *model*, raising clearly."""
    obj = model
    for part in dotted.split("."):
        if not hasattr(obj, part):
            raise AttributeError(f"path {dotted!r} broke at {part!r}")
        obj = getattr(obj, part)
    return obj
