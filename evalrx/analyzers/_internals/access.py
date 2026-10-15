"""Reach the live module behind a white-box handle, and the positions to act on.

Only the ``hf_local`` backend holds a torch module (``model._loaded`` lazily
loads and returns ``(module, processor)``). Everything that needs one goes
through :func:`torch_module`, so an API or fake model fails with one clear
message instead of an ``AttributeError`` three calls deep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.model import Model, Trace

#: Architectures the write-side analyzers can edit. ``standard``: every decoder
#: layer is a dense token×token softmax that the registered attention function
#: sees in full. ``hybrid_sparse`` (Qwen3.5, Qwen3-Next): softmax layers plus
#: gated-delta-rule linear-attention layers, which ``linear_mix`` edits — but
#: only when EVERY layer is one of the two kinds (checked per stack by
#: :attr:`DecoderStack.editable`; a Mamba mixer is neither). Latent /
#: unverified attention is not a plain token matrix and stays refused.
_EDITABLE_SEMANTICS = frozenset({"standard", "hybrid_sparse"})


def torch_module(model: "Model", analyzer: str = "this analyzer") -> Any:
    """The ``torch.nn.Module`` behind *model*; ``ValueError`` when there is none."""
    import torch.nn as nn

    loaded = getattr(model, "_loaded", None)  # HFLocalModel: lazy-loads on access
    module = loaded[0] if isinstance(loaded, tuple) else loaded
    if not isinstance(module, nn.Module):
        raise ValueError(
            f"{analyzer} edits or differentiates the forward pass, so it needs a local "
            f"white-box model (hf_local); {type(model).__name__} exposes no torch module."
        )
    return module


def tokenizer_of(model: "Model") -> Any:
    """The tokenizer behind a local handle, or ``None``."""
    loaded = getattr(model, "_loaded", None)
    if not isinstance(loaded, tuple) or len(loaded) < 2 or loaded[1] is None:
        return None
    processor = loaded[1]
    return getattr(processor, "tokenizer", processor)


def check_editable_attention(model: "Model", analyzer: str, stack: "DecoderStack | None" = None) -> None:
    """Refuse architectures whose token mixing cannot be edited layer by layer.

    The spec's declared semantics are checked first; with *stack*, every layer
    must also be a softmax attention or a supported linear-attention mixer, so
    an edit never silently skips a layer.
    """
    spec = getattr(model, "spec", None)
    semantics = getattr(getattr(spec, "attn_semantics", None), "value", None)
    if semantics is not None and semantics not in _EDITABLE_SEMANTICS:
        raise ValueError(
            f"{analyzer} needs softmax or gated-delta-rule decoder attention; "
            f"{getattr(spec, 'key', '?')} declares attn_semantics={semantics!r}, where an "
            "attention edit would not act on a token matrix (latent / unverified attention)."
        )
    if stack is not None and not stack.editable:
        bad = [i for i, kind in enumerate(stack.kinds) if kind is None]
        raise ValueError(
            f"{analyzer}: decoder layers {bad[:8]} have neither softmax attention nor a "
            "supported linear-attention mixer, so an edit would skip them."
        )


def is_delta_rule_mixer(module: Any) -> bool:
    """A gated-delta-rule linear-attention mixer (Qwen3.5 / Qwen3-Next GatedDeltaNet).

    Recognised by the delta-rule kernel and per-head output projection. Older
    transformers exposes ``chunk_gated_delta_rule`` on each instance; newer
    versions call ``torch_chunk_gated_delta_rule`` in the modeling module.
    """
    return (
        module is not None
        and (callable(getattr(module, "chunk_gated_delta_rule", None))
             or callable(getattr(sys.modules.get(type(module).__module__),
                                 "torch_chunk_gated_delta_rule", None)))
        and hasattr(getattr(module, "out_proj", None), "weight")
        and getattr(module, "head_v_dim", None) is not None
    )


@dataclass
class DecoderStack:
    """The language decoder's layers, their token mixers, and their owner.

    ``attn[i]`` is the softmax self-attention module or ``None``; ``linear[i]``
    the gated-delta-rule mixer or ``None``; ``kinds[i]`` is ``"softmax"``,
    ``"linear"`` or ``None`` (a mixer none of the edits understand).
    """

    layers: Any          # nn.ModuleList
    path: str            # dotted path of ``layers`` inside the module
    attn: list           # per layer: softmax attention submodule or None
    parent: Any          # module owning ``layers`` (the text model)
    linear: list = None  # per layer: delta-rule linear-attention mixer or None

    def __post_init__(self) -> None:
        if self.linear is None:
            self.linear = [None] * len(self.attn)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def kinds(self) -> "list[str | None]":
        return ["softmax" if a is not None else "linear" if m is not None else None
                for a, m in zip(self.attn, self.linear)]

    @property
    def mixers(self) -> list:
        """Per layer: the token mixer (softmax or linear), or ``None``."""
        return [a if a is not None else m for a, m in zip(self.attn, self.linear)]

    @property
    def fully_attentional(self) -> bool:
        return all(a is not None for a in self.attn)

    @property
    def editable(self) -> bool:
        """Every layer is a softmax attention or a supported linear mixer."""
        return all(kind is not None for kind in self.kinds)

    @property
    def n_softmax(self) -> int:
        return sum(a is not None for a in self.attn)

    @property
    def n_linear(self) -> int:
        return sum(m is not None for m in self.linear)


def decoder_stack(module: Any) -> DecoderStack:
    """Discover the decoder stack at runtime (never a hard-coded module path)."""
    from evalrx.models.backends.hf.discover import find_decoder_layers, resolve

    layers, path = find_decoder_layers(module)
    parent = module if "." not in path else resolve(module, path.rsplit(".", 1)[0])
    attn = [getattr(layer, "self_attn", None) for layer in layers]
    linear = [None if a is not None else (lambda m: m if is_delta_rule_mixer(m) else None)(
        getattr(layer, "linear_attn", None)) for layer, a in zip(layers, attn)]
    return DecoderStack(layers=layers, path=path, attn=attn, parent=parent, linear=linear)


def layer_points(n_layers: int, n_points: int) -> list[int]:
    """``n_points`` layer indices spread evenly over ``[0, n_layers)`` (all when fewer)."""
    if n_points <= 0 or n_points >= n_layers:
        return list(range(n_layers))
    step = (n_layers - 1) / (n_points - 1) if n_points > 1 else 0
    return sorted({int(round(i * step)) for i in range(n_points)})


def media_positions(trace: "Trace") -> "tuple[str, list[int]]":
    """``(slot, positions)`` of the media placeholder tokens in *trace*.

    Image/video tokens come from the TokenTypeMap (falling back to the
    ``image_token_mask`` extra); audio from ``audio_token_mask``. Returns
    ``("", [])`` when the sequence holds no media tokens.
    """
    ttm = getattr(trace, "token_type_map", None)
    if ttm is not None and getattr(ttm, "image_pos", None):
        return "image", list(ttm.image_pos)
    extras = getattr(trace, "extras", None) or {}
    for slot, key in (("image", "image_token_mask"), ("audio", "audio_token_mask")):
        mask = extras.get(key)
        if mask is not None:
            positions = [int(i) for i in mask.nonzero().flatten().tolist()]
            if positions:
                return slot, positions
    return "", []
