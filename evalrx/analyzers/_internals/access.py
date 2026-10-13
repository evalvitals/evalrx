"""Reach the live module behind a white-box handle, and the positions to act on.

Only the ``hf_local`` backend holds a torch module (``model._loaded`` lazily
loads and returns ``(module, processor)``). Everything that needs one goes
through :func:`torch_module`, so an API or fake model fails with one clear
message instead of an ``AttributeError`` three calls deep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.model import Model, Trace

#: Architectures whose decoder attention is a dense token×token softmax that a
#: registered attention function sees in full. Hybrid stacks route most layers
#: through linear/SSM attention with no query-key matrix; latent / unverified
#: attention is not a plain token matrix. Edits there would be partial or
#: meaningless, so the write-side analyzers refuse them.
_EDITABLE_SEMANTICS = frozenset({"standard"})


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


def check_editable_attention(model: "Model", analyzer: str) -> None:
    """Refuse architectures whose attention cannot be edited as a token matrix."""
    spec = getattr(model, "spec", None)
    semantics = getattr(getattr(spec, "attn_semantics", None), "value", None)
    if semantics is not None and semantics not in _EDITABLE_SEMANTICS:
        raise ValueError(
            f"{analyzer} needs standard dense decoder attention; {getattr(spec, 'key', '?')} "
            f"declares attn_semantics={semantics!r}, where an attention edit would be "
            "partial (hybrid/linear layers) or not a token matrix (latent attention)."
        )


@dataclass
class DecoderStack:
    """The language decoder's layers, their attention modules, and their owner.

    ``attn[i]`` is ``None`` for a layer with no softmax self-attention (the
    linear-attention layers of a hybrid stack).
    """

    layers: Any          # nn.ModuleList
    path: str            # dotted path of ``layers`` inside the module
    attn: list           # per layer: attention submodule or None
    parent: Any          # module owning ``layers`` (the text model)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def fully_attentional(self) -> bool:
        return all(a is not None for a in self.attn)


def decoder_stack(module: Any) -> DecoderStack:
    """Discover the decoder stack at runtime (never a hard-coded module path)."""
    from evalrx.models.backends.hf.discover import find_decoder_layers, resolve

    layers, path = find_decoder_layers(module)
    parent = module if "." not in path else resolve(module, path.rsplit(".", 1)[0])
    attn = [getattr(layer, "self_attn", None) for layer in layers]
    return DecoderStack(layers=layers, path=path, attn=attn, parent=parent)


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
