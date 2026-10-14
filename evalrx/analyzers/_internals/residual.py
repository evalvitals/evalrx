"""Read or overwrite the residual stream ENTERING a decoder layer.

The patch site is a forward pre-hook on decoder layer ``l``: the hidden state
the layer is about to read. Hooking layer INPUTS rather than outputs is
deliberate: Qwen3-VL ("DeepStack") adds visual features to the hidden state
between the first few layers, outside any layer. A value written into a layer's
OUTPUT would then get the receiving run's own visual features added on top; a
value written into the next layer's INPUT is exactly what that layer reads.

Site ``0`` is the embedding output after the media features were merged in.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from evalrx.analyzers._internals.access import decoder_stack, torch_module

if TYPE_CHECKING:
    from evalrx.core.model import Model


@dataclass
class ResidualSession:
    """``captured[layer]`` — ``(len(positions), dim)``; the full ``(1, seq, dim)``
    hidden state, graph attached, when the hooks ran with ``keep_grad``."""

    captured: "dict[int, Any]" = field(default_factory=dict)


@contextmanager
def residual_hooks(
    model: "Model",
    *,
    capture: "Mapping[int, Sequence[int] | None] | None" = None,
    replace: "Mapping[int, tuple[Sequence[int], Any]] | None" = None,
    keep_grad: bool = False,
    analyzer: str = "residual hooks",
) -> Iterator[ResidualSession]:
    """Install pre-hooks on decoder layers for the duration of the block.

    Args:
        capture:   ``{layer: positions}`` — record ``hidden[0, positions]``
                   (``None`` = every position), detached and cloned.
        replace:   ``{layer: (positions, values)}`` — overwrite
                   ``hidden[0, positions]`` with ``values`` (``(n, dim)``).
                   Replacement happens before capture at the same layer.
        keep_grad: record the whole hidden tensor un-detached instead, so a
                   caller can ``torch.autograd.grad`` a target with respect to it.
                   Under grad mode a state that does not require grad (frozen
                   weights upstream) is made a leaf that does — nothing upstream
                   needed a gradient, so no path is cut.
    """
    import torch

    module = torch_module(model, analyzer)
    stack = decoder_stack(module)
    capture = dict(capture or {})
    replace = dict(replace or {})
    for idx in (*capture, *replace):
        if not 0 <= idx < stack.n_layers:
            raise ValueError(f"{analyzer}: layer {idx} outside [0, {stack.n_layers})")
    session = ResidualSession()

    def make_hook(idx: int):
        def hook(mod, args, kwargs):
            if args:
                hidden = args[0]
            elif "hidden_states" in kwargs:
                hidden = kwargs["hidden_states"]
            else:  # pragma: no cover - no decoder layer we support calls it this way
                raise RuntimeError(f"{analyzer}: cannot find the hidden state passed to layer {idx}")
            if hidden.dim() != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"{analyzer}: expected a (1, seq, dim) hidden state, got "
                                   f"{tuple(hidden.shape)}")
            changed = False
            if idx in replace:
                positions, values = replace[idx]
                hidden = hidden.clone()
                hidden[0, list(positions)] = values.to(device=hidden.device, dtype=hidden.dtype)
                changed = True
            if idx in capture:
                if keep_grad:
                    if torch.is_grad_enabled() and not hidden.requires_grad:
                        hidden = hidden.detach().requires_grad_(True)
                        changed = True
                    session.captured[idx] = hidden
                else:
                    positions = capture[idx]
                    sel = hidden[0] if positions is None else hidden[0, list(positions)]
                    session.captured[idx] = sel.detach().clone()
            if not changed:
                return None
            if args:
                return (hidden, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": hidden}

        return hook

    handles = [
        stack.layers[idx].register_forward_pre_hook(make_hook(idx), with_kwargs=True)
        for idx in sorted(set(capture) | set(replace))
    ]
    try:
        yield session
    finally:
        for handle in handles:
            handle.remove()
