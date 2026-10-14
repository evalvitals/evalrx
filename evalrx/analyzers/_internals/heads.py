"""Per-head token-mixer outputs (read at one position, or read / overwritten everywhere),
and the final norm frozen to a linear map.

Every supported decoder token mixer — softmax attention, or the gated-delta-rule
linear attention of a hybrid stack — ends in an output projection whose INPUT
is the heads' outputs concatenated, ``(n_heads · head_dim)``. A forward
pre-hook on that projection reads them; head ``h``'s write into the residual
stream is ``W_O[:, h·hd:(h+1)·hd] @ x[h·hd:(h+1)·hd]``, and the per-head writes
sum to the projection's output (minus its bias, if any).

Direct logit attribution then reads each write through the final norm and the
unembedding. The norm is not linear, so — as is standard — its scale is FROZEN
at the clean final residual: RMSNorm becomes ``x ↦ s ⊙ x`` with
``s = w_eff / rms(x_final)``. ``w_eff`` is read off the module itself
(``norm(1)``), which covers Llama/Qwen ``w`` and Gemma ``1 + w`` alike.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from evalrx.analyzers._internals.access import decoder_stack, torch_module

if TYPE_CHECKING:
    from evalrx.core.model import Model

_OUT_PROJ_NAMES = ("o_proj", "out_proj", "dense")


def out_proj(attn: Any) -> Any:
    """The attention module's output projection (``None`` when not found)."""
    for name in _OUT_PROJ_NAMES:
        proj = getattr(attn, name, None)
        if proj is not None and hasattr(proj, "weight"):
            return proj
    return None


def n_heads_of(attn: Any, proj: Any) -> int:
    """Head count: the projection's input width over the head dim.

    Softmax attention exposes ``head_dim``; a gated-delta-rule mixer writes its
    value heads (``head_v_dim`` wide each) through ``out_proj``.
    """
    head_dim = getattr(attn, "head_dim", None) or getattr(attn, "head_v_dim", None)
    if head_dim is None:
        cfg = getattr(attn, "config", None)
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads if cfg is not None else None)
    if not head_dim or proj.weight.shape[1] % int(head_dim):
        raise ValueError(f"cannot split {type(attn).__name__}'s output projection into heads")
    return proj.weight.shape[1] // int(head_dim)


@dataclass
class HeadSession:
    """``head_inputs[layer]`` — ``(n_heads · head_dim,)`` at the position;
    ``final_residual`` — the final norm's input there."""

    head_inputs: "dict[int, Any]" = field(default_factory=dict)
    final_residual: Any = None


@contextmanager
def head_capture(model: "Model", position: int = -1, *, analyzer: str = "head capture") -> Iterator[HeadSession]:
    """Record every layer's per-head outputs and the final residual at *position*."""
    from evalrx.models.backends.hf.discover import get_final_norm

    module = torch_module(model, analyzer)
    stack = decoder_stack(module)
    norm = get_final_norm(module)
    if norm is None:
        raise ValueError(f"{analyzer}: no final norm found after the decoder layers")
    session = HeadSession()
    handles = []

    def grab_heads(layer: int):
        def hook(mod, args):
            session.head_inputs[layer] = args[0][0, position].detach()
        return hook

    def grab_final(mod, args):
        session.final_residual = args[0][0, position].detach()

    for layer, mixer in enumerate(stack.mixers):
        proj = out_proj(mixer) if mixer is not None else None
        if proj is None:
            raise ValueError(f"{analyzer}: layer {layer} has no token-mixer output projection")
        handles.append(proj.register_forward_pre_hook(grab_heads(layer)))
    handles.append(norm.register_forward_pre_hook(grab_final))
    try:
        yield session
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class MixerSession:
    """``inputs[layer]`` — the output projection's whole input ``(seq, n_heads · head_dim)``;
    ``final_residual`` — the final norm's input ``(seq, dim)``."""

    inputs: "dict[int, Any]" = field(default_factory=dict)
    final_residual: Any = None


@contextmanager
def mixer_hooks(
    model: "Model",
    *,
    capture: bool = False,
    replace: "dict[int, Any] | None" = None,
    analyzer: str = "mixer hooks",
) -> Iterator[MixerSession]:
    """Read and/or overwrite every token mixer's per-head outputs, all positions.

    ``replace[layer]`` — ``(seq, n_heads · head_dim)`` written over that layer's
    output-projection input (heads concatenated, as the projection reads them),
    so every head's write into the residual stream is set; the projection and
    everything after it recompute. With *capture*, the inputs are recorded
    (after any replacement) together with the final norm's input.
    """
    from evalrx.models.backends.hf.discover import get_final_norm

    module = torch_module(model, analyzer)
    stack = decoder_stack(module)
    replace = dict(replace or {})
    session = MixerSession()
    handles = []

    def hook_for(layer: int):
        def hook(mod, args):
            x = args[0]
            if x.dim() != 3 or x.shape[0] != 1:
                raise RuntimeError(f"{analyzer}: expected a (1, seq, width) projection input, "
                                   f"got {tuple(x.shape)}")
            if layer in replace:
                x = replace[layer].to(device=x.device, dtype=x.dtype).reshape(x.shape)
            if capture:
                session.inputs[layer] = x[0].detach()
            return (x, *args[1:]) if layer in replace else None
        return hook

    for layer, mixer in enumerate(stack.mixers):
        proj = out_proj(mixer) if mixer is not None else None
        if proj is None:
            raise ValueError(f"{analyzer}: layer {layer} has no token-mixer output projection")
        if capture or layer in replace:
            handles.append(proj.register_forward_pre_hook(hook_for(layer)))
    if capture:
        norm = get_final_norm(module)
        if norm is None:
            raise ValueError(f"{analyzer}: no final norm found after the decoder layers")

        def grab_final(mod, args):
            session.final_residual = args[0][0].detach()

        handles.append(norm.register_forward_pre_hook(grab_final))
    try:
        yield session
    finally:
        for handle in handles:
            handle.remove()


def frozen_norm_scale(norm: Any, x: Any) -> "tuple[Any, bool]":
    """``(scale, centre)``: ``norm(x + c) ≈ norm(x) + scale ⊙ (c - centre·mean(c))``.

    Exact (scale frozen at ``x``) for RMSNorm; LayerNorm additionally centres.
    """
    import torch

    xf = x.float()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)) or 1e-6)
    is_layernorm = isinstance(norm, torch.nn.LayerNorm) or (
        "layernorm" in type(norm).__name__.lower() and "rms" not in type(norm).__name__.lower())
    with torch.no_grad():
        if is_layernorm:
            weight = norm.weight.float() if getattr(norm, "weight", None) is not None else torch.ones_like(xf)
            return weight / torch.sqrt(xf.var(unbiased=False) + eps), True
        w_eff = norm(torch.ones_like(x)).float() * (1.0 + eps) ** 0.5  # undo rms(1) = sqrt(1 + eps)
        return w_eff / torch.sqrt(xf.pow(2).mean() + eps), False


def readout_direction(norm: Any, unembed: Any, x_final: Any, positive: int, negative: int) -> Any:
    """``v`` with ``margin ≈ x · v`` around *x_final* (the final norm frozen there).

    A write ``c`` into the final residual moves the ``positive − negative`` logit
    margin by ``c · v``: the frozen norm scale times the unembedding difference
    (mean-removed for LayerNorm, whose centring is linear).
    """
    scale, centre = frozen_norm_scale(norm, x_final)
    w_u = unembed.weight
    direction = (w_u[positive].float() - w_u[negative].float()).to(scale.device)
    v = scale * direction
    return v - v.mean() if centre else v
