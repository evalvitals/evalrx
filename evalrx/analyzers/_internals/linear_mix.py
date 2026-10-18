"""Gated-delta-rule linear-attention layers (Qwen3.5, Qwen3-Next): media edits and relevance.

A hybrid stack interleaves softmax attention with GatedDeltaNet layers, which
have no query×key matrix: a short causal convolution mixes neighbouring tokens,
then a gated delta-rule recurrence carries everything earlier forward in a
state. The softmax-only tools (:mod:`evalrx.analyzers._internals.attn_tap`)
cannot touch them, so a knockout there would be partial. This module gives the
two operations the analyzers need, with the module's own code doing the math.

**Media edits** (:func:`linear_media_edit`). For a linear layer, "the text does
not read the media" is defined as: the text positions get the output the layer
produces on the sequence WITH THE MEDIA TOKENS REMOVED. The layer is re-run on
that shortened sequence (``module.forward``, no cache) and the selected query
positions are blended::

    out[q] = absent[q] + w · (real[q] − absent[q])

``w = 0`` is the knockout, ``w = 1`` the unchanged layer, ``w = 2`` doubles the
media's contribution. Removing the tokens (rather than zeroing them) keeps the
convolution and the recurrence consistent: these layers carry no positional
encoding, so the shortened sequence is exactly "the media was never there".
Positions before the first media token are unaffected by causality; the
re-run's outputs there must match the real ones, which :func:`check_linear_fidelity`
verifies once per model.

**Relevance** (:func:`delta_rule_relevance`). With q, k, gates and β fixed, the
delta rule is LINEAR in the values: ``o = A · v`` per head for a data-dependent
lower-triangular ``A`` (T×T). Running the module's own pure-torch kernel with
the values replaced by the identity materialises ``A`` exactly; with
``∂y/∂A = (∂y/∂o) vᵀ`` the layer contributes Chefer's ``E_h[(∇A ⊙ A)⁺]`` like a
softmax layer. The convolution before the kernel (window 4) is not propagated:
relevance moves at most three tokens through it unaccounted.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock, get_ident
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from evalrx.analyzers._internals.access import decoder_stack, torch_module

if TYPE_CHECKING:
    from evalrx.core.model import Model


@dataclass
class LinearEditState:
    """What one :func:`linear_media_edit` block did."""

    calls: "list[int]" = field(default_factory=list)          # layer index per edited call
    prefix_rel_diff: "dict[int, float]" = field(default_factory=dict)  # layer -> max rel diff


def _hidden_arg(args: tuple, kwargs: dict) -> Any:
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    if args:
        return args[0]
    raise RuntimeError("linear-attention layer called without hidden_states")


@contextmanager
def linear_media_edit(
    model: "Model",
    media: Iterable[int],
    *,
    queries: "Iterable[int] | None" = None,
    layers: "Iterable[int] | None" = None,
    weight: float = 0.0,
    analyzer: str = "linear media edit",
) -> Iterator[LinearEditState]:
    """Blend linear-layer outputs at *queries* toward the media-absent re-run.

    ``queries=None`` = every non-media position; ``layers=None`` = every linear
    layer. Softmax layers are untouched (use the attention tap for those).
    """
    import torch

    stack = decoder_stack(torch_module(model, analyzer))
    media_set = {int(m) for m in media}
    query_set = None if queries is None else {int(q) for q in queries}
    chosen = None if layers is None else {int(i) for i in layers}
    state = LinearEditState()

    def make_hook(layer: int):
        def hook(mod, args, kwargs, output):
            hidden = _hidden_arg(args, kwargs)
            seq = hidden.shape[1]
            present = sorted(m for m in media_set if m < seq)
            if not present or weight == 1.0:
                return None
            first = present[0]
            keep = [i for i in range(seq) if i not in media_set]
            targets = [q for q in keep if q > first and (query_set is None or q in query_set)]
            if not targets:
                return None
            absent = mod.forward(hidden[:, keep], cache_params=None, attention_mask=None)
            index = {pos: i for i, pos in enumerate(keep)}
            sel = torch.tensor([index[q] for q in targets], device=output.device)
            tgt = torch.tensor(targets, device=output.device)
            before = [i for i, pos in enumerate(keep) if pos < first]
            if before:
                ref = output[:, before].float()
                diff = (absent[:, before].float() - ref).abs().max()
                state.prefix_rel_diff[layer] = float(diff / ref.abs().max().clamp(min=1e-6))
            real, alt = output.index_select(1, tgt).float(), absent.index_select(1, sel).float()
            out = output.clone()
            out[:, tgt] = (alt + weight * (real - alt)).to(output.dtype)
            state.calls.append(layer)
            return out

        return hook

    handles = [
        mod.register_forward_hook(make_hook(i), with_kwargs=True)
        for i, mod in enumerate(stack.linear)
        if mod is not None and (chosen is None or i in chosen)
    ]
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


def check_linear_fidelity(model: "Model", inputs: Any, media: Iterable[int], *,
                          rtol: float = 1e-2, analyzer: str = "linear media edit") -> "dict[str, Any]":
    """Does the media-absent re-run engage on every linear layer and match the prefix?

    The re-run calls ``module.forward`` on a shortened sequence without the
    cache/mask arguments the decoder passed; before the first media token it
    must reproduce the real outputs, or the re-run is not the same computation.
    Measured on this input and media span every call (one forward), not cached.
    """
    from evalrx.core.capability import Capability

    module = torch_module(model, analyzer)
    stack = decoder_stack(module)
    with linear_media_edit(model, media, weight=0.0, analyzer=analyzer) as state:
        model.forward(inputs, capture={Capability.LOGITS})
    engaged = len(set(state.calls))
    worst = max(state.prefix_rel_diff.values(), default=float("nan"))
    return {
        "linear_faithful": bool(engaged == stack.n_linear and state.prefix_rel_diff
                                and worst <= rtol),
        "linear_layers_engaged": engaged,
        "linear_layers_expected": stack.n_linear,
        "linear_prefix_max_rel_diff": round(worst, 6),
        "linear_rtol": rtol,
    }


def _torch_kernel(mod: Any) -> Callable:
    """The module family's pure-torch chunked delta-rule kernel (exact, any value width)."""
    kernel = getattr(sys.modules.get(type(mod).__module__), "torch_chunk_gated_delta_rule", None)
    if kernel is None:
        raise RuntimeError(f"{type(mod).__name__}: no torch_chunk_gated_delta_rule next to it")
    return kernel


def mixing_matrix(kernel: Callable, query: Any, key: Any, g: Any, beta: Any, **kwargs: Any) -> Any:
    """``A`` with ``o = A · v`` per head: ``(batch, heads, T, T)``, float32.

    The delta rule is linear in the values, so feeding the identity as values
    (value width T) returns ``A`` row by row.
    """
    import torch

    batch, seq, heads, _ = key.shape
    eye = torch.eye(seq, device=key.device, dtype=torch.float32)
    values = eye.view(1, seq, 1, seq).expand(batch, seq, heads, seq)
    out, _ = kernel(query.float(), key.float(), values, g=g.float(), beta=beta.float(),
                    initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=kwargs.get("use_qk_l2norm_in_kernel", False))
    return out.float().permute(0, 2, 1, 3)  # (B, T_query, H, T_key) -> (B, H, T, T)


# Serialize temporary module-global replacements across tapped forwards. Calls
# from other threads still use the original kernel via the dispatcher below.
_DELTA_KERNEL_LOCK = RLock()


@contextmanager
def _replace_attribute(owner: Any, name: str, value: Any) -> Iterator[None]:
    had_instance_attr = name in owner.__dict__
    previous = owner.__dict__.get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had_instance_attr:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def _module_kernel_forward(mod: Any, wrap_kernel: Callable[[Callable], Callable]) -> Callable:
    """Tap the module-level kernel only while this mixer's forward is running."""
    owner = sys.modules[type(mod).__module__]
    forward = mod.forward

    @wraps(forward)
    def tapped_forward(*args, **kwargs):
        with _DELTA_KERNEL_LOCK:
            original = owner.torch_chunk_gated_delta_rule
            tapped = wrap_kernel(original)
            thread = get_ident()

            @wraps(original)
            def dispatch(*kernel_args, **kernel_kwargs):
                kernel = tapped if get_ident() == thread else original
                return kernel(*kernel_args, **kernel_kwargs)

            with _replace_attribute(owner, "torch_chunk_gated_delta_rule", dispatch):
                return forward(*args, **kwargs)

    return tapped_forward


@contextmanager
def delta_rule_relevance(model: "Model", on_abar: "Callable[[int, Any], None]", *,
                         analyzer: str = "delta-rule relevance") -> "Iterator[dict[int, Any]]":
    """Wrap each linear layer's kernel so its backward yields Chefer's ``Ā``.

    Yields ``{layer: kernel output}`` recorded during the forward (graph
    attached); a backward pass through them fires one hook per layer that calls
    ``on_abar(layer, Ā)`` with ``Ā`` of shape (T, T).
    """
    import torch

    stack = decoder_stack(torch_module(model, analyzer))
    recorded: dict[int, Any] = {}

    def wrap(layer: int, kernel: Callable, original: Callable) -> Callable:
        def wrapped(query, key, value, g=None, beta=None, **kwargs):
            out, final_state = original(query, key, value, g=g, beta=beta, **kwargs)
            if out.requires_grad:
                held = [t.detach() for t in (query, key, value, g, beta)]

                def hook(grad: Any) -> None:
                    q, k, v, gg, bb = held
                    a = mixing_matrix(kernel, q, k, gg, bb, **kwargs)
                    grad_a = torch.einsum("bthd,bjhd->bhtj", grad.float(), v.float())
                    on_abar(layer, (grad_a * a).clamp(min=0).mean(dim=1)[0])

                out.register_hook(hook)
                recorded[layer] = out
            return out, final_state

        return wrapped

    # Capture the pure kernel before any forward dispatch is installed; the
    # backward hooks run after this context exits and must not re-enter a tap.
    with ExitStack() as patches:
        for layer, mod in enumerate(stack.linear):
            if mod is None:
                continue
            kernel = _torch_kernel(mod)
            original = getattr(mod, "chunk_gated_delta_rule", None)
            if callable(original):
                patches.enter_context(_replace_attribute(
                    mod, "chunk_gated_delta_rule", wrap(layer, kernel, original)))
            else:
                def wrap_kernel(original, layer=layer, kernel=kernel):
                    return wrap(layer, kernel, original)

                patches.enter_context(_replace_attribute(
                    mod, "forward", _module_kernel_forward(mod, wrap_kernel)))
        yield recorded
