"""An attention function we own: reproduces eager attention, and can edit or record it.

Mechanism (the transformers attention interface): each supported decoder
attention module computes q/k/v itself and hands the softmax part to
``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``. Registering
:func:`_tap_attention` under :data:`TAP_NAME` and pointing ONLY the decoder's
configs at it inside a ``with`` block routes the language model's attention
through code we control; the vision tower keeps its own config and
implementation.

Two things this module exists to get right:

* **The mask.** ``masking_utils`` builds NO mask for an implementation name it
  does not know (it assumes a vLLM/TGI-style kernel that masks internally), so a
  bare registration silently lets every token attend to the future. The tap
  registers eager's additive mask under the same name.
* **Fidelity.** "The tap is installed" proves nothing on an architecture that
  bypasses the interface. :func:`check_fidelity` runs an input with an
  edit-free tap: it must be called on every attention layer AND reproduce the
  native logits, or :func:`require_fidelity` refuses. Whether it reproduces
  them can depend on the input (bf16 ``sdpa`` sits near the tolerance), so
  analyzers check every case they report on via
  :class:`~evalrx.analyzers._internals.edits.FidelityGate`; nothing is cached.

Edits act on the pre-softmax scores, so they compose with the causal mask:

* ``block``     — attention knockout: the query cannot read the key at all
  (Geva et al., 2023). A query is never blocked from itself, so no row empties.
* ``scale``     — multiply the attention to the keys by ``w`` and renormalise,
  implemented exactly as ``+log w`` on the scores (``w = 0`` is ``block``).
* ``scale_raw`` — multiply post-softmax weights by ``w`` without renormalising
  (the PAI-style amplification; rows no longer sum to one).
* ``temp``      — attention temperature ``T`` over the keys: their scores are
  divided by ``T`` (``T < 1`` sharpens, ``T > 1`` flattens).
* ``temp_within`` — the same, but each row keeps the total weight it gave those
  keys: only how that weight is distributed AMONG them changes,
  ``p'_k ∝ p_k^(1/T)`` inside the set (a per-row shift of the divided scores).
Masked cells (future keys, padding) are never touched by a temperature.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional

from evalrx.analyzers._internals.access import decoder_stack, torch_module

if TYPE_CHECKING:
    from evalrx.core.model import Model

TAP_NAME = "evalrx_tap"
EDIT_KINDS = ("block", "scale", "scale_raw", "temp", "temp_within")
TEMPERATURE_KINDS = ("temp", "temp_within")

_ACTIVE: "list[TapState]" = []
_REGISTERED = False


@dataclass(frozen=True)
class AttentionEdit:
    """One edit applied to some decoder layers' attention.

    ``keys`` / ``queries`` are absolute sequence positions (``queries=None`` =
    every query row); ``layers=None`` = every decoder layer.
    """

    kind: str
    keys: "tuple[int, ...]"
    queries: "tuple[int, ...] | None" = None
    layers: "frozenset[int] | None" = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"edit kind must be one of {EDIT_KINDS}, got {self.kind!r}")
        if self.kind in TEMPERATURE_KINDS and not (self.weight > 0 and math.isfinite(self.weight)):
            raise ValueError(f"attention temperature must be a finite number > 0, got {self.weight}")
        if self.weight < 0:
            raise ValueError(f"attention weight must be >= 0, got {self.weight}")

    def applies(self, layer: int) -> bool:
        return self.layers is None or layer in self.layers


def knockout(keys: Iterable[int], *, queries: "Iterable[int] | None" = None,
             layers: "Iterable[int] | None" = None) -> AttentionEdit:
    """Block *queries* from attending to *keys* in *layers*."""
    return AttentionEdit(
        "block", tuple(int(k) for k in keys),
        None if queries is None else tuple(int(q) for q in queries),
        None if layers is None else frozenset(int(i) for i in layers),
    )


def reweight(keys: Iterable[int], weight: float, *, queries: "Iterable[int] | None" = None,
             layers: "Iterable[int] | None" = None, renormalize: bool = True) -> AttentionEdit:
    """Scale attention from *queries* to *keys* by *weight*."""
    return AttentionEdit(
        "scale" if renormalize else "scale_raw", tuple(int(k) for k in keys),
        None if queries is None else tuple(int(q) for q in queries),
        None if layers is None else frozenset(int(i) for i in layers),
        float(weight),
    )


def temperature(keys: Iterable[int], value: float, *, queries: "Iterable[int] | None" = None,
                layers: "Iterable[int] | None" = None, within: bool = False) -> AttentionEdit:
    """Attention temperature *value* over *keys* (``within``: keep each row's mass on them)."""
    return AttentionEdit(
        "temp_within" if within else "temp", tuple(int(k) for k in keys),
        None if queries is None else tuple(int(q) for q in queries),
        None if layers is None else frozenset(int(i) for i in layers),
        float(value),
    )


class TapState:
    """What the tap does while one ``attention_tap`` block is active."""

    def __init__(self, edits: Iterable[AttentionEdit], layer_of: "dict[int, int]",
                 on_probs: "Callable[[int, Any], None] | None") -> None:
        self.edits = list(edits)
        self.layer_of = layer_of
        self.on_probs = on_probs
        self.calls: list[int] = []  # decoder-layer index of every tapped call
        self._masks: dict = {}

    def _mask(self, index: int, edit: AttentionEdit, q_len: int, k_len: int, device: Any) -> Any:
        """(q_len, k_len) bool: which score cells *edit* touches."""
        import torch

        cache_key = (index, q_len, k_len, str(device))
        if cache_key in self._masks:
            return self._masks[cache_key]
        # Query rows are the LAST q_len positions (q_len < k_len only with a KV cache).
        q_abs = torch.arange(k_len - q_len, k_len, device=device)
        k_sel = torch.zeros(k_len, dtype=torch.bool, device=device)
        keys = [k for k in edit.keys if 0 <= k < k_len]
        if keys:
            k_sel[torch.tensor(keys, device=device)] = True
        if edit.queries is None:
            q_sel = torch.ones(q_len, dtype=torch.bool, device=device)
        else:
            q_full = torch.zeros(k_len, dtype=torch.bool, device=device)
            queries = [q for q in edit.queries if 0 <= q < k_len]
            if queries:
                q_full[torch.tensor(queries, device=device)] = True
            q_sel = q_full[q_abs]
        mask = q_sel[:, None] & k_sel[None, :]
        if edit.kind == "block" or edit.weight == 0:
            mask &= q_abs[:, None] != torch.arange(k_len, device=device)[None, :]
        self._masks[cache_key] = mask
        return mask

    def edit_scores(self, layer: int, scores: Any) -> Any:
        import torch

        q_len, k_len = scores.shape[-2:]
        for index, edit in enumerate(self.edits):
            if edit.kind == "scale_raw" or not edit.applies(layer):
                continue
            mask = self._mask(index, edit, q_len, k_len, scores.device)
            if edit.kind in TEMPERATURE_KINDS:
                if edit.weight != 1.0:
                    scores = _tempered(scores, mask, edit.weight, within=edit.kind == "temp_within")
            elif edit.kind == "block" or edit.weight == 0:
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            elif edit.weight != 1.0:
                scores = scores + mask.to(scores.dtype) * math.log(edit.weight)
        return scores

    def edit_probs(self, layer: int, probs: Any) -> Any:
        import torch

        q_len, k_len = probs.shape[-2:]
        for index, edit in enumerate(self.edits):
            if edit.kind != "scale_raw" or not edit.applies(layer) or edit.weight == 1.0:
                continue
            mask = self._mask(index, edit, q_len, k_len, probs.device)
            factor = torch.where(mask, edit.weight, 1.0).to(probs.dtype)
            probs = probs * factor
        return probs


def _tempered(scores: Any, mask: Any, t: float, *, within: bool) -> Any:
    """Divide the *mask* cells' scores by *t*; with *within*, shift each row so the
    softmax mass on those cells is what it was (``logsumexp`` over them unchanged)."""
    import torch

    floor = torch.finfo(scores.dtype).min
    sel = mask & (scores > floor / 2)          # never the masked (future / padding) cells
    s = scores.float()
    out = s / t
    if within:
        neg = torch.tensor(float("-inf"), device=s.device)
        before = torch.logsumexp(torch.where(sel, s, neg), dim=-1, keepdim=True)
        after = torch.logsumexp(torch.where(sel, out, neg), dim=-1, keepdim=True)
        shift = torch.where(torch.isfinite(before), before - after, torch.zeros_like(before))
        out = out + shift
    return torch.where(sel, out, s).to(scores.dtype)


def _tap_attention(module: Any, query: Any, key: Any, value: Any, attention_mask: Any,
                   scaling: "float | None" = None, dropout: float = 0.0,
                   softcap: "float | None" = None, **kwargs: Any) -> "tuple[Any, Any]":
    """Eager attention, op for op, with edit/record points around the softmax.

    Matches the per-family ``eager_attention_forward`` it replaces: GQA head
    repetition, scaling (``head_dim ** -0.5`` when the caller passes none, as
    Gemma/Qwen2-Audio default), optional logit soft-capping (Gemma), additive
    mask, fp32 softmax, dropout.
    """
    import torch
    from torch import nn

    if kwargs.get("s_aux") is not None or kwargs.get("sinks") is not None:
        raise NotImplementedError("attention sinks (s_aux) are not reproduced by the evalrx tap")
    state = _ACTIVE[-1] if _ACTIVE else None
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:  # == transformers' repeat_kv
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if softcap is not None:
        scores = scores / softcap
        scores = torch.tanh(scores)
        scores = scores * softcap
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : key.shape[-2]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    layer = state.layer_of.get(id(module)) if state is not None else None
    if layer is not None:
        state.calls.append(layer)
        scores = state.edit_scores(layer, scores)
    probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if layer is not None:
        probs = state.edit_probs(layer, probs)
        if state.on_probs is not None:
            state.on_probs(layer, probs)
    probs = nn.functional.dropout(probs, p=dropout, training=module.training)
    out = torch.matmul(probs, value).transpose(1, 2).contiguous()
    return out, probs


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask

    AttentionInterface.register(TAP_NAME, _tap_attention)
    AttentionMaskInterface.register(TAP_NAME, eager_mask)  # else: NO causal mask at all
    _REGISTERED = True


def _get_impl(config: Any) -> Any:
    return getattr(config, "_attn_implementation", None)


def _set_impl(config: Any, value: Any) -> None:
    # Write the backing field directly: the public setter recurses into
    # sub-configs, which on a composite config would retarget the vision tower.
    if "_attn_implementation_internal" in getattr(config, "__dict__", {}):
        config._attn_implementation_internal = value
    else:
        config._attn_implementation = value


@contextmanager
def attention_tap(
    model: "Model",
    edits: Iterable[AttentionEdit] = (),
    *,
    on_probs: "Callable[[int, Any], None] | None" = None,
    analyzer: str = "attention tap",
) -> Iterator[TapState]:
    """Route the decoder's attention through the tap for the duration of the block.

    ``on_probs(layer, probs)`` sees each layer's (edited) attention weights,
    shape ``(batch, heads, q, k)``, still attached to the autograd graph when
    the forward runs with gradients.
    """
    module = torch_module(model, analyzer)
    stack = decoder_stack(module)
    _ensure_registered()

    configs: list = []
    seen: set[int] = set()
    for owner in [*(a for a in stack.attn if a is not None), stack.parent]:
        cfg = getattr(owner, "config", None)
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            configs.append(cfg)
    if not configs:
        raise ValueError(f"{analyzer}: no config found on the decoder attention modules")

    layer_of = {id(a): i for i, a in enumerate(stack.attn) if a is not None}
    state = TapState(edits, layer_of, on_probs)
    saved = [(cfg, _get_impl(cfg)) for cfg in configs]
    _ACTIVE.append(state)
    try:
        for cfg, _ in saved:
            _set_impl(cfg, TAP_NAME)
        yield state
    finally:
        for cfg, impl in saved:
            _set_impl(cfg, impl)
        _ACTIVE.remove(state)


def check_fidelity(model: "Model", inputs: Any, *, rtol: float = 1e-2,
                   analyzer: str = "attention tap", base: Any = None) -> "dict[str, Any]":
    """Does an edit-free tap engage on every attention layer and reproduce the logits on *inputs*?

    Measured on this input every call (not cached): the tapped forward must
    engage every softmax layer, stay within ``rtol`` of the largest native
    logit at the last position, and keep the native top-1. *base* — the native
    logits of this same input, when the caller already has them — saves the
    native forward.
    """
    from evalrx.core.capability import Capability

    module = torch_module(model, analyzer)
    stack = decoder_stack(module)
    native = tuple(str(_get_impl(getattr(a, "config", None))) for a in stack.attn if a is not None)
    n_attn = sum(a is not None for a in stack.attn)
    if base is None:
        base = model.forward(inputs, capture={Capability.LOGITS}).require(Capability.LOGITS)
    with attention_tap(model, analyzer=analyzer) as state:
        tapped = model.forward(inputs, capture={Capability.LOGITS}).require(Capability.LOGITS)
    b = base[-1].float()
    t = tapped[-1].float().to(b.device)
    diff = float((b - t).abs().max())
    scale = float(b.abs().max()) or 1.0
    engaged = len(set(state.calls))
    argmax_same = int(b.argmax()) == int(t.argmax())
    return {
        "tap_faithful": bool(engaged == n_attn and diff <= rtol * scale and argmax_same),
        "tap_layers_engaged": engaged,
        "tap_layers_expected": n_attn,
        "tap_max_logit_diff": round(diff, 6),
        "tap_scale": round(scale, 6),
        "tap_rel_diff": round(diff / scale, 6),
        "tap_argmax_same": argmax_same,
        "tap_rtol": rtol,
        "native_attn_implementation": native[0] if native else None,
    }


class FidelityError(RuntimeError):
    """An edit mechanism failed its edit-free self-check on this model.

    A ``RuntimeError`` so every existing ``except RuntimeError`` still catches
    it; the subclass lets a caller tell "the measurement cannot be trusted"
    apart from an ordinary crash without parsing the message. ``verdict`` is
    the check's own record (engaged layers, logit diff, tolerance).
    """

    def __init__(self, message: str, verdict: "dict[str, Any]") -> None:
        super().__init__(message)
        self.verdict = dict(verdict)


def require_fidelity(model: "Model", inputs: Any, *, rtol: float = 1e-2,
                     analyzer: str = "attention tap") -> "dict[str, Any]":
    """:func:`check_fidelity`, raising when the tap cannot be trusted on *model*."""
    verdict = check_fidelity(model, inputs, rtol=rtol, analyzer=analyzer)
    if not verdict["tap_faithful"]:
        raise FidelityError(
            f"{analyzer}: the evalrx attention tap does not faithfully reproduce this "
            f"model's attention ({verdict}); refusing to report edited-attention effects. "
            "Engaged < expected layers means the architecture bypasses the transformers "
            "attention interface; a large logit diff means its attention is not eager-equivalent.",
            verdict,
        )
    return verdict


def active_state() -> Optional[TapState]:
    """The innermost active tap state (``None`` outside any ``attention_tap``)."""
    return _ACTIVE[-1] if _ACTIVE else None
