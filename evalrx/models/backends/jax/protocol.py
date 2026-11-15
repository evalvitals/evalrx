"""The adapter contract behind the ``jax_local`` backend.

A JAX model has no single API the way an HF torch model does: Flax Linen, Flax
NNX, MaxText, Penzai and hand-rolled ``lax.scan`` stacks all expose internals
differently. The backend therefore never touches a framework. It drives a
:class:`JaxModelAdapter`, and each framework (or a user's own training script)
implements this small protocol. ``evalrx.models.backends.jax.adapters.gemma`` is the reference
implementation. Design notes: ``docs/design_jax_backend.md`` (section 3.1).

Conventions the backend relies on:

* ``forward`` returns **framework arrays** (jax / numpy); the backend converts
  them to CPU torch tensors at the ``Trace`` boundary, so analyzers see exactly
  what ``hf_local`` produces.
* ``hidden`` follows the HF ``output_hidden_states`` layout: ``L + 1`` entries,
  the embedding output first, block ``l``'s residual output at index ``l + 1``,
  and the LAST entry already passed through the final norm.
* ``attn`` is one ``(heads, seq, seq)`` probability matrix per block, available
  only under reference (materialised-softmax) attention; ``None`` otherwise.
* When ``layers`` is given, an adapter may leave unrequested entries ``None``
  to save memory; the lists keep their full length and the backend subsets.

This module is torch-free and jax-free at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

#: capture keys the backend hands to ``JaxModelAdapter.forward``
CAPTURE_LOGITS = "logits"
CAPTURE_HIDDEN = "hidden_states"
CAPTURE_ATTN = "attentions"


@dataclass
class Encoding:
    """What the adapter's tokenizer / processor produced for one input.

    ``text`` keeps the rendered prompt so generation can re-tokenise the SAME
    string the forward pass saw (samplers that take text, e.g. the gemma
    library's, then agree with ``ids`` token for token). Multimodal adapters
    fill ``media`` (framework arrays) and the placeholder masks; ``grids`` is
    the per-image ``(t, h, w)`` patch grid when the encoder's token count per
    image is fixed, else empty (the Gemma 4 torch path leaves it empty too).
    """

    ids: list[int]
    tokens: list[str]
    text: str | None = None
    media: dict[str, Any] = field(default_factory=dict)
    image_token_mask: list[bool] | None = None
    audio_token_mask: list[bool] | None = None
    grids: list[tuple[int, int, int]] = field(default_factory=list)
    image_token_id: int | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def extended(self, ids: list[int], tokens: list[str]) -> "Encoding":
        """This encoding followed by *ids* (teacher forcing a generated continuation)."""
        n = len(ids)
        return Encoding(
            ids=list(self.ids) + list(ids),
            tokens=list(self.tokens) + list(tokens),
            text=None,
            media=dict(self.media),
            image_token_mask=None if self.image_token_mask is None else list(self.image_token_mask) + [False] * n,
            audio_token_mask=None if self.audio_token_mask is None else list(self.audio_token_mask) + [False] * n,
            grids=list(self.grids),
            image_token_id=self.image_token_id,
        )


@dataclass
class SamplingParams:
    """Backend-neutral decoding controls (the names PipelineSpec / hf_local use)."""

    max_new_tokens: int
    temperature: float = 0.0        # <= 0 -> greedy
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None         # PRNG seed for sampled decoding; None -> adapter default
    stop: list[str] = field(default_factory=list)

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class ForwardOut(NamedTuple):
    """One teacher-forced forward pass over ``Encoding.ids`` (no batch dim)."""

    logits: Any                     # (S, V) after the model's own final soft-cap, or None
    hidden: list[Any] | None        # L+1 x (S, D): embeddings first, last entry post final-norm
    attn: list[Any] | None          # L x (H, S, S) probabilities under reference attention
    extras: dict[str, Any]          # adapter-specific additions to Trace.extras


class GenerateOut(NamedTuple):
    ids: list[int]                  # generated token ids, end/pad tokens stripped
    text: str                       # decoded, special tokens skipped


@dataclass(frozen=True)
class NormParams:
    """The head-side RMSNorm, so logit-lens readouts stay faithful.

    ``y = x * rsqrt(mean(x^2) + eps) * w`` with ``w = scale`` (gemma library) or
    ``w = 1 + scale`` (HF Gemma stores the offset form). ``scale=None`` means a
    parameter-free norm.
    """

    scale: Any
    eps: float = 1e-6
    plus_one: bool = False


class JaxModelAdapter(Protocol):
    """What a framework must provide for ``JaxLocalModel`` to drive it.

    Cheap to construct (no jax import); ``load()`` does the heavy lifting.
    """

    n_layers: int
    modalities: frozenset[str]          # {"text"} | {"text","image"} | {"text","image","audio"}
    reference_attention: bool           # True -> ``forward`` can return attention probabilities

    def load(self) -> None: ...

    # --- encoding: the framework owns its tokenizer / processor ------------
    def encode(self, inputs: Any, *, chat_template: bool) -> Encoding: ...
    def render_chat(self, messages: list, tools: list | None = None) -> Encoding: ...
    def decode(self, ids: list[int]) -> str: ...

    # --- read internals ------------------------------------------------------
    def forward(
        self,
        enc: Encoding,
        *,
        capture: frozenset[str],
        layers: tuple[int, ...] | None = None,
    ) -> ForwardOut: ...

    def unembed(self) -> Any: ...                          # (V, D) array the model really applies
    def final_norm_params(self) -> NormParams | None: ...

    # --- generation --------------------------------------------------------
    def generate(self, enc: Encoding, params: SamplingParams) -> GenerateOut: ...
