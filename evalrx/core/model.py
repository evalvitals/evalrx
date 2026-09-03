"""Model — the abstract contract every analyzable model implements.

The key design move: a model exposes **one** rich forward pass,
``forward(inputs, capture={...}) -> Trace``, where ``capture`` names the
internals to record (attention, hidden states, …).  Analyzers consume the
``Trace``, not the model's framework-specific guts.  This decouples "run the
model and grab internals" from "interpret the internals", so one model serves
every compatible analyzer and one analyzer works across every compatible model.

The hybrid convenience API (``model.call_attention(prompt)``) is provided here
by ``__getattr__``: it resolves ``attention`` in the analyzer registry, checks
the model supports it, and delegates to ``AttentionAnalyzer().run(self, ...)``.
No mixins, no per-model wiring — the shim is derived from capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalrx.core.capability import Capability

if TYPE_CHECKING:
    import torch

    from evalrx.core.tokentype import TokenTypeMap
    from evalrx.core.tool import ChatTurn


@dataclass
class CaptureSpec:
    """Optional knobs that bound what ``forward`` records, to control memory.

    Capturing every layer/head of attention for a long sequence is O(L^2) and
    can be tens of GB.  A backend that honours this spec captures only the
    requested layers/heads and can keep tensors off the CPU.  ``None`` fields
    mean "everything" (the simple default).  Backends that don't support
    sub-selection ignore it — the ``capture`` set alone still works.
    """

    layers: "list[int] | None" = None    # which decoder layers to keep (None = all)
    heads: "list[int] | None" = None     # which attention heads to keep (None = all)
    to_cpu: bool = True                   # move captured tensors to CPU (frees GPU mem)
    with_grad: bool = False               # keep the graph for GRADIENTS (else no_grad)


@dataclass
class TokenLogprob:
    """Logprob of one output token (+ optional top-k alternatives).

    The black-box uncertainty primitive: retrievable from an API that returns
    ``logprobs`` (OpenAI) — no model internals needed.
    """

    token: str
    logprob: float
    top: dict[str, float] = field(default_factory=dict)  # token -> logprob (top-k alternatives)


@dataclass
class Trace:
    """Captured internals from a single forward pass.

    Only the fields named in the ``capture`` set are populated; the rest stay
    ``None``.  ``provided`` records what was actually captured.
    """

    tokens: list[str]
    token_ids: list[int]
    provided: set[Capability] = field(default_factory=set)
    attentions: "list[torch.Tensor] | None" = None      # per layer: (heads, seq, seq)
    hidden_states: "list[torch.Tensor] | None" = None    # per layer: (seq, dim)
    logits: "torch.Tensor | None" = None                 # (seq, vocab)
    token_type_map: "TokenTypeMap | None" = None         # VLM: image-token positions + patch grid
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    def require(self, capability: Capability) -> Any:
        """Return the field for *capability*, raising if it wasn't captured."""
        attr = {
            Capability.ATTENTION: "attentions",
            Capability.HIDDEN_STATES: "hidden_states",
            Capability.LOGITS: "logits",
        }.get(capability)
        value = getattr(self, attr) if attr else None
        if value is None:
            raise ValueError(
                f"Trace does not contain {capability.value!r}. "
                f"Was it included in the forward(capture=...) set? "
                f"Captured: {sorted(c.value for c in self.provided)}"
            )
        return value


class Model(ABC):
    """Abstract base for every analyzable model.

    Subclasses must:
      1. set the ``capabilities`` class attribute,
      2. implement :meth:`generate` and :meth:`forward`.

    They get ``call_<analyzer>(data, **kwargs)`` for free via :meth:`__getattr__`.
    """

    #: Capabilities this model provides. Override in subclasses.
    capabilities: frozenset[Capability] = frozenset()

    #: Modalities this model handles (e.g. {"text"}, {"text","image"}). Analyzers
    #: declare ``applies_to_modalities`` and are matched against this.
    modalities: frozenset[str] = frozenset({"text"})

    def unembed_weight(self):
        """Optional: the unembedding matrix ``(vocab, dim)`` for logit-lens analyzers.

        Returns ``None`` unless a white-box backend exposes it.
        """
        return None

    @abstractmethod
    def generate(self, inputs: Any, **kwargs) -> str:
        """Produce a text response for *inputs* (a prompt or :class:`Inputs`)."""

    @abstractmethod
    def forward(
        self,
        inputs: Any,
        capture: set[Capability],
        spec: "CaptureSpec | None" = None,
    ) -> Trace:
        """Run one forward pass, capturing the requested internals into a :class:`Trace`.

        ``capture`` names *what* to record; the optional ``spec`` bounds *how
        much* (layers/heads/device) for memory control.  Backends free to ignore
        ``spec`` when they cannot sub-select.
        """


    def chat(self, messages: list, tools: "list | None" = None) -> "ChatTurn":
        """One tool-aware turn for agent mode (the ``Agent`` loop calls this).

        Returns a :class:`~evalrx.core.tool.ChatTurn` (assistant text + any
        native tool_calls).  Only tool-capable handles override this; the default
        signals that this model cannot drive an agent loop.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement chat(); it is not TOOL_CALLS-capable."
        )

    def logprobs(self, inputs: Any, **kwargs) -> "list[TokenLogprob]":
        """Generate and return per-output-token logprobs (the LOGPROBS capability).

        Black-box-feasible (OpenAI-style ``logprobs``).  Only LOGPROBS-capable
        handles override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement logprobs(); it is not LOGPROBS-capable."
        )


    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------

    def supports(self, required: frozenset[Capability] | set[Capability]) -> bool:
        """True iff this model provides every capability in *required*."""
        return set(required).issubset(self.capabilities)

    # ------------------------------------------------------------------
    # Hybrid convenience shim:  model.call_<analyzer>(data, **kwargs)
    # ------------------------------------------------------------------

    def __getattr__(self, attr: str) -> Callable[..., Any]:
        # __getattr__ only fires for attributes not found normally, so this
        # never shadows real methods/fields.
        if attr.startswith("call_"):
            analyzer_name = attr[len("call_"):]
            from evalrx.core.registry import registry

            if registry.analyzers.has(analyzer_name):
                analyzer_cls = registry.analyzers.get(analyzer_name)

                def _shim(data: Any, **kwargs) -> Any:
                    return analyzer_cls(**kwargs).run(self, data)

                _shim.__name__ = attr
                _shim.__doc__ = (
                    f"Convenience shim → {analyzer_cls.__name__}(**kwargs).run(self, data). "
                    f"Auto-derived from capabilities + registry."
                )
                return _shim

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {attr!r}"
        )
