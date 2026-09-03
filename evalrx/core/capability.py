"""Capability vocabulary — the contract between models and analyzers.

A *model* declares the capabilities it **provides** (e.g. a local Qwen exposes
``ATTENTION`` and ``HIDDEN_STATES``; an API model provides only ``GENERATE``).
An *analyzer* declares the capabilities it **requires**.  The registry matches
the two, so an agent can ask "what can I run on this model?" without touching
any model internals.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """What a model can expose to analyzers.

    Inherits from ``str`` so values are JSON-serialisable and print cleanly,
    which matters when results are handed to an LLM agent.
    """

    GENERATE = "generate"           # produce text output
    TOOL_CALLS = "tool_calls"       # emit/consume tool calls (agent mode)
    LOGPROBS = "logprobs"           # top-k token logprobs (API-feasible; OpenAI yes / Gemini no)
    LOGITS = "logits"               # full next-token logit tensor (white-box forward)
    ATTENTION = "attention"         # per-layer, per-head attention weights
    HIDDEN_STATES = "hidden_states" # per-layer residual-stream activations
    EMBEDDINGS = "embeddings"       # input/output embedding vectors
    ACTIVATIONS = "activations"     # arbitrary hooked module activations
    GRADIENTS = "gradients"         # input/parameter gradients (saliency)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Capability.{self.name}"


class CapabilityError(RuntimeError):
    """Raised when an analyzer is run on a model that lacks required capabilities."""

    def __init__(
        self,
        analyzer: str,
        model: str,
        missing: set[Capability],
    ) -> None:
        self.analyzer = analyzer
        self.model = model
        self.missing = missing
        missing_names = sorted(c.value for c in missing)
        super().__init__(
            f"Analyzer '{analyzer}' cannot run on model '{model}': "
            f"missing capabilities {missing_names}. "
            f"This analysis requires a model that provides them "
            f"(typically a white-box / locally-deployed model)."
        )
