"""Backend — a runtime that turns a ModelSpec into a Model.

The orthogonal decomposition: **capabilities come from the backend**,
**identity comes from the spec**.  The same :class:`~evalrx.core.spec.ModelSpec`
drives an API call (``api``), a high-throughput logprob batch (``vllm_offline``),
or a full internals capture (``hf_local``) — only the capability set differs.

This module is torch-free; concrete backends import their heavy deps lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.core.capability import Capability

if TYPE_CHECKING:
    from evalrx.core.model import Model
    from evalrx.core.spec import ModelSpec


@dataclass
class RuntimeConfig:
    """How to run a model — orthogonal to *which* model (that's the spec)."""

    device: str = "auto"
    dtype: str = "bfloat16"
    attn_impl: Optional[str] = None          # hf_local forces "eager" when attention is wanted
    max_new_tokens: int = 512
    #: hf_local, TEXT-ONLY specs: render ``inputs.prompt`` as one user turn through
    #: the tokenizer's chat template (with ``spec.chat_template_kwargs``, e.g.
    #: ``enable_thinking=False``) before generate()/logprobs(). Off by default: a
    #: raw prompt is tokenised verbatim (completion mode), which is what every
    #: caller got before 2026-08-21 and what a pre-formatted prompt needs. The
    #: multimodal encode path always renders the template.
    apply_chat_template: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)   # vllm_offline LLM(**)
    client_kwargs: dict[str, Any] = field(default_factory=dict)   # api client opts (e.g. logprobs=True)
    generate_fn: Optional[Callable[..., str]] = None              # api: simple text generate
    chat_fn: Optional[Callable[..., Any]] = None                  # api: tool-aware chat (returns ChatTurn)
    logprobs_fn: Optional[Callable[..., Any]] = None              # api: returns list[TokenLogprob]


class Backend(ABC):
    """Base for runtimes.  Declares the capability set it can provide."""

    kind: str = "base"
    capabilities: frozenset[Capability] = frozenset()

    @abstractmethod
    def build(self, spec: "ModelSpec", runtime: RuntimeConfig) -> "Model":
        """Construct a :class:`~evalrx.core.model.Model` for *spec* (lazy load)."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(kind={self.kind!r})"
