"""Token-level uncertainty from the logit distribution.

A small, high-ROI white-box analyzer: from a forward pass it summarises the
next-token predictive distribution (entropy + top-k) — a cheap signal for
"where was the model unsure?" that needs only the ``LOGITS`` capability, so it
runs on any model that provides it (and, unlike attention, does not need eager).

``torch`` is imported inside ``_run`` so this module imports torch-free.

References:
- Uncertainty Estimation in Autoregressive Structured Prediction (predictive entropy)
  Malinin & Gales, ICLR 2021 — arXiv:2002.07650
- Semantic Uncertainty: ... for Natural Language Generation
  Kuhn, Gal & Farquhar, ICLR 2023 — arXiv:2302.09664
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@dataclass
class UncertaintyResult(Result):
    """Per-position predictive entropy (artifact) + agent-readable summary."""

    tokens: list[str] = field(default_factory=list)

    @property
    def entropy(self):  # (seq_len,)
        return self.artifacts["entropy"]


@register_analyzer("token_entropy")
class TokenEntropyAnalyzer(Analyzer):
    """Summarise next-token predictive entropy from a model's logits.

    Hyper-parameters:
        top_k: number of top next-token candidates to report at the final position.
    """

    name = "token_entropy"
    requires = frozenset({Capability.LOGITS})

    def __init__(self, top_k: int = 5) -> None:
        super().__init__(top_k=top_k)

    def _run(self, model: "Model", cases: "CaseBatch") -> UncertaintyResult:
        import torch

        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.LOGITS})
        logits = trace.require(Capability.LOGITS)            # (seq, vocab)

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(dim=-1)            # (seq,)

        last = logprobs[-1]
        k = min(self.top_k, last.shape[-1])
        topk = torch.topk(last, k)
        top_next = [
            {"token_id": int(i), "logprob": round(float(v), 4)}
            for v, i in zip(topk.values, topk.indices)
        ]

        result = UncertaintyResult(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            tokens=trace.tokens,
            artifacts={"entropy": entropy, "logits": logits},
        )
        result.findings = {
            "seq_len": len(trace.tokens),
            "mean_entropy": round(float(entropy.mean()), 4),
            "max_entropy": round(float(entropy.max()), 4),
            "final_token_entropy": round(float(entropy[-1]), 4),
            "top_next_tokens": top_next,
        }
        return result
