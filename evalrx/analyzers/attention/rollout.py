"""Attention rollout (Abnar & Zuidema, 2020).

Composes per-layer attention (head-averaged, residual-augmented, row-normalised)
across the stack to estimate how much each input token influences the final
position.  Operates on a ``Trace`` (``requires=ATTENTION``).

Caveat: rollout is a heuristic over raw attention — see "Attention is not
Explanation" (Jain & Wallace) before over-claiming causality.

References:
- Quantifying Attention Flow in Transformers (attention rollout)
  Abnar & Zuidema, ACL 2020 — arXiv:2005.00928
- Caveat — Attention is not Explanation, Jain & Wallace, NAACL 2019 — arXiv:1902.10186
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
class RolloutResult(Result):
    tokens: list[str] = field(default_factory=list)

    @property
    def rollout(self):
        return self.artifacts["rollout"]


@register_analyzer("attention_rollout")
class AttentionRolloutAnalyzer(Analyzer):
    """Attention-rollout influence of each token on the final position."""

    name = "attention_rollout"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, top_k: int = 5) -> None:
        super().__init__(top_k=top_k)

    def _run(self, model: "Model", cases: "CaseBatch") -> RolloutResult:
        import torch

        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.ATTENTION})
        attns = trace.require(Capability.ATTENTION)  # list of (heads, seq, seq)
        seq = attns[0].shape[-1]
        eye = torch.eye(seq)
        rollout = eye.clone()
        for a in attns:
            A = a.float().mean(0) + eye           # head-average + residual
            A = A / A.sum(dim=-1, keepdim=True)   # row-normalise
            rollout = A @ rollout
        last = rollout[-1]
        k = min(self.top_k, seq)
        topk = torch.topk(last, k)
        top = [{"token": trace.tokens[i], "weight": round(float(v), 4)} for v, i in zip(topk.values, topk.indices)]
        result = RolloutResult(
            analyzer=self.name, model=repr(model), cases=cases,
            tokens=trace.tokens, artifacts={"rollout": rollout},
        )
        result.findings = {"seq_len": seq, "n_layers": len(attns), "top_rollout_tokens": top}
        return result
