"""Linear CKA — representational similarity across layers.

Computes pairwise linear Centered Kernel Alignment between layers' hidden states
(sequence positions as samples).  ``requires=HIDDEN_STATES``.

Construct-validity caveat: CKA/cosine-geometry methods were validated on CLIP-style
two-tower encoders. A decoder-only MLLM has no joint two-tower space, so cross-modal
geometry numbers don't carry the meaning the papers claim — scope this to a frozen
CLIP/SigLIP tower's activations, or read it only as within-stream layer similarity.

References:
- Similarity of Neural Network Representations Revisited (linear CKA)
  Kornblith, Norouzi, Lee & Hinton, ICML 2019 — arXiv:1905.00414
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


def linear_cka(X, Y) -> float:
    """Linear CKA between two activation matrices ``(n, d1)`` and ``(n, d2)``."""

    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    hsic = (Y.T @ X).pow(2).sum()
    denom = (X.T @ X).norm() * (Y.T @ Y).norm()
    return float(hsic / denom) if denom > 0 else 0.0


@register_analyzer("cka")
class CKAAnalyzer(Analyzer):
    """Pairwise linear-CKA matrix across the model's layers for one input."""

    name = "cka"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.HIDDEN_STATES})
        hidden = trace.require(Capability.HIDDEN_STATES)  # list of (seq, dim)
        L = len(hidden)
        mat = torch.zeros(L, L)
        for i in range(L):
            for j in range(i, L):
                c = linear_cka(hidden[i], hidden[j])
                mat[i, j] = mat[j, i] = c
        off = mat[~torch.eye(L, dtype=torch.bool)]
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"cka_matrix": mat},
            findings={
                "n_layers": L,
                "mean_offdiagonal_cka": round(float(off.mean()), 4),
                "adjacent_layer_cka": [round(float(mat[i, i + 1]), 4) for i in range(L - 1)],
                "_caveat": "CKA validated on CLIP two-tower encoders; scope to a tower, not decoder MLLM cross-modal geometry.",
            },
        )
