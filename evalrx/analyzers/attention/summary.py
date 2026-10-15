"""Attention analysis for white-box models.

``AttentionAnalyzer`` is a :class:`~evalrx.core.analyzer.Analyzer`: configure
it with hyper-parameters, then ``run`` it on any model that provides the
``ATTENTION`` capability.  It returns an :class:`AttentionResult`, whose
``findings`` are a light, agent-readable summary and whose ``artifacts`` hold the
raw per-layer/per-head tensors for plotting and numeric work.

References:
- "What's in the Image?" — vision attention deep-dive
- "VL-Cache" — modality-aware KV-cache
- "When Attention Sink Emerges in Language Models" — https://arxiv.org/abs/2309.17453
- "MLLMs Know Where to Look" (relative attention) — https://arxiv.org/abs/2502.17422
  · code: https://github.com/saccharomycetes/mllms_know
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


# ======================================================================
# Result
# ======================================================================

@dataclass
class AttentionResult(Result):
    """Attention analysis output for a single case.

    The heavy per-layer attention tensors live in ``artifacts["attentions"]``
    (each ``(heads, seq, seq)``).  ``findings`` carries the agent-readable
    summary.  The convenience methods below operate on the artifacts.
    """

    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)

    # -- artifact accessors --------------------------------------------
    @property
    def attentions(self) -> list[torch.Tensor]:
        return self.artifacts["attentions"]

    @property
    def num_layers(self) -> int:
        return len(self.attentions)

    @property
    def num_heads(self) -> int:
        return self.attentions[0].shape[0]

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    # -- core aggregation ----------------------------------------------
    def aggregate(self, layer: int = -1, head: int | str = "mean") -> torch.Tensor:
        """Return a ``(seq_len, seq_len)`` attention matrix.

        Args:
            layer: Layer index (negative indexing supported; ``-1`` = last).
            head:  Head index, or ``"mean"`` to average over heads.
        """
        attn = self.attentions[layer]
        return attn.mean(dim=0) if head == "mean" else attn[int(head)]

    def layer_head_matrix(self) -> torch.Tensor:
        """Stacked tensor of shape ``(layers, heads, seq_len, seq_len)``."""
        return torch.stack(self.attentions, dim=0)

    def to_numpy(self, layer: int = -1, head: int | str = "mean") -> np.ndarray:
        return self.aggregate(layer=layer, head=head).cpu().float().numpy()

    # -- inspection ----------------------------------------------------
    def top_attended_tokens(
        self,
        query_pos: int = -1,
        k: int = 10,
        layer: int = -1,
        head: int | str = "mean",
    ) -> list[tuple[str, float]]:
        """Top-k tokens attended to by the token at *query_pos*."""
        row = self.aggregate(layer=layer, head=head)[query_pos]
        k = min(k, self.seq_len)
        topk = torch.topk(row, k)
        return [(self.tokens[i], float(v)) for v, i in zip(topk.values, topk.indices)]

    def attention_entropy(self, layer: int = -1, head: int | str = "mean") -> torch.Tensor:
        """Shannon entropy of the attention distribution per query position."""
        mat = self.aggregate(layer=layer, head=head).float().clamp(min=1e-9)
        return -(mat * mat.log()).sum(dim=-1)

    # -- visualisation -------------------------------------------------
    def plot(self, layer: int = -1, head: int | str = "mean", ax=None, figsize=(8, 7)):
        """Plot the attention matrix as a heatmap (requires ``evalrx[viz]``)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Visualisation requires matplotlib. Install: pip install evalrx[viz]"
            ) from None

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=figsize)
        mat = self.to_numpy(layer=layer, head=head)
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
        ax.set_xticks(range(self.seq_len))
        ax.set_xticklabels(self.tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(self.seq_len))
        ax.set_yticklabels(self.tokens, fontsize=7)
        ax.set_title(f"Attention | layer={layer} head={head}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if created:
            plt.tight_layout()
            return fig
        return ax


# ======================================================================
# Analyzer
# ======================================================================

@register_analyzer("attention")
class AttentionAnalyzer(Analyzer):
    """Extract and summarise attention weights from a white-box model.

    Hyper-parameters (sklearn-style, passed to ``__init__``):
        layer: Default layer for the ``findings`` summary (``-1`` = last).
        head:  Default head for the summary (``"mean"`` or an int).
        top_k: Number of top attended tokens to include in ``findings``.

    Example::

        result = AttentionAnalyzer(layer=-1, top_k=5).run(qwen, "The capital of France is")
        result.summary()
        result.plot()
    """

    name = "attention"
    requires = frozenset({Capability.ATTENTION})
    signal_docs = {
        "last_query_entropy": ("Answer-token attention spread", "Normalized entropy of the final prompt position's attention at the summary layer (1 = spread evenly over all tokens, 0 = on one token)."),
        "last_query_top1_share": ("Answer-token peak attention", "Largest single share of the final prompt position's attention at the summary layer."),
        "media_mass_mid": ("Attention on image/audio", "Share of the final prompt position's attention that lands on image/audio tokens, averaged over heads and the middle half of the layers; low = the answer position barely reads the media."),
    }

    def __init__(self, layer: int = -1, head: int | str = "mean", top_k: int = 10,
                 max_cases: int = 0) -> None:
        super().__init__(layer=layer, head=head, top_k=top_k, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> AttentionResult:
        # The first case keeps the full per-layer tensors (single-case
        # ergonomics: aggregate/plot/top_attended_tokens); every analysed case
        # contributes one per_case row for the M2 stats layer.
        result: AttentionResult | None = None
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            trace = model.forward(case.inputs, capture={Capability.ATTENTION})
            attentions = trace.require(Capability.ATTENTION)
            if result is None:
                result = AttentionResult(
                    analyzer=self.name,
                    model=repr(model),
                    cases=cases,
                    tokens=trace.tokens,
                    token_ids=trace.token_ids,
                    artifacts={"attentions": attentions},
                )
            per_case.append({"sample_id": case.id, **self._case_columns(trace, attentions)})
        if result is None:
            raise ValueError("attention: empty case batch")
        result.findings = self._summarise(result)
        result.findings["n_cases_analyzed"] = len(per_case)
        result.findings["per_case"] = per_case
        return result

    def _case_columns(self, trace: Any, attentions: "list[torch.Tensor]") -> dict[str, Any]:
        """Per-case scalars: last-query concentration, and its media share mid-stack."""
        from evalrx.analyzers._internals.access import media_positions

        attn = attentions[self.layer]
        row = (attn.float().mean(0) if self.head == "mean" else attn[int(self.head)].float())[-1]
        p = row.clamp(min=1e-12) / row.sum().clamp(min=1e-12)
        ent = float(-(p * p.log()).sum()) / float(np.log(max(p.numel(), 2)))
        out: dict[str, Any] = {
            "last_query_entropy": round(ent, 4),
            "last_query_top1_share": round(float(row.max()), 4),
        }
        _, media = media_positions(trace)
        if media:
            n = len(attentions)
            mid = range(n // 4, max(n // 4 + 1, (3 * n) // 4))
            shares = [float(attentions[i].float().mean(0)[-1][media].sum()) for i in mid]
            out["media_mass_mid"] = round(sum(shares) / len(shares), 4)
        return out

    def _summarise(self, result: AttentionResult) -> dict[str, Any]:
        """Build the light, agent-readable summary stored in ``findings``."""
        top = result.top_attended_tokens(
            query_pos=-1, k=self.top_k, layer=self.layer, head=self.head
        )
        entropy = result.attention_entropy(layer=self.layer, head=self.head)
        return {
            "num_layers": result.num_layers,
            "num_heads": result.num_heads,
            "seq_len": result.seq_len,
            "summary_layer": self.layer,
            "summary_head": self.head,
            "top_attended_tokens": [{"token": t, "weight": round(w, 4)} for t, w in top],
            "mean_attention_entropy": round(float(entropy.mean()), 4),
        }
