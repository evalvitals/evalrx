"""Logit lens (nostalgebraist) — project intermediate hidden states through the unembed.

Reveals *when* (which layer) the model's prediction forms.  Needs ``HIDDEN_STATES``
plus the model's unembedding matrix (``model.unembed_weight()``).  The cheapest,
highest-value white-box signal — build it first.

Faithfulness: when the model exposes ``final_norm()`` (RMSNorm-family models —
Llama, Qwen, ...), each PREMATURE layer's hidden state is normalized before
unembedding, matching ``lm_head(norm(h_i))``; raw projection distorts
trajectories on those models.  The final hidden-states entry is projected
as-is: HF backends return it already post final-norm, so re-normalising it
would square the RMS gain and distort the reference distribution.  Findings
note whether the norm was applied.

References:
- interpreting GPT: the logit lens — nostalgebraist (2020), LessWrong
  https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


@register_analyzer("logit_lens")
class LogitLensAnalyzer(Analyzer):
    """Per-layer readout at one position, plus per-case decision-depth signals.

    Per case (``findings["per_case"]``, consumable by M2's stats layer):
        decision_layer / decision_frac: first layer whose top-1 equals the
            final top-1 — late decisions mean the answer formed in the last
            layers (prior-override candidates).
        final_top1_prob:  confidence of the emitted token at the final layer.
        interim_peak_prob / late_drop: peak probability of the final top-1
            across layers and its drop to the final layer (non-monotonic
            trajectories flag suppression dynamics).

    Hyper-parameters:
        pos:       query position to read (default ``-1``, the last token).
        top_k:     number of candidates to report per layer (first case only).
        max_cases: cap on cases analyzed (one forward each).
    """

    name = "logit_lens"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    # Default cap sized for ENRICHED diagnosis batches: the per-layer fail/pass
    # contrast is powered by the scarce FAIL group, and a small cap (e.g. 32)
    # starves it — on a curated batch the stratified subsample then carries only
    # ~16 FAILs, too few for M5 to clear the e-value bar. 128 covers typical
    # mined fail counts (tens) plus controls at one forward each.
    def __init__(self, pos: int = -1, top_k: int = 3, max_cases: int = 128) -> None:
        super().__init__(pos=pos, top_k=top_k, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        W = model.unembed_weight()
        if W is None:
            raise ValueError(
                f"{type(model).__name__} exposes no unembed_weight(); logit-lens needs it "
                "(white-box local backend)."
            )
        W = W.detach().float()  # (vocab, dim), once — not per layer
        device = W.device
        norm = model.final_norm() if hasattr(model, "final_norm") else None
        norm_dtype = next(norm.parameters()).dtype if norm is not None and any(
            True for _ in norm.parameters()) else None

        # Label-stratified subsample: a plain head is mostly PASS on an enriched
        # batch, starving the FAIL group that the downstream contrast needs.
        per_case, per_layer_top, n_layers = [], None, 0
        for case in cases.stratified_head(self.max_cases):
            trace = model.forward(case.inputs, capture={Capability.HIDDEN_STATES})
            hidden = trace.require(Capability.HIDDEN_STATES)  # list per layer: (seq, dim)
            n_layers = len(hidden)

            layer_logits = []
            with torch.no_grad():
                for layer_index, h in enumerate(hidden):
                    vec = h[self.pos].detach().to(device)  # device-align: states may be on CPU
                    # HF backends return the FINAL hidden-states entry already
                    # post final-norm (this analyzer requests the full stack, so
                    # the last element IS that entry); re-normalising it would
                    # square the RMS gain and shift the final distribution the
                    # decision-depth columns are anchored to.
                    if norm is not None and layer_index < len(hidden) - 1:
                        vec = norm(vec.to(norm_dtype)) if norm_dtype is not None else norm(vec)
                    layer_logits.append(vec.float() @ W.T)  # (vocab,)

                final = layer_logits[-1]
                top1 = int(final.argmax())
                p_top1 = [float(torch.softmax(lg, dim=-1)[top1]) for lg in layer_logits]
                argmaxes = [int(lg.argmax()) for lg in layer_logits]
            decision_layer = next(
                (i for i in range(n_layers) if all(a == top1 for a in argmaxes[i:])), n_layers - 1)
            peak = max(p_top1)
            per_case.append({
                "sample_id": case.id,
                "decision_layer": decision_layer,
                "decision_frac": round(decision_layer / max(1, n_layers - 1), 4),
                "final_top1_prob": round(p_top1[-1], 4),
                "interim_peak_prob": round(peak, 4),
                "late_drop": round(peak - p_top1[-1], 4),
            })

            if per_layer_top is None:  # detailed per-layer table for the first case only
                per_layer_top = []
                for layer_idx, lg in enumerate(layer_logits):
                    k = min(self.top_k, lg.shape[-1])
                    topk = torch.topk(lg, k)
                    per_layer_top.append({
                        "layer": layer_idx,
                        "top": [{"token_id": int(i), "logit": round(float(v), 3)}
                                for v, i in zip(topk.values, topk.indices)],
                    })

        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"per_case": per_case},
            findings={
                "n_layers": n_layers, "pos": self.pos,
                "n_cases_analyzed": len(per_case),
                "final_norm_applied": norm is not None,
                "per_layer_top": per_layer_top or [],
                "per_case": per_case,
            },
        )
