"""Premature-layer vs final-layer divergence — the DoLa/DeCo family's diagnostic half.

DoLa's observation: factual knowledge surfaces in specific layers, and tokens
whose final distribution diverges sharply from premature layers are where
"what the layers know" and "what gets decoded" disagree — a hallucination
indicator.  DeCo's follow-up: hallucinated targets are often *correct* in
preceding layers and suppressed at the top.  This analyzer measures those
dynamics per case WITHOUT changing decoding: it emits divergence/agreement
columns that, when they separate FAIL from PASS in M2, point the fix search at
layer-contrast decoding repairs (DoLa/DeCo/VISTA family).

White-box: ``requires=HIDDEN_STATES`` plus the model's unembedding
(``unembed_weight()``), same access contract as :class:`LogitLensAnalyzer`.

References:
- DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
  Models — Chuang et al., ICLR 2024 — arXiv:2309.03883
- MLLM Can See? Dynamic Correction Decoding for Hallucination Mitigation
  (DeCo) — Wang et al., ICLR 2025 — arXiv:2410.11779
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("layer_contrast")
class LayerContrastAnalyzer(Analyzer):
    """DoLa-style premature/final layer divergence columns (JSD, agreement, contrast margin).

    Hyper-parameters:
        pos:             query position to read (default ``-1``, the last token).
        skip_first_frac: fraction of early layers excluded from the premature
                         candidate set (embedding-adjacent layers are noise).
        max_cases:       label-stratified cap (one forward per case); 0 (the default) = every case.
    """

    name = "layer_contrast"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, pos: int = -1, skip_first_frac: float = 0.25, max_cases: int = 0) -> None:
        super().__init__(pos=pos, skip_first_frac=float(skip_first_frac), max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        W = model.unembed_weight()
        if W is None:
            raise ValueError(
                f"{type(model).__name__} exposes no unembed_weight(); layer-contrast "
                "needs it (white-box local backend)."
            )
        W = W.detach().float()
        device = W.device
        norm = model.final_norm() if hasattr(model, "final_norm") else None
        norm_dtype = next(norm.parameters()).dtype if norm is not None and any(
            True for _ in norm.parameters()) else None

        def _jsd(p: "torch.Tensor", q: "torch.Tensor") -> float:
            m = 0.5 * (p + q)
            kl_pm = torch.sum(p * (torch.log(p + 1e-12) - torch.log(m + 1e-12)))
            kl_qm = torch.sum(q * (torch.log(q + 1e-12) - torch.log(m + 1e-12)))
            return float(0.5 * (kl_pm + kl_qm))

        per_case: list[dict[str, Any]] = []
        n_layers = 0
        for case in cases.stratified_head(self.max_cases):
            trace = model.forward(case.inputs, capture={Capability.HIDDEN_STATES})
            hidden = trace.require(Capability.HIDDEN_STATES)  # list per layer: (seq, dim)
            n_layers = len(hidden)
            with torch.no_grad():
                probs = []
                for index, h in enumerate(hidden):
                    vec = h[self.pos].detach().to(device)
                    # HF backends return the FINAL hidden-states entry already
                    # post final-norm; re-normalising it would square the RMS
                    # gain and distort every divergence below.
                    if norm is not None and index < len(hidden) - 1:
                        vec = norm(vec.to(norm_dtype)) if norm_dtype is not None else norm(vec)
                    probs.append(torch.softmax(vec.float() @ W.T, dim=-1))
                final = probs[-1]
                top1 = int(final.argmax())
                start = min(int(self.skip_first_frac * n_layers), max(0, n_layers - 2))
                candidates = list(range(start, n_layers - 1))
                if not candidates:
                    candidates = [max(0, n_layers - 2)]
                jsds = [_jsd(final, probs[layer]) for layer in candidates]
                jsd_max = max(jsds)
                max_layer = candidates[jsds.index(jsd_max)]
                agree = sum(1 for layer in candidates if int(probs[layer].argmax()) == top1)
                # DoLa contrast margin at the max-divergence premature layer:
                # positive = the final layer amplifies its top-1 over the
                # premature layer; negative = the top-1 was STRONGER early and
                # got suppressed at the top (DeCo's hallucination signature).
                contrast_margin = float(
                    torch.log(final[top1] + 1e-12) - torch.log(probs[max_layer][top1] + 1e-12)
                )
            per_case.append({
                "sample_id": case.id,
                "jsd_max": round(jsd_max, 4),
                "jsd_mean": round(sum(jsds) / len(jsds), 4),
                "jsd_max_layer_frac": round(max_layer / max(1, n_layers - 1), 4),
                "layer_agreement_frac": round(agree / len(candidates), 4),
                "contrast_margin": round(contrast_margin, 4),
                "final_top1_prob": round(float(final[top1]), 4),
            })

        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_layers": n_layers,
            "pos": self.pos,
            "premature_layer_window": f"[{self.skip_first_frac:.2f}L, L-1)",
            "per_case": per_case,
            "_caveat": (
                "Divergence DIAGNOSTICS only — no decoding intervention is "
                "performed. If jsd_max / negative contrast_margin separate FAIL "
                "from PASS in M2, the indicated repair family is layer-contrast "
                "decoding (DoLa/DeCo/VISTA), an L0/L3 fix. Columns read one "
                "position (pos) of a teacher-forced forward; generation-time "
                "dynamics can differ."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
