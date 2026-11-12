"""Grad-CAM on the decoder's image-token states — which image regions the answer depends on.

Selvaraju et al. weight a CNN feature map's channels by the spatially averaged
gradient of the class score, sum, and keep the positive part. A VLM decoder has
no CNN feature map, but at any decoder layer its image tokens ARE one: a grid of
``(h, w)`` positions, each carrying a ``dim``-channel state. So, at decoder
layer ``l`` (the residual stream entering it):

    α_d   = mean over image tokens t of  ∂margin / ∂h[t, d]
    cam_t = ReLU( Σ_d α_d · h[t, d] )

with ``margin`` the next-token logit margin (see
:mod:`evalrx.analyzers._internals.targets`). The map is rebuilt on the image's
post-merge patch grid, so it overlays on the photo like ``relative_attention``.

This is an ADAPTATION of Grad-CAM, not the vision-tower original: it reads the
image tokens after the language model has started mixing them, which is where
an answer's dependence on the image is decided, and it needs no
per-architecture vision-tower target layer.

Alongside the map, ``gc_media_grad_share`` is a cruder, map-free reading: the
share of first-order (gradient × input) sensitivity at that layer carried by
image tokens rather than text tokens.

References:
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
  Selvaraju et al., ICCV 2017 — arXiv:1610.02391
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("gradcam")
class GradCAMAnalyzer(Analyzer):
    """Grad-CAM over image tokens at one decoder layer, per case.

    Hyper-parameters:
        layer:     decoder layer whose INPUT residual stream is read. ``float``
                   in (0, 1) = fractional depth (default 0.5); ``int`` =
                   absolute index (negative allowed).
        target:    ``"prediction"`` (clean top-1 vs top-2) or ``"gold"``.
        max_cases: label-stratified cap; 0 (the default) = every case.
                   Cost: one forward + one backward per case.
    """

    name = "gradcam"
    description = (
        "Gradient-weighted map of which image regions the answer token depends on "
        "(Grad-CAM over the decoder's image tokens), plus the image tokens' share of "
        "gradient×input sensitivity."
    )
    requires = frozenset({Capability.GRADIENTS, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video"})
    requires_modalities = frozenset({"image", "video"})
    signal_docs = {
        "gc_margin": ("Answer margin", "Logit margin of the target answer token the map explains."),
        "gc_media_grad_share": ("Image share of sensitivity", "Share of the answer's first-order (gradient × input) sensitivity carried by image tokens at the probed layer; low = the answer hinges on the text."),
        "gc_empty": ("Grad-CAM map empty", "No image token pushed the answer up (the positive part of the map is zero everywhere)."),
        "gc_entropy": ("Grad-CAM spread", "Normalized entropy of the Grad-CAM map over image tokens (1 = diffuse, 0 = one patch)."),
        "gc_top1_share": ("Grad-CAM peak share", "Share of the Grad-CAM map on its single strongest image token."),
        "gc_top5_share": ("Grad-CAM top-5 share", "Share of the Grad-CAM map on its five strongest image tokens."),
        "gc_center_offset": ("Grad-CAM off-centre", "Distance of the map's centre of mass from the image centre (0 = centre, 1 = corner)."),
        "gc_edge_mass": ("Grad-CAM on border", "Share of the map on the outermost ring of patches."),
    }

    def __init__(self, layer: "int | float" = 0.5, target: str = "prediction", max_cases: int = 0) -> None:
        super().__init__(layer=layer, target=target, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch"):
        import numpy as np
        import torch

        from evalrx.analyzers._internals.access import decoder_stack, media_positions, torch_module
        from evalrx.analyzers._internals.maps import (
            SpatialMapResult,
            group_maps,
            image_token_scores,
            map_features,
        )
        from evalrx.analyzers._internals.residual import residual_hooks
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.analyzers.attention.relative_attn import resolve_attention_layer
        from evalrx.core.model import CaptureSpec

        stack = decoder_stack(torch_module(model, self.name))
        layer = resolve_attention_layer(self.layer, stack.n_layers) % stack.n_layers
        spec = CaptureSpec(with_grad=True, to_cpu=False)

        per_case: list[dict[str, Any]] = []
        per_case_maps: dict[str, Any] = {}
        skipped: Counter = Counter()
        first_id: "str | None" = None
        for case in cases.stratified_head(self.max_cases):
            with residual_hooks(model, capture={layer: None}, keep_grad=True,
                                analyzer=self.name) as hooks:
                trace = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            slot, media = media_positions(trace)
            if slot != "image":
                skipped["no_image_tokens"] += 1
                continue
            logits = trace.require(Capability.LOGITS)[-1:]
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            hidden = hooks.captured[layer]
            if not hidden.requires_grad:
                raise RuntimeError(
                    f"{self.name}: the forward pass kept no autograd graph — the backend "
                    "ignores CaptureSpec(with_grad=True), so there is nothing to differentiate."
                )
            m_value = margin.value(logits)
            (grad,) = torch.autograd.grad(margin(logits), hidden)
            h, g = hidden.detach()[0].float(), grad[0].float()
            idx = torch.tensor(media, device=h.device)
            cam = torch.relu(h[idx] @ g[idx].mean(0)).cpu().numpy()
            gxi = (g * h).sum(-1).abs()
            share = float(gxi[idx].sum() / gxi.sum().clamp(min=1e-12))
            flat, spatial = image_token_scores(trace, cam)
            del trace, hooks, hidden, grad, logits, h, g

            row: dict[str, Any] = {
                "sample_id": case.id,
                "gc_margin": round(m_value, 4),
                "gc_media_grad_share": round(share, 4),
                "gc_empty": bool(flat.sum() <= 0.0),
            }
            row.update(map_features(flat, spatial, "gc"))
            per_case.append(row)
            per_case_maps[case.id] = np.asarray(spatial, dtype=np.float16)
            if first_id is None:
                first_id = case.id

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        artifacts: dict[str, Any] = {
            "spatial_map": per_case_maps[first_id].astype(np.float64),
            "per_case_maps": per_case_maps,
        }
        group_artifacts, group_findings = group_maps(per_case_maps, cases)
        artifacts.update(group_artifacts)
        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "layer_used": layer,
            "n_layers": stack.n_layers,
            "target": self.target,
            "map_shape": list(artifacts["spatial_map"].shape),
            "mean_media_grad_share": round(sum(r["gc_media_grad_share"] for r in per_case) / n, 4),
            "frac_empty": round(sum(r["gc_empty"] for r in per_case) / n, 4),
            "per_case": per_case,
            "spatial_map_case_id": first_id,
            **group_findings,
            "_caveat": (
                "Grad-CAM adapted to decoder image tokens (residual stream entering "
                "layer_used), explaining the next-token margin at the end of the prompt. "
                "A gradient map is a first-order local explanation: it says what would "
                "move the margin, not what the model attended to."
            ),
        }
        return SpatialMapResult(analyzer=self.name, model=repr(model), cases=cases,
                                artifacts=artifacts, findings=findings)
