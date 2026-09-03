"""Grad-CAM on the vision tower (Stage 2).

Gradient-weighted activation maps over the ViT feature map → spatial saliency.
``requires=GRADIENTS`` (backward pass); VLM vision tower only. Needs a
``reshape_transform`` for ViT and the per-arch vision-tower target layer.

References:
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
  Selvaraju et al., ICCV 2017 — arXiv:1610.02391
- Code: https://github.com/jacobgil/pytorch-grad-cam
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer


@register_analyzer("gradcam")
class GradCAMAnalyzer(Analyzer):
    name = "gradcam"
    requires = frozenset({Capability.GRADIENTS})
    applies_to_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: register a hook on the vision-tower target block, backward through a "
            "target logit, weight activations by gradients (pytorch-grad-cam + reshape_transform)."
        )
