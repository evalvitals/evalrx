"""VL-SHAP — Shapley attribution over image regions for VLMs.

Black-box (``LOGPROBS``): partition the image into a grid of regions, mask
subsets, score each masked image by the model's output logprob, and compute each
region's Shapley value → a spatial attribution map over the image.

The region scorer is injectable (``region_score_fn(kept_regions: set) -> float``)
for decoupled testing; the default builds a grid-masked image (PIL) and scores it
via ``model.logprobs`` (mean logprob).  ``requires=LOGPROBS``; VLM-only.

Shapley values for ML:
  "A Unified Approach to Interpreting Model Predictions"
  Lundberg & Lee, NeurIPS 2017 — https://arxiv.org/abs/1705.07874
  Code: https://github.com/shap/shap

Region-level visual attribution for VLMs (MM-SHAP framework):
  "MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions
  in Vision and Language Models"
  Parcalabescu & Frank, ACL 2022 — https://arxiv.org/abs/2212.08158
  Code: https://github.com/coastalcph/mm-shap
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

from evalrx.analyzers.perturbation._shapley import shapley_values
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.case import Inputs
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("vl_shap")
class VLShapAnalyzer(Analyzer):
    """Per-region Shapley attribution over a grid of image regions."""

    name = "vl_shap"
    requires = frozenset({Capability.LOGPROBS})
    applies_to_modalities = frozenset({"image"})
    # Its default scorer dereferences case.inputs.image, so an omni model on an
    # audio batch matched here and then raised on the first case.
    requires_modalities = frozenset({"image"})

    def __init__(
        self,
        n_regions: int = 16,
        region_score_fn: Optional[Callable[[set], float]] = None,
        n_samples: int = 64,
        top_k: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(n_regions=n_regions, region_score_fn=region_score_fn,
                         n_samples=n_samples, top_k=top_k, seed=seed)

    def _default_region_value(self, model: "Model", case) -> Callable[[set], float]:
        from PIL import Image  # lazy

        image = case.inputs.image
        if image is None:
            raise ValueError("VLShapAnalyzer default scorer needs case.inputs.image (a PIL image).")
        side = max(1, int(math.isqrt(self.n_regions)))
        W, H = image.size
        cw, ch = W / side, H / side

        def value(kept: set) -> float:
            masked = image.copy()
            black = Image.new(image.mode, (max(1, int(cw)), max(1, int(ch))), 0)
            for r in range(self.n_regions):
                if r in kept:
                    continue
                gx, gy = r % side, r // side
                masked.paste(black, (int(gx * cw), int(gy * ch)))
            lps = model.logprobs(Inputs(prompt=case.inputs.prompt, image=masked))
            return sum(t.logprob for t in lps) / len(lps) if lps else 0.0
        return value

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        players = list(range(self.n_regions))
        value = self.region_score_fn or self._default_region_value(model, case)
        shap = shapley_values(players, value, n_samples=self.n_samples, seed=self.seed)
        top = sorted(
            ({"region": r, "shapley": round(shap[r], 4)} for r in players),
            key=lambda d: -abs(d["shapley"]),
        )[: self.top_k]
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"region_shapley": shap},
            findings={
                "n_regions": self.n_regions,
                "grid_side": max(1, int(math.isqrt(self.n_regions))),
                "top_regions": top,
                "total_abs_attribution": round(sum(abs(v) for v in shap.values()), 4),
            },
        )
