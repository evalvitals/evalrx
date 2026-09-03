"""OPERA — over-trust penalty + retrospection-allocation on attention (Stage 2).

Diagnoses object hallucination from the attention pattern (over-trust on a few
summary tokens). ``requires=ATTENTION`` (+ decode-loop control for the mitigation).
White-box, VLM.

References:
- OPERA: Alleviating Hallucination in MLLMs via Over-trust Penalty and Retrospection-Allocation
  Huang et al., CVPR 2024 — arXiv:2311.17911
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer


@register_analyzer("opera")
class OPERAAnalyzer(Analyzer):
    name = "opera"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"image"})
    # An omni model DECLARES image even on an audio benchmark, so the registry
    # match alone still offers this on a batch with no images in it.
    requires_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError("Stage 2: attention over-trust diagnosis (needs decode-loop control).")
