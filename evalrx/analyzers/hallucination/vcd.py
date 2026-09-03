"""VCD — Visual Contrastive Decoding (Leng et al.) (Stage 2).

Contrast logits on the original vs a distorted image to suppress
language-prior-driven hallucination. Needs decode-time logit control (in-process
HF), VLM. Modeled here as an analyzer that reports the contrastive shift.

References:
- Mitigating Object Hallucinations in LVLMs through Visual Contrastive Decoding (VCD)
  Leng et al., CVPR 2024 — arXiv:2311.16922
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer


@register_analyzer("vcd")
class VCDAnalyzer(Analyzer):
    name = "vcd"
    requires = frozenset({Capability.LOGITS})
    applies_to_modalities = frozenset({"image"})
    # An omni model DECLARES image even on an audio benchmark, so the registry
    # match alone still offers this on a batch with no images in it.
    requires_modalities = frozenset({"image"})

    def _run(self, model, cases):
        raise NotImplementedError("Stage 2: contrastive decoding over original vs distorted image (logit control).")
