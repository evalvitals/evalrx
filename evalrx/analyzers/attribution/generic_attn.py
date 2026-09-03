"""Generic attention-model explainability (Chefer et al., 2021) (Stage 2).

Relevance propagation through attention layers using attention weights AND their
gradients — hence ``requires={ATTENTION, GRADIENTS}`` (white-box, not black-box,
correcting the original taxonomy). Built for enc-dec / bi-modal transformers; not
decoder-only-native.

References:
- Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers
  Chefer, Gur & Wolf, ICCV 2021 — arXiv:2103.15679
- Predecessor — Transformer Interpretability Beyond Attention Visualization,
  Chefer, Gur & Wolf, CVPR 2021 — arXiv:2012.09838
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer


@register_analyzer("generic_attention")
class GenericAttentionExplainability(Analyzer):
    name = "generic_attention"
    requires = frozenset({Capability.ATTENTION, Capability.GRADIENTS})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: Chefer-style relevance via attention × gradient-of-attention propagation."
        )
