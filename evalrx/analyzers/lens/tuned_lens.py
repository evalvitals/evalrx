"""Tuned lens (Belrose et al.) — per-layer learned affine translators before unembed.

More faithful than the raw logit lens, but requires PER-MODEL trained translators.
Stage 2: train/load translators, then apply like the logit lens.

References:
- Eliciting Latent Predictions from Transformers with the Tuned Lens
  Belrose et al., 2023 — arXiv:2303.08112
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer


@register_analyzer("tuned_lens")
class TunedLensAnalyzer(Analyzer):
    """Tuned-lens projection of hidden states (needs trained translators)."""

    name = "tuned_lens"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: load/train per-layer affine translators (e.g. via the `tuned-lens` package), "
            "then project hidden states through them before the unembed."
        )
