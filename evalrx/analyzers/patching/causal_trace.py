"""Activation patching / causal tracing (Meng et al., ROME) (Stage 2).

The only CAUSAL white-box method here: read a clean run's activations and WRITE
them into a corrupted run to localise where information is causally used.
``requires=HIDDEN_STATES`` (read+write hooks — via nnsight). Memory ∝ layers×positions.

References:
- Locating and Editing Factual Associations in GPT (ROME, causal tracing)
  Meng et al., NeurIPS 2022 — arXiv:2202.05262
- Attribution Patching (scalable approximation): Neel Nanda, 2023.
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer


@register_analyzer("causal_trace")
class CausalTraceAnalyzer(Analyzer):
    name = "causal_trace"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model, cases):
        raise NotImplementedError(
            "Stage 2: clean/corrupt runs + patch cached activations at each (layer, position) "
            "via nnsight; report the causal-effect map. Needs read+write hooks (beyond HF flags)."
        )
