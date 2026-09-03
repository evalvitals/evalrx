"""Embedding-geometry analyzers (HIDDEN_STATES). CLIP/SigLIP-tower-scoped — see caveats."""

from evalrx.analyzers.geometry.cka import CKAAnalyzer, linear_cka
from evalrx.analyzers.geometry.linear_probe import LinearProbeAnalyzer

__all__ = ["CKAAnalyzer", "linear_cka", "LinearProbeAnalyzer"]
