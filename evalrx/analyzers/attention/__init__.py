"""Attention analyzers (require the ATTENTION capability → white-box / HF eager).

NOTE: these import torch at module load; on the light (pure-API) install the whole
subpackage is skipped by ``analyzers/__init__`` (you can't run them without torch).
"""

from evalrx.analyzers.attention.relative_attn import RelativeAttentionAnalyzer
from evalrx.analyzers.attention.rollout import AttentionRolloutAnalyzer, RolloutResult
from evalrx.analyzers.attention.sink import AttentionSinkAnalyzer
from evalrx.analyzers.attention.summary import AttentionAnalyzer, AttentionResult

__all__ = [
    "AttentionAnalyzer",
    "AttentionResult",
    "AttentionRolloutAnalyzer",
    "RolloutResult",
    "AttentionSinkAnalyzer",
    "RelativeAttentionAnalyzer",
]
