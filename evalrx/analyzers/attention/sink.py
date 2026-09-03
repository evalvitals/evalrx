"""Attention sink (Gu et al., "When Attention Sink Emerges in LMs").

Measures how much attention mass collapses onto the first token (the "sink"),
per layer, averaged over heads and query positions.  Operates on a ``Trace``.

References:
- Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
  Xiao et al., ICLR 2024 — arXiv:2309.17453
- When Attention Sink Emerges in Language Models: An Empirical View
  Gu et al., ICLR 2025 — arXiv:2410.10781
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("attention_sink")
class AttentionSinkAnalyzer(Analyzer):
    """Per-layer attention mass on the first token (sink strength)."""

    name = "attention_sink"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, sink_pos: int = 0) -> None:
        super().__init__(sink_pos=sink_pos)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.ATTENTION})
        attns = trace.require(Capability.ATTENTION)
        per_layer = []
        for a in attns:
            A = a.float().mean(0)                  # (seq, seq), head-averaged
            per_layer.append(round(float(A[:, self.sink_pos].mean()), 4))
        mean_sink = round(sum(per_layer) / max(len(per_layer), 1), 4)
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"per_layer_sink": per_layer},
            findings={
                "n_layers": len(attns),
                "sink_token": trace.tokens[self.sink_pos] if trace.tokens else None,
                "mean_sink_mass": mean_sink,
                "per_layer_sink": per_layer,
            },
        )
