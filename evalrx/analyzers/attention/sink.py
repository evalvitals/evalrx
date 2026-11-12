"""Attention sink (Gu et al., "When Attention Sink Emerges in LMs").

Measures how much attention mass collapses onto the first token (the "sink"),
per layer, averaged over heads and query positions.  Operates on a ``Trace``.

Runs over the batch (one attention forward per case) and emits per-case sink
columns, so M2 can test whether FAIL cases dump more attention on the sink —
the "attention is parked, not reading anything" signature.

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
    """Per-layer attention mass on the first token (sink strength), per case.

    Hyper-parameters:
        sink_pos:  position treated as the sink (default 0, the first token).
        max_cases: label-stratified cap (one attention forward each); 0 (the
                   default) = every case.
    """

    name = "attention_sink"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"text", "image"})
    signal_docs = {
        "sink_mass_mean": ("Attention on sink token", "Share of attention every query puts on the first token, averaged over heads and layers; high = attention parked on the sink instead of reading content."),
        "sink_mass_last_query": ("Answer token's sink share", "Share of the final prompt position's attention on the first token, averaged over heads and layers."),
        "sink_peak_layer_frac": ("Strongest-sink depth", "Relative depth (0 = first layer, 1 = last) of the layer with the most attention on the sink."),
    }

    def __init__(self, sink_pos: int = 0, max_cases: int = 0) -> None:
        super().__init__(sink_pos=sink_pos, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict] = []
        layer_sums: list[float] = []
        n_summed = 0
        first = None  # (n_layers, sink token) of the first analysed case
        for case in cases.stratified_head(self.max_cases):
            trace = model.forward(case.inputs, capture={Capability.ATTENTION})
            attns = trace.require(Capability.ATTENTION)
            per_layer, last_query = [], []
            for a in attns:
                A = a.float().mean(0)                  # (seq, seq), head-averaged
                per_layer.append(float(A[:, self.sink_pos].mean()))
                last_query.append(float(A[-1, self.sink_pos]))
            if first is None:
                first = (len(attns), trace.tokens[self.sink_pos] if trace.tokens else None)
                layer_sums = [0.0] * len(attns)
            if len(per_layer) == len(layer_sums):
                layer_sums = [s + v for s, v in zip(layer_sums, per_layer)]
                n_summed += 1
            n_layers = len(per_layer)
            peak = max(range(n_layers), key=per_layer.__getitem__) if n_layers else 0
            per_case.append({
                "sample_id": case.id,
                "sink_mass_mean": round(sum(per_layer) / max(n_layers, 1), 4),
                "sink_mass_last_query": round(sum(last_query) / max(n_layers, 1), 4),
                "sink_peak_layer_frac": round(peak / max(1, n_layers - 1), 4),
            })

        if first is None:
            raise ValueError("attention_sink: empty case batch")
        n = len(per_case)
        per_layer_sink = [round(s / n_summed, 4) for s in layer_sums]
        mean_sink = round(sum(r["sink_mass_mean"] for r in per_case) / n, 4)
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"per_layer_sink": per_layer_sink},
            findings={
                "n_layers": first[0],
                "sink_token": first[1],
                "n_cases_analyzed": n,
                "mean_sink_mass": mean_sink,
                "per_layer_sink": per_layer_sink,
                "per_case": per_case,
            },
        )
