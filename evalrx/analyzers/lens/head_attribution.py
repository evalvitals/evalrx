"""Direct logit attribution per attention head — which heads write the answer.

The residual stream at the final prompt position is a sum of writes: the
embedding, every attention head, every MLP. Read through the unembedding (with
the final norm's scale frozen at the clean run — see
:mod:`evalrx.analyzers._internals.heads`), each write contributes a fixed amount
to the next-token logit margin, and the contributions add up to the margin.
This analyzer reports the attention heads' share: how much of the margin they
write directly, how concentrated it is in a few heads, and at what depth.

"Direct" means the head's own write only; a head that matters by feeding later
layers (its effect on what they compute) scores low here. Combine with
``attention_knockout`` / ``causal_trace`` for indirect effects.

On a hybrid stack (Qwen3.5, Qwen3-Next) the gated-delta-rule linear-attention
layers write through the same kind of per-head output projection, so their
value heads are attributed alongside the softmax heads (``layer_kinds`` and
``heads_per_layer`` say which is which; rows of the maps are zero-padded to the
widest layer).

The per-head mean attribution over the batch, and its FAIL−PASS difference,
are artifacts (``(layers, heads)`` maps), so a head that writes the wrong
answer on failures stands out as a hot cell.

References:
- A Mathematical Framework for Transformer Circuits
  Elhage et al., Anthropic 2021 — transformer-circuits.pub/2021/framework
- Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small
  Wang et al., ICLR 2023 — arXiv:2211.00593
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("head_attribution")
class HeadAttributionAnalyzer(Analyzer):
    """Per-head direct contribution to the next-token logit margin, per case.

    Hyper-parameters:
        target:    ``"prediction"`` (clean top-1 vs top-2) or ``"gold"``.
        top_k:     heads listed for the first case.
        max_cases: label-stratified cap; 0 (the default) = every case.
                   Cost: one logits forward per case.
    """

    name = "head_attribution"
    description = (
        "Splits the answer token's logit margin into each attention head's direct write "
        "(direct logit attribution); reports how much the heads write and how concentrated."
    )
    requires = frozenset({Capability.HIDDEN_STATES, Capability.LOGITS})
    applies_to_modalities = frozenset({"text", "image"})
    signal_docs = {
        "dla_margin": ("Answer margin", "Logit margin of the target answer token being decomposed."),
        "dla_heads_total": ("Margin written by heads", "Sum of all attention heads' direct contributions to the answer margin (logits)."),
        "dla_heads_share": ("Heads' share of margin", "dla_heads_total as a fraction of the margin (the rest is written by MLPs and the embedding)."),
        "dla_top_head_share": ("Top head's share", "Largest single head's share of all heads' absolute contribution; high = one head decides the answer."),
        "dla_top_head_layer_frac": ("Top head's depth", "Relative depth (0 = first layer, 1 = last) of the head with the largest absolute contribution."),
        "dla_late_share": ("Late heads' share", "Share of all heads' absolute contribution coming from the last third of the layers."),
        "dla_negative_share": ("Heads opposing answer", "Share of all heads' absolute contribution that pushes AGAINST the answer token."),
        "dla_linear_share": ("Linear-attention share", "Hybrid stacks only: share of all heads' absolute contribution written by the linear-attention (gated delta rule) layers rather than softmax attention."),
    }

    def __init__(self, target: str = "prediction", top_k: int = 10, max_cases: int = 0) -> None:
        super().__init__(target=target, top_k=top_k, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import numpy as np
        import torch

        from evalrx.analyzers._internals.access import decoder_stack, torch_module
        from evalrx.analyzers._internals.heads import (
            frozen_norm_scale,
            head_capture,
            n_heads_of,
            out_proj,
        )
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.models.backends.hf.discover import get_final_norm, get_unembed

        module = torch_module(model, self.name)
        stack = decoder_stack(module)
        norm = get_final_norm(module)
        unembed = get_unembed(module)
        projs = [out_proj(m) if m is not None else None for m in stack.mixers]
        if any(p is None for p in projs):
            raise ValueError(f"{self.name}: some decoder layers have no recognised token mixer "
                             "with an output projection; head attribution would miss them.")
        n_heads = [n_heads_of(m, p) for m, p in zip(stack.mixers, projs)]
        n_layers = stack.n_layers
        linear_layers = [i for i, kind in enumerate(stack.kinds) if kind == "linear"]

        per_case: list[dict[str, Any]] = []
        maps: dict[str, Any] = {}
        skipped: Counter = Counter()
        max_err = 0.0
        first_top: list = []
        for case in cases.stratified_head(self.max_cases):
            with torch.no_grad(), head_capture(model, analyzer=self.name) as cap:
                trace = model.forward(case.inputs, capture={Capability.LOGITS})
            logits = trace.require(Capability.LOGITS)[-1:]
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            m_value = margin.value(logits)
            with torch.no_grad():
                x_final = cap.final_residual
                scale, centre = frozen_norm_scale(norm, x_final)
                w_u = unembed.weight
                direction = (w_u[margin.positive].float() - w_u[margin.negative].float()).to(scale.device)
                v = scale * direction
                if centre:
                    v = v - v.mean()
                readout = float(x_final.float().to(v.device) @ v)
                bias = getattr(unembed, "bias", None)
                if bias is not None:
                    readout += float(bias[margin.positive] - bias[margin.negative])
                max_err = max(max_err, abs(readout - m_value))
                dla = torch.zeros(n_layers, max(n_heads))
                for layer, (proj, heads) in enumerate(zip(projs, n_heads)):
                    w_o = proj.weight.float()
                    per_input = (w_o.T @ v.to(w_o.device)) * cap.head_inputs[layer].float().to(w_o.device)
                    dla[layer, :heads] = per_input.view(heads, -1).sum(-1).cpu()
            del trace, cap

            flat = dla.flatten()
            absolute = flat.abs()
            total_abs = float(absolute.sum()) or 1.0
            top = int(absolute.argmax())
            late = n_layers - n_layers // 3
            row: dict[str, Any] = {
                "sample_id": case.id,
                "dla_margin": round(m_value, 4),
                "dla_heads_total": round(float(flat.sum()), 4),
                "dla_top_head_share": round(float(absolute[top]) / total_abs, 4),
                "dla_top_head_layer_frac": round((top // dla.shape[1]) / max(1, n_layers - 1), 4),
                "dla_late_share": round(float(dla[late:].abs().sum()) / total_abs, 4),
                "dla_negative_share": round(float(flat.clamp(max=0).abs().sum()) / total_abs, 4),
            }
            if abs(m_value) > 1e-6:
                row["dla_heads_share"] = round(float(flat.sum()) / m_value, 4)
            if linear_layers:
                row["dla_linear_share"] = round(float(dla[linear_layers].abs().sum()) / total_abs, 4)
            per_case.append(row)
            maps[case.id] = dla.numpy()
            if not first_top:
                order = torch.argsort(absolute, descending=True)[: self.top_k]
                first_top = [{"layer": int(i) // dla.shape[1], "head": int(i) % dla.shape[1],
                              "dla": round(float(flat[i]), 4)} for i in order]

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        label_of = {c.id: getattr(getattr(c, "label", None), "value", None) for c in cases}
        artifacts: dict[str, Any] = {"head_dla_mean_map": np.mean(list(maps.values()), axis=0)}
        fail = [m for cid, m in maps.items() if label_of.get(cid) == "fail"]
        passed = [m for cid, m in maps.items() if label_of.get(cid) == "pass"]
        if fail and passed:
            artifacts["head_dla_diff_map_fail_minus_pass"] = np.mean(fail, axis=0) - np.mean(passed, axis=0)
        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": n_layers,
            "n_heads": max(n_heads),
            "heads_per_layer": n_heads,
            "layer_kinds": stack.kinds,
            "target": self.target,
            "top_heads_first_case": first_top,
            "mean_heads_total": round(sum(r["dla_heads_total"] for r in per_case) / n, 4),
            "linearization_max_error": round(max_err, 4),
            "per_case": per_case,
            "_caveat": (
                "Direct effects only, read at the final prompt position with the final "
                "norm's scale frozen at the clean run. linearization_max_error is the gap "
                "between the frozen-norm readout of the full residual and the true margin: "
                "bf16 rounding of the model's own logits (~1% of their magnitude), or more "
                "when the model soft-caps its logits."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases,
                      artifacts=artifacts, findings=findings)
