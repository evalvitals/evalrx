"""Generic attention-model explainability (Chefer et al., 2021) for a decoder-only VLM.

Relevance propagation through the self-attention stack using attention weights
AND their gradients. For self-attention, Chefer et al. initialise ``R = I`` and,
layer by layer,

    Ā_l = E_heads[ (∇A_l ⊙ A_l)⁺ ]          R ← R + Ā_l · R

so ``R[i, j]`` is how much token ``j`` contributes to the representation token
``i`` predicts from. The row of the final prompt position is the relevance of
every input token to the next-token prediction; here the gradients are of the
next-token logit margin (see :mod:`evalrx.analyzers._internals.targets`).

Only that one row is needed, so it is propagated as a vector from the top
layer down — ``v ← v + v · Ā_l`` for ``l = L-1 … 0`` — O(seq²) per layer
instead of materialising ``R``. The identity's own contribution (the final
position relevant to itself) is dropped before shares are computed.

Attention weights and their gradients come from the evalrx attention tap
(:mod:`evalrx.analyzers._internals.attn_tap`), checked once per model to
reproduce the native logits; the per-layer ``Ā_l`` is formed in a tensor hook
during the backward pass, so no full set of attention gradients is ever held.

On a hybrid stack (Qwen3.5, Qwen3-Next) the gated-delta-rule linear-attention
layers have no softmax matrix, but they are linear in their values: ``o = A·v``
for a data-dependent causal ``A``, which
:mod:`evalrx.analyzers._internals.linear_mix` materialises with the module's own
kernel, and ``∇A = (∂y/∂o) vᵀ``. Those layers then enter the same rule. Their
``A`` is not row-normalised like a softmax, and the short convolution before the
delta rule (three tokens back) is not propagated.

References:
- Generic Attention-model Explainability for Interpreting Bi-Modal and
  Encoder-Decoder Transformers — Chefer, Gur & Wolf, ICCV 2021 — arXiv:2103.15679
- Transformer Interpretability Beyond Attention Visualization
  Chefer, Gur & Wolf, CVPR 2021 — arXiv:2012.09838
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("generic_attention")
class GenericAttentionExplainability(Analyzer):
    """Chefer relevance of each input token to the next-token prediction, per case.

    Hyper-parameters:
        target:        ``"prediction"`` (clean top-1 vs top-2) or ``"gold"``.
        top_k:         relevant tokens listed for the first case.
        fidelity_rtol: tolerance of the edit-free attention tap vs native logits.
        max_cases:     label-stratified cap; 0 (the default) = every case.
                       Cost: one forward + one backward per case, plus a
                       native and a tapped forward for the per-case edit-free
                       self-check when the stack has softmax layers.
    """

    name = "generic_attention"
    description = (
        "Chefer gradient-weighted attention relevance of every input token to the "
        "answer token; reports how much of it lands on the image/audio tokens and "
        "where in the image."
    )
    requires = frozenset({Capability.ATTENTION, Capability.GRADIENTS, Capability.LOGITS})
    applies_to_modalities = frozenset({"text", "image"})
    signal_docs = {
        "rel_margin": ("Answer margin", "Logit margin of the target answer token being explained."),
        "rel_media_share": ("Relevance on image/audio", "Share of the answer token's attention relevance that traces back to image/audio tokens; low = the answer was built from the text."),
        "rel_input_entropy": ("Relevance spread", "Normalized entropy of relevance over all input tokens (1 = spread evenly, 0 = one token)."),
        "rel_input_top1_share": ("Top token's relevance", "Share of all relevance on the single most relevant input token."),
        "rel_entropy": ("Image relevance spread", "Normalized entropy of relevance over image tokens (1 = diffuse, 0 = one patch)."),
        "rel_top1_share": ("Image relevance peak", "Share of image relevance on its single strongest image token."),
        "rel_top5_share": ("Image relevance top-5", "Share of image relevance on its five strongest image tokens."),
        "rel_center_offset": ("Image relevance off-centre", "Distance of the image relevance's centre of mass from the image centre."),
        "rel_edge_mass": ("Image relevance on border", "Share of image relevance on the outermost ring of patches."),
    }

    def __init__(self, target: str = "prediction", top_k: int = 10,
                 fidelity_rtol: float = 1e-2, max_cases: int = 0) -> None:
        super().__init__(target=target, top_k=top_k, fidelity_rtol=fidelity_rtol,
                         max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch"):
        import numpy as np
        import torch

        from evalrx.analyzers._internals.access import (
            check_editable_attention,
            decoder_stack,
            media_positions,
            torch_module,
        )
        from evalrx.analyzers._internals.attn_tap import attention_tap
        from evalrx.analyzers._internals.edits import FidelityGate
        from evalrx.analyzers._internals.linear_mix import delta_rule_relevance
        from evalrx.analyzers._internals.maps import (
            SpatialMapResult,
            group_maps,
            image_token_scores,
            map_features,
        )
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.core.model import CaptureSpec

        stack = decoder_stack(torch_module(model, self.name))
        check_editable_attention(model, self.name, stack)
        n_layers = stack.n_layers
        spec = CaptureSpec(with_grad=True, to_cpu=False)

        per_case: list[dict[str, Any]] = []
        per_case_maps: dict[str, Any] = {}
        skipped: Counter = Counter()
        # the tap only records here; the linear layers' relevance has no media re-run to check
        gate = FidelityGate(model, rtol=self.fidelity_rtol, analyzer=self.name, linear=False)
        first: "dict[str, Any] | None" = None
        def measure(case: Any) -> None:
            nonlocal first
            gate.check(case.id, case.inputs)
            abar: dict[int, Any] = {}
            recorded: dict[int, Any] = {}

            def on_abar(layer: int, value: Any) -> None:
                abar[layer] = value

            def on_probs(layer: int, probs: Any) -> None:
                if not probs.requires_grad:
                    return
                held = probs.detach()

                def hook(grad: Any, layer: int = layer, held: Any = held) -> None:
                    # Ā_l = E_heads[(∇A ⊙ A)⁺] for the single sequence in the batch
                    abar[layer] = (grad.float() * held.float()).clamp(min=0).mean(dim=1)[0]

                probs.register_hook(hook)
                recorded[layer] = probs

            with attention_tap(model, on_probs=on_probs, analyzer=self.name), \
                    delta_rule_relevance(model, on_abar, analyzer=self.name) as linear_out:
                trace = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            recorded.update(linear_out)
            logits = trace.require(Capability.LOGITS)[-1:]
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                return
            if not recorded:
                raise RuntimeError(
                    f"{self.name}: the forward pass kept no autograd graph through attention — "
                    "the backend ignores CaptureSpec(with_grad=True)."
                )
            m_value = margin.value(logits)
            # Asking for the first layer's mixing gradient sends the backward
            # pass through every layer's mixer, firing each layer's hook once.
            torch.autograd.grad(margin(logits), recorded[min(recorded)])
            recorded.clear()  # drop the graph-holding tensors; the closure keeps the dict
            del linear_out, logits
            if len(abar) != n_layers:
                raise RuntimeError(f"{self.name}: relevance reached {len(abar)} of {n_layers} layers")

            seq = trace.seq_len
            v = torch.zeros(seq, device=abar[0].device)
            v[-1] = 1.0
            for layer in reversed(range(n_layers)):
                v = v + v @ abar[layer]
            rel = v[:-1].clamp(min=0).cpu().double().numpy()  # drop the final position's self term
            abar.clear()
            total = float(rel.sum())

            row: dict[str, Any] = {"sample_id": case.id, "rel_margin": round(m_value, 4)}
            if total > 0:
                p = rel / total
                nz = p[p > 0]
                row["rel_input_entropy"] = round(float(-(nz * np.log(nz)).sum() / np.log(max(p.size, 2))), 4)
                row["rel_input_top1_share"] = round(float(p.max()), 4)
            slot, media = media_positions(trace)
            media = [i for i in media if i < seq - 1]
            if media and total > 0:
                row["rel_media_share"] = round(float(rel[media].sum() / total), 4)
                if slot == "image":
                    flat, spatial = image_token_scores(trace, rel[media])
                    row.update(map_features(flat, spatial, "rel"))
                    per_case_maps[case.id] = np.asarray(spatial, dtype=np.float16)
            if first is None:
                order = np.argsort(rel)[::-1][: self.top_k]
                first = {
                    "case_id": case.id,
                    "top": [{"position": int(i), "token": trace.tokens[int(i)],
                             "relevance": round(float(rel[i] / total), 4) if total > 0 else 0.0}
                            for i in order],
                }
            per_case.append(row)
            del trace

        for case in cases.stratified_head(self.max_cases):
            try:
                measure(case)
            except torch.cuda.OutOfMemoryError:
                # one case's graph did not fit (a long media sequence); the rest still get their reading
                skipped["out_of_memory"] += 1
                oom = True
            else:
                oom = False
            if oom:   # the case's tensors went with the except clause's traceback; now the cache can go
                torch.cuda.empty_cache()

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        n = len(per_case)
        shares = [r["rel_media_share"] for r in per_case if "rel_media_share" in r]
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": n_layers,
            "target": self.target,
            "tap_fidelity": gate.summary(),
            "top_relevant_tokens": first["top"] if first else [],
            "mean_media_share": round(sum(shares) / len(shares), 4) if shares else None,
            "per_case": per_case,
            "_caveat": (
                "Chefer et al.'s self-attention rule applied to a decoder-only stack, "
                "explaining the next-token margin at the end of the prompt. Relevance "
                "follows attention paths only; information carried by MLPs within a "
                "position is attributed to that position."
            ),
        }
        artifacts: dict[str, Any] = {}
        if per_case_maps:
            first_map_id = next(iter(per_case_maps))
            artifacts["spatial_map"] = per_case_maps[first_map_id].astype(np.float64)
            artifacts["per_case_maps"] = per_case_maps
            findings["spatial_map_case_id"] = first_map_id
            group_artifacts, group_findings = group_maps(per_case_maps, cases)
            artifacts.update(group_artifacts)
            findings.update(group_findings)
        return SpatialMapResult(analyzer=self.name, model=repr(model), cases=cases,
                                artifacts=artifacts, findings=findings)
