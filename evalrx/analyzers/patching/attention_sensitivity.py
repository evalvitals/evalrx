"""Attention sensitivity — would more (or less) attention to the media change the answer?

Scales the attention every text position pays to the media tokens by ``w``
(renormalised: exactly ``+log w`` on the pre-softmax scores, so each row still
sums to one) in a band of layers, and measures the next-token margin at
``w = down`` and ``w = up``. The slope of the margin in ``log w`` says which way
the media pulls the answer:

* positive — more media attention strengthens the model's answer: the answer is
  media-supported;
* negative — more media attention weakens it: the answer is held up by the text
  despite the media (the language-prior signature), and amplifying visual
  attention — the PAI family of decoding repairs — would push the model off it;
* near zero — the answer is insensitive to how much the media is attended to.

A diagnostic only: no decoding is changed. With ``target="gold"`` the slope is
for the gold token instead, i.e. whether amplifying the media would move the
model TOWARD the right answer.

On a hybrid stack (Qwen3.5, Qwen3-Next) the gated-delta-rule linear-attention
layers have no attention weights to rescale; there the media's contribution to
the text positions' layer output is scaled instead —
``absent + w · (real − absent)``, with ``absent`` the layer re-run without the
media tokens (:mod:`evalrx.analyzers._internals.linear_mix`).

References:
- Paying More Attention to Image: A Training-Free Method for Alleviating
  Hallucination in LVLMs (PAI) — Liu et al., ECCV 2024 — arXiv:2407.21771
- Dissecting Recall of Factual Associations in Auto-Regressive Language Models
  Geva et al., EMNLP 2023 — arXiv:2304.14767
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model

_BANDS = ("all", "early", "mid", "late")


@register_analyzer("attention_sensitivity")
class AttentionSensitivityAnalyzer(Analyzer):
    """Margin response to scaling text→media attention up and down, per case.

    Hyper-parameters:
        down / up:     attention multipliers (renormalised); defaults 0.5 and 2.0.
        band:          layers edited — ``"all"`` or the ``"early"`` / ``"mid"`` /
                       ``"late"`` third of the stack.
        target:        ``"prediction"`` or ``"gold"``.
        fidelity_rtol: tolerance of the edit-free attention tap vs native logits.
        max_cases:     label-stratified cap; 0 (the default) = every case.
                       Cost: 3 logits-only forwards per case, plus the per-case
                       edit-free self-check (1 tapped forward; +1 media-absent
                       re-run on a hybrid stack).
    """

    name = "attention_sensitivity"
    description = (
        "Scales how much the text attends to the image/audio tokens (×0.5, ×2) and "
        "measures which way the answer's margin moves; negative slope = more looking "
        "weakens the answer."
    )
    requires = frozenset({Capability.ATTENTION, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "rw_margin_clean": ("Clean answer margin", "Logit margin of the target answer token with attention untouched."),
        "rw_delta_down": ("Margin change, less looking", "Change in the answer margin when attention to the media is scaled down."),
        "rw_delta_up": ("Margin change, more looking", "Change in the answer margin when attention to the media is scaled up."),
        "rw_slope": ("Margin per log-attention", "Slope of the answer margin in log(media attention scale); negative = more attention to the media weakens the answer."),
        "rw_flips_up": ("Answer flips, more looking", "Scaling media attention up changes the model's top next token."),
    }

    def __init__(self, down: float = 0.5, up: float = 2.0, band: str = "all",
                 target: str = "prediction", fidelity_rtol: float = 1e-2, max_cases: int = 0) -> None:
        if band not in _BANDS:
            raise ValueError(f"band must be one of {_BANDS}, got {band!r}")
        if not (0 < down < 1 < up):
            raise ValueError(f"need 0 < down < 1 < up, got down={down}, up={up}")
        super().__init__(down=down, up=up, band=band, target=target,
                         fidelity_rtol=fidelity_rtol, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        from evalrx.analyzers._internals.access import (
            check_editable_attention,
            decoder_stack,
            media_positions,
            torch_module,
        )
        from evalrx.analyzers._internals.edits import FidelityGate, media_flow_edit
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.analyzers.patching.attention_knockout import layer_bands
        from evalrx.core.model import CaptureSpec

        stack = decoder_stack(torch_module(model, self.name))
        check_editable_attention(model, self.name, stack)
        layers = None if self.band == "all" else layer_bands(stack.n_layers)[self.band]
        spec = CaptureSpec(to_cpu=False)
        log_span = math.log(self.up) - math.log(self.down)

        per_case: list[dict[str, Any]] = []
        skipped: Counter = Counter()
        gate = FidelityGate(model, rtol=self.fidelity_rtol, analyzer=self.name)
        for case in cases.stratified_head(self.max_cases):
            clean = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            _, media = media_positions(clean)
            if not media:
                skipped["no_media_tokens"] += 1
                continue
            logits = clean.require(Capability.LOGITS)[-1:].float()
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            gate.check(case.id, case.inputs, media, base=logits)
            media_set = set(media)
            queries = [i for i in range(clean.seq_len) if i not in media_set]
            top1 = int(logits[-1].argmax())
            del clean

            def scaled(weight: float) -> Any:
                with media_flow_edit(model, media, queries=queries, layers=layers, weight=weight,
                                     analyzer=self.name):
                    out = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
                return out.require(Capability.LOGITS)[-1:].float()

            m_clean = margin.value(logits)
            low, high = scaled(self.down), scaled(self.up)
            m_low, m_high = margin.value(low), margin.value(high)
            per_case.append({
                "sample_id": case.id,
                "rw_margin_clean": round(m_clean, 4),
                "rw_delta_down": round(m_low - m_clean, 4),
                "rw_delta_up": round(m_high - m_clean, 4),
                "rw_slope": round((m_high - m_low) / log_span, 4),
                "rw_flips_up": margin.top1_changed(high, top1),
            })

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": stack.n_layers,
            "band": self.band,
            "weights": [self.down, self.up],
            "target": self.target,
            "layer_kinds": {"softmax": stack.n_softmax, "linear": stack.n_linear},
            "tap_fidelity": gate.summary(),
            "mean_slope": round(sum(r["rw_slope"] for r in per_case) / n, 4),
            "frac_negative_slope": round(sum(r["rw_slope"] < 0 for r in per_case) / n, 4),
            "per_case": per_case,
            "_caveat": (
                "Diagnostic only — measured on the next-token margin at the end of the "
                "prompt with attention to the media rescaled (rows renormalised) in the "
                "chosen band; no decoding was changed."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
