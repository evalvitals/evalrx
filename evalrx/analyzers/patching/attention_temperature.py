"""Attention temperature — does sharpening or spreading the model's look change the answer?

Divides the pre-softmax attention scores of the text positions after the media
by a temperature ``T`` in a band of layers (``T < 1`` sharpens, ``T > 1``
flattens) and measures the next-token margin at ``T = sharp`` and ``T = flat``.

``scope="within"`` (default) tempers only the scores on the media keys AND
keeps each row's total weight on the media fixed: the model looks at the image
exactly as much as before, only more focused (``p_k ∝ p_k^(1/T)`` inside the
image) or more spread out. That separates FOCUS from AMOUNT — the amount is
what ``attention_sensitivity`` scales. The slope of the margin in ``log T``:

* negative — sharper focus inside the image strengthens the answer: the model
  reads the image too diffusely for this answer;
* positive — spreading the attention over the image strengthens it: the model
  is over-focused (e.g. on a few sink tokens) and misses the rest;
* near zero — the answer does not depend on how the look is distributed.

``scope="all"`` is plain attention temperature over every visible key (text
and media together), which also moves the media's share.

A diagnostic only. With ``target="gold"`` the slope is for the gold token: would
focusing (or spreading) move the model TOWARD the right answer. Only softmax
attention has a temperature: on a hybrid stack (Qwen3.5) the linear-attention
layers are left as they are (``tt_edited_layers`` counts the edited ones).

References:
- Attention temperature as a decoding-time intervention — no single
  originating paper; related visual-attention interventions: PAI (Liu et al.,
  ECCV 2024, arXiv:2407.21771), VAR (Kang et al., ICLR 2025, arXiv:2503.03321).
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
SCOPES = ("within", "all")


@register_analyzer("attention_temperature")
class AttentionTemperatureAnalyzer(Analyzer):
    """Margin response to sharpening / flattening the text→media attention, per case.

    Hyper-parameters:
        sharp / flat:  temperatures below / above 1; defaults 0.5 and 2.0.
        scope:         ``"within"`` (media keys only, media mass per row kept) or
                       ``"all"`` (every visible key).
        band:          layers edited — ``"all"`` or the ``"early"`` / ``"mid"`` /
                       ``"late"`` third of the stack (softmax layers only).
        target:        ``"prediction"`` or ``"gold"``.
        fidelity_rtol: tolerance of the edit-free attention tap vs native logits.
        max_cases:     label-stratified cap; 0 (the default) = every case.
                       Cost: 3 logits-only forwards per case.
    """

    name = "attention_temperature"
    description = (
        "Sharpens and flattens the text's attention over the image/audio tokens (same amount "
        "of looking, different focus) and measures which way the answer's margin moves."
    )
    requires = frozenset({Capability.ATTENTION, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "tt_margin_clean": ("Clean answer margin", "Logit margin of the target answer token with attention untouched."),
        "tt_delta_sharp": ("Margin change, sharper", "Change in the answer margin when the attention over the media is sharpened (T = sharp)."),
        "tt_delta_flat": ("Margin change, flatter", "Change in the answer margin when the attention over the media is flattened (T = flat)."),
        "tt_slope": ("Margin per log-temperature", "Slope of the answer margin in log T; negative = sharper focus strengthens the answer, positive = spreading the look strengthens it."),
        "tt_flips_sharp": ("Answer flips, sharper", "Sharpening changes the model's top next token."),
        "tt_flips_flat": ("Answer flips, flatter", "Flattening changes the model's top next token."),
        "tt_media_entropy": ("Focus inside the media", "Normalized entropy of the answer position's attention distribution over the media tokens in the edited layers (1 = spread evenly, 0 = one token)."),
    }

    def __init__(self, sharp: float = 0.5, flat: float = 2.0, scope: str = "within",
                 band: str = "all", target: str = "prediction", fidelity_rtol: float = 1e-2,
                 max_cases: int = 0) -> None:
        if band not in _BANDS:
            raise ValueError(f"band must be one of {_BANDS}, got {band!r}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        if not (0 < sharp < 1 < flat) or not math.isfinite(flat):
            raise ValueError(f"need 0 < sharp < 1 < flat, got sharp={sharp}, flat={flat}")
        super().__init__(sharp=sharp, flat=flat, scope=scope, band=band, target=target,
                         fidelity_rtol=fidelity_rtol, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.access import (
            check_editable_attention,
            decoder_stack,
            media_positions,
            torch_module,
        )
        from evalrx.analyzers._internals.attn_tap import (
            attention_tap,
            require_fidelity,
            temperature,
        )
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.analyzers.patching.attention_knockout import layer_bands
        from evalrx.core.model import CaptureSpec

        stack = decoder_stack(torch_module(model, self.name))
        check_editable_attention(model, self.name, stack)
        chosen = range(stack.n_layers) if self.band == "all" else layer_bands(stack.n_layers)[self.band]
        layers = [i for i in chosen if stack.attn[i] is not None]
        if not layers:
            raise ValueError(f"{self.name}: no softmax attention layer in band {self.band!r} "
                             f"(layer kinds: {stack.kinds}); there is no temperature to set")
        spec = CaptureSpec(to_cpu=False)
        log_span = math.log(self.flat) - math.log(self.sharp)

        per_case: list[dict[str, Any]] = []
        skipped: Counter = Counter()
        fidelity: "dict | None" = None
        for case in cases.stratified_head(self.max_cases):
            probe = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            _, media = media_positions(probe)
            if not media:
                skipped["no_media_tokens"] += 1
                continue
            seq_len = probe.seq_len
            del probe
            queries = list(range(max(media) + 1, seq_len))
            if not queries:
                skipped["no_text_after_media"] += 1
                continue
            if fidelity is None:
                fidelity = require_fidelity(model, case.inputs, rtol=self.fidelity_rtol, analyzer=self.name)
            keys = media if self.scope == "within" else list(range(seq_len))
            entropies: list[float] = []

            def focus(layer: int, probs: Any) -> None:
                if layer in layers:
                    row = probs[0, :, -1, media].float()           # (heads, n_media)
                    row = row / row.sum(-1, keepdim=True).clamp(min=1e-12)
                    ent = -(row * row.clamp(min=1e-12).log()).sum(-1) / math.log(max(2, len(media)))
                    entropies.append(float(ent.mean()))

            with torch.no_grad(), attention_tap(model, on_probs=focus, analyzer=self.name):
                clean = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            logits = clean.require(Capability.LOGITS)[-1:].float()
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            top1 = int(logits[-1].argmax())
            del clean

            def tempered(value: float) -> Any:
                edit = temperature(keys, value, queries=queries, layers=layers,
                                   within=self.scope == "within")
                with torch.no_grad(), attention_tap(model, [edit], analyzer=self.name):
                    out = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
                return out.require(Capability.LOGITS)[-1:].float()

            m_clean = margin.value(logits)
            sharp, flat = tempered(self.sharp), tempered(self.flat)
            m_sharp, m_flat = margin.value(sharp), margin.value(flat)
            per_case.append({
                "sample_id": case.id,
                "tt_margin_clean": round(m_clean, 4),
                "tt_delta_sharp": round(m_sharp - m_clean, 4),
                "tt_delta_flat": round(m_flat - m_clean, 4),
                "tt_slope": round((m_flat - m_sharp) / log_span, 4),
                "tt_flips_sharp": margin.top1_changed(sharp, top1),
                "tt_flips_flat": margin.top1_changed(flat, top1),
                "tt_media_entropy": round(sum(entropies) / len(entropies), 4) if entropies else None,
            })
            if per_case[-1]["tt_media_entropy"] is None:
                del per_case[-1]["tt_media_entropy"]

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": stack.n_layers,
            "band": self.band,
            "scope": self.scope,
            "temperatures": [self.sharp, self.flat],
            "target": self.target,
            "tt_edited_layers": len(layers),
            "layer_kinds": {"softmax": stack.n_softmax, "linear": stack.n_linear},
            "tap_fidelity": fidelity,
            "mean_slope": round(sum(r["tt_slope"] for r in per_case) / n, 4),
            "frac_negative_slope": round(sum(r["tt_slope"] < 0 for r in per_case) / n, 4),
            "frac_flips_sharp": round(sum(r["tt_flips_sharp"] for r in per_case) / n, 4),
            "frac_flips_flat": round(sum(r["tt_flips_flat"] for r in per_case) / n, 4),
            "per_case": per_case,
            "_caveat": (
                "Diagnostic only — the next-token margin at the end of the prompt with the "
                "attention of the text after the media tempered in the chosen band"
                + (" over the media keys, each row keeping its total media weight"
                   if self.scope == "within" else " over every visible key")
                + "; softmax layers only, no decoding was changed."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
