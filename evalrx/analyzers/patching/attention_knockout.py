"""Attention knockout — block text→media attention and measure what the answer loses.

Geva et al. trace information flow by preventing chosen query positions from
attending to chosen key positions in a window of layers, then measuring the
change in the prediction. Here the keys are the media placeholder tokens
(image / video frames / audio) and the queries are the text tokens: with every
such edge cut, the text can no longer read the media directly.

Per case, the next-token margin at the end of the prompt (see
:mod:`evalrx.analyzers._internals.targets`) is measured clean, with the edges
cut in EVERY layer, and with them cut only in the early / middle / late third
of the stack. A large full-stack drop says the answer was built from the media;
a drop near zero says the answer did not come from the media at all — the
language-prior signature. The band columns say WHERE the reading happened.

Mechanism: :mod:`evalrx.analyzers._internals.attn_tap` routes the decoder's
attention through an eager-equivalent function that masks the cut edges to
−inf before the softmax (an exact knockout, rows stay normalised). Before
reporting anything the analyzer verifies, once per model, that the edit-free
tap engages on every attention layer and reproduces the native logits.

References:
- Dissecting Recall of Factual Associations in Auto-Regressive Language Models
  Geva et al., EMNLP 2023 — arXiv:2304.14767
- Cross-modal Information Flow in Multimodal Large Language Models
  Zhang et al., CVPR 2025 — arXiv:2411.18620
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

_BANDS = ("early", "mid", "late")


def layer_bands(n_layers: int) -> "dict[str, list[int]]":
    """Split ``range(n_layers)`` into contiguous early / mid / late thirds."""
    a, b = round(n_layers / 3), round(2 * n_layers / 3)
    return {"early": list(range(0, a)), "mid": list(range(a, b)), "late": list(range(b, n_layers))}


@register_analyzer("attention_knockout")
class AttentionKnockoutAnalyzer(Analyzer):
    """Cut text→media attention (all layers, then per third) and measure the margin drop.

    Hyper-parameters:
        target:        ``"prediction"`` (clean top-1 vs top-2, label free) or
                       ``"gold"`` (gold first token vs strongest other token).
        queries:       ``"text"`` — every non-media position loses the media
                       (Geva's setting); ``"last"`` — only the final position does.
        fidelity_rtol: tolerance of the edit-free tap vs native logits (relative
                       to the largest logit) before the analyzer refuses.
        max_cases:     label-stratified cap; 0 (the default) = every case.
                       Cost: 5 logits-only forwards per case.
    """

    name = "attention_knockout"
    description = (
        "Blocks the text from attending to the image/audio tokens and measures how "
        "much the answer's margin drops; near-zero drop = the answer ignored the media."
    )
    requires = frozenset({Capability.ATTENTION, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "ko_margin_clean": ("Clean answer margin", "Logit margin of the target answer token with nothing blocked."),
        "ko_delta_full": ("Drop when media blocked", "How much the answer margin falls when the text cannot attend to the media tokens in any layer; near zero means the answer did not use the media."),
        "ko_rel_drop_full": ("Relative drop, media blocked", "ko_delta_full as a fraction of the clean margin (1 = the margin vanished entirely)."),
        "ko_flips_top1": ("Answer flips w/o media", "Blocking the media changes the model's top next token."),
        "ko_delta_early": ("Drop, early layers", "Margin drop when media attention is blocked only in the first third of the layers."),
        "ko_delta_mid": ("Drop, middle layers", "Margin drop when media attention is blocked only in the middle third of the layers."),
        "ko_delta_late": ("Drop, late layers", "Margin drop when media attention is blocked only in the last third of the layers."),
    }

    def __init__(self, target: str = "prediction", queries: str = "text",
                 fidelity_rtol: float = 1e-2, max_cases: int = 0) -> None:
        if queries not in ("text", "last"):
            raise ValueError(f"queries must be 'text' or 'last', got {queries!r}")
        super().__init__(target=target, queries=queries, fidelity_rtol=fidelity_rtol,
                         max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        from evalrx.analyzers._internals.access import (
            check_editable_attention,
            decoder_stack,
            media_positions,
            torch_module,
        )
        from evalrx.analyzers._internals.attn_tap import attention_tap, knockout, require_fidelity
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.core.model import CaptureSpec

        check_editable_attention(model, self.name)
        stack = decoder_stack(torch_module(model, self.name))
        if not stack.fully_attentional:
            raise ValueError(f"{self.name}: some decoder layers have no softmax attention "
                             "(hybrid stack); a knockout there would be partial.")
        bands = layer_bands(stack.n_layers)
        spec = CaptureSpec(to_cpu=False)

        def last_logits(inputs: Any) -> Any:
            logits = model.forward(inputs, capture={Capability.LOGITS}, spec=spec).require(Capability.LOGITS)
            return logits[-1:].float()

        per_case: list[dict[str, Any]] = []
        skipped: Counter = Counter()
        slots: Counter = Counter()
        fidelity: "dict | None" = None
        for case in cases.stratified_head(self.max_cases):
            clean = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            slot, media = media_positions(clean)
            if not media:
                skipped["no_media_tokens"] += 1
                continue
            logits = clean.require(Capability.LOGITS)[-1:].float()
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            if fidelity is None:
                fidelity = require_fidelity(model, case.inputs, rtol=self.fidelity_rtol,
                                            analyzer=self.name)
            media_set = set(media)
            queries = ([clean.seq_len - 1] if self.queries == "last"
                       else [i for i in range(clean.seq_len) if i not in media_set])

            def cut(layers: "list[int] | None") -> Any:
                edit = knockout(media, queries=queries, layers=layers)
                with attention_tap(model, [edit], analyzer=self.name):
                    return last_logits(case.inputs)

            m_clean = margin.value(logits)
            full = cut(None)
            delta_full = m_clean - margin.value(full)
            row: dict[str, Any] = {
                "sample_id": case.id,
                "ko_margin_clean": round(m_clean, 4),
                "ko_delta_full": round(delta_full, 4),
                "ko_flips_top1": margin.top1_changed(full, int(logits[-1].argmax())),
            }
            if m_clean > 1e-6:
                row["ko_rel_drop_full"] = round(delta_full / m_clean, 4)
            for band in _BANDS:
                row[f"ko_delta_{band}"] = round(m_clean - margin.value(cut(bands[band])), 4)
            per_case.append(row)
            slots[slot] += 1

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": stack.n_layers,
            "layer_bands": {k: [v[0], v[-1] + 1] if v else [] for k, v in bands.items()},
            "media_slot": slots.most_common(1)[0][0],
            "target": self.target,
            "queries": self.queries,
            "tap_fidelity": fidelity,
            "mean_delta_full": round(sum(r["ko_delta_full"] for r in per_case) / n, 4),
            "frac_flips_top1": round(sum(bool(r["ko_flips_top1"]) for r in per_case) / n, 4),
            "per_case": per_case,
            "_caveat": (
                "Measured on the next-token margin at the end of the prompt (the first "
                "answer token, teacher-forced). A band knockout leaves media information "
                "already copied into text positions by other layers intact, so band drops "
                "need not sum to the full drop."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
