"""Causal tracing (Meng et al., ROME) over the media tokens of a VLM / ALM.

The causal method of the family: corrupt the input, then WRITE the clean run's
hidden state back at one (layer, position) site of the corrupted run and see
how much of the answer comes back. Meng et al. corrupt the subject tokens'
embeddings with Gaussian noise; here the corrupted tokens are the media
(image / video / audio) placeholder tokens, and two sites are restored at each
probed layer ``l`` (the residual stream ENTERING layer ``l``):

* ``media`` — every media token. Restoring them early undoes the corruption;
  the layer where this stops working is where the media's content has already
  been copied out into the text positions (the hand-off).
* ``last``  — the final prompt position, where the answer is read out.
  Restoring it late recovers the answer; the layer where it starts to is where
  the media's contribution has arrived there.

Recovery at a site is ``(m_restored - m_corrupt) / (m_clean - m_corrupt)`` on
the next-token logit margin (:mod:`evalrx.analyzers._internals.targets`). When
corrupting the media barely moves the margin (``|m_clean - m_corrupt| <
min_effect``) there is nothing to recover; those cases still report
``ct_media_effect`` — a near-zero effect is itself the "answer did not depend on
the media" finding — but no recovery columns.

Corruption: ``"noise"`` (Meng's; Gaussian noise of ``noise_scale`` × the
media embeddings' own std, added to the embedding output — and, on DeepStack
models (Qwen3-VL), to each re-injected copy of the visual features; works for
any media)
or ``"gray"`` (each image replaced by a flat image of its mean colour, same size,
so the token count and M-RoPE grid stay identical — verified per case).

References:
- Locating and Editing Factual Associations in GPT (ROME, causal tracing)
  Meng et al., NeurIPS 2022 — arXiv:2202.05262
- Understanding Information Storage and Transfer in Multi-modal Large Language Models
  Basu et al., NeurIPS 2024 — arXiv:2406.04236
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Iterator

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, Inputs
    from evalrx.core.model import Model

CORRUPTIONS = ("noise", "gray")


def _gray(image: Any) -> Any:
    """A flat image of *image*'s mean colour, same size (``None`` if undecodable)."""
    from PIL import Image, ImageStat

    from evalrx.analyzers.attention.relative_attn import _resolve_image

    img = _resolve_image(image)
    if not hasattr(img, "convert"):
        return None
    rgb = img.convert("RGB")
    mean = tuple(int(round(v)) for v in ImageStat.Stat(rgb).mean)
    return Image.new("RGB", rgb.size, mean)


def gray_inputs(inputs: "Inputs") -> "Inputs | None":
    """*inputs* with every image / video frame replaced by its flat mean-colour version."""
    from dataclasses import replace

    def swap(media: Any) -> Any:
        if media is None:
            return None
        items = list(media) if isinstance(media, (list, tuple)) else [media]
        grayed = [_gray(m) for m in items]
        if any(g is None for g in grayed):
            raise ValueError("undecodable image")
        return grayed if isinstance(media, (list, tuple)) else grayed[0]

    try:
        image, video = swap(getattr(inputs, "image", None)), swap(getattr(inputs, "video", None))
    except ValueError:
        return None
    if image is None and video is None:
        return None
    return replace(inputs, image=image, video=video)


def _merge(*plans: "dict[int, tuple[list[int], Any]]") -> "dict[int, tuple[list[int], Any]]":
    """Combine replace plans; a later plan wins at a shared (layer, position)."""
    import torch

    rows: dict[int, dict[int, Any]] = {}
    for plan in plans:
        for layer, (positions, values) in plan.items():
            slot = rows.setdefault(layer, {})
            for pos, value in zip(positions, values):
                slot[int(pos)] = value
    return {layer: (list(slot), torch.stack(list(slot.values())))
            for layer, slot in rows.items()}


def deepstack_mergers(module: Any) -> list:
    """The vision tower's DeepStack mergers (Qwen3-VL family), in injection order.

    DeepStack adds a second copy of the visual features to the decoder's hidden
    state after each of the first few layers — outside the embedding, outside
    any decoder layer. Corrupting only the embedding would leave those copies
    clean, so noise corruption also perturbs these mergers' outputs.
    """
    mergers: list = []
    for name, sub in module.named_modules():
        if name.endswith("deepstack_merger_list"):
            mergers.extend(list(sub))
    return mergers


def corrupted_variants(
    inputs: "Inputs",
    corruption: str,
    *,
    slot: str,
    media: "list[int]",
    embeddings: Any,
    deepstack: bool,
    noise_scale: float,
    n_noise: int,
    seed: int,
) -> "tuple[list[tuple[Any, dict, Any]], str]":
    """The corrupted runs of one case: ``(variants, "")`` or ``([], skip_reason)``.

    A variant is ``(inputs, layer-0 replace plan, (DeepStack noise store,
    generator))`` — one per noise sample (Gaussian noise of ``noise_scale`` ×
    the media embeddings' std written over *embeddings* at the *media*
    positions; on DeepStack models the re-injected copies are noised from the
    same generator on the variant's first forward, see :func:`_noised`), or the
    single gray-image input. A gray variant still has to be checked for token
    alignment by the caller.
    """
    import torch

    if corruption == "gray":
        grayed = gray_inputs(inputs) if slot == "image" else None
        if grayed is None:
            return [], "gray_needs_an_image"
        return [(grayed, {}, (None, None))], ""
    std = float(embeddings.float().std()) * noise_scale
    gen = torch.Generator(device="cpu").manual_seed(seed)
    variants = []
    for _ in range(max(1, n_noise)):
        noise = torch.randn(embeddings.shape, generator=gen).to(embeddings.device, embeddings.dtype) * std
        store = {} if deepstack and slot == "image" else None
        variants.append((inputs, {0: (media, embeddings + noise)}, (store, gen)))
    return variants, ""


@contextmanager
def _noised(mergers: list, store: dict, scale: float, gen: Any) -> Iterator[None]:
    """Add fixed noise to each merger output; drawn once per variant, then reused."""
    import torch

    def make(i: int):
        def hook(mod, args, output):
            if i not in store:
                std = float(output.detach().float().std()) * scale
                store[i] = (torch.randn(output.shape, generator=gen) * std).to(output.device, output.dtype)
            return output + store[i]
        return hook

    handles = [m.register_forward_hook(make(i)) for i, m in enumerate(mergers)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@register_analyzer("causal_trace")
class CausalTraceAnalyzer(Analyzer):
    """Corrupt the media tokens, restore clean states per layer, measure recovery.

    Hyper-parameters:
        corruption:  ``"noise"`` (default, any media) or ``"gray"`` (images/video).
        noise_scale: noise std as a multiple of the media embeddings' std (Meng: 3).
        n_noise:     noise samples averaged per case (``"noise"`` only).
        n_points:    probed layers, spread evenly over the stack.
        min_effect:  smallest ``|m_clean - m_corrupt|`` (logits) worth tracing. Recovery
                     divides by this effect; on a bf16 model logits are quantised to
                     ~0.06–0.125 at typical magnitudes, so a smaller effect yields
                     recoveries dominated by rounding.
        target:      ``"prediction"`` or ``"gold"``.
        seed:        noise seed (offset per case, so reruns reproduce).
        max_cases:   label-stratified cap; 0 (the default) = every case.
                     Cost per case: 1 + n_noise · (1 + 2 · n_points) logits-only forwards.
    """

    name = "causal_trace"
    description = (
        "Corrupts the image/audio tokens and restores clean hidden states layer by layer "
        "to find where the media's influence on the answer hands off to the text."
    )
    requires = frozenset({Capability.HIDDEN_STATES, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "ct_margin_clean": ("Clean answer margin", "Logit margin of the target answer token with the media intact."),
        "ct_media_effect": ("Margin lost to corrupt media", "How much the answer margin falls when the media tokens are corrupted; near zero = the answer did not depend on the media."),
        "ct_media_effect_rel": ("Relative loss, corrupt media", "ct_media_effect as a fraction of the clean margin."),
        "ct_effect_traceable": ("Media effect traceable", "Corrupting the media moved the margin by at least min_effect, so recovery could be traced."),
        "ct_media_recovery_mid": ("Recovery via media, mid", "Share of the lost margin recovered by restoring the clean media tokens in the middle third of the layers."),
        "ct_media_handoff_frac": ("Media hand-off depth", "Relative depth where restoring the media tokens stops recovering half the margin: the media's content has moved into the text by then."),
        "ct_last_rise_frac": ("Answer read-out depth", "Relative depth where restoring the final prompt position first recovers half the margin."),
        "ct_last_recovery_max": ("Best recovery at answer pos.", "Largest share of the lost margin recovered by restoring the final prompt position at any probed layer."),
    }

    def __init__(self, corruption: str = "noise", noise_scale: float = 3.0, n_noise: int = 1,
                 n_points: int = 8, min_effect: float = 1.0, target: str = "prediction",
                 seed: int = 0, max_cases: int = 0) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {corruption!r}")
        super().__init__(corruption=corruption, noise_scale=noise_scale, n_noise=n_noise,
                         n_points=n_points, min_effect=min_effect, target=target, seed=seed,
                         max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        from evalrx.analyzers._internals.access import (
            decoder_stack,
            layer_points,
            media_positions,
            torch_module,
        )
        from evalrx.analyzers._internals.residual import residual_hooks
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.core.model import CaptureSpec

        module = torch_module(model, self.name)
        stack = decoder_stack(module)
        mergers = deepstack_mergers(module)
        n_layers = stack.n_layers
        points = layer_points(n_layers, self.n_points)
        mid = [p for p in points if n_layers / 3 <= p < 2 * n_layers / 3] or points
        spec = CaptureSpec(to_cpu=False)

        def margin_with(variant: "tuple[Any, dict, Any]", margin: Any, restore: "dict | None" = None) -> float:
            inputs, plan, (store, gen) = variant
            replace = _merge(plan, restore) if restore else plan
            noised = _noised(mergers, store, self.noise_scale, gen) if store is not None else nullcontext()
            with noised, residual_hooks(model, replace=replace, analyzer=self.name):
                logits = model.forward(inputs, capture={Capability.LOGITS}, spec=spec).require(
                    Capability.LOGITS)
            return margin.value(logits[-1:])

        per_case: list[dict[str, Any]] = []
        curves: dict[str, dict[str, list[float]]] = {}
        skipped: Counter = Counter()
        slots: Counter = Counter()
        sites = sorted(set(points) | {0})
        for index, case in enumerate(cases.stratified_head(self.max_cases)):
            with residual_hooks(model, capture={layer: None for layer in sites},
                                analyzer=self.name) as clean:
                probe = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            slot, media = media_positions(probe)
            if not media:
                skipped["no_media_tokens"] += 1
                continue
            logits = probe.require(Capability.LOGITS)[-1:]
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            m_clean = margin.value(logits)
            last = probe.seq_len - 1
            clean_media = {layer: (media, clean.captured[layer][media]) for layer in sites}
            clean_last = {layer: ([last], clean.captured[layer][[last]]) for layer in sites}
            probe.logits = None  # keep token ids for the alignment check, not the (seq, vocab) logits
            del clean, logits

            variants, why = corrupted_variants(
                case.inputs, self.corruption, slot=slot, media=media, embeddings=clean_media[0][1],
                deepstack=bool(mergers), noise_scale=self.noise_scale, n_noise=self.n_noise,
                seed=self.seed * 100003 + index)
            if not variants:
                skipped[why] += 1
                continue
            if self.corruption == "gray":
                check = model.forward(variants[0][0], capture={Capability.LOGITS}, spec=spec)
                if check.token_ids != probe.token_ids:
                    skipped["corruption_misaligned"] += 1
                    continue

            m_corrupt = sum(margin_with(v, margin) for v in variants) / len(variants)
            effect = m_clean - m_corrupt
            row: dict[str, Any] = {
                "sample_id": case.id,
                "ct_margin_clean": round(m_clean, 4),
                "ct_media_effect": round(effect, 4),
                "ct_effect_traceable": abs(effect) >= self.min_effect,
            }
            if m_clean > 1e-6:
                row["ct_media_effect_rel"] = round(effect / m_clean, 4)
            slots[slot] += 1
            if abs(effect) < self.min_effect:
                per_case.append(row)
                continue

            def recovery(restore: dict) -> float:
                restored = sum(margin_with(v, margin, restore) for v in variants) / len(variants)
                return (restored - m_corrupt) / effect

            media_curve = [recovery({p: clean_media[p]}) for p in points]
            last_curve = [recovery({p: clean_last[p]}) for p in points]
            depth = [p / max(1, n_layers - 1) for p in points]
            row["ct_media_recovery_mid"] = round(
                sum(r for p, r in zip(points, media_curve) if p in mid) / len(mid), 4)
            # On DeepStack models a media restore before the last re-injection is
            # partial by construction (later injections stay corrupted), so the
            # hand-off is only looked for from that depth on.
            row["ct_media_handoff_frac"] = round(
                next((d for p, d, r in zip(points, depth, media_curve)
                      if p >= len(mergers) and r < 0.5), 1.0), 4)
            row["ct_last_rise_frac"] = round(
                next((d for d, r in zip(depth, last_curve) if r >= 0.5), 1.0), 4)
            row["ct_last_recovery_max"] = round(max(last_curve), 4)
            per_case.append(row)
            curves[case.id] = {"media": [round(r, 4) for r in media_curve],
                               "last": [round(r, 4) for r in last_curve]}

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        label_of = {c.id: getattr(getattr(c, "label", None), "value", None) for c in cases}

        def mean_curve(label: str, site: str) -> "list[float] | None":
            rows = [c[site] for cid, c in curves.items() if label_of.get(cid) == label]
            if not rows:
                return None
            return [round(sum(col) / len(rows), 4) for col in zip(*rows)]

        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_traced": len(curves),
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": n_layers,
            "layers_probed": points,
            "media_slot": slots.most_common(1)[0][0] if slots else None,
            "corruption": self.corruption,
            "deepstack_layers": len(mergers),
            "target": self.target,
            "mean_media_effect": round(sum(r["ct_media_effect"] for r in per_case) / n, 4),
            "frac_traceable": round(len(curves) / n, 4),
            "mean_recovery_curves": {
                f"{label}_{site}": mean_curve(label, site)
                for label in ("fail", "pass") for site in ("media", "last")
                if mean_curve(label, site) is not None
            },
            "per_case": per_case,
            "_caveat": (
                "Recovery is measured on the next-token margin at the end of the prompt. "
                "Sites are the residual stream ENTERING each probed layer; restoring the "
                "media tokens at layer 0 undoes the corruption by construction"
                + (" except on DeepStack models, where visual features are re-injected "
                   f"after the first {len(mergers)} layers: restoring the media before that "
                   "depth is partial because later injections stay corrupted."
                   if mergers else ".")
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases,
                      artifacts={"recovery_curves": curves, "layers_probed": points},
                      findings=findings)
