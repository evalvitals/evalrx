"""Path patching — which heads carry the media's effect DIRECTLY to the answer?

Path patching (Wang et al., IOI, Appendix B) measures one head's effect along a
chosen path only. Here the path is a head → the answer logits: in the clean
run, the sender head's output is replaced by its value in the run whose media
was corrupted (``causal_trace``'s corruptions and noise draws), while EVERY
other token-mixing head — attention heads, and the value heads of a hybrid
stack's linear-attention layers — is frozen at its clean output. MLPs and norms
recompute. What the answer margin then loses travels from the sender to the
logits directly or through MLPs, never through another head:

    direct share(h) = (m_clean − m_patched) / (m_clean − m_corrupt)

— the share of the media's effect on the answer that flows through ``h``'s
direct path. Without the freeze the same patch measures ``h``'s TOTAL effect
(through later heads too).

Patching every head exactly costs a forward pass per head, so the heads are
screened first: each head's media-dependent output at the answer position is
read through the frozen final norm into the margin (direct logit attribution of
``o_corrupt − o_clean``, as in ``head_attribution``), and the ``top_k`` heads by
that estimate are patched for real. The screen ignores MLP mediation and the
norm's change; ``pp_screen_spearman`` says how well it ranked the heads that
were patched.

Reported per case: the strongest direct-path head (exact), the joint direct
and total shares of the top-k set patched together, the screen's agreement,
and on hybrids how much of the direct effect runs through linear-attention
heads. Across cases: which heads recur as the media's direct carriers.

References:
- Interpretability in the Wild: a Circuit for Indirect Object Identification in
  GPT-2 small — Wang et al., ICLR 2023 — arXiv:2211.00593
- Locating and Editing Factual Associations in GPT (corruption) — Meng et al.,
  NeurIPS 2022 — arXiv:2202.05262
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


def head_name(layer: int, head: int) -> str:
    return f"L{layer}.H{head}"


def sender_inputs(base: "dict[int, Any]", donor: "dict[int, Any]", senders: "list[tuple[int, int]]",
                  n_heads: "list[int]", *, freeze: bool) -> "dict[int, Any]":
    """Output-projection inputs for a patched run: the *senders*' head slices from
    *donor*, everything else from *base* — at every layer when *freeze* (path
    patching: no other head may respond), else only at the senders' layers
    (the senders' total effect)."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in senders:
        by_layer[layer].append(head)
    out = {}
    for layer in (range(len(n_heads)) if freeze else sorted(by_layer)):
        x = base[layer]
        if by_layer.get(layer):
            x = x.clone()
            width = x.shape[-1] // n_heads[layer]
            for head in by_layer[layer]:
                x[:, head * width:(head + 1) * width] = donor[layer][:, head * width:(head + 1) * width]
        out[layer] = x
    return out


def _spearman(a: "list[float]", b: "list[float]") -> "float | None":
    """Rank correlation (no tie correction — the inputs are continuous)."""
    n = len(a)
    if n < 3:
        return None

    def ranks(xs: "list[float]") -> "list[float]":
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    ra, rb = ranks(a), ranks(b)
    mean = (n - 1) / 2
    cov = sum((x - mean) * (y - mean) for x, y in zip(ra, rb))
    var = sum((x - mean) ** 2 for x in ra)
    return cov / var if var > 0 else None


@register_analyzer("path_patching")
class PathPatchingAnalyzer(Analyzer):
    """Exact head → logits path patching of the media's effect, screened by direct logit attribution.

    Hyper-parameters:
        corruption:  ``"noise"`` (default, any media) or ``"gray"`` (images/video),
                     exactly as in ``causal_trace``.
        noise_scale: noise std as a multiple of the media embeddings' std (Meng: 3).
        top_k:       heads patched exactly per case, chosen by the screen;
                     0 = every head (one forward each).
        min_effect:  smallest ``|m_clean - m_corrupt|`` (logits) worth decomposing;
                     the shares are divided by it.
        target:      ``"prediction"`` or ``"gold"``.
        seed:        noise seed — the same per-case noise as ``causal_trace``.
        max_cases:   label-stratified cap; 0 (the default) = every case.
                     Cost per case: ``top_k + 4`` logits-only forwards.
    """

    name = "path_patching"
    description = (
        "Patches one head at a time with its output under corrupted image/audio, every "
        "other head frozen, to find which heads carry the media's effect directly to the "
        "answer (and how much of it the top heads carry together)."
    )
    requires = frozenset({Capability.HIDDEN_STATES, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "pp_margin_clean": ("Clean answer margin", "Logit margin of the target answer token with the media intact."),
        "pp_media_effect": ("Margin lost to corrupt media", "How much the answer margin falls when the media tokens are corrupted; near zero = the answer did not depend on the media."),
        "pp_media_effect_rel": ("Relative loss, corrupt media", "pp_media_effect as a fraction of the clean margin."),
        "pp_effect_traceable": ("Media effect traceable", "Corrupting the media moved the margin by at least min_effect, so it could be decomposed over heads."),
        "pp_top_head": ("Top direct-path head", "The head (L<layer>.H<head>) whose direct path to the answer carries the largest share of the media's effect (exact patch)."),
        "pp_top_head_share": ("Top head's direct share", "Share of the media's effect lost when only that head's direct path sees the corrupted media."),
        "pp_top_head_depth_frac": ("Top head depth", "Relative depth of that head's layer."),
        "pp_top_head_linear": ("Top head is linear attention", "That head belongs to a linear-attention (gated delta rule) layer of a hybrid stack."),
        "pp_topk_direct_share": ("Top-k heads, direct share", "Share of the media's effect lost when the top-k screened heads' direct paths are corrupted together, every other head frozen."),
        "pp_topk_total_share": ("Top-k heads, total share", "The same patch without freezing the other heads: the top-k heads' total effect, through later heads too."),
        "pp_screen_spearman": ("Screen vs exact (rank)", "Rank correlation between the direct-logit-attribution screen and the exact direct shares of the patched heads."),
        "pp_linear_share": ("Direct effect via linear heads", "Share of the patched heads' absolute exact direct effect carried by linear-attention heads (hybrid stacks)."),
    }

    def __init__(self, corruption: str = "noise", noise_scale: float = 3.0, top_k: int = 8,
                 min_effect: float = 1.0, target: str = "prediction", seed: int = 0,
                 max_cases: int = 0) -> None:
        from evalrx.analyzers.patching.causal_trace import CORRUPTIONS

        if corruption not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {corruption!r}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        super().__init__(corruption=corruption, noise_scale=noise_scale, top_k=top_k,
                         min_effect=min_effect, target=target, seed=seed, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.access import decoder_stack, media_positions, torch_module
        from evalrx.analyzers._internals.heads import (
            mixer_hooks,
            n_heads_of,
            out_proj,
            readout_direction,
        )
        from evalrx.analyzers._internals.residual import residual_hooks
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.analyzers.patching.causal_trace import (
            _noised,
            corrupted_variants,
            deepstack_mergers,
        )
        from evalrx.core.model import CaptureSpec
        from evalrx.models.backends.hf.discover import get_final_norm, get_unembed

        module = torch_module(model, self.name)
        stack = decoder_stack(module)
        mergers = deepstack_mergers(module)
        norm, unembed = get_final_norm(module), get_unembed(module)
        projs = [out_proj(m) if m is not None else None for m in stack.mixers]
        if any(p is None for p in projs):
            raise ValueError(f"{self.name}: some decoder layers have no recognised token mixer "
                             "with an output projection; their heads could not be frozen.")
        n_heads = [n_heads_of(m, p) for m, p in zip(stack.mixers, projs)]
        n_layers = stack.n_layers
        kinds = stack.kinds
        all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads[layer])]
        spec = CaptureSpec(to_cpu=False)

        def forward(variant: Any, *, capture: bool = False, replace: "dict | None" = None,
                    residual0: bool = False) -> "tuple[Any, Any, dict]":
            inputs, plan, (store, gen) = variant
            noised = _noised(mergers, store, self.noise_scale, gen) if store is not None else nullcontext()
            with torch.no_grad(), noised, \
                    residual_hooks(model, capture={0: None} if residual0 else None, replace=plan,
                                   analyzer=self.name) as res, \
                    mixer_hooks(model, capture=capture, replace=replace, analyzer=self.name) as mix:
                trace = model.forward(inputs, capture={Capability.LOGITS}, spec=spec)
            return trace, mix, res.captured

        def with_senders(base: dict, donor: dict, senders: "list[tuple[int, int]]",
                         freeze: bool) -> dict:
            return sender_inputs(base, donor, senders, n_heads, freeze=freeze)

        per_case: list[dict[str, Any]] = []
        tables: dict[str, list[dict[str, Any]]] = {}
        recurring: dict[str, list[float]] = defaultdict(list)
        screen_pairs: list[tuple[float, float]] = []
        skipped: Counter = Counter()
        slots: Counter = Counter()
        for index, case in enumerate(cases.stratified_head(self.max_cases)):
            clean_variant = (case.inputs, {}, (None, None))
            trace, clean_mix, clean_res = forward(clean_variant, capture=True, residual0=True)
            slot, media = media_positions(trace)
            if not media:
                skipped["no_media_tokens"] += 1
                continue
            logits = trace.require(Capability.LOGITS)[-1:]
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            m_clean = margin.value(logits)
            token_ids, last = trace.token_ids, trace.seq_len - 1
            variants, why = corrupted_variants(
                case.inputs, self.corruption, slot=slot, media=media,
                embeddings=clean_res[0][media], deepstack=bool(mergers),
                noise_scale=self.noise_scale, n_noise=1, seed=self.seed * 100003 + index)
            del trace, logits, clean_res
            if not variants:
                skipped[why] += 1
                continue
            donor_variant = variants[0]
            trace, donor_mix, _ = forward(donor_variant, capture=True)
            if trace.token_ids != token_ids:  # a gray image that re-tokenised differently
                skipped["corruption_misaligned"] += 1
                continue
            m_corrupt = margin.value(trace.require(Capability.LOGITS)[-1:])
            del trace
            effect = m_clean - m_corrupt
            row: dict[str, Any] = {
                "sample_id": case.id,
                "pp_margin_clean": round(m_clean, 4),
                "pp_media_effect": round(effect, 4),
                "pp_effect_traceable": abs(effect) >= self.min_effect,
            }
            if m_clean > 1e-6:
                row["pp_media_effect_rel"] = round(effect / m_clean, 4)
            slots[slot] += 1
            if abs(effect) < self.min_effect:
                per_case.append(row)
                continue

            clean_in, donor_in = clean_mix.inputs, donor_mix.inputs
            # Screen: the media-dependent part of each head's output at the answer
            # position, read through the frozen final norm into the margin.
            with torch.no_grad():
                v = readout_direction(norm, unembed, clean_mix.final_residual[last],
                                      margin.positive, margin.negative)
                screen: dict[tuple[int, int], float] = {}
                for layer, proj in enumerate(projs):
                    w_o = proj.weight.float()
                    delta = (donor_in[layer][last] - clean_in[layer][last]).float().to(w_o.device)
                    per_head = ((w_o.T @ v.to(w_o.device)) * delta).view(n_heads[layer], -1).sum(-1)
                    for head, value in enumerate(per_head.tolist()):
                        screen[(layer, head)] = -value / effect  # margin LOST, as a share
            k = len(all_heads) if self.top_k == 0 else min(self.top_k, len(all_heads))
            chosen = sorted(all_heads, key=lambda lh: -abs(screen[lh]))[:k]

            def share(replace: dict) -> float:
                trace, _, _ = forward(clean_variant, replace=replace)
                return (m_clean - margin.value(trace.require(Capability.LOGITS)[-1:])) / effect

            table = []
            for layer, head in chosen:
                direct = share(with_senders(clean_in, donor_in, [(layer, head)], freeze=True))
                table.append({"head": head_name(layer, head), "layer": layer, "head_index": head,
                              "kind": kinds[layer], "screen": round(screen[(layer, head)], 4),
                              "direct": round(direct, 4)})
                screen_pairs.append((screen[(layer, head)], direct))
            top = max(table, key=lambda t: abs(t["direct"]))
            row.update({
                "pp_top_head": top["head"],
                "pp_top_head_share": top["direct"],
                "pp_top_head_depth_frac": round(top["layer"] / max(1, n_layers - 1), 4),
                "pp_top_head_linear": top["kind"] == "linear",
                "pp_topk_direct_share": round(share(with_senders(clean_in, donor_in, chosen, freeze=True)), 4),
                "pp_topk_total_share": round(share(with_senders(clean_in, donor_in, chosen, freeze=False)), 4),
            })
            rho = _spearman([t["screen"] for t in table], [t["direct"] for t in table])
            if rho is not None:
                row["pp_screen_spearman"] = round(rho, 4)
            if "linear" in kinds:
                mass = sum(abs(t["direct"]) for t in table) or 1.0
                row["pp_linear_share"] = round(
                    sum(abs(t["direct"]) for t in table if t["kind"] == "linear") / mass, 4)
            per_case.append(row)
            tables[case.id] = table
            for t in sorted(table, key=lambda t: -abs(t["direct"]))[:3]:
                recurring[t["head"]].append(t["direct"])
            del clean_mix, donor_mix, clean_in, donor_in

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        traced = [r for r in per_case if "pp_top_head" in r]
        n = len(per_case)

        def mean(key: str) -> "float | None":
            values = [r[key] for r in traced if key in r]
            return round(sum(values) / len(values), 4) if values else None

        pooled = _spearman([a for a, _ in screen_pairs], [b for _, b in screen_pairs])
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_traced": len(traced),
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": n_layers,
            "heads_per_layer": n_heads,
            "layer_kinds": kinds,
            "media_slot": slots.most_common(1)[0][0] if slots else None,
            "corruption": self.corruption,
            "top_k": self.top_k,
            "target": self.target,
            "mean_media_effect": round(sum(r["pp_media_effect"] for r in per_case) / n, 4),
            "mean_topk_direct_share": mean("pp_topk_direct_share"),
            "mean_topk_total_share": mean("pp_topk_total_share"),
            "mean_screen_spearman": mean("pp_screen_spearman"),
            "screen_spearman_pooled": round(pooled, 4) if pooled is not None else None,
            "recurring_heads": [
                {"head": name, "n_cases": len(shares), "mean_direct_share": round(sum(shares) / len(shares), 4)}
                for name, shares in sorted(recurring.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
            ],
            "per_case": per_case,
            "_caveat": (
                "Exact path patches on the next-token margin at the end of the prompt: a head's "
                "output under corrupted media, every other token-mixing head frozen at its clean "
                "output, MLPs and norms recomputed. Only the top_k heads by a direct-logit-"
                "attribution screen are patched, so a head the screen ranks low is never "
                "measured — check pp_screen_spearman."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases,
                      artifacts={"head_tables": tables}, findings=findings)
