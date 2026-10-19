"""Attribution patching — causal tracing of every single token at every layer, from gradients.

``causal_trace`` patches GROUPS of positions exactly — all media tokens, or the
final prompt position — one forward pass per (layer, group). Attribution
patching estimates the effect of patching each SINGLE token ``t`` at each layer
``l`` at once, with a first-order expansion of the answer margin:

    effect(l, t) ≈ (h_clean[l, t] − h_corrupt[l, t]) · ḡ[l, t]

where ``h[l, t]`` is the residual stream ENTERING decoder layer ``l`` (the site
``causal_trace`` patches, under the same corruptions and noise draws) and ``ḡ``
is ``∂margin / ∂h[l, t]``:

* ``ig_steps = 1`` — taken at one end: the CORRUPTED run for
  ``direction="denoise"`` (Nanda: the estimate of restoring the clean token
  into the corrupted run) or the CLEAN run for ``"noise"`` (Syed et al.,
  Kramár et al.: corrupting that one token of the clean run);
* ``ig_steps > 1`` (default) — averaged along the straight path that moves the
  decoder's media input from its corrupted to its clean value (EAP-IG, Hanna et
  al.): the media embeddings and, on DeepStack models, every re-injected copy of
  the visual features. The pixels are not interpolated, so a gray image's path
  does not run through the vision tower.

Divided by the media effect ``m_clean − m_corrupt``, ``effect(l, t)`` is the
share of the media's effect that token carries at that depth. Reported, for
single tokens only:

* ``last`` — the final prompt position, where the answer is read out: its
  share per layer and the depth where it first carries half (read-out depth);
* ``relay`` — at each layer, the strongest single text token other than the
  last one: the question word or template token through which the media's
  information reaches the answer; across cases, which tokens those are.

Not reported: the media tokens as a group. Restoring hundreds of image tokens
together is far from additive, and the sum of their single-token estimates
does not track the joint patch (on Qwen3-VL-2B its correlation with the exact
group patch was below 0.4 under every setting, against 0.9 and above for the
last position); ``causal_trace`` measures the group exactly.

Each case checks itself: the reported sites — the last position at
``n_verify`` evenly spaced interior layers and the strongest relay token at its
layer — are also patched for real. A case whose mean gap exceeds
``verify_tol`` is kept, flagged ``ap_verified = False``, and left out of the
aggregate curves and the relay-token tally; ``verification`` pools estimate
against exact over all cases. The estimate fails mostly where corrupting the
media made the model MORE confident (a saturated margin, where gradients
under-read the effect — Kramár et al.).

References:
- Attribution Patching: Activation Patching At Industrial Scale — Nanda, 2023
  (neelnanda.io/mechanistic-interpretability/attribution-patching)
- Attribution Patching Outperforms Automated Circuit Discovery
  Syed, Rager & Conmy, BlackboxNLP 2024 — arXiv:2310.10348
- AtP*: An efficient and scalable method for localizing LLM behaviour to
  components — Kramár et al., 2024 — arXiv:2403.00745
- Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model
  Mechanisms (EAP-IG) — Hanna, Pezzelle & Belinkov, COLM 2024 — arXiv:2403.17806
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
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model

DIRECTIONS = ("denoise", "noise")
SITES = ("last", "relay")


def _token_text(tokenizer: Any, token_id: int) -> "str | None":
    """One token decoded, or ``None`` when the tokenizer cannot decode."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 - no decoder → no token text, never a guess
        return None
    return text if isinstance(text, str) else None


@contextmanager
def _recorded(mergers: list, out: dict) -> Iterator[None]:
    """Record each DeepStack merger's output into ``out[i]``."""
    def make(i: int):
        def hook(mod, args, output):
            out[i] = output.detach()
        return hook

    handles = [m.register_forward_hook(make(i)) for i, m in enumerate(mergers)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def path_point(alpha: float, *, inputs: Any, media: "list[int]", clean_media: Any,
               corrupt_media: Any, deltas: "dict | None") -> Any:
    """The run *alpha* of the way from the corrupted (0) to the clean (1) media input.

    Runs on the clean *inputs*, writes ``clean + (1 − alpha)(corrupt − clean)``
    over the media embeddings at layer 0 and adds ``(1 − alpha)`` × each
    DeepStack re-injection's corrupt-minus-clean difference (*deltas*).
    """
    base = clean_media.float()
    mixed = (base + (1.0 - alpha) * (corrupt_media.float() - base)).to(corrupt_media.dtype)
    store = None if deltas is None else {i: d * (1.0 - alpha) for i, d in deltas.items()}
    return inputs, {0: (media, mixed)}, (store, None)


def _pooled(pairs: "list[dict[str, Any]]") -> "dict[str, Any] | None":
    """Estimate-vs-exact agreement over verified sites."""
    if not pairs:
        return None
    est = [p["estimate"] for p in pairs]
    exact = [p["exact"] for p in pairs]
    errors = [abs(a - b) for a, b in zip(est, exact)]
    out: dict[str, Any] = {
        "n_sites": len(pairs),
        "mae": round(sum(errors) / len(errors), 4),
        "max_abs_err": round(max(errors), 4),
    }
    if len(pairs) >= 3:
        n = len(pairs)
        me, mx = sum(est) / n, sum(exact) / n
        cov = sum((a - me) * (b - mx) for a, b in zip(est, exact))
        var_e = sum((a - me) ** 2 for a in est)
        var_x = sum((b - mx) ** 2 for b in exact)
        if var_e > 0 and var_x > 0:
            out["pearson"] = round(cov / (var_e * var_x) ** 0.5, 4)
    return out


@register_analyzer("attribution_patching")
class AttributionPatchingAnalyzer(Analyzer):
    """Gradient estimate of every single-token patch effect of the media, per case.

    Hyper-parameters:
        corruption:  ``"noise"`` (default, any media) or ``"gray"`` (images/video),
                     exactly as in ``causal_trace``.
        noise_scale: noise std as a multiple of the media embeddings' std (Meng: 3).
        n_noise:     noise samples averaged per case (``"noise"`` only).
        ig_steps:    gradient points along the corrupt → clean path (midpoint rule);
                     1 = plain attribution patching at one end (see ``direction``).
        direction:   ``"denoise"`` (restore clean into corrupt; default) or
                     ``"noise"`` (corrupt one token of the clean run): the
                     patch the estimate is checked against, and with
                     ``ig_steps=1`` the run whose gradient is used.
        n_verify:    interior layers where the last position's estimate is checked
                     by an exact patch (plus one check of the strongest relay
                     token); 0 = no exact patches (and every case is aggregated).
        verify_tol:  largest mean |estimate − exact| (in shares of the media
                     effect) at which a case counts as verified.
        min_effect:  smallest ``|m_clean - m_corrupt|`` (logits) worth attributing;
                     the estimates are divided by it (see ``causal_trace``).
        target:      ``"prediction"`` or ``"gold"``.
        seed:        noise seed — the same per-case noise as ``causal_trace``.
        max_cases:   label-stratified cap; 0 (the default) = every case.
                     Cost per case and noise sample: 2 forwards, ``ig_steps``
                     forward + backward passes, ``n_verify + 1`` forwards.
    """

    name = "attribution_patching"
    description = (
        "Gradient estimate of causal tracing for every single token at every layer: how "
        "much of the image/audio's effect the answer position and each question token "
        "carry at each depth, checked against exact patches."
    )
    requires = frozenset({Capability.GRADIENTS, Capability.HIDDEN_STATES, Capability.LOGITS})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "ap_margin_clean": ("Clean answer margin", "Logit margin of the target answer token with the media intact."),
        "ap_media_effect": ("Margin lost to corrupt media", "How much the answer margin falls when the media tokens are corrupted; near zero = the answer did not depend on the media."),
        "ap_media_effect_rel": ("Relative loss, corrupt media", "ap_media_effect as a fraction of the clean margin."),
        "ap_effect_traceable": ("Media effect traceable", "Corrupting the media moved the margin by at least min_effect, so the effect could be attributed."),
        "ap_last_rise_frac": ("Answer read-out depth (est.)", "Relative depth where the final prompt position first carries half of the media's effect, estimated at every layer; late = the media's content reaches the answer late."),
        "ap_last_share_mid": ("Answer position share, mid", "Mean share of the media's effect carried by the final prompt position over the middle third of the layers."),
        "ap_relay_share": ("Strongest relay token share", "Largest share of the media's effect carried by one text token other than the last, at any layer; high = the media reaches the answer through that token."),
        "ap_relay_depth_frac": ("Strongest relay depth", "Relative depth at which that token carries it."),
        "ap_relay_pos_from_end": ("Strongest relay position", "That token's distance from the end of the prompt, in tokens."),
        "ap_relay_token": ("Strongest relay token", "That token's text (when the tokenizer can decode it)."),
        "ap_verify_mae": ("Estimate vs exact patch", "Mean absolute gap between the estimated and the exactly patched share at the verified sites; large = do not read this case's curves."),
        "ap_verified": ("Estimate verified", "ap_verify_mae is within verify_tol: this case's curves and relay token can be read (only these enter the aggregates)."),
    }

    def __init__(self, corruption: str = "noise", noise_scale: float = 3.0, n_noise: int = 1,
                 ig_steps: int = 5, direction: str = "denoise", n_verify: int = 2,
                 verify_tol: float = 0.25, min_effect: float = 1.0, target: str = "prediction",
                 seed: int = 0, max_cases: int = 0) -> None:
        from evalrx.analyzers.patching.causal_trace import CORRUPTIONS

        if corruption not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {corruption!r}")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        if ig_steps < 1:
            raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")
        super().__init__(corruption=corruption, noise_scale=noise_scale, n_noise=n_noise,
                         ig_steps=ig_steps, direction=direction, n_verify=n_verify,
                         verify_tol=verify_tol, min_effect=min_effect, target=target, seed=seed,
                         max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.access import (
            decoder_stack,
            layer_points,
            media_positions,
            tokenizer_of,
            torch_module,
        )
        from evalrx.analyzers._internals.residual import residual_hooks
        from evalrx.analyzers._internals.seeds import case_seed
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.analyzers.patching.causal_trace import (
            _merge,
            _noised,
            corrupted_variants,
            deepstack_mergers,
        )
        from evalrx.core.model import CaptureSpec

        module = torch_module(model, self.name)
        stack = decoder_stack(module)
        mergers = deepstack_mergers(module)
        n_layers = stack.n_layers
        layers = list(range(n_layers))
        depth = [layer / max(1, n_layers - 1) for layer in layers]
        mid = [layer for layer in layers if n_layers / 3 <= layer < 2 * n_layers / 3] or layers
        verify_layers = layer_points(n_layers, self.n_verify + 2)[1:-1] if self.n_verify > 0 else []
        denoise = self.direction == "denoise"
        integrated = self.ig_steps > 1
        alphas = [(j + 0.5) / self.ig_steps for j in range(self.ig_steps)]
        tokenizer = tokenizer_of(model)

        def run(variant: Any, *, grad: bool = False, capture: bool = False,
                restore: "dict | None" = None, record: "dict | None" = None) -> "tuple[Any, dict]":
            inputs, plan, (store, gen) = variant
            replace = _merge(plan, restore) if restore else plan
            noised = _noised(mergers, store, self.noise_scale, gen) if store is not None else nullcontext()
            recording = _recorded(mergers, record) if record is not None else nullcontext()
            with recording, noised, residual_hooks(
                    model, capture=dict.fromkeys(layers) if capture else None, replace=replace,
                    keep_grad=grad, analyzer=self.name) as hooks:
                trace = model.forward(inputs, capture={Capability.LOGITS},
                                      spec=CaptureSpec(with_grad=grad, to_cpu=False))
            return trace, hooks.captured

        def margin_of(margin: Any, variant: Any, restore: dict) -> float:
            trace, _ = run(variant, restore=restore)
            return margin.value(trace.require(Capability.LOGITS)[-1:])

        def gradients(margin: Any, trace: Any, states: dict) -> "tuple[list, list]":
            """``(∂margin/∂h[l], h[l])`` per layer, both ``(seq, dim)``."""
            hidden = [states[layer] for layer in layers]
            if not all(h.requires_grad for h in hidden):
                raise RuntimeError(
                    f"{self.name}: the forward pass kept no autograd graph — the backend "
                    "ignores CaptureSpec(with_grad=True), so there is nothing to differentiate."
                )
            grads = torch.autograd.grad(margin(trace.require(Capability.LOGITS)[-1:]), hidden)
            return [g[0] for g in grads], [h[0].detach() for h in hidden]

        def attributed(clean_h: list, corrupt_h: list, grads: list) -> Any:
            """Σ_d (h_clean − h_corrupt) · ḡ, per (layer, token)."""
            return torch.stack([
                ((clean_h[layer].float() - corrupt_h[layer].float()) * grads[layer].float()).sum(-1)
                for layer in layers
            ])

        per_case: list[dict[str, Any]] = []
        curves: dict[str, dict[str, Any]] = {}
        checks: dict[str, list[dict[str, Any]]] = {}
        relay_tokens: Counter = Counter()
        trusted: set = set()  # traced cases whose estimates passed their exact check
        skipped: Counter = Counter()
        slots: Counter = Counter()
        for case in cases.stratified_head(self.max_cases):
            clean_variant = (case.inputs, {}, (None, None))
            # A gray image's DeepStack copies are interpolated too: record both runs' copies.
            clean_rec: "dict | None" = {} if (integrated and mergers and self.corruption == "gray") else None
            clean_grad_needed = not (denoise or integrated)
            trace, states = run(clean_variant, grad=clean_grad_needed, capture=True, record=clean_rec)
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
            if clean_grad_needed:
                clean_grad, clean_h = gradients(margin, trace, states)
            else:
                clean_grad, clean_h = None, [states[layer] for layer in layers]
            token_ids, last = trace.token_ids, trace.seq_len - 1
            del trace, states, logits

            variants, why = corrupted_variants(
                case.inputs, self.corruption, slot=slot, media=media, embeddings=clean_h[0][media],
                deepstack=bool(mergers), noise_scale=self.noise_scale, n_noise=self.n_noise,
                seed=case_seed(self.seed, case))
            if not variants:
                skipped[why] += 1
                continue

            contrib: Any = None
            m_corrupt, aligned = 0.0, True
            corrupt_runs: list[list] = []  # "noise" checks write these corrupt states into the clean run
            for variant in variants:
                corrupt_rec: "dict | None" = {} if clean_rec is not None else None
                trace, states = run(variant, grad=denoise and not integrated, capture=True,
                                    record=corrupt_rec)
                if trace.token_ids != token_ids:  # a gray image that re-tokenised differently
                    aligned = False
                    break
                m_corrupt += margin.value(trace.require(Capability.LOGITS)[-1:])
                if integrated:
                    corrupt_h = [states[layer] for layer in layers]
                    del trace, states
                    store = variant[2][0]
                    deltas = ({i: corrupt_rec[i] - clean_rec[i] for i in corrupt_rec} if corrupt_rec
                              else (dict(store) if store else None))
                    term = None
                    for alpha in alphas:  # ḡ averaged along the corrupt → clean media path
                        point = path_point(alpha, inputs=case.inputs, media=media,
                                           clean_media=clean_h[0][media],
                                           corrupt_media=corrupt_h[0][media], deltas=deltas)
                        trace, states = run(point, grad=True, capture=True)
                        grads, _ = gradients(margin, trace, states)
                        step = attributed(clean_h, corrupt_h, grads) / len(alphas)
                        term = step if term is None else term + step
                        del trace, states, grads, step
                elif denoise:
                    grads, corrupt_h = gradients(margin, trace, states)
                    term = attributed(clean_h, corrupt_h, grads)
                    del trace, states, grads
                else:
                    corrupt_h = [states[layer] for layer in layers]
                    term = attributed(clean_h, corrupt_h, clean_grad)
                    del trace, states
                if not denoise:
                    corrupt_runs.append(corrupt_h)
                contrib = term if contrib is None else contrib + term
                del corrupt_h, term
            if not aligned:
                skipped["corruption_misaligned"] += 1
                continue
            contrib = contrib / len(variants)
            m_corrupt /= len(variants)

            effect = m_clean - m_corrupt
            row: dict[str, Any] = {
                "sample_id": case.id,
                "ap_margin_clean": round(m_clean, 4),
                "ap_media_effect": round(effect, 4),
                "ap_effect_traceable": abs(effect) >= self.min_effect,
            }
            if m_clean > 1e-6:
                row["ap_media_effect_rel"] = round(effect / m_clean, 4)
            slots[slot] += 1
            if abs(effect) < self.min_effect:
                per_case.append(row)
                continue

            share = contrib / effect                          # (layers, seq)
            text = torch.ones(share.shape[1], dtype=torch.bool, device=share.device)
            text[media] = False
            text[last] = False
            last_curve = share[:, last].tolist()
            row["ap_last_rise_frac"] = round(next((d for d, r in zip(depth, last_curve) if r >= 0.5), 1.0), 4)
            row["ap_last_share_mid"] = round(sum(last_curve[layer] for layer in mid) / len(mid), 4)
            sites = [(layer, last, "last", last_curve[layer]) for layer in verify_layers]
            relay_curve: "list[float] | None" = None
            relay_at: "list[int] | None" = None
            if bool(text.any()):
                best, where = share.masked_fill(~text, float("-inf")).max(1)
                relay_curve, relay_at = best.tolist(), where.tolist()
                peak = max(layers, key=lambda layer: relay_curve[layer])
                if relay_curve[peak] > 0:
                    pos = relay_at[peak]
                    row["ap_relay_share"] = round(relay_curve[peak], 4)
                    row["ap_relay_depth_frac"] = round(depth[peak], 4)
                    row["ap_relay_pos_from_end"] = last - pos
                    token = _token_text(tokenizer, token_ids[pos]) if tokenizer is not None else None
                    if token is not None:
                        row["ap_relay_token"] = token
                    if verify_layers:
                        sites.append((peak, pos, "relay", relay_curve[peak]))

            pairs: list[dict[str, Any]] = []
            for layer, pos, site, estimate in sites:
                if denoise:
                    restore = {layer: ([pos], clean_h[layer][[pos]])}
                    patched = sum(margin_of(margin, v, restore) for v in variants) / len(variants)
                    exact = (patched - m_corrupt) / effect
                else:
                    patched = sum(margin_of(margin, clean_variant, {layer: ([pos], run_h[layer][[pos]])})
                                  for run_h in corrupt_runs) / len(corrupt_runs)
                    exact = (m_clean - patched) / effect
                pairs.append({"layer": layer, "site": site,
                              "estimate": round(estimate, 4), "exact": round(exact, 4)})
            if pairs:
                row["ap_verify_mae"] = round(
                    sum(abs(p["estimate"] - p["exact"]) for p in pairs) / len(pairs), 4)
                row["ap_verified"] = row["ap_verify_mae"] <= self.verify_tol
                checks[case.id] = pairs
            if row.get("ap_verified", True):
                trusted.add(case.id)
                if "ap_relay_token" in row:
                    relay_tokens[row["ap_relay_token"]] += 1
            per_case.append(row)
            curves[case.id] = {"last": [round(r, 4) for r in last_curve]}
            if relay_curve is not None:
                curves[case.id]["relay"] = [round(r, 4) for r in relay_curve]
                curves[case.id]["relay_pos_from_end"] = [last - p for p in relay_at]
            del contrib, share, clean_h, clean_grad, corrupt_runs

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        label_of = {c.id: getattr(getattr(c, "label", None), "value", None) for c in cases}

        def mean_curve(label: str, site: str) -> "list[float] | None":
            rows = [c[site] for cid, c in curves.items()
                    if cid in trusted and label_of.get(cid) == label and site in c]
            if not rows:
                return None
            return [round(sum(col) / len(rows), 4) for col in zip(*rows)]

        every_pair = [p for pairs in checks.values() for p in pairs]
        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_traced": len(curves),
            "n_verified": len(trusted) if verify_layers else None,
            "verify_tol": self.verify_tol,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": n_layers,
            "media_slot": slots.most_common(1)[0][0] if slots else None,
            "corruption": self.corruption,
            "ig_steps": self.ig_steps,
            "direction": self.direction,
            "deepstack_layers": len(mergers),
            "target": self.target,
            "mean_media_effect": round(sum(r["ap_media_effect"] for r in per_case) / n, 4),
            "frac_traceable": round(len(curves) / n, 4),
            "verify_layers": verify_layers,
            "verification": ({site: _pooled([p for p in every_pair if p["site"] == site]) for site in SITES}
                             | {"all": _pooled(every_pair)}) if every_pair else None,
            "relay_tokens": relay_tokens.most_common(10),
            "mean_share_curves": {
                f"{label}_{site}": mean_curve(label, site)
                for label in ("fail", "pass") for site in SITES
                if mean_curve(label, site) is not None
            },
            "per_case": per_case,
            "_caveat": (
                "First-order (gradient) estimates of single-token residual-stream patch effects "
                "on the next-token margin at the end of the prompt, as a share of the media's "
                "effect" + (f", with the gradient averaged over {self.ig_steps} points of the "
                            "corrupt → clean media path" if integrated else "")
                + ". The aggregate curves and relay tokens cover only cases whose estimates "
                "passed their own exact check (ap_verified); the media tokens' joint effect "
                "is not estimated here (it is not additive) — causal_trace measures it exactly."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases,
                      artifacts={"share_curves": curves, "verification": checks},
                      findings=findings)
