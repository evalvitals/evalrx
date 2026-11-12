"""Patchscope — read a hidden state by letting the model itself decode it.

Logit lens and tuned lens push an intermediate residual straight to the
vocabulary. A Patchscope (Ghandeharioun et al.) instead writes the residual
into a second forward pass — a few-shot token-identity prompt,
``"cat -> cat; 1135 -> 1135; hello -> hello; ?"`` — at its placeholder
position and the SAME layer, and reads what the model predicts there: the
remaining layers do the decoding. Ghandeharioun et al. find this reads the
model's eventual next token better than either lens from mid depth on, with no
training.

Two sources per case:

* ``last`` (always) — the final prompt position, where the answer is formed,
  at every layer (or ``n_points`` of them): the probability the decoder gives
  the target token (the model's own answer, or the gold one with
  ``target="gold"``) and whether it ranks first. The depth where the answer
  first ranks first; the depth where it peaks; and the gap between that peak
  and what the model finally outputs — positive = the answer was readable more
  confidently mid-stack than at the output (with ``target="gold"`` on failing
  cases: formed, then suppressed).
* ``media`` (``media_layers > 0``) — every image / audio token at a few
  evenly spaced layers above the embedding, decoded the same way: the word the
  model's own machinery reads off each media token ("what it sees"), kept as a
  per-token word map and tallied (content words only) across cases.

The patch is checked once per model: patching a target position with the
target's own residual must leave its logits unchanged.

References:
- Patchscopes: A Unifying Framework for Inspecting Hidden Representations of
  Language Models — Ghandeharioun et al., ICML 2024 — arXiv:2401.06102
- interpreting GPT: the logit lens — nostalgebraist, 2020 (the baseline it improves on)
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

IDENTITY_PROMPT = "cat -> cat; 1135 -> 1135; hello -> hello; ?"
# Function words the decoder emits for almost any state; kept out of the word tally.
_STOP = frozenset("""a an and are as at be but by for from has have in into is it its of on or
that the their there this to was were with""".split())


def _word(tokenizer: Any, token_id: int) -> "str | None":
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 - no decoder → no word, never a guess
        return None
    return text if isinstance(text, str) else None


@register_analyzer("patchscope")
class PatchscopeAnalyzer(Analyzer):
    """Token-identity Patchscope readout of the answer position (and, optionally, the media tokens).

    Hyper-parameters:
        target:        ``"prediction"`` (the model's own top-1) or ``"gold"``.
        target_prompt: the decoding prompt; its LAST token is the placeholder
                       that receives the patch (default: token identity, 3 shots).
        n_points:      layers read at the answer position; 0 (default) = all.
        media_layers:  layers at which every media token is decoded too, spread over
                       the stack above the embedding (which decodes to noise); 0 = off.
        batch_rows:    rows per batched decoding forward.
        max_cases:     label-stratified cap; 0 (the default) = every case.
                       Cost per case: 1 forward + one batched forward of the
                       short target prompt (+ media tokens × media_layers rows).
    """

    name = "patchscope"
    description = (
        "Decodes the answer position's hidden state at every layer with the model itself "
        "(Patchscope, token-identity prompt): when the answer becomes readable, and whether "
        "it was readable more strongly mid-stack than at the output; optionally decodes the "
        "image/audio tokens into words."
    )
    requires = frozenset({Capability.HIDDEN_STATES, Capability.LOGITS})
    applies_to_modalities = frozenset({"text", "image", "video", "audio"})
    signal_docs = {
        "ps_final_prob": ("Answer prob. at the output", "Probability the model itself gives the target token."),
        "ps_first_depth_frac": ("Answer readable from (depth)", "First relative depth at which the Patchscope readout of the answer position ranks the target token first (1.0 = never before the last layer)."),
        "ps_peak_prob": ("Peak readable answer prob.", "Highest probability the Patchscope readout gives the target token at any depth."),
        "ps_peak_depth_frac": ("Peak readout depth", "Relative depth of that peak."),
        "ps_late_drop": ("Readable answer lost late", "Peak readout probability minus the model's final probability of the target token; positive = more confident mid-stack than at the output (suppressed late)."),
        "ps_agree_mid": ("Mid-stack readout agrees", "Share of the middle third of the layers whose readout ranks the model's final answer first."),
    }

    def __init__(self, target: str = "prediction", target_prompt: str = IDENTITY_PROMPT,
                 n_points: int = 0, media_layers: int = 0, batch_rows: int = 64,
                 max_cases: int = 0) -> None:
        super().__init__(target=target, target_prompt=target_prompt, n_points=n_points,
                         media_layers=media_layers, batch_rows=batch_rows, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.access import (
            decoder_stack,
            layer_points,
            media_positions,
            tokenizer_of,
            torch_module,
        )
        from evalrx.analyzers._internals.residual import batched_patch, residual_hooks
        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.core.model import CaptureSpec

        module = torch_module(model, self.name)
        stack = decoder_stack(module)
        n_layers = stack.n_layers
        layers = layer_points(n_layers, self.n_points)
        media_at = layer_points(n_layers, self.media_layers + 1)[1:] if self.media_layers > 0 else []
        mid = [layer for layer in layers if n_layers / 3 <= layer < 2 * n_layers / 3] or layers
        depth = {layer: layer / max(1, n_layers - 1) for layer in range(n_layers)}
        tokenizer = tokenizer_of(model)
        special = set(getattr(tokenizer, "all_special_tokens", None) or [])
        if tokenizer is None:
            raise ValueError(f"{self.name}: needs the model's tokenizer to build the target prompt")
        prompt_ids = tokenizer(self.target_prompt, add_special_tokens=False)["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = prompt_ids[0]
        if len(prompt_ids) < 2:
            raise ValueError(f"{self.name}: target_prompt must tokenize to at least two tokens")
        placeholder = len(prompt_ids) - 1
        device = next(module.parameters()).device
        spec = CaptureSpec(to_cpu=False)

        def decode(sources: "list[tuple[int, Any]]") -> Any:
            """Placeholder logits ``(rows, vocab)``: row ``i`` carries ``sources[i] = (layer, vector)``."""
            out = []
            for start in range(0, len(sources), max(1, self.batch_rows)):
                chunk = sources[start:start + self.batch_rows]
                patches: dict[int, list] = {}
                for row, (layer, vector) in enumerate(chunk):
                    patches.setdefault(layer, []).append((row, placeholder, vector))
                ids = torch.tensor([prompt_ids] * len(chunk), device=device)
                with torch.no_grad(), batched_patch(model, patches, analyzer=self.name):
                    logits = module(input_ids=ids).logits
                out.append(logits[:, placeholder].float())
            return torch.cat(out)

        # The patch itself must be inert: writing the target's own residual back.
        check_layer = layers[len(layers) // 2]
        with torch.no_grad():
            own: dict[str, Any] = {}

            def grab(mod, args, kwargs):
                own["h"] = (args[0] if args else kwargs["hidden_states"])[0, placeholder].detach().clone()

            handle = stack.layers[check_layer].register_forward_pre_hook(grab, with_kwargs=True)
            try:
                plain = module(input_ids=torch.tensor([prompt_ids], device=device)).logits[0, placeholder].float()
            finally:
                handle.remove()
            patched = decode([(check_layer, own["h"])])[0]
        patch_diff = float((patched - plain).abs().max())
        if patch_diff > 1e-3 * max(1.0, float(plain.abs().max())):
            raise RuntimeError(f"{self.name}: re-writing a residual with itself changed the logits "
                               f"by {patch_diff:.3g}; the patch does not reach the layer input")

        per_case: list[dict[str, Any]] = []
        curves: dict[str, dict[str, Any]] = {}
        media_words: dict[str, dict[int, list]] = {}
        word_counts: Counter = Counter()
        skipped: Counter = Counter()
        for case in cases.stratified_head(self.max_cases):
            with torch.no_grad(), residual_hooks(model, capture=dict.fromkeys(set(layers) | set(media_at)),
                                                 analyzer=self.name) as res:
                trace = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            last = trace.seq_len - 1
            _, media = media_positions(trace) if media_at else (None, [])
            logits = trace.require(Capability.LOGITS)[-1:]
            margin, why = build_margin(self.target, logits, model=model, case=case)
            if margin is None:
                skipped[why] += 1
                continue
            final_probs = torch.softmax(logits[-1].float(), -1)
            final_top = int(final_probs.argmax())
            target_id = margin.positive
            del trace

            states = res.captured  # layer -> (seq, dim)
            readout = torch.softmax(decode([(layer, states[layer][last]) for layer in layers]), -1)
            p_target = readout[:, target_id].tolist()
            top1 = readout.argmax(-1).tolist()
            peak = max(range(len(layers)), key=lambda i: p_target[i])
            p_final = float(final_probs[target_id])
            row: dict[str, Any] = {
                "sample_id": case.id,
                "ps_final_prob": round(p_final, 4),
                "ps_first_depth_frac": round(next(
                    (depth[layer] for layer, t in zip(layers, top1) if t == target_id), 1.0), 4),
                "ps_peak_prob": round(p_target[peak], 4),
                "ps_peak_depth_frac": round(depth[layers[peak]], 4),
                "ps_late_drop": round(p_target[peak] - p_final, 4),
                "ps_agree_mid": round(
                    sum(top1[layers.index(layer)] == final_top for layer in mid) / len(mid), 4),
            }
            per_case.append(row)
            curves[case.id] = {"p_target": [round(p, 4) for p in p_target],
                               "top1": [_word(tokenizer, t) or str(t) for t in top1]}

            if media_at and media:
                rows = [(layer, states[layer][pos]) for layer in media_at for pos in media]
                ids = decode(rows).argmax(-1).tolist()
                words: dict[int, list] = {}
                for (layer, _), token in zip(rows, ids):
                    word = _word(tokenizer, token)
                    words.setdefault(layer, []).append(word if word is not None else str(token))
                    key = (word or "").strip().lower()
                    if word not in special and key.isalpha() and key not in _STOP:
                        word_counts[key] += 1
                media_words[case.id] = words
            del res, states

        if not per_case:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        label_of = {c.id: getattr(getattr(c, "label", None), "value", None) for c in cases}

        def mean_curve(label: str) -> "list[float] | None":
            rows = [c["p_target"] for cid, c in curves.items() if label_of.get(cid) == label]
            if not rows:
                return None
            return [round(sum(col) / len(rows), 4) for col in zip(*rows)]

        n = len(per_case)
        findings: dict[str, Any] = {
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "n_layers": n_layers,
            "layers_read": layers,
            "target": self.target,
            "target_prompt": self.target_prompt,
            "patch_check_max_diff": round(patch_diff, 6),
            "mean_late_drop": round(sum(r["ps_late_drop"] for r in per_case) / n, 4),
            "mean_first_depth_frac": round(sum(r["ps_first_depth_frac"] for r in per_case) / n, 4),
            "mean_p_target_curves": {label: mean_curve(label) for label in ("fail", "pass")
                                     if mean_curve(label) is not None},
            "per_case": per_case,
            "_caveat": (
                "Each layer's residual at the final prompt position is decoded by the model "
                "itself, patched into the placeholder of the target prompt at the same layer. "
                "A readout says what the model's remaining layers make of that state in a "
                "copy-the-token context, not what the original run did with it."
            ),
        }
        if media_at:
            findings["media_layers_read"] = media_at
            findings["media_word_counts"] = word_counts.most_common(20)
        return Result(analyzer=self.name, model=repr(model), cases=cases,
                      artifacts={"readout_curves": curves, "media_words": media_words},
                      findings=findings)
