"""EmbedLens — what do the media tokens mean when they enter the language model?

Fan et al. read each projected visual token, at the language model's input,
through its nearest neighbours in the input-embedding table (cosine
similarity): many tokens sit next to concrete words — objects, colours, text in
the image — before any decoder layer has run, while others carry no
image-specific meaning. Here, for every media token (image / video / audio) at
the input of decoder layer 0:

* its ``top_k`` nearest vocabulary tokens and the best cosine similarity;
* its activation norm.

The paper sorts tokens into *sink* / *dead* / *alive* by checkpoint-specific
neighbour ids. Without such ids, the batch supplies the evidence:

* **recurring** — the token's nearest word is the nearest word of media tokens
  in at least ``recurring_frac`` of the images: a direction the projector emits
  whatever the image shows, so it says nothing about this one (dead-/sink-like);
* **high-norm** — norm at least ``norm_ratio`` × the image's median token norm
  (the massive-activation signature of sink tokens);
* **specific** — the rest: tokens whose nearest word is image-specific (alive-like).

And one grounding readout per case: each content word of the prompt against
the media tokens — the best cosine similarity any media token reaches to that
word's embedding. A question about an object the image does contain should
find a media token close to the object's word.

The paper reads the INPUT (``layer=0``, the default). Whether a model's
projector writes into vocabulary space there is itself the finding: on a
LLaVA-style model most media tokens sit near concrete words. On Qwen3.5-0.8B
they do not — the mean best cosine is 0.15, 90% of tokens share a recurring
nearest word, and the queried object's word is no closer to the image tokens
when the object is present than when it is absent (POPE adversarial, AUROC
≈ 0.5). The same lens at a late layer (``layer=0.85``–``1.0``) separates them
(AUROC 0.76–0.82) and 87% of the tokens turn image-specific (rocky, snow,
wooden, buses, ...): there the image tokens have been turned toward words.
``layer`` reads any depth.

References:
- What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in
  Multimodal Large Language Models — Fan et al., CVPR 2026 — arXiv:2603.00510
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


def content_word(tokenizer: Any, token_id: int, special: "set[str]") -> "str | None":
    """The token as a lower-case content word, or ``None`` (punctuation, function word, special)."""
    from evalrx.analyzers.lens.patchscope import _STOP, _word

    word = _word(tokenizer, token_id)
    if word is None or word in special:
        return None
    key = word.strip().lower()
    return key if key.isalpha() and len(key) >= 2 and key not in _STOP else None


@register_analyzer("embed_lens")
class EmbedLensAnalyzer(Analyzer):
    """Nearest vocabulary words of every media token at the decoder input, per case.

    Hyper-parameters:
        layer:          residual stream read — ``0`` (default, the decoder input as in
                        the paper), an index, or a fractional depth in (0, 1).
        top_k:          nearest vocabulary tokens kept per media token.
        recurring_frac: a nearest word shared by media tokens in at least this
                        fraction of the images is "recurring" (not image-specific);
                        needs at least ``min_images`` analysed images.
        min_images:     fewest images from which recurring words are inferred.
        norm_ratio:     a token is "high-norm" at ``norm ≥ norm_ratio × median``.
        vocab_chunk:    embedding rows compared per block (bounds memory).
        max_cases:      label-stratified cap; 0 (the default) = every case.
                        Cost: 1 forward per case + one pass over the embedding table.
    """

    name = "embed_lens"
    description = (
        "Reads each image/audio token, as it enters the language model, through its nearest "
        "vocabulary words: how many carry image-specific meaning, which repeat across every "
        "image, and whether a token sits near the words the question asks about."
    )
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"image", "video", "audio"})
    requires_modalities = frozenset({"image", "video", "audio"})
    signal_docs = {
        "el_n_media_tokens": ("Media tokens", "Number of media tokens entering the language model."),
        "el_mean_max_sim": ("Nearest-word similarity", "Mean, over media tokens, of the cosine similarity to their nearest vocabulary embedding."),
        "el_frac_specific": ("Image-specific tokens", "Share of media tokens whose nearest word is neither recurring across images nor carried by a high-norm token (alive-like)."),
        "el_frac_recurring": ("Recurring-word tokens", "Share of media tokens whose nearest word recurs across most images: not specific to this one (dead-/sink-like)."),
        "el_frac_high_norm": ("High-norm tokens", "Share of media tokens with norm ≥ norm_ratio × the image's median token norm (sink-like)."),
        "el_prompt_best_word": ("Best-grounded prompt word", "The prompt content word some media token is closest to."),
        "el_prompt_best_sim": ("Its grounding similarity", "Cosine similarity of that word's embedding to its closest media token."),
    }

    def __init__(self, layer: "int | float" = 0, top_k: int = 5, recurring_frac: float = 0.5,
                 min_images: int = 3, norm_ratio: float = 4.0, vocab_chunk: int = 32768,
                 max_cases: int = 0) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not 0 < recurring_frac <= 1:
            raise ValueError(f"recurring_frac must be in (0, 1], got {recurring_frac}")
        super().__init__(layer=layer, top_k=top_k, recurring_frac=recurring_frac,
                         min_images=min_images, norm_ratio=norm_ratio, vocab_chunk=vocab_chunk,
                         max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.access import (
            decoder_stack,
            media_positions,
            tokenizer_of,
            torch_module,
        )
        from evalrx.analyzers._internals.residual import residual_hooks
        from evalrx.analyzers.attention.relative_attn import resolve_attention_layer
        from evalrx.analyzers.lens.patchscope import _word
        from evalrx.core.model import CaptureSpec

        module = torch_module(model, self.name)
        n_layers = decoder_stack(module).n_layers
        layer = resolve_attention_layer(self.layer, n_layers) % n_layers
        table = module.get_input_embeddings().weight
        tokenizer = tokenizer_of(model)
        special = set(getattr(tokenizer, "all_special_tokens", None) or [])
        with torch.no_grad():
            row_norms = table.float().norm(dim=-1).clamp(min=1e-12)       # (vocab,)

        def nearest(x: Any) -> "tuple[Any, Any]":
            """Top-k cosine neighbours of *x* ``(n, dim)``: ``(sims, ids)``, each ``(n, k)``."""
            x = x.float().to(table.device)
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            best_s = best_i = None
            for start in range(0, table.shape[0], max(1, self.vocab_chunk)):
                block = table[start:start + self.vocab_chunk].float()
                sims = (x @ block.T) / row_norms[start:start + self.vocab_chunk]
                k = min(self.top_k, sims.shape[1])
                s, i = sims.topk(k, dim=-1)
                i = i + start
                if best_s is None:
                    best_s, best_i = s, i
                else:
                    merged_s = torch.cat([best_s, s], -1)
                    merged_i = torch.cat([best_i, i], -1)
                    s, order = merged_s.topk(min(self.top_k, merged_s.shape[1]), dim=-1)
                    best_s, best_i = s, merged_i.gather(-1, order)
            return best_s, best_i

        rows: list[dict[str, Any]] = []
        per_token: dict[str, dict[str, Any]] = {}
        prompt_sims: dict[str, dict[str, float]] = {}
        images_with: Counter = Counter()        # top-1 id -> number of DISTINCT images it is top-1 in
        seen_media: set = set()                  # fingerprints of the media already counted
        skipped: Counter = Counter()
        spec = CaptureSpec(to_cpu=False)
        for case in cases.stratified_head(self.max_cases):
            with torch.no_grad(), residual_hooks(model, capture={layer: None}, analyzer=self.name) as res:
                trace = model.forward(case.inputs, capture={Capability.LOGITS}, spec=spec)
            _, media = media_positions(trace)
            if not media:
                skipped["no_media_tokens"] += 1
                continue
            with torch.no_grad():
                x = res.captured[layer][media]                              # (n, dim)
                sims, ids = nearest(x)
                norms = x.float().norm(dim=-1)
                media_set = set(media)
                words = {}
                for pos, tid in enumerate(trace.token_ids):
                    if pos not in media_set:
                        word = content_word(tokenizer, tid, special) if tokenizer is not None else None
                        if word is not None and word not in words:
                            words[word] = int(tid)
                grounding = {}
                if words:
                    w = table[list(words.values())].float()
                    w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    xn = x.float().to(w.device)
                    xn = xn / xn.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    best = (xn @ w.T).max(0).values.tolist()
                    grounding = {word: round(s, 4) for word, s in zip(words, best)}
            top1 = ids[:, 0].tolist()
            # The media tokens precede the question, so several questions about one
            # image give identical media states: count each image once.
            fingerprint = (tuple(x.shape), round(float(x.float().sum()), 2),
                           tuple(round(v, 3) for v in x[0, :8].float().tolist()))
            if fingerprint not in seen_media:
                seen_media.add(fingerprint)
                images_with.update(set(top1))
            median = float(norms.median()) or 1e-12
            per_token[case.id] = {
                "top1": top1,
                "top_ids": ids.tolist(),
                "max_sim": [round(v, 4) for v in sims[:, 0].tolist()],
                "high_norm": [bool(v >= self.norm_ratio * median) for v in norms.tolist()],
            }
            prompt_sims[case.id] = grounding
            row: dict[str, Any] = {
                "sample_id": case.id,
                "el_n_media_tokens": len(media),
                "el_mean_max_sim": round(float(sims[:, 0].mean()), 4),
                "el_frac_high_norm": round(sum(per_token[case.id]["high_norm"]) / len(media), 4),
            }
            if grounding:
                best_word = max(grounding, key=grounding.get)
                row["el_prompt_best_word"] = best_word
                row["el_prompt_best_sim"] = grounding[best_word]
            rows.append(row)
            del trace, res, x, sims, ids

        if not rows:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        n_cases, n_images = len(rows), len(seen_media)
        recurring: set = set()
        if n_images >= self.min_images:
            recurring = {tid for tid, count in images_with.items() if count >= self.recurring_frac * n_images}
        specific_words: Counter = Counter()
        recurring_mass: Counter = Counter()
        total_tokens = 0
        for row in rows:
            tokens = per_token[row["sample_id"]]
            n = len(tokens["top1"])
            total_tokens += n
            high = tokens["high_norm"]
            rec = [not h and tid in recurring for tid, h in zip(tokens["top1"], high)]
            row["el_frac_recurring"] = round(sum(rec) / n, 4)
            row["el_frac_specific"] = round(1.0 - row["el_frac_recurring"] - row["el_frac_high_norm"], 4)
            for tid, h in zip(tokens["top1"], high):
                if tid in recurring:
                    recurring_mass[tid] += 1
                elif not h and tokenizer is not None:
                    word = content_word(tokenizer, tid, special)
                    if word is not None:
                        specific_words[word] += 1

        def label(tid: int) -> str:
            word = _word(tokenizer, tid) if tokenizer is not None else None
            return word if word is not None else str(tid)

        mean = lambda key: round(sum(r[key] for r in rows) / n_cases, 4)  # noqa: E731
        findings: dict[str, Any] = {
            "n_cases_analyzed": n_cases,
            "n_distinct_media": n_images,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "top_k": self.top_k,
            "layer_used": layer,
            "n_layers": n_layers,
            "recurring_inferred": n_images >= self.min_images,
            "recurring_words": [
                {"word": label(tid), "token_id": int(tid), "n_images": images_with[tid],
                 "token_share": round(recurring_mass[tid] / max(1, total_tokens), 4)}
                for tid in sorted(recurring, key=lambda t: -recurring_mass[t])
            ],
            "mean_max_sim": mean("el_mean_max_sim"),
            "mean_frac_specific": mean("el_frac_specific"),
            "mean_frac_recurring": mean("el_frac_recurring"),
            "mean_frac_high_norm": mean("el_frac_high_norm"),
            "specific_word_counts": specific_words.most_common(20),
            "per_case": rows,
            "_caveat": (
                f"Nearest neighbours by cosine similarity in the input-embedding table, read at the "
                f"input of decoder layer {layer}. 'Recurring' is inferred from this batch (a nearest word "
                "shared across at least recurring_frac of the distinct images), not from "
                "checkpoint-specific token ids; with fewer than min_images distinct images nothing "
                "is called recurring. The rule is the paper's, for the input layer: read deep, a "
                "scene word common to most of the batch's images can recur too."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases,
                      artifacts={"tokens": per_token, "prompt_sims": prompt_sims},
                      findings=findings)
