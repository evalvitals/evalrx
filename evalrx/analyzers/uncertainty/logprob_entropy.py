"""Perplexity + predictive entropy from output-token logprobs (black-box).

Unlike :class:`TokenEntropyAnalyzer` (which needs the full LOGITS tensor from a
white-box forward), this works from ``model.logprobs()`` — i.e. OpenAI-style
``logprobs`` — so it runs on an API model.  ``requires=LOGPROBS``.

Perplexity is the exponential of mean negative log-likelihood — lower is better.
Token-level predictive entropy (mean over top-k distribution) measures local
uncertainty; high entropy on early tokens signals broad uncertainty.

Predictive entropy for uncertainty estimation:
  "Dropout as a Bayesian Approximation: Representing Model Uncertainty in
  Deep Learning" — Gal & Ghahramani, ICML 2016 — https://arxiv.org/abs/1506.02142

LLM self-knowledge via logprobs:
  "Language Models (Mostly) Know What They Know"
  Kadavath et al., 2022 — https://arxiv.org/abs/2207.05221
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("logprob_entropy")
class LogprobEntropyAnalyzer(Analyzer):
    """Sequence perplexity + mean top-k predictive entropy from output logprobs."""

    name = "logprob_entropy"
    requires = frozenset({Capability.LOGPROBS})
    applies_to_modalities = frozenset({"text", "image"})

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        toks = model.logprobs(case.inputs)            # list[TokenLogprob]
        lps = [t.logprob for t in toks]
        mean_lp = sum(lps) / len(lps) if lps else 0.0
        perplexity = math.exp(-mean_lp) if lps else float("inf")

        entropies = []
        for t in toks:
            if not t.top:
                continue
            ps = [math.exp(v) for v in t.top.values()]
            z = sum(ps) or 1.0
            ps = [p / z for p in ps]
            entropies.append(-sum(p * math.log(p) for p in ps if p > 0))

        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"token_logprobs": toks},
            findings={
                "n_tokens": len(toks),
                "mean_logprob": round(mean_lp, 4),
                "perplexity": round(perplexity, 4) if math.isfinite(perplexity) else None,
                "min_token_logprob": round(min(lps), 4) if lps else None,
                "mean_top_entropy": round(sum(entropies) / len(entropies), 4) if entropies else None,
            },
        )
