"""RISE — Randomized Input Sampling for Explanation (Petsiuk et al.), text variant.

Black-box, model-agnostic: mask random subsets of input tokens, score each masked
output, and attribute importance to a token as the mean score when it is kept.
``requires=GENERATE`` plus an injected ``score_fn(output)->float`` (correctness /
verifier signal — task-specific, so the caller provides it).

Cost note: this is the dominant compute (n_masks forward passes per sample). For
API models, run it through the existing async caller with an on-disk cache keyed
by the mask hash so a crashed sweep resumes.

References:
- RISE: Randomized Input Sampling for Explanation of Black-box Models
  Petsiuk, Das & Saenko, BMVC 2018 — arXiv:1806.07421
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Optional

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("rise")
class RISEAnalyzer(Analyzer):
    """Token-occlusion saliency over the input prompt.

    Hyper-parameters:
        score_fn:  ``output_text -> float`` scalar (required).
        n_masks:   number of random masks (cost driver).
        keep_prob: probability a token is kept in a mask.
        mask_token: replacement for dropped tokens.
        top_k:     tokens to report.
        seed:      RNG seed (reproducibility).
    """

    name = "rise"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text"})

    def __init__(
        self,
        score_fn: Optional[Callable[[str], float]] = None,
        n_masks: int = 50,
        keep_prob: float = 0.5,
        mask_token: str = "___",
        top_k: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(
            score_fn=score_fn, n_masks=n_masks, keep_prob=keep_prob,
            mask_token=mask_token, top_k=top_k, seed=seed,
        )

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        if self.score_fn is None:
            raise ValueError("RISEAnalyzer requires score_fn(output)->float (a task-specific verifier).")
        rng = random.Random(self.seed)
        case = cases[0]
        words = str(case.inputs).split()
        n = len(words)
        sal = [0.0] * n
        cnt = [0] * n
        n_calls = 0
        for _ in range(self.n_masks):
            mask = [rng.random() < self.keep_prob for _ in range(n)]
            if not any(mask):
                continue
            masked = " ".join(w if m else self.mask_token for w, m in zip(words, mask))
            score = float(self.score_fn(model.generate(masked)))
            n_calls += 1
            for j, keep in enumerate(mask):
                if keep:
                    sal[j] += score
                    cnt[j] += 1
        importance = [(sal[j] / cnt[j]) if cnt[j] else 0.0 for j in range(n)]
        ranked = sorted(zip(words, importance), key=lambda x: -x[1])[: self.top_k]
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"importance": importance, "words": words},
            findings={
                "n_words": n,
                "n_masks_run": n_calls,
                "top_tokens": [{"token": w, "importance": round(v, 4)} for w, v in ranked],
            },
        )
