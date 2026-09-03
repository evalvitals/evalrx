"""Hugging Face adapter for VCD's released per-token contrastive sampler."""

from __future__ import annotations

import math
from typing import Any


class VCDLogitsProcessor:
    """Contrast clean scores with a cached noisy-image generation path.

    The authors' sampling loop performs the second multimodal forward at every
    decode step.  This processor owns the corresponding noisy KV cache, which
    lets an unmodified Hugging Face LLaVA ``generate`` call execute the same
    score formula.
    """

    def __init__(self, model: Any, noisy_inputs: dict[str, Any], *, alpha: float, beta: float) -> None:
        self.model = model
        self.noisy_inputs = noisy_inputs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._noisy_output: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        with torch.no_grad():
            if self._noisy_output is None:
                self._noisy_output = self.model(
                    **self.noisy_inputs,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                self._noisy_output = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=self._noisy_output.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        noisy_scores = self._noisy_output.logits[:, -1, :]
        cutoff = scores.max(dim=-1, keepdim=True).values + math.log(self.beta)
        contrastive = (1.0 + self.alpha) * scores - self.alpha * noisy_scores
        return contrastive.masked_fill(scores < cutoff, -float("inf"))
