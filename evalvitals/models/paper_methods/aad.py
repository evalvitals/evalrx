"""Hugging Face adapter for AAD's released per-token contrastive sampler.

AAD (Hsu et al. 2025, arXiv:2506.07233, "Reducing Object Hallucination in
Large Audio-Language Models via Audio-Aware Decoding") contrasts the
with-audio next-token distribution against the SAME prompt with the audio
waveform replaced by silence (zeros), at every decode step:

    logits' = (1 + alpha) * logit(y_t | audio, text, y<t)
              - alpha * logit(y_t | silence, text, y<t)

No adaptive-plausibility cutoff -- unlike VCD, the released formula (and this
adapter, to stay faithful to it) does not mask sub-threshold candidates.
"""

from __future__ import annotations

from typing import Any


class AADLogitsProcessor:
    """Contrast clean (real-audio) scores with a cached silent-audio pass.

    Mirrors VCDLogitsProcessor's KV-cache reuse (the released reference
    implementation reruns the full growing sequence every step; caching the
    silent branch's own past_key_values gets the identical formula without
    that O(n^2) cost).
    """

    def __init__(self, model: Any, silent_inputs: dict[str, Any], *, alpha: float) -> None:
        self.model = model
        self.silent_inputs = silent_inputs
        self.alpha = float(alpha)
        self._silent_output: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        with torch.no_grad():
            if self._silent_output is None:
                self._silent_output = self.model(
                    **self.silent_inputs,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                self._silent_output = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=self._silent_output.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        silent_scores = self._silent_output.logits[:, -1, :]
        return (1.0 + self.alpha) * scores - self.alpha * silent_scores
