"""Analyzers — functional taxonomy keyed by required capability (not black/white-box).

Every analyzer declares ``requires`` (capabilities) + ``applies_to_modalities`` and
self-registers on import.  Categories:

    perturbation/   input masking (RISE; VL-SHAP/MM-SHAP)        — GENERATE/LOGPROBS
    uncertainty/    entropy / self-consistency / verbalized conf  — LOGITS/GENERATE
    hallucination/  POPE / CHAIR / OPERA / VCD                    — GENERATE/ATTENTION
    attention/      summary / rollout / sink / relative-attn      — ATTENTION
    attribution/    Grad-CAM / Chefer generic-attention           — GRADIENTS
    lens/           logit-lens / tuned-lens                       — HIDDEN_STATES
    patching/       causal tracing / activation patching          — HIDDEN_STATES (read+write)
    geometry/       CKA / linear-probe                            — HIDDEN_STATES
    reasoning/      extraction/termination hygiene; arithmetic,
                    self-repair, step-value, knowledge split       — GENERATE
    agent/          loop / ignored-obs / first-error / counterfactual — Trajectory

Import is torch-tolerant: the ``attention`` subpackage loads torch at import, so on
the light (pure-API) install it is skipped and its analyzers are simply not offered.
"""

from evalvitals.analyzers import (  # noqa: F401  -- import to self-register
    agent,
    attribution,
    geometry,
    hallucination,
    lens,
    patching,
    perturbation,
    reasoning,
    uncertainty,
)
from evalvitals.analyzers.base import Analyzer, Result

try:  # attention/summary imports torch+numpy at module load
    from evalvitals.analyzers import attention  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover - light install
    if e.name not in {"torch", "numpy"}:
        raise
    attention = None  # type: ignore

__all__ = ["Analyzer", "Result"]
