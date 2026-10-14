"""Analyzers — functional taxonomy keyed by required capability (not black/white-box).

Every analyzer declares ``requires`` (capabilities) + ``applies_to_modalities`` and
self-registers on import.  Categories (full tables: docs/analyzers.md):

    perturbation/   RISE / VL-SHAP / MM-SHAP / modality ablation /
                    prompt contrast / MC format / CoT faithfulness /
                    context-SHAP / perturbation battery            — GENERATE/LOGPROBS
    uncertainty/    entropy / self-consistency / verbalized conf /
                    calibration / coverage-verification gap        — LOGITS/LOGPROBS/GENERATE
    hallucination/  POPE / CHAIR / SelfCheck / OPERA / VCD         — GENERATE/ATTENTION/LOGITS
    attention/      summary / rollout / sink / relative-attn       — ATTENTION
    attribution/    Grad-CAM / Chefer generic-attention            — GRADIENTS
    lens/           logit-lens / tuned-lens / layer contrast /
                    head logit attribution / patchscope            — HIDDEN_STATES
    patching/       causal tracing / attribution patching /
                    path patching / attention knockout /
                    attention sensitivity                          — HIDDEN_STATES / ATTENTION (write)
    geometry/       CKA / linear-probe                             — HIDDEN_STATES
    reasoning/      extraction/termination hygiene; arithmetic,
                    self-repair, step-value, knowledge split,
                    contamination                                  — GENERATE
    agent/          loop / ignored-obs / first-error / rubric /
                    counterfactual / reliability / tool-SHAP       — Trajectory

Import is torch-tolerant: the ``attention`` subpackage loads torch at import, so on
the light (pure-API) install it is skipped and its analyzers are simply not offered.

``_internals/`` is the private write-side plumbing (attention tap, residual hooks,
logit-margin targets) that the patching / attribution / head-attribution analyzers
install around an ordinary ``model.forward`` call; it imports torch lazily.
"""

from evalrx.analyzers import (  # noqa: F401  -- import to self-register
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
from evalrx.analyzers.base import Analyzer, Result

try:  # attention/summary imports torch+numpy at module load
    from evalrx.analyzers import attention  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover - light install
    if e.name not in {"torch", "numpy"}:
        raise
    attention = None  # type: ignore

__all__ = ["Analyzer", "Result"]
