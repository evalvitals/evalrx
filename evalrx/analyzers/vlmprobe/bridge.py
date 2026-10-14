"""Adapt EvalRX's live local model and native preprocessing to VLMProbe.

Diagnostic ``Trace`` objects deliberately detach their tensors. They cannot be
used for autograd or activation interventions; this bridge uses the exact live
model instead. All optional dependencies are imported only when requested.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evalrx.core.case import FailureCase, Inputs


def import_vlmprobe():
    try:
        import vlmprobe
    except ModuleNotFoundError as exc:
        if exc.name not in {"vlmprobe", "vlm_probing", "torch"}:
            raise
        raise ImportError(
            'VLMProbe analysis requires the optional dependency: pip install "evalrx[vlmprobe]". '
            "Automatic Hugging Face adapters also require VLMProbe's audited Transformers "
            "version; see docs/vlmprobe.md."
        ) from exc
    return vlmprobe


def resolve_prober(model, explicit=None):
    """Use an explicit adapter or bind the already-loaded EvalRX HF handle."""
    from evalrx.models.backends.hf.model import HFLocalModel

    if explicit is not None:
        if isinstance(model, HFLocalModel) and explicit.model is not model._loaded[0]:
            raise ValueError("prober must be bound to the exact model wrapped by EvalRX")
        return explicit
    if not isinstance(model, HFLocalModel):
        raise TypeError(
            "Automatic VLMProbe binding requires an EvalRX HF-local model (evalrx.wrap/load). "
            "For another backend, provide prober=... and input_builder=... with native inputs. "
            "Detached API/JAX/vLLM traces are not editable Torch activations."
        )
    raw_model, processor = model._loaded
    return import_vlmprobe().Prober(raw_model, processor=processor)


def native_inputs(model, value: Any):
    """Reuse EvalRX's chat/image encoding; explicit native inputs pass through."""
    if isinstance(value, FailureCase):
        value = value.inputs
    if isinstance(value, Mapping) or (hasattr(value, "kwargs") and hasattr(value, "layout")):
        return value
    from evalrx.models.backends.hf.model import HFLocalModel

    if not isinstance(model, HFLocalModel):
        raise TypeError("Provide native model kwargs or ProbeInputs for reference/target inputs")
    if isinstance(value, str):
        value = Inputs(prompt=value)
    if not isinstance(value, Inputs):
        raise TypeError("Expected Inputs, FailureCase, prompt text, native kwargs, or ProbeInputs")
    if value.audio is not None or value.video is not None:
        raise ValueError("Automatic VLMProbe preprocessing supports text and images; use a custom input_builder")
    if value.image is not None and not model.spec.needs_multimodal_encode:
        raise ValueError("The wrapped text-only model cannot consume this image")
    if model.spec.needs_multimodal_encode:
        raw_model, processor = model._loaded
        values, _, _, _ = model._encode_vlm(value, raw_model, processor)
        values = dict(values)
    else:
        values = dict(model._encode(value.prompt))
    # Match the native HFLocalModel forward path, which rejects processor-only
    # token_type_ids on multimodal architectures. ModelSpec supplies cache flags.
    values.pop("token_type_ids", None)
    return values


def make_metric(value):
    """Explicit task scores; callable scores keep their autograd graph intact."""
    if value is None or not isinstance(value, Mapping):
        return value
    config = dict(value)
    kind = config.pop("kind", "token_margin")
    package = import_vlmprobe()
    factories = {
        "token_margin": package.TokenMargin,
        "class_score": package.ClassScore,
        "sequence_logprob": package.SequenceLogProb,
    }
    if kind not in factories:
        raise ValueError(f"Unknown metric kind {kind!r}; choose {sorted(factories)}")
    return factories[kind](**config)
