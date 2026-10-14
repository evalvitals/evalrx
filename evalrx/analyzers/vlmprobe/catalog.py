"""Lightweight inventory of VLMProbe's public, model-bound method factories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    family: str
    factory: str
    needs_metric: bool = False
    calibrated: bool = False


METHOD_SPECS = {
    "lens.logit": MethodSpec("lens", "logit"),
    "lens.tuned": MethodSpec("lens", "tuned", calibrated=True),
    "lens.jacobian": MethodSpec("lens", "jacobian", calibrated=True),
    "lens.attention": MethodSpec("lens", "attention", calibrated=True),
    "lens.embed": MethodSpec("lens", "embed"),
    "lens.patchscope": MethodSpec("lens", "patchscope"),
    "attention.profile": MethodSpec("attention", "profile"),
    "attention.rollout": MethodSpec("attention", "rollout"),
    "attention.grad_cam": MethodSpec("attention", "grad_cam", needs_metric=True),
    "attention.attribution": MethodSpec("attention", "attribution", needs_metric=True),
    "attention.tam": MethodSpec("attention", "tam", needs_metric=True),
    "attention.beyond_intuition": MethodSpec("attention", "beyond_intuition", needs_metric=True),
    "attention.dtd_lrp": MethodSpec("attention", "dtd_lrp"),
    "attention.relevance": MethodSpec("attention", "relevance", needs_metric=True),
    "attention.head_logits": MethodSpec("attention", "head_logits"),
    "attention.reweight": MethodSpec("attention", "reweight", needs_metric=True),
    "attention.temperature": MethodSpec("attention", "temperature", needs_metric=True),
    "causal.patch": MethodSpec("causal", "patch", needs_metric=True),
    "causal.attribute": MethodSpec("causal", "attribute", needs_metric=True),
    "causal.path": MethodSpec("causal", "path", needs_metric=True),
    "causal.knockout": MethodSpec("causal", "knockout", needs_metric=True),
    "causal.steer": MethodSpec("causal", "steer", needs_metric=True),
    "causal.vsv": MethodSpec("causal", "vsv", needs_metric=True),
    "causal.eap_ig": MethodSpec("causal", "eap_ig", needs_metric=True),
}


def list_methods() -> list[str]:
    """Return all supported factory keys without importing Torch or VLMProbe."""
    return sorted(METHOD_SPECS)
