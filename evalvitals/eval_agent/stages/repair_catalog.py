"""Runtime-discovered, pre-audited repair capabilities.

This registry deliberately lives outside :mod:`fix_agent`.  Diagnosis reports
mechanisms; the repair agent receives only the structurally executable entries
returned here and decides which mechanism match (if any) is supported by the
evidence.  Adding an executor therefore extends this catalog without adding a
method-specific routing branch to the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalvitals.eval_agent.stages.fix_tiers import FixTier


@dataclass(frozen=True)
class RepairMethod:
    """A discoverable repair executor plus the evidence-facing description."""

    name: str
    tier: FixTier
    kind: str
    executor: str
    description: str
    source: str = "paper_default"
    payload: dict[str, Any] = field(default_factory=dict)
    baseline_executor: str | None = None


def method_names() -> frozenset[str]:
    """Stable names understood by explicit frozen-experiment allowlists."""

    return frozenset(method.name for method in _METHODS)


def discover_methods(
    model: Any,
    *,
    max_tier: FixTier,
    has_images: bool,
    has_audio: bool,
    tasks: set[str],
    binary_hallucination_supported: bool,
    allow_adapted: bool,
    prior_names: frozenset[str] = frozenset(),
) -> list[RepairMethod]:
    """Return methods that can physically execute for this model and batch.

    This is capability discovery, not mechanism matching.  In particular, it
    never reads hypothesis text.  The judge receives the resulting catalog and
    may decline every entry when the diagnosis does not support one.
    """

    fidelity_fn = getattr(model, "paper_method_fidelity", None)

    def fidelity(key: str) -> str:
        return fidelity_fn(key) if callable(fidelity_fn) else "unavailable"

    available: list[RepairMethod] = []
    for method in _METHODS:
        if method.name in prior_names or method.tier > max_tier:
            continue
        if not callable(getattr(model, method.executor, None)):
            continue
        if method.baseline_executor and not callable(
            getattr(model, method.baseline_executor, None)
        ):
            continue
        if not _structurally_eligible(
            method,
            fidelity=fidelity,
            has_images=has_images,
            has_audio=has_audio,
            tasks=tasks,
            binary_hallucination_supported=binary_hallucination_supported,
            allow_adapted=allow_adapted,
        ):
            continue
        available.append(method)
    return available


def _structurally_eligible(
    method: RepairMethod,
    *,
    fidelity,
    has_images: bool,
    has_audio: bool,
    tasks: set[str],
    binary_hallucination_supported: bool,
    allow_adapted: bool,
) -> bool:
    """Architecture/task gates only; never infer a mechanism from labels."""

    key = method.kind
    if key == "opera":
        return (
            has_images
            and tasks == {"yes_no"}
            and binary_hallucination_supported
            and fidelity("opera") == "native_binary_specialization"
        )
    if key in {"vicrop", "vicrop_consensus"}:
        return has_images and fidelity("vicrop") == "native_selector_specialization"
    if key == "ifcd":
        return (
            has_images
            and tasks == {"yes_no"}
            and binary_hallucination_supported
            and fidelity("ifcd") == "adapted_truthx_artifact"
            and allow_adapted
        )
    if key == "pai":
        return (
            has_images
            and (tasks != {"yes_no"} or binary_hallucination_supported)
            and fidelity("pai") == "native_attention_cfg_specialization"
        )
    if key == "tcd":
        method_fidelity = fidelity("tcd")
        return (
            has_audio
            and tasks in ({"multiple_choice"}, {"multiple_choice_letter"})
            and (
                method_fidelity == "native_layer_matched_stability"
                or (method_fidelity == "adapted_truncated_layer_stability" and allow_adapted)
            )
        )
    return False


_METHODS: tuple[RepairMethod, ...] = (
    RepairMethod(
        name="opera_overtrust_binary",
        tier=FixTier.L3A_INTERNALS_READ,
        kind="opera",
        executor="generate_opera_binary",
        source="paper_default_binary_specialization",
        payload={"num_attn_candidates": 5, "penalty_weight": 1.0},
        description=(
            "Penalises binary next-token candidates that neglect image attention; "
            "targets object or attribute hallucination caused by language priors "
            "overriding visual evidence."
        ),
    ),
    RepairMethod(
        name="vicrop_relative_attention",
        tier=FixTier.L3A_INTERNALS_READ,
        kind="vicrop",
        executor="generate_vicrop",
        payload={"layer": 14},
        description=(
            "Uses attention to locate and crop a relevant image region before "
            "re-answering; targets small or local visual detail missed at native resolution."
        ),
    ),
    RepairMethod(
        name="vicrop_consensus_guard",
        tier=FixTier.L3A_INTERNALS_READ,
        kind="vicrop_consensus",
        executor="generate_vicrop_consensus",
        source="safety_guard",
        payload={"layer": 14},
        description=(
            "Attention-guided crop and re-answer with a label-free consensus guard; "
            "a safer transfer variant for small or local visual detail."
        ),
    ),
    RepairMethod(
        name="ifcd_truthx_contrast",
        tier=FixTier.L3B_INTERNALS_WRITE,
        kind="ifcd",
        executor="generate_ifcd",
        source="paper_adapted_truthx_artifact",
        payload={"alpha": 0.1, "beta": 0.1, "edit_strength": 0.5, "top_layers": 15},
        description=(
            "Contrasts internal representations against a trained truthfulness direction; "
            "targets visual hallucination driven by language priors."
        ),
    ),
    RepairMethod(
        name="pai_image_attention",
        tier=FixTier.L3B_INTERNALS_WRITE,
        kind="pai",
        executor="generate_pai",
        source="paper_default_attention_cfg",
        payload={"alpha": 0.2, "guidance_scale": 2.0, "start_layer": 2, "end_layer": 32},
        description=(
            "Amplifies image-attention logits during decoding with classifier-free "
            "guidance; targets visual evidence being overridden by language priors."
        ),
    ),
    RepairMethod(
        name="tcd_temporal_blur",
        tier=FixTier.L3A_INTERNALS_READ,
        kind="tcd",
        executor="generate_tcd",
        baseline_executor="generate_tcd_baseline",
        description=(
            "Contrasts decoding against a temporally blurred audio view; targets "
            "under-weighted transient acoustic detail, not a flat knowledge gap or "
            "an unrelated option-position bias."
        ),
    ),
)
