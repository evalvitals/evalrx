"""Declarative discovery of pre-audited model repair capabilities.

The fix agent only sees compatible descriptions from this registry.  It has no
method names, benchmark names, or method-specific executor branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalvitals.core.capability import Capability
from evalvitals.eval_agent.stages.fix_tiers import FixTier


@dataclass(frozen=True)
class RepairMethod:
    name: str
    tier: FixTier
    executor: str
    description: str
    source: str = "registered"
    payload: dict[str, Any] = field(default_factory=dict)
    baseline_executor: str | None = None
    fidelity_key: str | None = None
    accepted_fidelities: frozenset[str] = frozenset()
    adapted_fidelities: frozenset[str] = frozenset()
    required_inputs: frozenset[str] = frozenset()
    accepted_tasks: frozenset[str] = frozenset()
    requires_logprobs: bool = False
    pass_baseline_answer: bool = False


def method_names() -> frozenset[str]:
    return frozenset(method.name for method in _METHODS)


def supports_tier(model: Any, tier: FixTier) -> bool:
    return any(
        method.tier == tier and callable(getattr(model, method.executor, None))
        for method in _METHODS
    )


def discover_methods(
    model: Any,
    *,
    max_tier: FixTier,
    has_images: bool,
    has_audio: bool,
    tasks: set[str],
    allow_adapted: bool,
    prior_names: frozenset[str] = frozenset(),
) -> list[RepairMethod]:
    """Return physically executable repairs without consulting labels or data IDs."""

    fidelity_fn = getattr(model, "paper_method_fidelity", None)
    capabilities = getattr(model, "capabilities", frozenset())
    inputs = frozenset(
        name for name, present in (("image", has_images), ("audio", has_audio)) if present
    )
    available: list[RepairMethod] = []
    for method in _METHODS:
        if method.name in prior_names or method.tier > max_tier:
            continue
        if not callable(getattr(model, method.executor, None)):
            continue
        if method.baseline_executor and not callable(getattr(model, method.baseline_executor, None)):
            continue
        if not method.required_inputs.issubset(inputs):
            continue
        if method.accepted_tasks and not tasks.issubset(method.accepted_tasks):
            continue
        if method.requires_logprobs and Capability.LOGPROBS not in capabilities:
            continue
        if method.fidelity_key:
            fidelity = fidelity_fn(method.fidelity_key) if callable(fidelity_fn) else "unavailable"
            accepted = method.accepted_fidelities | (
                method.adapted_fidelities if allow_adapted else frozenset()
            )
            if fidelity not in accepted:
                continue
        available.append(method)
    return available


_YES_NO = frozenset({"yes_no"})
_AUDIO_MC = frozenset({"multiple_choice", "multiple_choice_letter"})

_METHODS: tuple[RepairMethod, ...] = (
    RepairMethod(
        "vcd_diffusion_noise", FixTier.L3A_INTERNALS_READ, "generate_vcd",
        "Contrast original and corrupted-image decoding when visual claims follow language priors.",
        payload={"alpha": 1.0, "beta": 0.1, "noise_step": 500},
        baseline_executor="generate_vcd_baseline",
        fidelity_key="vcd", accepted_fidelities=frozenset({
            "exact", "native_binary_specialization", "per_item_seeded_sampler_specialization",
        }),
        # VCD contrasts every generated token; it is not a binary-only
        # executor.  A yes/no gate hid it from open-ended VQA before the
        # repair agent could inspect it.
        required_inputs=frozenset({"image"}), requires_logprobs=True,
    ),
    RepairMethod(
        "aad_silence_contrast", FixTier.L3A_INTERNALS_READ, "generate_aad",
        "Contrast normal and silenced-audio decoding when claims ignore acoustic evidence.",
        payload={"alpha": 0.5}, fidelity_key="aad",
        accepted_fidelities=frozenset({"native_silence_contrast"}),
        required_inputs=frozenset({"audio"}), accepted_tasks=_YES_NO,
    ),
    RepairMethod(
        "icd_instruction_disturbance", FixTier.L3A_INTERNALS_READ, "generate_instruction_cd",
        "Contrast normal and instruction-disturbed decoding when instruction priors override vision.",
        payload={"alpha": 1.0, "beta": 0.1, "qformer_mode": "normal"}, fidelity_key="icd",
        accepted_fidelities=frozenset({"exact", "native_binary_specialization"}),
        adapted_fidelities=frozenset({"adapted"}), required_inputs=frozenset({"image"}),
        accepted_tasks=_YES_NO, requires_logprobs=True,
    ),
    RepairMethod(
        "icd_instruction_disturbance_question", FixTier.L3A_INTERNALS_READ,
        "generate_instruction_cd",
        "Use architecture-native question-conditioned instruction contrast for visual grounding.",
        payload={"alpha": 1.0, "beta": 0.1, "qformer_mode": "question"}, fidelity_key="icd",
        accepted_fidelities=frozenset({"exact", "native_binary_specialization"}),
        required_inputs=frozenset({"image"}), accepted_tasks=_YES_NO, requires_logprobs=True,
    ),
    RepairMethod(
        "opera_overtrust_binary", FixTier.L3A_INTERNALS_READ, "generate_opera_binary",
        "Penalise binary token candidates that neglect image attention.",
        payload={"num_attn_candidates": 5, "penalty_weight": 1.0}, fidelity_key="opera",
        accepted_fidelities=frozenset({"native_binary_specialization"}),
        required_inputs=frozenset({"image"}), accepted_tasks=_YES_NO,
    ),
    RepairMethod(
        "vicrop_relative_attention", FixTier.L3A_INTERNALS_READ, "generate_vicrop",
        "Use attention to locate and crop visual detail before re-answering.",
        payload={"layer": 14}, fidelity_key="vicrop",
        accepted_fidelities=frozenset({"native_selector_specialization"}),
        required_inputs=frozenset({"image"}),
    ),
    RepairMethod(
        "vicrop_consensus_guard", FixTier.L3A_INTERNALS_READ, "generate_vicrop_consensus",
        "Use an attention crop with a label-free baseline-consensus guard.",
        payload={"layer": 14}, fidelity_key="vicrop",
        accepted_fidelities=frozenset({"native_selector_specialization"}),
        required_inputs=frozenset({"image"}), pass_baseline_answer=True,
    ),
    RepairMethod(
        "ifcd_truthx_contrast", FixTier.L3B_INTERNALS_WRITE, "generate_ifcd",
        "Contrast representations against a trained truthfulness direction.",
        payload={"alpha": 0.1, "beta": 0.1, "edit_strength": 0.5, "top_layers": 15},
        fidelity_key="ifcd", adapted_fidelities=frozenset({"adapted_truthx_artifact"}),
        required_inputs=frozenset({"image"}), accepted_tasks=_YES_NO,
    ),
    RepairMethod(
        "pai_image_attention", FixTier.L3B_INTERNALS_WRITE, "generate_pai",
        "Amplify image-attention logits with guidance when language overrides vision.",
        payload={"alpha": 0.2, "guidance_scale": 2.0, "start_layer": 2, "end_layer": 32},
        fidelity_key="pai", accepted_fidelities=frozenset({"native_attention_cfg_specialization"}),
        required_inputs=frozenset({"image"}),
    ),
    RepairMethod(
        "tcd_temporal_blur", FixTier.L3A_INTERNALS_READ, "generate_tcd",
        "Contrast decoding against temporally blurred audio to recover transient detail.",
        baseline_executor="generate_tcd_baseline", fidelity_key="tcd",
        accepted_fidelities=frozenset({"native_layer_matched_stability"}),
        adapted_fidelities=frozenset({"adapted_truncated_layer_stability"}),
        required_inputs=frozenset({"audio"}), accepted_tasks=_AUDIO_MC,
    ),
)
