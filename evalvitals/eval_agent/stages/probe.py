"""M1 — StrategyProbe: select which analyzers to run given a model.

The probe inspects the model's capabilities and modalities to determine what
kind of model it is (VLM, agent, or plain LLM), then returns a ranked list of
analyzer names from the registry that are both compatible with the model and
ordered by their diagnostic value for that model kind.

Usage::

    probe = StrategyProbe()
    kind  = probe.detect_kind(model)          # ModelKind.VLM / AGENT / LLM
    names = probe.select(model, max_analyzers=4)  # e.g. ["attention", "logit_lens", ...]
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from evalvitals.core.capability import Capability
from evalvitals.core.registry import registry


def _carries_trajectories(data: Any) -> bool:
    """True when *data* (CaseBatch / iterable of cases) has any trajectory-carrying case."""
    try:
        return any(getattr(c, "trajectory", None) is not None for c in data)
    except TypeError:
        return False

if TYPE_CHECKING:
    from evalvitals.core.model import Model


class ModelKind(str, Enum):
    VLM   = "vlm"    # image modality — hallucination + attention analyzers first
    AGENT = "agent"  # TOOL_CALLS — trajectory / behavioral analyzers first
    LLM   = "llm"    # text-only — interpretability analyzers first


# Maps failure-mode tags (from M3 hypotheses) to the analyzers most likely to
# surface evidence for that mode.  Used in cycle 2+ to focus the probe on what
# the diagnosis agent flagged rather than running the same generic priority list.
_FAILURE_MODE_TO_ANALYZERS: dict[str, list[str]] = {
    "attention_sink":            ["attention_sink"],
    "attention":                 ["attention", "attention_rollout"],
    "hallucination":             ["pope", "chair", "selfcheck_consistency"],
    "low_consistency":           ["self_consistency"],
    "unstable_generation":       ["self_consistency"],
    "overconfidence":            ["verbalized_confidence", "calibration"],
    "miscalibrated_confidence":  ["verbalized_confidence", "calibration"],
    "format_bias":               ["format_sensitivity"],
    "position_bias":             ["format_sensitivity"],
    "unfaithful_reasoning":      ["cot_faithfulness"],
    "post_hoc_reasoning":        ["cot_faithfulness"],
    "context_ignored":           ["context_shap"],
    "premature_layer_divergence": ["layer_contrast"],
    "confident_inconsistency":   ["self_consistency", "verbalized_confidence"],
    "loop":                      ["loop_detect"],
    "ignored_obs":               ["ignored_obs"],
    # Decision-layer mechanisms — discriminated by prompt interventions.
    "language_prior_bias":       ["prompt_contrast"],
    "negative_bias":             ["prompt_contrast", "pope"],
    "threshold_miscalibration":  ["prompt_contrast"],
    "prompt_formatting_bias":    ["prompt_contrast"],
    "instruction_following":     ["prompt_contrast"],
    "entropy":                   ["token_entropy", "logprob_entropy"],
    "perplexity":                ["logprob_entropy"],
    "logit_lens":                ["logit_lens"],
    "representational_collapse": ["cka"],
    "numerical_hallucination":   ["self_consistency", "verbalized_confidence"],
}

# Per-kind ordered priority: high → low diagnostic value for false attribution.
# Analyzers not in the list are appended alphabetically after the ranked ones.
_PRIORITY: dict[str, list[str]] = {
    ModelKind.VLM: [
        "pope", "chair",                          # hallucination metrics
        "attention", "attention_rollout",          # where is the model looking?
        "attention_sink",                          # sink collapse?
        "prompt_contrast",                         # are failures prompt-repairable?
        "mm_shap",                                # text vs image reliance
        "logprob_entropy", "self_consistency",
    ],
    ModelKind.AGENT: [
        "loop_detect", "ignored_obs",             # behavioral heuristics
        "first_error_judge", "trajectory_rubric", # LLM-judge localisation + classification
        "counterfactual", "reliability_probe",    # re-run probes: step perturbation, pass@k
        "tool_shap",                              # re-run probe: tool-subset Shapley
    ],
    ModelKind.LLM: [
        "selfcheck_consistency",                  # text hallucination (black-box)
        "format_sensitivity",                     # MC position bias vs content-tracking
        "cot_faithfulness",                       # is the reasoning load-bearing?
        "calibration",                            # ECE / overconfidence vs labels
        "attention", "logit_lens",                # interpretability
        "layer_contrast",                          # DoLa/DeCo divergence signal
        "token_entropy", "logprob_entropy",
        "attention_sink", "attention_rollout",
        "prompt_contrast",                         # are failures prompt-repairable?
        "context_shap",                            # RAG context dependence
        "cka", "self_consistency", "verbalized_confidence",
    ],
}


def get_analyzer_catalog(model: "Model") -> dict[str, str]:
    """Return ``{name: description}`` for all analyzers compatible with *model*.

    Descriptions come from each analyzer class's ``description`` attribute or,
    if absent, the first non-empty line of its docstring.  Used by
    :class:`~evalvitals.eval_agent.probe_agent.ProbeAgent` to build the LLM
    selection prompt so the judge understands what each analyzer measures.
    """
    compatible = set(registry.analyzers.names_compatible_with(model))
    catalog: dict[str, str] = {}
    for name in sorted(compatible):
        cls = registry.analyzers.get(name)
        if cls is None:
            continue
        desc: str | None = getattr(cls, "description", None)
        if not desc and cls.__doc__:
            for line in cls.__doc__.strip().splitlines():
                line = line.strip()
                if line:
                    desc = line
                    break
        catalog[name] = desc or name
    return catalog


class StrategyProbe:
    """Selects analyzers appropriate for a given model.

    Args:
        priority_override: Replaces the built-in per-kind priority tables
            (keyed by ``ModelKind``).  Useful for domain-specific orderings.
    """

    def __init__(self, priority_override: dict[str, list[str]] | None = None) -> None:
        self._priority = priority_override or _PRIORITY

    def detect_kind(self, model: "Model", data: Any = None) -> ModelKind:
        """Infer VLM / AGENT / LLM from the data shape, capabilities and modalities.

        Trajectory-carrying *data* is the definitive agent signal and wins
        outright: a VLM that drove a tool loop should get the AGENT analyzer
        priority (loop detection, first-error attribution), not the VLM one.
        Without trajectories, image modality takes priority over TOOL_CALLS so
        that VLMs that merely *support* tool use (e.g. Qwen3-VL) are treated
        as VLMs, not agents.
        """
        if data is not None and _carries_trajectories(data):
            return ModelKind.AGENT
        if "image" in getattr(model, "modalities", frozenset({"text"})):
            return ModelKind.VLM
        if Capability.TOOL_CALLS in getattr(model, "capabilities", frozenset()):
            return ModelKind.AGENT
        return ModelKind.LLM

    def select(
        self,
        model: "Model",
        max_analyzers: int | None = None,
        hint_failure_modes: list[str] | None = None,
        data: Any = None,
    ) -> list[str]:
        """Return compatible analyzer names ranked by diagnostic priority.

        Args:
            model:              The model to analyse.
            max_analyzers:      If given, cap the returned list at this length.
            hint_failure_modes: Failure-mode tags from outstanding M3 hypotheses.
                                Analyzers that match a hint are promoted to the
                                front of the ranked list for focused follow-up.
            data:               The case batch about to be probed, when available.
                                Trajectory-carrying cases flip the ranking to the
                                AGENT priority list (see :meth:`detect_kind`).

        Returns:
            Ordered list of registered analyzer names.  Hint-matched items
            come first, then the standard priority-list items, then remaining
            compatible analyzers sorted alphabetically.
        """
        kind = self.detect_kind(model, data)
        compatible = set(registry.analyzers.names_compatible_with(model))
        priority = self._priority.get(kind, [])

        ranked = [name for name in priority if name in compatible]
        ranked += sorted(compatible - set(ranked))

        if hint_failure_modes:
            # Promote analyzers that map to outstanding failure modes, preserving
            # their relative order and avoiding duplicates.
            boosted = dict.fromkeys(
                a
                for mode in hint_failure_modes
                for a in _FAILURE_MODE_TO_ANALYZERS.get(mode.lower().replace(" ", "_"), [])
                if a in compatible
            )
            ranked = list(boosted) + [a for a in ranked if a not in boosted]

        if max_analyzers is not None:
            ranked = ranked[:max_analyzers]
        return ranked
