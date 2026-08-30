"""M1 — StrategyProbe: select which analyzers to run given a model and a batch.

Ranking composes per-SLOT priority lists rather than looking up one per-kind
list, because a model is not one kind.  ``LLM`` / ``VLM`` / ``ALM`` / ``AVLM``
are four subsets of ``{text, image, audio, video}``, and an enum over the
subsets grows as 2^n while composing over the members grows as n.

Which slots to compose over is decided by the BATCH, not by the model.  An omni
model declares every modality it can consume; evaluated on an audio benchmark it
still declares image, and ranking on the declaration put image analyzers at the
top of an audio run.  :func:`~evalvitals.core.case.probed_modalities` answers
what is actually under test, and the model's declaration is the fallback for a
batch that fills no media slot at all — no evidence, rather than evidence of
absence.

Usage::

    probe = StrategyProbe()
    slots = probe.routed_slots(model, cases)   # {"text", "audio"} for an ALM run
    kind  = probe.detect_kind(model)           # ModelKind.VLM / ALM / AVLM / AGENT / LLM
    names = probe.select(model, max_analyzers=4, data=cases)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable

from evalvitals.core.capability import Capability
from evalvitals.core.case import MEDIA_SLOTS, probed_modalities
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
    """A coarse label for a model's modality set.

    Kept for display, logging and the ``priority_override`` API — NOT for
    routing.  It is a lossy rendering of a set (an AVLM answering a text-only
    question is still AVLM here), which is exactly why :meth:`StrategyProbe.select`
    composes over :meth:`StrategyProbe.routed_slots` instead of branching on this.
    """

    VLM   = "vlm"    # image — hallucination + attention analyzers first
    ALM   = "alm"    # audio, no image — listening / grounding analyzers first
    AVLM  = "avlm"   # audio AND image — both, audio ranked first
    AGENT = "agent"  # trajectories or TOOL_CALLS — behavioral analyzers first
    LLM   = "llm"    # text only — interpretability analyzers first


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
    # Text-reasoning mechanisms (2026-08). The hygiene pair is listed under the
    # names an M3 hypothesis uses when it blames the harness rather than the model.
    "answer_extraction":         ["answer_extraction_audit"],
    "parse_failure":             ["answer_extraction_audit"],
    "truncation":                ["termination_audit"],
    "degenerate_repetition":     ["termination_audit"],
    "premature_termination":     ["termination_audit"],
    "arithmetic_error":          ["arith_audit"],
    "computation_slip":          ["arith_audit"],
    "chain_break":               ["arith_audit", "step_rollout_value"],
    "reasoning_break":           ["step_rollout_value", "arith_audit"],
    "overthinking":              ["cot_faithfulness", "step_rollout_value"],
    "self_correction_failure":   ["self_repair"],
    "knowledge_gap":             ["knowledge_reasoning_split"],
    "compositionality_gap":      ["knowledge_reasoning_split"],
    "selection_failure":         ["coverage_verification_gap"],
    "verification_gap":          ["coverage_verification_gap"],
    "brittleness":               ["perturbation_battery"],
    "surface_form_sensitivity":  ["perturbation_battery", "format_sensitivity"],
    "memorization":              ["contamination_score", "perturbation_battery"],
    "contamination":             ["contamination_score"],
    "semantic_uncertainty":      ["self_consistency"],
}

# Per-SLOT ordered priority: high → low diagnostic value for false attribution.
#
# Composed, not looked up: ``select`` concatenates the lists for the slots the
# batch actually fills (agent, then video, audio, image, then always text) and
# appends the remaining compatible analyzers alphabetically.  Adding a modality
# is one more entry here; it does not multiply the table the way a per-kind
# table does, and an AVLM needs no entry of its own — it is audio + image.
_SLOT_PRIORITY: dict[str, list[str]] = {
    "agent": [
        "loop_detect", "ignored_obs",             # behavioral heuristics
        "first_error_judge", "trajectory_rubric", # LLM-judge localisation + classification
        "counterfactual", "reliability_probe",    # re-run probes: step perturbation, pass@k
        "tool_shap",                              # re-run probe: tool-subset Shapley
    ],
    "video": [
        "modality_ablation",                      # is the video load-bearing at all?
        "relative_attn",                          # attention across frames
        "mm_shap",                                # per-slot reliance
    ],
    "audio": [
        # Grounding first, for the same reason the text list leads with hygiene:
        # an ALM that answers from the language prior alone produces failures that
        # mimic every mechanism below, and nothing else here would reveal it.
        "modality_ablation",                      # does replacing the audio change the answer?
        "mm_shap",                                # audio vs text reliance
        "answer_extraction_audit",                # is the FAIL label real or a parse miss?
        "termination_audit",                      # truncated / degenerate / gave up?
        "prompt_contrast",                        # are failures prompt-repairable?
        "selfcheck_consistency",                  # hallucinated content in the description
        "self_consistency", "logprob_entropy",
        "attention", "attention_sink",            # where does it attend among audio tokens?
        "verbalized_confidence", "calibration",
    ],
    "image": [
        "pope", "chair",                          # hallucination metrics
        "modality_ablation",                      # is the image load-bearing at all?
        "attention", "attention_rollout",          # where is the model looking?
        "attention_sink",                          # sink collapse?
        "prompt_contrast",                         # are failures prompt-repairable?
        "mm_shap",                                # text vs image reliance
        "logprob_entropy", "self_consistency",
    ],
    "text": [],   # filled below from the legacy LLM list
}

# Back-compat view: ``StrategyProbe(priority_override=...)`` and the docs are
# written against ModelKind keys.  Composition is the default path; an override
# still selects one list by kind, exactly as before.
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
        # Hygiene first: both produce confounds that mimic every mechanism
        # column below, so a finding read before them is not interpretable.
        "answer_extraction_audit",                # is the FAIL label real or a parse miss?
        "termination_audit",                      # truncated / degenerate / gave up?
        "arith_audit",                            # computation slip vs chain break (free)
        "selfcheck_consistency",                  # text hallucination (black-box)
        "format_sensitivity",                     # MC position bias vs content-tracking
        "cot_faithfulness",                       # is the reasoning load-bearing?
        "coverage_verification_gap",              # cannot solve vs cannot select
        "perturbation_battery",                   # invariance breaks / missing sensitivity
        "self_repair",                            # detect / correct / DAMAGE
        "knowledge_reasoning_split",              # missing fact vs broken composition
        "calibration",                            # ECE / overconfidence vs labels
        "attention", "logit_lens",                # interpretability
        "layer_contrast",                          # DoLa/DeCo divergence signal
        "token_entropy", "logprob_entropy",
        "attention_sink", "attention_rollout",
        "prompt_contrast",                         # are failures prompt-repairable?
        "context_shap",                            # RAG context dependence
        "cka", "self_consistency", "verbalized_confidence",
        "step_rollout_value",                      # where the chain broke (expensive)
        "contamination_score",                     # is the benchmark measuring recall?
    ],
}

# text IS the legacy LLM list: every model has a prompt, so this is the floor
# every composition starts from rather than a peer of the media slots.
_SLOT_PRIORITY["text"] = _PRIORITY[ModelKind.LLM]

# Slots each ModelKind stands for, in composition order. Only the back-compat
# ``_PRIORITY`` view and ``detect_kind`` read this; ``select`` uses the batch.
_KIND_SLOTS: dict[str, tuple[str, ...]] = {
    ModelKind.AGENT: ("agent", "text"),
    ModelKind.AVLM:  ("audio", "image", "text"),
    ModelKind.ALM:   ("audio", "text"),
    ModelKind.VLM:   ("image", "text"),
    ModelKind.LLM:   ("text",),
}


def _compose(slots: "Iterable[str]") -> "list[str]":
    """Concatenate the slot lists in ranking order, first occurrence wins.

    ``dict.fromkeys`` rather than a set: an analyzer listed under two slots (a
    VLM run also ranks ``mm_shap`` under image) must keep the position its
    highest-priority slot gave it, and a set would lose the order entirely.
    """
    ordered: list[str] = []
    for slot in slots:
        ordered.extend(_SLOT_PRIORITY.get(slot, ()))
    return list(dict.fromkeys(ordered))


# ALM / AVLM complete the back-compat view; they never had an entry because the
# enum had no member for them.
for _kind, _slots in _KIND_SLOTS.items():
    _PRIORITY.setdefault(_kind, _compose(_slots))


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
        #: ``None`` selects the composed-per-slot path. A caller-supplied table
        #: is keyed by ModelKind and is looked up, not composed — an override is
        #: an explicit statement about ordering and must not be reordered by the
        #: batch's modality slots.
        self._priority = priority_override

    @staticmethod
    def slot_starved(names: "set[str]", data: Any = None, model: "Model | None" = None) -> "set[str]":
        """Analyzers whose required modality slot the batch never fills.

        The registry match is an intersection against the MODEL's declaration,
        so one shared member (``"text"``, which every model has) is enough to
        make an image analyzer "compatible" with an audio run. It is then
        selected, runs, reads an empty slot and reports a number computed over
        nothing — the same silent-emptiness failure ``requires_trajectories``
        was added to close.

        A batch that fills NO media slot is ambiguous — media that was never
        persisted looks identical to media that was never there — so nothing is
        dropped, unless *model* is given and declares no media modality either.
        Then there is no ambiguity left to protect: a text-only model cannot have
        had an image, and keeping the gate off would offer a text run every
        media analyzer in the registry.

        ``data=None`` is no evidence at all; nothing is dropped.
        """
        if data is None:
            return set()
        filled = probed_modalities(data)
        if not (filled & set(MEDIA_SLOTS)):
            declared = set(getattr(model, "modalities", None) or ()) if model is not None else None
            if declared is None or (declared & set(MEDIA_SLOTS)):
                return set()
        starved: set[str] = set()
        for name in names:
            cls = registry.analyzers.get(name)
            required = set(getattr(cls, "requires_modalities", frozenset()) or ())
            if required and not (required & filled):
                starved.add(name)
        return starved

    # -- what a run is about -------------------------------------------
    @staticmethod
    def is_agent_run(model: "Model", data: Any = None) -> bool:
        """Whether this run should be ranked as an agent run.

        Orthogonal to modality: a VLM driving a tool loop is both. Trajectories
        in the batch are definitive; without them, declared TOOL_CALLS is the
        weaker fallback.
        """
        if data is not None and _carries_trajectories(data):
            return True
        return Capability.TOOL_CALLS in getattr(model, "capabilities", frozenset())

    @staticmethod
    def routed_slots(model: "Model", data: Any = None) -> set[str]:
        """The modality slots ranking should compose over.

        Normally the model's declared modalities intersected with the slots the
        batch actually fills, because both have to hold: an analyzer needs the
        model to accept the modality AND the data to contain it.

        Falls back to the model's declaration when the batch fills no media slot
        at all. A batch with no images is not evidence that images are not under
        test — it may be a text-only sample of a larger benchmark, or a batch
        whose media never got persisted — and treating "nothing observed" as
        "modality absent" is the same collapse the contract's rule 4 forbids.
        The two are recorded separately in ``AnalyzerSelection`` so a reader can
        see which path was taken.
        """
        declared = set(getattr(model, "modalities", frozenset({"text"})) or {"text"})
        probed = probed_modalities(data) if data is not None else {"text"}
        if not (probed & set(MEDIA_SLOTS)):
            return declared | {"text"}
        return (declared & probed) | {"text"}

    def detect_kind(self, model: "Model", data: Any = None) -> ModelKind:
        """Coarse label for the run — display and ``priority_override`` only.

        Trajectory-carrying *data* is the definitive agent signal and wins
        outright: a VLM that drove a tool loop should get the AGENT analyzer
        priority (loop detection, first-error attribution), not the VLM one.
        Without trajectories, media modality takes priority over TOOL_CALLS so
        that VLMs that merely *support* tool use (e.g. Qwen3-VL) are treated
        as VLMs, not agents.

        Reads the MODEL's declaration, not the batch — the label answers "what
        is this model", and narrowing it by the data would make an omni model
        report a different identity per benchmark. :meth:`routed_slots` is the
        one that narrows, and it is what ranking actually uses.
        """
        if data is not None and _carries_trajectories(data):
            return ModelKind.AGENT
        modalities = set(getattr(model, "modalities", frozenset({"text"})) or {"text"})
        has_image = bool(modalities & {"image", "video"})
        has_audio = "audio" in modalities
        if has_image and has_audio:
            return ModelKind.AVLM
        if has_image:
            return ModelKind.VLM
        if has_audio:
            return ModelKind.ALM
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
        compatible = set(registry.analyzers.names_compatible_with(model))
        compatible -= self.slot_starved(compatible, data, model)

        if self._priority is not None:
            # A caller-supplied table is an allowlist as well as an ordering.
            # In particular, benchmark ``pinned`` mode must not silently fill
            # an unsupported pinned slot with an unrelated alphabetical probe.
            priority = self._priority.get(self.detect_kind(model, data), [])
            ranked = [name for name in priority if name in compatible]
        else:
            slots = self.routed_slots(model, data)
            media = [s for s in ("video", "audio", "image") if s in slots]
            # Trajectories in the batch are definitive and lead outright. Merely
            # DECLARING tool support does not: a VLM that supports tools (Qwen3-VL)
            # is still being asked a vision question, so media outranks the
            # capability — the capability only leads when no media slot is in play.
            agent_leads = (data is not None and _carries_trajectories(data)) or (
                not media and self.is_agent_run(model, data)
            )
            priority = _compose((["agent"] if agent_leads else []) + media + ["text"])
            ranked = [name for name in priority if name in compatible]
            ranked += sorted(compatible - set(ranked))

        if hint_failure_modes:
            # Promote analyzers that map to outstanding failure modes, preserving
            # their relative order and avoiding duplicates.
            boosted = dict.fromkeys(
                a
                for mode in hint_failure_modes
                for a in _FAILURE_MODE_TO_ANALYZERS.get(mode.lower().replace(" ", "_"), [])
                if a in compatible and (self._priority is None or a in ranked)
            )
            ranked = list(boosted) + [a for a in ranked if a not in boosted]

        if max_analyzers is not None:
            ranked = ranked[:max_analyzers]
        return ranked
