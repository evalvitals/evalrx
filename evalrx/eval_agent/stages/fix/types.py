"""Data model of the fix stage: candidates, validations, outcomes, context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalrx.eval_agent.stages.fix.tiers import FixTier

if TYPE_CHECKING:
    from evalrx.core.case import FailureCase
    from evalrx.eval_agent.run_context import Trial


@dataclass
class FixCandidate:
    """One proposed repair, compiled later to an ab_runner-style strategy.

    Attributes:
        tier:        Intervention space the candidate lives in.
        name:        Short identifier -- a slug, used to key trial folders and
                     to join validations back to their candidate.  It is not
                     shown to a reader; ``description`` is.
        description: One plain sentence saying what this candidate does, for
                     someone with no background in the field.  Supplied by the
                     judge where the judge invented the candidate, and by
                     :data:`~.descriptions.BUILTIN_DESCRIPTIONS` for the host's own.  Empty
                     is allowed and means "nothing readable to say" -- readers
                     are shown a blank rather than the slug in title case.
        kind:        ``"template"`` (L1) | ``"spec"`` (L2 declarative) |
                     ``"code"`` (agent-written pipeline) |
                     ``"registered_repair"`` (runtime-discovered executor) |
                     ``"visual_search"`` (L2 question-guided crop through an
                     opt-in backend).
        payload:     Kind-specific — template: ``{"prompt_template": ...}``;
                     spec: a :class:`~.tools.PipelineSpec` dict;
                     code: ``{"code": "<python source>"}``.
        source:      ``"judge"``, ``"cli:<provider>"`` or ``"default"``.
        predicate:   Optional ``(case) -> bool`` applicability gate.  When set,
                     the candidate is only applied to (and only judged on) the
                     cases it returns True for — a *conditional* fix.  When
                     ``None``, applicability is inferred structurally (a spec
                     that does not change a case's input is a no-op there).
        trial:       Optional :class:`~evalrx.eval_agent.run_context.Trial`
                     — when set (a ``RunContext`` is in play), this candidate's
                     code, the sandbox it ran in, and its record/result all
                     live under ``trial.root`` instead of being scattered
                     across ``tools/`` / ``workspace/`` / ``fixes/``.
    """

    tier: FixTier
    name: str
    payload: "dict[str, Any]"
    kind: str = "spec"
    source: str = "judge"
    description: str = ""
    predicate: "Callable[[FailureCase], bool] | None" = None
    trial: "Trial | None" = None


@dataclass
class FixValidation:
    """Paired validation of one candidate against the unmodified baseline.

    Safety (``n_broken``) and coverage are scoped to the cases the candidate is
    *applicable* to — a fix is not blamed for cases it never touched.  Cases
    whose baseline answer is unstable across repeats (sampling noise) are held
    out as ``n_unstable`` so a stochastic flip is not mistaken for a regression.
    """

    candidate: FixCandidate
    n_pairs: int = 0
    n_baseline_correct: int = 0  # among the paired, applicable cases
    n_candidate_correct: int = 0  # among the paired, applicable cases
    n_fixed: int = 0
    n_broken: int = 0
    fixed_cases: "list[str]" = field(default_factory=list)
    broken_cases: "list[str]" = field(default_factory=list)
    effect: "float | None" = None
    reject: bool = False
    fixed: bool = False
    summary: str = ""
    # Applicability + noise accounting (defects 1 & 2).
    n_applicable: int = 0  # cases the candidate actually touched
    coverage: "float | None" = None  # applicable FAILs / total FAILs in subset
    n_unstable: int = 0  # cases dropped as baseline-unstable (noise)
    # Failing cases a coded pipeline ALSO got right with the model frozen to
    # its recorded answers (PipelineCoder.frozen_model_control) — the repair was
    # the pipeline's own computation, not the model's, so they leave the paired
    # test. Baseline-correct cases are never dropped on this ground.
    n_model_independent: int = 0
    e_value: "float | None" = None
    # The bar ``e_value`` was judged against (1/alpha). Recorded beside the
    # number so a reader of "1.5 to 1" can see it needed 20 to 1, not guess.
    e_threshold: "float | None" = None
    # Coarse verdict (defect 4): fixed | partial | unsafe | regressed |
    # no_effect | not_executed | model_independent.  Richer than the boolean
    # ``fixed`` for triage.
    verdict: str = ""
    # Non-empty when the candidate never EXECUTED (sandbox crash, timeout,
    # bridge contract violation) — distinct from "executed and not effective".
    # Escalation must not treat these as evidence that the tier is exhausted.
    exec_error: str = ""
    # What the candidate actually PRODUCED, per case id (the aggregated /
    # final answer that was scored). Persisted by RunLogger as
    # ``outputs.jsonl`` beside the record, never inlined in the JSONL event.
    # Without this a "regressed" verdict cannot be told apart from a truncated
    # chain, a format slip, or a genuinely wrong answer after the fact.
    outputs: "dict[str, str]" = field(default_factory=dict)
    # Model calls that hit the decode cap while this candidate ran (delta of
    # the model's ``n_truncated`` counter when it exposes one; ``None`` when
    # the backend has no such telemetry). A candidate whose breaks coincide
    # with truncations was undone by its decoding budget, not by its idea.
    n_truncated: "int | None" = None
    # Which paired test decided: ``"mcnemar"`` (one sample per arm; McNemar +
    # Bernoulli-mixture e-value) or ``"paired_rates"`` (per-case pass rates
    # from k baseline / m candidate samples; betting e-value on the rate
    # difference). ``n_fixed``/``n_broken`` are always MODAL flips (baseline
    # rate < 0.5 -> candidate rate >= 0.5 and the reverse) so they read the
    # same under both; the effect/e-value under paired_rates weigh each case
    # by the size of the move, so an unstable case counts fractionally.
    noise_model: str = "mcnemar"
    baseline_rate: "float | None" = None   # mean per-case baseline pass rate (paired cases)
    candidate_rate: "float | None" = None  # mean per-case candidate pass rate
    n_baseline_samples: int = 1
    n_candidate_samples: int = 1
    e_value_regression: "float | None" = None  # mirror e-value (candidate WORSE), paired_rates only


@dataclass
class FixOutcome:
    """Everything the fix module did, plus the escalation recommendation.

    ``recommendation`` is ``None`` when a candidate validated; otherwise
    ``{"recommend_tier": "L3a", "reason": ...}`` — the caller decides whether
    to re-run with a higher ``max_tier`` (never automatic).
    """

    max_tier: FixTier
    routed: "list[dict[str, str]]" = field(default_factory=list)
    attempted: "list[FixValidation]" = field(default_factory=list)
    best: "FixValidation | None" = None
    fixed: bool = False
    recommendation: "dict[str, Any] | None" = None
    # Feedback edge back into diagnosis (defect 3): set when a candidate helps
    # one subset and hurts another — evidence the mechanism is subset-specific
    # and the hypothesis should be re-scoped, not that no fix exists.
    refine_signal: "dict[str, Any] | None" = None
    # Number of feedback-driven propose->validate rounds actually run (>= 1).
    repair_rounds: int = 0
    # Candidate names whose e-value survives e-BH FDR control across the whole
    # tested family (the multiplicity correction for best-of-N selection). A
    # candidate is eligible to be `best` only if it is BOTH individually `fixed`
    # AND in this set. Empty when no candidate carried an e-value.
    ebh_survivors: "list[str]" = field(default_factory=list)
    # In held-out mode candidates are authored/tuned on EXPLORE, then exactly
    # one frozen candidate is sent to CONFIRM.  Keep a compact audit trail of
    # that selection phase without mixing its statistics into the final gate.
    selection_attempted: "list[dict[str, Any]]" = field(default_factory=list)
    selected_on_explore: "str | None" = None
    # A skipped repair is not a failed repair.  This explicit state is used by
    # the report and Langfuse lifecycle event instead of inferring intent from
    # an empty ``attempted`` list.
    stage_status: str = "completed"
    skip_reason: str | None = None

    def to_dict(self) -> "dict[str, Any]":
        return {
            "max_tier": self.max_tier.label,
            "repair_rounds": self.repair_rounds,
            "routed": self.routed,
            "attempted": [
                {
                    "tier": v.candidate.tier.label,
                    "name": v.candidate.name,
                    "kind": v.candidate.kind,
                    "source": v.candidate.source,
                    "payload": v.candidate.payload,
                    # Self-contained attempt folder (see RunContext.new_trial) —
                    # None when no RunContext is in play (legacy flat layout).
                    "trial_root": (str(v.candidate.trial.root) if v.candidate.trial else None),
                    "n_pairs": v.n_pairs,
                    "n_baseline_correct": v.n_baseline_correct,
                    "n_candidate_correct": v.n_candidate_correct,
                    "n_fixed": v.n_fixed,
                    "n_broken": v.n_broken,
                    "fixed_cases": v.fixed_cases,
                    "broken_cases": v.broken_cases,
                    "effect": v.effect,
                    "reject": v.reject,
                    "fixed": v.fixed,
                    "summary": v.summary,
                    "n_applicable": v.n_applicable,
                    "coverage": v.coverage,
                    "n_unstable": v.n_unstable,
                    "n_model_independent": v.n_model_independent,
                    "e_value": v.e_value,
                    "e_threshold": v.e_threshold,
                    "verdict": v.verdict,
                    "n_truncated": v.n_truncated,
                    "noise_model": v.noise_model,
                    "baseline_rate": v.baseline_rate,
                    "candidate_rate": v.candidate_rate,
                    "n_baseline_samples": v.n_baseline_samples,
                    "n_candidate_samples": v.n_candidate_samples,
                    "e_value_regression": v.e_value_regression,
                    "outputs": dict(v.outputs),
                }
                for v in self.attempted
            ],
            "best": self.best.candidate.name if self.best else None,
            "fixed": self.fixed,
            "recommendation": self.recommendation,
            "refine_signal": self.refine_signal,
            "ebh_survivors": self.ebh_survivors,
            "selection_attempted": self.selection_attempted,
            "selected_on_explore": self.selected_on_explore,
            "stage_status": self.stage_status,
            "skip_reason": self.skip_reason,
        }


def truncated_count(model: Any) -> "int | None":
    """The model's ``n_truncated`` counter when it exposes one, else ``None``."""
    value = getattr(model, "n_truncated", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


@dataclass
class FixContext:
    """What the proposer (judge / coder) may see BESIDES the hypotheses.

    Everything is optional; the loop builds one from its report in
    ``run_fix``. The two rules that matter:

    * ``example_cases`` expose only prompt, baseline output, and PASS/FAIL.
      Expected/gold answers are always withheld. They are normally disjoint
      from the candidate-validation batch: the loop passes its EXPLORE split
      and scores the fix on CONFIRM.
    * ``evidence`` / ``refuted`` are read-only narrative: what M2/M4/explore
      established and what M5's intervention experiment knocked down. They
      steer *what* to propose; validation still decides *whether* it works.

    Attributes:
        example_cases:      Cases the proposer may see in full (see above).
        evidence:           Host-built summary of M2 statistics, M4 test
                            verdicts and exploratory notes.
        refuted:            Hypotheses an M5 experiment REFUTED (statement +
                            why), so the proposer does not build on them.
        scoring_note:       How outputs are scored / the expected final-answer
                            format (e.g. "last 'Answer:' line, '(D)' == 'D'").
        baseline_decoding:  ``{"max_tokens": .., "temperature": ..}`` the
                            baseline was generated with; the agent also uses
                            ``max_tokens`` as the floor a candidate may not go
                            below.
        task_note:          One-paragraph task / protocol description.
        hypotheses_note:    Status caveat printed right under the hypotheses
                            (e.g. "UNVERIFIED: M4 found no significant evidence
                            …") when the loop hands the fix unverified leads.
    """

    example_cases: "Any | None" = None
    evidence: str = ""
    refuted: "list[str]" = field(default_factory=list)
    scoring_note: str = ""
    baseline_decoding: "dict[str, Any]" = field(default_factory=dict)
    task_note: str = ""
    hypotheses_note: str = ""
