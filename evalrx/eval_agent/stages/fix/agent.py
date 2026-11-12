"""Fix module — tiered, validated repair attempts after the diagnosis loop.

Design (intervention-space tiers, see :mod:`.tiers`): the allowed tier is
an **input** (default L2).  The agent proposes candidate fixes inside the
allowed tiers, compiles every candidate to the same shape — a per-case success
function, exactly :mod:`~evalrx.eval_agent.ab_runner`'s *strategy*
contract — and validates each against the unmodified baseline with the paired
machinery from :mod:`evalrx.stats` (McNemar + e-value, never a bare p).

There is **no automatic tier escalation**: when no candidate within the allowed
tier validates, the outcome carries a *recommendation* to raise the tier,
routed from the verified hypotheses' mechanisms (:func:`route_min_tier`).

What the agent *does* retry within the allowed tier is the **proposal** itself:
``max_repair_rounds`` (default 1) lets it run up to N propose→validate rounds.
After a round in which nothing validates, the per-candidate results (how many
cases each fixed / broke, net effect, or an execution error — see
:meth:`FixAgent._format_prior`) are summarised and fed back to the judge/coder,
which then proposes *different* strategies — never re-running an identical
candidate (:meth:`FixAgent._signature`), never raising the tier.  The loop
stops as soon as a candidate validates, or when a round yields no new
candidate.

Executors by tier:

* **L0** — runtime configuration: bounded decoding controls such as
  ``max_tokens``. Only proposed from explicit execution telemetry, never from
  an LLM's guess about a response being short.
* **L1** — prompt transforms (judge-proposed templates).
* **L2 declarative** — catalog-tool pipelines (:mod:`.tools`): cheap,
  deterministic, validated first.
* **L2 coded** — the coding agent (CLI agent first, judge fallback) writes a
  brand-new pipeline as Python: multiple model calls per case, branching on
  intermediate outputs — only the model itself is unchanged.  The code runs
  sandboxed with bridged model access (:mod:`.pipeline`); labels and
  rubrics never reach it, so it cannot cheat by echoing gold answers.
* **L3a** — internals read (:mod:`.internals`): contrastive decoders read
  and combine logits/attention from paired forward passes. Coded pipelines
  are L3a only when their source actually calls the bridged
  ``model_attend()``; merely exposing that optional API does not promote an
  otherwise black-box L2 scaffold.
* **L3b** — internals write (:mod:`.internals`): pre-audited intervention
  primitives (v1: visual embedding boost via a forward hook) — the judge
  selects and parameterises; never free codegen against the model handle.
* **L4** — parameter space (:mod:`.internals`): the judge writes a
  :class:`~.internals.FinetuneSpec` recipe. v1's executor
  (:func:`~.internals.run_lora_repair`) runs exactly one shape —
  ``method="lora"`` on ``target="llm"``, trained on ``finetune_pool`` (a
  caller-supplied diagnosis-only :class:`CaseBatch`, never the validation
  split) and validated through this same paired machinery. Every other
  recipe shape, or no ``finetune_pool`` at all, is recorded but not
  executed, so the escalation decision always has something concrete to
  act on either way.

A *fixed* verdict means: the paired test rejects with positive net effect —
the candidate repairs significantly more cases than it breaks. With one
sample per arm that is McNemar + the Bernoulli-mixture e-value on the
discordant pairs; with ``baseline_repeats``/``candidate_repeats`` > 1 each
case is a per-arm PASS RATE and the test is the betting e-value on the paired
rate differences (:func:`evalrx.stats.compare_paired_rates`) — a
stochastic model's flaky cases are weighed by how far the candidate moves
them, neither dropped nor mistaken for repairs.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from evalrx.analyzers.perturbation.prompt_contrast import _default_score
from evalrx.eval_agent.hypothesis import hypothesis_id
from evalrx.eval_agent.stages.fix.codegen import PipelineCoder
from evalrx.eval_agent.stages.fix.proposal import (
    MAX_JUDGE_CANDIDATES,
    FixProposer,
    enforce_generation_floor,
)
from evalrx.eval_agent.stages.fix.recorder import FixRecorder
from evalrx.eval_agent.stages.fix.tiers import FixTier, parse_tier, route_min_tier
from evalrx.eval_agent.stages.fix.types import (
    FixCandidate,
    FixContext,
    FixOutcome,
    FixValidation,
)
from evalrx.eval_agent.stages.fix.validation import Baseline, CandidateValidator
from evalrx.eval_agent.stages.fix.verdict import (
    ebh_survivors,
    no_fix_recommendation,
    recommend,
    refine_signal,
)

if TYPE_CHECKING:
    from evalrx.agent_runtime.cli_types import CliAgentConfig
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model
    from evalrx.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class FixAgent:
    """Propose and validate tiered fixes for the loop's verified hypotheses.

    Args:
        judge:            LLM proposing candidates (deterministic defaults
                          when ``None`` or unparseable).
        max_tier:         Highest allowed intervention tier (runtime config,
                          prompt, scaffold, …; default L2).
        score_fn:         ``(case, output) -> bool | None``; defaults to the
                          rubric scorer shared with prompt_contrast.
        run_logger:       Optional RunLogger — records the outcome as a
                          ``fix`` event and coded pipelines as ``tool_codegen``.
        cli_config:       CLI coding-agent config; when set (provider != "llm")
                          it writes the L2 coded pipeline (judge fallback).
        allow_codegen:    Gate for the L2 coded-pipeline path (sandboxed,
                          bridged model access).  Declarative candidates do
                          not depend on this.
        sandbox:          Workdir provider for coded pipelines (fresh temp
                          dir when ``None``).
        exec_timeout_sec: Wall-clock limit for one coded-pipeline session
                          (includes the bridged model calls).
        max_validation_cases: When > 0 and the batch is larger, validate every
                          candidate on a label-stratified subset of this size
                          (all-FAIL-first; deterministic).  Every candidate
                          validation costs >= one model call per case, so an
                          unbounded batch makes coded pipelines time out.
        baseline_repeats: Samples per case for the unmodified baseline
                          (default 1 = the frozen ``observed`` output). With
                          ``k > 1`` the frozen sample counts as one and ``k-1``
                          fresh samples are drawn, giving each case a baseline
                          PASS RATE; the paired test then runs on per-case rate
                          differences (``noise_model="paired_rates"``, betting
                          e-value) instead of one-sample McNemar. A case whose
                          baseline flips across samples is *unstable* — it is
                          REPORTED and weighed by how much the candidate moves
                          its rate, no longer dropped (dropping removed exactly
                          the cases a variance-reduction scaffold repairs;
                          keeping one sample per arm let sampling noise pose as
                          fixes and breaks — the 2B runs' one-break-short
                          verdicts).
        candidate_repeats: Passes per case for a template / spec candidate
                          (default 1). Coded pipelines, internals primitives
                          and fine-tune recipes always run once. Any value > 1
                          also switches the test to paired rates.
        alpha:            Significance level for the e-value gate (default 0.05;
                          rejects when e >= 1/alpha).  Also sets the power
                          ceiling used to flag underpowered-by-design runs.
        run_context:      Optional :class:`~evalrx.eval_agent.run_context.RunContext`.
                          When set (directly, or inherited from ``run_logger``),
                          coded-pipeline sandbox workdirs are allocated durably
                          under ``workspace/`` instead of an ephemeral temp dir.
        max_repair_rounds: Number of feedback-driven propose->validate rounds
                          (default 1 = single-shot).  When > 1 and a round
                          validates nothing, the failed candidates' results are
                          fed back to the judge/coder (see
                          :meth:`_format_prior`), which proposes *different*
                          strategies within the SAME tier (no tier escalation).
                          Stops early on the first validated fix or when a
                          round adds no new candidate.
        max_judge_candidates: Maximum candidates to request and validate for
                          each judge-proposed tier. Defaults to three. Lower
                          this for a bounded screening experiment; doing so is
                          also reflected in the multiplicity correction.
        finetune_pool:    Diagnosis-split-only cases available for L4's LoRA
                          executor to train on — MUST be disjoint from
                          whatever batch is passed to
                          :meth:`propose_and_validate` (never the selection
                          or confirmation split; training and validating on
                          the same cases is leakage, not a fix). ``None``
                          (default) means L4 candidates are recorded but not
                          executed, same as before this executor existed.
                          See :func:`~.internals.run_lora_repair`.
        verbose:          When ``True``, print this agent's own tier-routing /
                          candidate-generation / validation-verdict narration
                          to stdout (``evalrx.enable_console_logging()``).
                          Redundant when the owning ``VLDiagnoseLoop`` was
                          already constructed with ``verbose=True``.
        baseline_generation_kwargs: The decoding controls the BASELINE was
                          generated with (``{"max_tokens": 4096,
                          "temperature": 0.6}``). ``max_tokens`` becomes the
                          floor for every candidate: a judge-proposed
                          ``max_tokens`` below it is raised to it (measured:
                          on qwen3.5-2b/bbh_tracking7 the judge set 900 where
                          the baseline had 4096 and 30% of baseline outputs
                          already exceed 900 tokens — the candidate "regressed"
                          by truncation, not by its idea). When ``None`` the
                          floor is inferred from ``model.max_tokens`` or the
                          cases' ``metadata["generation_config"]``, else no
                          floor is enforced. Both values are also shown to the
                          proposer.
        concurrency:      Threads used to run a declarative candidate over
                          the validation batch, and how many bridged model
                          calls the coded-pipeline host services at once
                          (default 1 = serial, unchanged behaviour).
        scoring_note:     Free-text description of how outputs are scored /
                          the expected final-answer format, shown to the
                          proposer (a per-call ``FixContext.scoring_note``
                          overrides it).
        floor_candidates: Names of built-in default candidates that are ALWAYS
                          in the tested family for a text-only batch, on top of
                          whatever the judge proposes (default:
                          ``("self_consistency_5",)`` — 5 samples, majority
                          vote on the extracted final answer). Before this the
                          defaults were a *fallback* used only when the judge
                          returned nothing, so the most basic variance-reduction
                          scaffold was never even tested next to the judge's
                          ideas. Empty tuple / ``None`` disables the floor.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        max_tier: "str | FixTier" = FixTier.L2_SCAFFOLD,
        min_tier: "str | FixTier | None" = None,
        score_fn: "Callable[[FailureCase, str], Optional[bool]] | None" = None,
        run_logger: "Any | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        allow_codegen: bool = True,
        sandbox: "Any | None" = None,
        exec_timeout_sec: int = 600,
        max_validation_cases: int = 0,
        baseline_repeats: int = 1,
        candidate_repeats: int = 1,
        alpha: float = 0.05,
        run_context: "Any | None" = None,
        max_repair_rounds: int = 1,
        max_judge_candidates: int = MAX_JUDGE_CANDIDATES,
        allow_adapted_paper_methods: bool = False,
        paper_methods_only: bool = False,
        candidate_allowlist: "Iterable[str] | None" = None,
        finetune_pool: "CaseBatch | None" = None,
        verbose: bool = False,
        baseline_generation_kwargs: "dict[str, Any] | None" = None,
        concurrency: int = 1,
        scoring_note: str = "",
        floor_candidates: "Iterable[str] | None" = ("self_consistency_5",),
        prewritten_code: str = "",
        candidate_model: "Model | None" = None,
        deployed_spec: "dict[str, Any] | None" = None,
        progress_callback: "Callable[[dict[str, Any]], None] | None" = None,
    ) -> None:
        if verbose:
            # Surfaces this module's own logger.info()/.warning() calls (tier
            # routing, candidate generation, validation verdicts) — see
            # VLDiagnoseLoop's verbose= for the same convenience one layer up.
            from evalrx.logging_utils import enable_console_logging

            enable_console_logging()

        self._progress_callback = progress_callback
        self.max_tier = parse_tier(max_tier)
        # Normally a fixed-ceiling run considers every cheaper tier. The loop's
        # auto-escalation path sets this transiently so each ladder station
        # evaluates only the newly opened intervention space.
        self.min_tier = parse_tier(min_tier) if min_tier is not None else None
        self.alpha = float(alpha)
        self.max_validation_cases = max_validation_cases
        self._allow_codegen = allow_codegen
        self._allow_adapted_paper_methods = bool(allow_adapted_paper_methods)
        # Winner-as-new-baseline (spec deploys, recursive rounds): when the
        # caller runs the whole loop against a DEPLOYED pipeline handle
        # (tools.SpecPipelineModel), `candidate_model` is the raw model —
        # baseline arms keep measuring the deployed pipeline while proposals
        # and candidate arms run on the raw handle, so every candidate is a
        # full REPLACEMENT pipeline paired against the deployed one.
        # `deployed_spec` (the deployed PipelineSpec as a dict) is shown to
        # the proposer so its candidates are edits of a known incumbent
        # rather than blind wraps.
        self._candidate_model = candidate_model
        # When set (directly, or via the RunLogger's bound RunContext), every
        # attempt gets a durable trial folder under the run directory.
        self._run_context = run_context or getattr(run_logger, "_context", None)
        # Per-run state, set by the public entry points.
        self._baseline_rates: "dict[str, Optional[float]]" = {}
        self._max_tokens_floor: "int | None" = None

        score = score_fn or _default_score
        concurrency = max(1, int(concurrency))
        self.recorder = FixRecorder(run_logger)
        self.coder = PipelineCoder(
            judge=judge,
            cli_config=cli_config,
            allow_codegen=allow_codegen,
            score=score,
            recorder=self.recorder,
            run_context=self._run_context,
            sandbox=sandbox,
            exec_timeout_sec=exec_timeout_sec,
            concurrency=concurrency,
            max_repair_rounds=max(1, int(max_repair_rounds)),
            prewritten_code=prewritten_code,
        )
        self.proposer = FixProposer(
            judge=judge,
            recorder=self.recorder,
            coder=self.coder,
            max_judge_candidates=max_judge_candidates,
            allow_adapted_paper_methods=allow_adapted_paper_methods,
            paper_methods_only=paper_methods_only,
            candidate_allowlist=candidate_allowlist,
            baseline_generation_kwargs=baseline_generation_kwargs,
            scoring_note=scoring_note,
            floor_candidates=floor_candidates,
            deployed_spec=deployed_spec,
        )
        self.validator = CandidateValidator(
            score=score,
            recorder=self.recorder,
            coder=self.coder,
            progress=self._progress,
            concurrency=concurrency,
            baseline_repeats=max(1, int(baseline_repeats)),
            candidate_repeats=max(1, int(candidate_repeats)),
            finetune_pool=finetune_pool,
        )

    @property
    def run_logger(self) -> "Any | None":
        """The run logger fix events go to (callers may swap it between runs)."""
        return self.recorder.run_logger

    @run_logger.setter
    def run_logger(self, value: "Any | None") -> None:
        self.recorder.run_logger = value

    @property
    def max_repair_rounds(self) -> int:
        """Feedback-driven propose->validate rounds per call (see the class docstring)."""
        return self.coder.max_repair_rounds

    @max_repair_rounds.setter
    def max_repair_rounds(self, value: int) -> None:
        self.coder.max_repair_rounds = max(1, int(value))

    @property
    def max_judge_candidates(self) -> int:
        """Candidates requested and validated per judge-proposed tier."""
        return self.proposer.max_judge_candidates

    @max_judge_candidates.setter
    def max_judge_candidates(self, value: int) -> None:
        self.proposer.max_judge_candidates = max(1, int(value))

    @property
    def codegen_available(self) -> bool:
        """True when the L2 coded-pipeline path has a code-writing backend."""
        return self.coder.available

    # -- public ---------------------------------------------------------

    def propose_and_validate(
        self,
        model: "Model",
        data: "CaseBatch",
        hypotheses: "list[Hypothesis]",
        prior_attempts: "list[FixValidation] | None" = None,
        context: "FixContext | None" = None,
        proposal_data: "CaseBatch | None" = None,
        *,
        max_tier: "FixTier | str | None" = None,
        min_tier: "FixTier | str | None" = None,
        record: bool = True,
    ) -> FixOutcome:
        """Generate candidates within the allowed tiers, validate, recommend.

        *context* (optional :class:`FixContext`) is what the proposer sees
        besides the hypotheses — full example cases from a DISJOINT split,
        the M2/M4/explore evidence, M5-refuted hypotheses, the scoring rule
        and the baseline decoding budget.  ``proposal_data``, when supplied,
        is the discovery partition available to the repair author; ``data``
        remains untouched confirmation data and permits only one round.

        ``max_tier`` / ``min_tier`` narrow the tier window for this call only
        (default: the agent's own).  ``record=False`` writes nothing to the
        run logger, for a caller that records one combined outcome itself.
        """
        window = (
            parse_tier(max_tier) if max_tier is not None else self.max_tier,
            parse_tier(min_tier) if min_tier is not None else self.min_tier,
        )
        if record:
            return self._search(model, data, hypotheses, prior_attempts, context,
                                proposal_data, *window)
        with self.recorder.muted():
            return self._search(model, data, hypotheses, prior_attempts, context,
                                proposal_data, *window)

    def _search(
        self,
        model: "Model",
        data: "CaseBatch",
        hypotheses: "list[Hypothesis]",
        prior_attempts: "list[FixValidation] | None",
        context: "FixContext | None",
        proposal_data: "CaseBatch | None",
        max_tier: FixTier,
        min_tier: "FixTier | None",
    ) -> FixOutcome:
        """:meth:`propose_and_validate` within an explicit tier window."""
        outcome = FixOutcome(max_tier=max_tier)
        # Winner-as-new-baseline: `model` as handed in is what the baseline
        # arm must measure (the deployed pipeline, when one is configured);
        # proposals and candidate arms run on the raw handle so a candidate
        # REPLACES the deployed pipeline instead of nesting inside it.
        baseline_model = model
        if self._candidate_model is not None:
            model = self._candidate_model
        self._max_tokens_floor = self.proposer.resolve_max_tokens_floor(model, data)
        routed_tiers: "list[FixTier]" = []
        for h in hypotheses:
            tier, why = route_min_tier(h)
            routed_tiers.append(tier)
            outcome.routed.append(
                {
                    # Full-statement hash, NOT derived from the truncated
                    # "hypothesis" string below — must match the id the same
                    # Hypothesis object got in its M3 log entry.
                    "hypothesis_id": hypothesis_id(h),
                    "hypothesis": getattr(h, "statement", str(h))[:160],
                    "min_tier": tier.label,
                    "rationale": why,
                }
            )

        data = self._validation_subset(data)
        authoring_data = proposal_data if proposal_data is not None else data
        baseline = self._baseline(baseline_model, data)
        if not any(v is not None for v in baseline.scores.values()):
            logger.warning("FixAgent: no scorable case (no rubrics); nothing to validate")
            outcome.recommendation = self.recommend(
                routed_tiers,
                model=model,
                data=data,
                reason_prefix="no case carries a scoring rubric, so no fix can be validated",
                max_tier=max_tier,
            )
            self.recorder.outcome(outcome)
            return outcome

        # With zero repairable failure mass in the fresh baseline no candidate
        # can ever validate (n_fixed stays 0, the e-value ceiling is 1 <
        # 1/alpha), yet the search would still spend its whole judge/model
        # budget — a saturated live cell burned an hour on 15 candidates this
        # way. The paired test uses the FRESH baseline, not the stale case
        # labels, so the gate must too. Tiers above L2 are exempt: an L3a
        # candidate pairs against its own matched sampling control
        # (``baseline_executor``), so a clean greedy baseline does not bound
        # its e-value.
        repairable = True
        if max_tier <= FixTier.L2_SCAFFOLD:
            if self.validator.rates_mode:
                repairable = any(
                    r is not None and r < 1.0 for r in baseline.rates.values()
                )
            else:
                repairable = any(v is False for v in baseline.scores.values())
        if not repairable:
            n_scorable = sum(1 for v in baseline.scores.values() if v is not None)
            logger.warning(
                "FixAgent: all %d scorable case(s) pass the fresh baseline — no candidate "
                "can be validated here; skipping candidate proposal",
                n_scorable,
            )
            outcome.recommendation = {
                "recommend_tier": None,
                "action": "gather_more_failures",
                "reason": (
                    f"nothing to repair: all {n_scorable} scorable validation case(s) pass "
                    f"the fresh baseline, so even a perfect candidate tops out at e=1.0 "
                    f"(< {1.0 / self.alpha:.0f} needed) — collect failing cases instead of "
                    "spending the candidate budget."
                ),
            }
            self.recorder.outcome(outcome)
            return outcome

        # Feedback-driven repair rounds: propose -> validate; if nothing
        # validates, summarise the failures (this call's own attempts, PLUS
        # any prior_attempts carried over from an earlier escalation tier) and
        # ask for DIFFERENT candidates within the same tier (never escalating).
        # Stop on first validated fix or when a round adds no new candidate.
        seen: "set[tuple[str, str, str]]" = set()
        round_limit = 1 if proposal_data is not None else self.max_repair_rounds
        for round_idx in range(round_limit):
            self._progress("proposal_started", completed=round_idx, total=round_limit)
            combined_prior = list(prior_attempts or []) + outcome.attempted
            prior_text = (
                self._format_prior(combined_prior, authoring_data)
                if combined_prior
                else ""
            )
            prior_names: "frozenset[str]" = frozenset(v.candidate.name for v in combined_prior)
            new_candidates: "list[FixCandidate]" = []
            for candidate in self._propose(
                hypotheses, authoring_data, model, prior_text, prior_names, context=context,
                window=(max_tier, min_tier),
            ):
                sig = self._signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                new_candidates.append(candidate)
            if not new_candidates:
                logger.info(
                    "FixAgent: repair round %d produced no NEW candidate; stopping", round_idx + 1
                )
                break
            round_fixed = False
            for candidate in new_candidates:
                # Coded candidates already have a trial (allocated at proposal
                # time, since the CLI agent needs a workdir immediately).
                # Declarative candidates (template/spec/primitive) get one
                # here — AFTER dedup — so a repeated default proposal never
                # burns a folder for nothing.
                if candidate.trial is None and self._run_context is not None:
                    candidate.trial = self._run_context.new_trial(
                        "fixes", f"{candidate.tier.label}_{candidate.name}"
                    )
                logger.info(
                    "FixAgent: validating tier=%s candidate=%s kind=%s source=%s",
                    candidate.tier.label,
                    candidate.name,
                    candidate.kind,
                    candidate.source,
                )
                validation = self._validate(candidate, model, data, baseline)
                outcome.attempted.append(validation)
                logger.info(
                    "FixAgent: result tier=%s candidate=%s verdict=%s "
                    "effect=%s fixed=%d broken=%d pairs=%d",
                    candidate.tier.label,
                    candidate.name,
                    validation.verdict,
                    (
                        "n/a"
                        if validation.effect is None
                        else f"{validation.effect:+.4f}"
                    ),
                    validation.n_fixed,
                    validation.n_broken,
                    validation.n_pairs,
                )
                round_fixed = round_fixed or validation.fixed
            outcome.repair_rounds = round_idx + 1
            if round_fixed:
                break
            if round_idx + 1 < round_limit:
                logger.info(
                    "FixAgent: repair round %d validated no fix; feeding "
                    "%d failed attempt(s) back for round %d",
                    round_idx + 1,
                    len(new_candidates),
                    round_idx + 2,
                )

        outcome.refine_signal = refine_signal(outcome.attempted, data)
        # Multiplicity control over the candidate family (best-of-N): every
        # candidate cleared only its OWN paired gate (e >= 1/alpha) against the
        # SAME baseline, so picking the max is a multiple-comparisons hazard.
        # e-BH composes the family's e-values into one FDR guarantee; a candidate
        # is a winner only if it is BOTH individually `fixed` AND an e-BH survivor.
        tested = [v for v in outcome.attempted if v.e_value is not None]
        survivors = ebh_survivors(tested, self.alpha)
        outcome.ebh_survivors = sorted(v.candidate.name for v in tested if id(v) in survivors)
        winners = [v for v in outcome.attempted if v.fixed and id(v) in survivors]
        if winners:
            outcome.best = max(winners, key=lambda v: (v.effect or 0.0, -v.n_broken))
            outcome.fixed = True
        else:
            culled = [v for v in outcome.attempted if v.fixed and id(v) not in survivors]
            if culled:
                names = ", ".join(sorted(v.candidate.name for v in culled))
                outcome.recommendation = {
                    "recommend_tier": max_tier.label,
                    "reason": (
                        f"{len(culled)} candidate(s) cleared their own gate "
                        f"({names}) but did NOT survive e-BH FDR across the "
                        f"{len(tested)}-candidate family — best-of-N multiplicity, "
                        "not a validated fix. Gather more failing cases (more "
                        "power per candidate) or propose fewer, stronger candidates."
                    ),
                }
            else:
                outcome.recommendation = self.no_fix_recommendation(
                    outcome.attempted, routed_tiers, data, model, max_tier=max_tier
                )
        self.recorder.outcome(outcome)
        return outcome

    def validate_candidate(
        self,
        model: "Model",
        data: "CaseBatch",
        candidate: FixCandidate,
    ) -> FixValidation:
        """Confirm one pre-selected candidate on an untouched batch.

        Candidate selection must happen before this method is called. Unlike
        :meth:`propose_and_validate`, this method never consults the judge and
        validates exactly one pre-registered candidate, so no best-of-N
        selection correction is needed on the confirmation split.  It always
        uses the whole batch: ``max_validation_cases`` budgets the search, and
        dropping untouched confirmation cases would only cost power.
        """
        # Same handle split as propose_and_validate: baseline arm = deployed
        # pipeline (when configured), candidate arm = raw model.
        baseline_model = model
        if self._candidate_model is not None:
            model = self._candidate_model
        self._max_tokens_floor = self.proposer.resolve_max_tokens_floor(model, data)
        enforce_generation_floor([candidate], self._max_tokens_floor)
        return self._validate(candidate, model, data, self._baseline(baseline_model, data))

    def recommend(
        self,
        routed: "list[FixTier]",
        *,
        model: "Model | None" = None,
        data: "CaseBatch | None" = None,
        reason_prefix: str = "",
        max_tier: "FixTier | None" = None,
    ) -> "dict[str, Any] | None":
        """The next tier to try above the ceiling, skipping unsupported ones.

        The ceiling is *max_tier* (default: this agent's).  See
        :func:`~.verdict.recommend`.
        """
        return recommend(
            routed, max_tier if max_tier is not None else self.max_tier,
            model=model, data=data, reason_prefix=reason_prefix,
            allow_adapted=self._allow_adapted_paper_methods,
            allow_codegen=self._allow_codegen,
        )

    def no_fix_recommendation(
        self,
        attempted: "list[FixValidation]",
        routed_tiers: "list[FixTier]",
        data: "CaseBatch",
        model: "Model",
        *,
        max_tier: "FixTier | None" = None,
    ) -> "dict[str, Any] | None":
        """What to do when none of *attempted* validated, under this agent's settings.

        The ceiling is *max_tier* (default: this agent's).  See
        :func:`~.verdict.no_fix_recommendation`.
        """
        return no_fix_recommendation(
            attempted, routed_tiers, data, model,
            max_tier=max_tier if max_tier is not None else self.max_tier,
            alpha=self.alpha,
            baseline_rates=self._baseline_rates,
            paired_rates=self.validator.rates_mode,
            allow_adapted=self._allow_adapted_paper_methods,
            allow_codegen=self._allow_codegen,
        )

    # -- no-fix recommendation (escalate vs. gather data vs. retry exec) ----

    def _validation_subset(self, data: "CaseBatch") -> "CaseBatch":
        """Label-stratified, deterministic subset for candidate validation."""
        cap = self.max_validation_cases
        if not cap or len(data) <= cap:
            return data
        import random as _random

        from evalrx.core.case import CaseBatch, Label

        rng = _random.Random(0)
        fails = [c for c in data if c.label == Label.FAIL]
        passes = [c for c in data if c.label != Label.FAIL]
        rng.shuffle(fails)
        rng.shuffle(passes)
        n_fail = min(len(fails), max(cap // 2, cap - len(passes)))
        keep = fails[:n_fail] + passes[: cap - n_fail]
        logger.info(
            "FixAgent: validating on %d/%d cases (%d fail, %d pass)",
            len(keep),
            len(data),
            n_fail,
            len(keep) - n_fail,
        )
        return CaseBatch(keep)

    # -- candidate generation --------------------------------------------

    # -- proposer context ---------------------------------------------------

    @staticmethod
    def _signature(candidate: FixCandidate) -> "tuple[str, str, str]":
        """Identity of a candidate, to skip re-validating an identical one.

        Coded pipelines carry fresh source each round, so they never collide;
        templates / specs / primitives dedup on their defining payload. The
        candidate's ``name`` is always part of the signature, so separately
        registered experiments never collapse merely because their defaults
        happen to match.
        """
        p = candidate.payload
        if candidate.kind == "template":
            key = str(p.get("prompt_template", ""))
        elif candidate.kind == "primitive":
            key = json.dumps(
                {"primitive": p.get("primitive"), "params": p.get("params")},
                sort_keys=True,
                default=str,
            )
        elif candidate.kind == "code":
            key = str(p.get("code", ""))
        else:
            key = json.dumps(
                {k: v for k, v in p.items() if k != "exec_error"}, sort_keys=True, default=str
            )
        return (candidate.kind, candidate.name, key)

    @staticmethod
    def _format_prior(
        attempts: "list[FixValidation]", data: "CaseBatch | None" = None
    ) -> str:
        """Format failed prior attempts as a context block for judge prompts.

        Only aggregate outcomes and the attempted implementation are exposed.
        Raw IDs, prompts, and answers would let a later round memorize EXPLORE
        cases, so they never enter repair feedback.
        """
        del data
        items = []
        implementations = []
        heterogeneous = []
        for v in attempts:
            c = v.candidate
            if c.kind == "finetune_spec":
                continue
            effect = f"effect={v.effect:+.2f}" if v.effect is not None else "did not execute"
            trunc = (
                f"; {v.n_truncated} model call(s) hit the decode cap — its breaks are "
                "truncation, not the idea: give the model MORE room, never less"
                if v.n_truncated else ""
            )
            items.append(
                f"- [{c.tier.label}/{c.kind}] {c.name}: "
                f"{v.n_fixed} fixed / {v.n_broken} broken ({effect}{trunc})"
            )
            code = c.payload.get("code") if c.kind == "code" else None
            if isinstance(code, str) and code.strip():
                implementations.append(
                    f"PREVIOUS IMPLEMENTATION ({c.name}):\n{code[:3000]}"
                )
            if v.n_fixed > 0 and v.n_broken > 0:
                heterogeneous.append(
                    f"  '{c.name}' helped {v.n_fixed} and hurt {v.n_broken} cases. "
                    "The effect is heterogeneous, but case-level content is withheld; "
                    "replace the mechanism or derive a label-free gate from runtime "
                    "signals available on every future case."
                )
        if not items:
            return ""
        block = (
            "\n\nPRIOR ATTEMPTS THAT DID NOT WORK — reason from these failures "
            "and design something FUNDAMENTALLY DIFFERENT (different mechanism, "
            "not just different parameters):\n" + "\n".join(items)
        )
        if heterogeneous:
            block += (
                "\n\nHETEROGENEITY — a prior fix helped some cases and broke "
                "others. Prefer a CONDITIONAL fix (apply only where it helps) "
                "over a stronger global transform:\n" + "\n".join(heterogeneous)
            )
        if implementations:
            block += (
                "\n\nEXPLORE-TESTED IMPLEMENTATION(S) TO REVISE:\n"
                + "\n\n".join(implementations)
            )
        return block

    # -- strategy compilation + validation --------------------------------

    def _progress(self, phase: str, **fields: Any) -> None:
        """Operational counters only: never expose scores or held-out labels."""
        if self._progress_callback is not None:
            try:
                self._progress_callback({"phase": phase, **fields})
            except Exception:
                logger.warning("Repair progress callback failed", exc_info=True)

    def _propose(
        self,
        hypotheses: "list[Hypothesis]",
        data: "CaseBatch",
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        context: "FixContext | None" = None,
        window: "tuple[FixTier, FixTier | None] | None" = None,
    ) -> "list[FixCandidate]":
        """Candidates within *window* (``(max_tier, min_tier)``; default: the agent's)."""
        max_tier, min_tier = window or (self.max_tier, self.min_tier)
        return self.proposer.propose(
            hypotheses, data, model, prior_text, prior_names, context,
            max_tier=max_tier, min_tier=min_tier,
            max_tokens_floor=self._max_tokens_floor,
        )

    def _baseline(self, model: "Model", data: "CaseBatch") -> Baseline:
        """Measure the baseline and keep its rates for :meth:`no_fix_recommendation`."""
        baseline = self.validator.baseline(model, data)
        self._baseline_rates = baseline.rates
        return baseline

    def _validate(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch", baseline: Baseline
    ) -> FixValidation:
        self._progress("candidate_started", candidate=candidate.name, total=len(data))
        try:
            result = self.validator.validate(
                candidate, model, data, baseline,
                alpha=self.alpha, max_tokens_floor=self._max_tokens_floor,
            )
        except Exception:
            self._progress("candidate_failed", candidate=candidate.name)
            raise
        self._progress("candidate_completed", candidate=candidate.name,
                       completed=len(data), total=len(data))
        return result

    # -- recommendation + logging ------------------------------------------

