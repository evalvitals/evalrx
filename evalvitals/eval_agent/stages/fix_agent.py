"""Fix module — tiered, validated repair attempts after the diagnosis loop.

Design (intervention-space tiers, see :mod:`fix_tiers`): the allowed tier is
an **input** (default L2).  The agent proposes candidate fixes inside the
allowed tiers, compiles every candidate to the same shape — a per-case success
function, exactly :mod:`~evalvitals.eval_agent.ab_runner`'s *strategy*
contract — and validates each against the unmodified baseline with the paired
machinery from :mod:`evalvitals.stats` (McNemar + e-value, never a bare p).

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
* **L2 declarative** — catalog-tool pipelines (:mod:`fix_tools`): cheap,
  deterministic, validated first.
* **L2 coded** — the coding agent (CLI agent first, judge fallback) writes a
  brand-new pipeline as Python: multiple model calls per case, branching on
  intermediate outputs — only the model itself is unchanged.  The code runs
  sandboxed with bridged model access (:mod:`fix_pipeline`); labels and
  rubrics never reach it, so it cannot cheat by echoing gold answers.
* **L3a** — internals read (:mod:`fix_internals`): no canned primitive; the
  L2 coded pipeline gets a bridged ``model_attend()`` (read-only attention
  heatmap) and authors its own peak-find -> crop -> re-ask scaffold when the
  tier allows.
* **L3b** — internals write (:mod:`fix_internals`): pre-audited intervention
  primitives (v1: visual embedding boost via a forward hook) — the judge
  selects and parameterises; never free codegen against the model handle.
* **L4** — parameter space (:mod:`fix_internals`): the judge writes a
  :class:`~.fix_internals.FinetuneSpec` recipe. v1's executor
  (:func:`~.fix_internals.run_lora_repair`) runs exactly one shape —
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
rate differences (:func:`evalvitals.stats.compare_paired_rates`) — a
stochastic model's flaky cases are weighed by how far the candidate moves
them, neither dropped nor mistaken for repairs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from evalvitals.analyzers.perturbation.prompt_contrast import _default_score
from evalvitals.eval_agent.prompts.fix_agent import (
    _L1_PROMPT,
    _L2_CODE_PROMPT,
    _L2_PROMPT,
    _L3_PROMPT,
    _L4_PROMPT,
    _PAPER_METHOD_PROMPT,
    _REPAIR_PROMPT_BODY,
)
from evalvitals.eval_agent.stages.fix_internals import (
    INTERNALS_PRIMITIVES,
    FinetuneSpec,
    primitives_catalog_text,
    run_lora_repair,
)
from evalvitals.eval_agent.stages.fix_pipeline import (
    CodedPipelineResult,
    run_coded_pipeline,
    score_outputs,
)
from evalvitals.eval_agent.stages.fix_tiers import FixTier, parse_tier, route_min_tier
from evalvitals.eval_agent.stages.fix_tools import (
    PipelineSpec,
    catalog_text,
    run_pipeline,
    safe_format,
    score_to_bool,
    spec_changes_input,
)
from evalvitals.eval_agent.stages.probe_generator import _extract_code
from evalvitals.stats import compare, compare_paired_rates
from evalvitals.stats.ebh import ebh
from evalvitals.stats.evalue import evalue_bernoulli

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.hypothesis import Hypothesis
    from evalvitals.eval_agent.run_context import Trial

logger = logging.getLogger(__name__)


_MAX_JUDGE_CANDIDATES = 3
#: How many FAIL / PASS cases the proposer sees in full (prompt, the model's
#: baseline output, expected answer when allowed). Before this the judge saw
#: only the first 160 characters of a few failing prompts (1000 characters in
#: total) — no model output, no answer format, no PASS contrast — and designed
#: blind: on bbh_tracking7 it never saw an option list or an "Answer: (X)".
_EXAMPLE_FAILS = 4
_EXAMPLE_PASSES = 2
_EXAMPLE_PROMPT_CHARS = 1600
_EXAMPLE_OUTPUT_CHARS = 1000

_TEXT_ONLY_CATALOG_NOTE = (
    "(this batch is text-only: there are NO image tools; leave image_ops empty)"
)

_FLOOR_DESCRIPTIONS = {
    "self_consistency_5": "5 independent samples, majority vote on the extracted final answer",
    "self_refine": "answer -> critique -> revise, three calls",
    "least_to_most": "decompose -> solve with the decomposition, two calls",
    "chain_of_verification": "answer -> list checks -> answer after the checks, three calls",
}


def _clip(text: Any, limit: int, *, tail_share: float = 0.35) -> str:
    """Head + tail of *text* within *limit* characters (the answer sits at the end)."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    tail = max(1, int(limit * tail_share))
    head = max(1, limit - tail)
    return text[:head] + f"\n[… {len(text) - head - tail} chars elided …]\n" + text[-tail:]


def _format_examples(
    cases: Any,
    *,
    with_gold: bool,
    n_fail: int = _EXAMPLE_FAILS,
    n_pass: int = _EXAMPLE_PASSES,
) -> str:
    """Render FAIL (and a few PASS) cases in full for the proposer.

    Each example shows the prompt, the model's baseline output and — only when
    *with_gold* (the cases are disjoint from the validation batch) — the
    expected answer. Deterministic: first *n_fail* FAILs and first *n_pass*
    PASSes in batch order.
    """
    fails: "list[Any]" = []
    passes: "list[Any]" = []
    for case in list(cases or []):
        label = getattr(getattr(case, "label", None), "value", None)
        (fails if label == "fail" else passes).append(case)
    chosen = [("FAIL", c) for c in fails[:n_fail]] + [("PASS", c) for c in passes[:n_pass]]
    if not chosen:
        return "- (none)"
    blocks: "list[str]" = []
    for tag, case in chosen:
        inp = getattr(case, "inputs", None)
        prompt = _clip(getattr(inp, "prompt", ""), _EXAMPLE_PROMPT_CHARS, tail_share=0.3)
        observed = getattr(case, "observed", None)
        lines = [f"### {tag} case {getattr(case, 'id', '?')}", "PROMPT:", prompt]
        if observed is not None:
            raw = str(observed)
            lines += [f"MODEL OUTPUT (baseline, {len(raw)} chars):",
                      _clip(raw, _EXAMPLE_OUTPUT_CHARS, tail_share=0.7)]
        else:
            lines.append("MODEL OUTPUT (baseline): (not recorded)")
        if with_gold:
            gold = (getattr(case, "metadata", None) or {}).get(
                "gold", getattr(case, "expected", None))
            if gold is not None:
                lines.append(f"EXPECTED: {gold}")
        blocks.append("\n".join(lines))
    header = (
        "(from the diagnosis split — the fix is validated on a DISJOINT split; "
        "EXPECTED is shown so you can see the answer FORMAT, never to hard-code answers)"
        if with_gold else
        "(from the validation batch — expected answers withheld)"
    )
    return header + "\n\n" + "\n\n".join(blocks)

def _binary_answer(value: Any) -> "str | None":
    match = re.search(r"\b(yes|no)\b", str(value).lower())
    return match.group(1) if match else None


def _binary_hallucination_direction(data: "CaseBatch") -> "tuple[bool, int, int]":
    """Return whether binary evidence supports a false-``Yes`` repair.

    VCD, ICD, OPERA, PAI, and IFCD suppress answers that assert an object
    unsupported by the image. A labelled adapter exposes this direction through
    expected/observed answers. If it can, do not deploy a suppressive repair
    into a false-negative dominant slice; otherwise preserve generic support.
    """
    false_yes = false_no = 0
    for case in data:
        if getattr(getattr(case, "label", None), "value", None) != "fail":
            continue
        expected = _binary_answer(getattr(case, "expected", None))
        observed = _binary_answer(getattr(case, "observed", None))
        if expected == "no" and observed == "yes":
            false_yes += 1
        elif expected == "yes" and observed == "no":
            false_no += 1
    return (false_yes + false_no == 0 or false_yes >= false_no, false_yes, false_no)


def _false_yes_predicate(case: Any) -> bool:
    """Per-case gate: baseline asserted the object, gold says it isn't there.

    VCD/ICD/OPERA/PAI/IFCD are all *suppressive* -- they push the decoded
    answer away from asserting an object the image doesn't support. That is
    the right direction only on this per-case subpopulation. Unlike
    :func:`_binary_hallucination_direction` (a whole-batch go/no-go gate),
    this is meant to be attached to a :class:`FixCandidate` as its
    ``predicate`` so the *same* candidate can be run gated (touching only
    these cases) alongside its ungated sibling. Computed from ``expected``/
    ``observed`` on the baseline already recorded for this case -- never
    from a later selection/confirmation outcome.
    """
    return (
        _binary_answer(getattr(case, "expected", None)) == "no"
        and _binary_answer(getattr(case, "observed", None)) == "yes"
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FixCandidate:
    """One proposed repair, compiled later to an ab_runner-style strategy.

    Attributes:
        tier:        Intervention space the candidate lives in.
        name:        Short identifier.
        kind:        ``"template"`` (L1) | ``"spec"`` (L2 declarative) |
                     ``"code"`` (L2 agent-written pipeline) | ``"vcd"``
                     (L0 contrastive decoding through an opt-in backend) |
                     ``"opera"`` (L3a attention-over-trust penalty for a
                     one-token binary decision) | ``"ifcd"`` (L3b paired
                     TruthX internal edits) | ``"tcd"`` (L3a gated temporal
                     contrastive decoding for audio multi-choice QA) |
                     ``"visual_search"`` (L2 question-guided crop through an
                     opt-in backend).
        payload:     Kind-specific — template: ``{"prompt_template": ...}``;
                     spec: a :class:`~.fix_tools.PipelineSpec` dict;
                     code: ``{"code": "<python source>"}``.
        source:      ``"judge"``, ``"cli:<provider>"`` or ``"default"``.
        predicate:   Optional ``(case) -> bool`` applicability gate.  When set,
                     the candidate is only applied to (and only judged on) the
                     cases it returns True for — a *conditional* fix.  When
                     ``None``, applicability is inferred structurally (a spec
                     that does not change a case's input is a no-op there).
        trial:       Optional :class:`~evalvitals.eval_agent.run_context.Trial`
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
    # its recorded answers (fix_pipeline.frozen_model_control) — the repair was
    # the pipeline's own computation, not the model's, so they leave the paired
    # test. Baseline-correct cases are never dropped on this ground.
    n_model_independent: int = 0
    e_value: "float | None" = None
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
        }


def _truncated_count(model: Any) -> "int | None":
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

    * ``example_cases`` are shown IN FULL (prompt, the model's baseline output,
      the expected answer, PASS/FAIL). They MUST be disjoint from the batch
      the candidates are validated on — the loop passes its EXPLORE split, the
      fix is scored on CONFIRM. When no example cases are given, examples are
      drawn from the validation batch itself and the expected answer is
      withheld (a template that encodes gold answers of the cases it is scored
      on would be a leak, not a repair).
    * ``evidence`` / ``refuted`` are read-only narrative: what M2/M5/explore
      established and what M4's intervention experiment knocked down. They
      steer *what* to propose; validation still decides *whether* it works.

    Attributes:
        example_cases:      Cases the proposer may see in full (see above).
        evidence:           Host-built summary of M2 statistics, M5 test
                            verdicts and exploratory notes.
        refuted:            Hypotheses an M4 experiment REFUTED (statement +
                            why), so the proposer does not build on them.
        scoring_note:       How outputs are scored / the expected final-answer
                            format (e.g. "last 'Answer:' line, '(D)' == 'D'").
        baseline_decoding:  ``{"max_tokens": .., "temperature": ..}`` the
                            baseline was generated with; the agent also uses
                            ``max_tokens`` as the floor a candidate may not go
                            below.
        task_note:          One-paragraph task / protocol description.
        hypotheses_note:    Status caveat printed right under the hypotheses
                            (e.g. "UNVERIFIED: M5 found no significant evidence
                            …") when the loop hands the fix unverified leads.
    """

    example_cases: "Any | None" = None
    evidence: str = ""
    refuted: "list[str]" = field(default_factory=list)
    scoring_note: str = ""
    baseline_decoding: "dict[str, Any]" = field(default_factory=dict)
    task_note: str = ""
    hypotheses_note: str = ""


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
        run_context:      Optional :class:`~evalvitals.eval_agent.run_context.RunContext`.
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
                          See :func:`~.fix_internals.run_lora_repair`.
        verbose:          When ``True``, print this agent's own tier-routing /
                          candidate-generation / validation-verdict narration
                          to stdout (``evalvitals.enable_console_logging()``).
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
        max_judge_candidates: int = _MAX_JUDGE_CANDIDATES,
        allow_adapted_paper_methods: bool = False,
        paper_methods_only: bool = False,
        candidate_allowlist: "Iterable[str] | None" = None,
        finetune_pool: "CaseBatch | None" = None,
        verbose: bool = False,
        baseline_generation_kwargs: "dict[str, Any] | None" = None,
        concurrency: int = 1,
        scoring_note: str = "",
        floor_candidates: "Iterable[str] | None" = ("self_consistency_5",),
    ) -> None:
        if verbose:
            # Surfaces this module's own logger.info()/.warning() calls (tier
            # routing, candidate generation, validation verdicts) — see
            # VLDiagnoseLoop's verbose= for the same convenience one layer up.
            from evalvitals.logging_utils import enable_console_logging

            enable_console_logging()

        self._judge = judge
        self._finetune_pool = finetune_pool
        self.max_tier = parse_tier(max_tier)
        self._score = score_fn or _default_score
        self.run_logger = run_logger
        self._cli_config = cli_config
        self._allow_codegen = allow_codegen
        self._sandbox = sandbox
        # When set (directly, or via the RunLogger's bound RunContext), the
        # sandbox workdir is allocated durably under workspace/ instead of an
        # ephemeral tempfile.mkdtemp() that the sandbox deletes on success.
        self._run_context = run_context or getattr(run_logger, "_context", None)
        self._exec_timeout_sec = exec_timeout_sec
        self.max_validation_cases = max_validation_cases
        self._baseline_repeats = max(1, int(baseline_repeats))
        self._candidate_repeats = max(1, int(candidate_repeats))
        self._baseline_rates: "dict[str, Optional[float]]" = {}
        self._baseline_n: "dict[str, int]" = {}
        self._alpha = float(alpha)
        self.max_repair_rounds = max(1, int(max_repair_rounds))
        self.max_judge_candidates = max(1, int(max_judge_candidates))
        self._allow_adapted_paper_methods = bool(allow_adapted_paper_methods)
        self._paper_methods_only = bool(paper_methods_only)
        self._candidate_allowlist = (
            frozenset(str(name) for name in candidate_allowlist)
            if candidate_allowlist is not None
            else None
        )
        self._last_repair_prompt = ""
        self._last_usage: dict | None = None
        self._baseline_generation_kwargs = dict(baseline_generation_kwargs or {})
        self._concurrency = max(1, int(concurrency))
        self._scoring_note = str(scoring_note or "")
        self._floor_candidates = (
            tuple(str(n) for n in floor_candidates) if floor_candidates else ()
        )
        # Per-candidate scratch: case id -> the final output that was scored.
        # Filled by the strategy closures / run_pipeline capture while a
        # candidate runs; _validate moves it onto the FixValidation.
        self._captured: "dict[str, str]" = {}
        self._max_tokens_floor: "int | None" = None

    @property
    def codegen_available(self) -> bool:
        """True when the L2 coded-pipeline path has a code-writing backend."""
        return self._allow_codegen and (
            self._judge is not None
            or (self._cli_config is not None and self._cli_config.provider != "llm")
        )

    # -- public ---------------------------------------------------------

    def propose_and_validate(
        self,
        model: "Model",
        data: "CaseBatch",
        hypotheses: "list[Hypothesis]",
        prior_attempts: "list[FixValidation] | None" = None,
        context: "FixContext | None" = None,
        proposal_data: "CaseBatch | None" = None,
    ) -> FixOutcome:
        """Generate candidates within the allowed tiers, validate, recommend.

        *context* (optional :class:`FixContext`) is what the proposer sees
        besides the hypotheses — full example cases from a DISJOINT split,
        the M2/M5/explore evidence, M4-refuted hypotheses, the scoring rule
        and the baseline decoding budget.  ``proposal_data``, when supplied,
        is the discovery partition available to the repair author; ``data``
        remains untouched confirmation data and permits only one round.
        """
        outcome = FixOutcome(max_tier=self.max_tier)
        self._max_tokens_floor = self._resolve_max_tokens_floor(model, data)
        routed_tiers: "list[FixTier]" = []
        for h in hypotheses:
            tier, why = route_min_tier(h)
            routed_tiers.append(tier)
            outcome.routed.append(
                {
                    "hypothesis": getattr(h, "statement", str(h))[:160],
                    "min_tier": tier.label,
                    "rationale": why,
                }
            )

        data = self._validation_subset(data)
        authoring_data = proposal_data if proposal_data is not None else data
        baseline, unstable = self._baseline(model, data)
        if not any(v is not None for v in baseline.values()):
            logger.warning("FixAgent: no scorable case (no rubrics); nothing to validate")
            outcome.recommendation = self._recommend(
                routed_tiers,
                model=model,
                reason_prefix=("no case carries a scoring rubric, so no fix can be validated"),
            )
            self._emit(outcome)
            return outcome

        # Feedback-driven repair rounds: propose -> validate; if nothing
        # validates, summarise the failures (this call's own attempts, PLUS
        # any prior_attempts carried over from an earlier escalation tier) and
        # ask for DIFFERENT candidates within the same tier (never escalating).
        # Stop on first validated fix or when a round adds no new candidate.
        seen: "set[tuple[str, str, str]]" = set()
        round_limit = 1 if proposal_data is not None else self.max_repair_rounds
        for round_idx in range(round_limit):
            combined_prior = list(prior_attempts or []) + outcome.attempted
            prior_text = (
                self._format_prior(combined_prior, authoring_data)
                if combined_prior
                else ""
            )
            prior_names: "frozenset[str]" = frozenset(v.candidate.name for v in combined_prior)
            new_candidates: "list[FixCandidate]" = []
            for candidate in self._propose(
                hypotheses, authoring_data, model, prior_text, prior_names, context=context
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
                validation = self._validate(candidate, model, data, baseline, unstable)
                outcome.attempted.append(validation)
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

        outcome.refine_signal = self._refine_signal(outcome.attempted, data)
        # Multiplicity control over the candidate family (best-of-N): every
        # candidate cleared only its OWN paired gate (e >= 1/alpha) against the
        # SAME baseline, so picking the max is a multiple-comparisons hazard.
        # e-BH composes the family's e-values into one FDR guarantee; a candidate
        # is a winner only if it is BOTH individually `fixed` AND an e-BH survivor.
        tested = [v for v in outcome.attempted if v.e_value is not None]
        survivors = self._ebh_survivors(tested)
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
                    "recommend_tier": self.max_tier.label,
                    "reason": (
                        f"{len(culled)} candidate(s) cleared their own gate "
                        f"({names}) but did NOT survive e-BH FDR across the "
                        f"{len(tested)}-candidate family — best-of-N multiplicity, "
                        "not a validated fix. Gather more failing cases (more "
                        "power per candidate) or propose fewer, stronger candidates."
                    ),
                }
            else:
                outcome.recommendation = self._no_fix_recommendation(
                    outcome.attempted, routed_tiers, data, model
                )
        self._emit(outcome)
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
        selection correction is needed on the confirmation split.
        """
        data = self._validation_subset(data)
        self._max_tokens_floor = self._resolve_max_tokens_floor(model, data)
        self._enforce_generation_floor([candidate])
        baseline, unstable = self._baseline(model, data)
        return self._validate(candidate, model, data, baseline, unstable)

    def _ebh_survivors(self, tested: "list[FixValidation]") -> "set[int]":
        """id()s of validations whose e-value survives e-BH across the family.

        The e-values come from compare() (the validated core), never from an
        LLM, so e-BH is sound. With m candidates, the sole-survivor bar rises to
        m/alpha (vs the per-candidate 1/alpha) — the correct best-of-N tax.
        A single tested candidate (m=1) reduces to the per-candidate gate, so
        non-competitive runs are unaffected.
        """
        if not tested:
            return set()
        evalues = [float(v.e_value) for v in tested]
        return {id(tested[i]) for i in ebh(evalues, alpha=self._alpha)}

    # -- no-fix recommendation (escalate vs. gather data vs. retry exec) ----

    def _no_fix_recommendation(
        self,
        attempted: "list[FixValidation]",
        routed_tiers: "list[FixTier]",
        data: "CaseBatch",
        model: "Model",
    ) -> "dict[str, Any] | None":
        """Decide what 'no candidate validated' actually means.

        Three distinct causes the old single-path recommendation conflated:

        * **never executed** — engineering failure; retry within the tier, do
          not escalate (a crash is not evidence the tier is exhausted).
        * **underpowered by design** — even a flawless fix of every failure
          could not reach significance with this few failures; gather more
          failing cases instead of climbing the (more invasive) tier ladder.
        * **genuinely exhausted** — executed, powered, still no fix; escalate.
        """
        executed = [v for v in attempted if v.n_pairs > 0]
        # A model-independent candidate DID execute; it just is not a repair of
        # the model. It is neither an engineering failure nor evidence the tier
        # is exhausted, so it sits in neither list.
        never_ran = [v for v in attempted
                     if v.n_pairs == 0 and v.verdict != "model_independent"]
        if never_ran and not executed:
            return {
                "recommend_tier": self.max_tier.label,
                "action": "fix_execution",
                "reason": (
                    "no candidate EXECUTED — escalating would be premature; "
                    f"fix candidate execution and retry within {self.max_tier.label}. "
                    "Failures: "
                    + "; ".join(
                        f"{v.candidate.name}: {(v.exec_error or v.summary)[:120]}"
                        for v in never_ran[:3]
                    )
                ),
            }

        # Underpowered-by-design: with n_fail failures the best possible result
        # (every failure repaired, nothing broken) yields an e-value ceiling of
        # evalue_bernoulli(n_fail, n_fail); if that cannot clear 1/alpha, no fix
        # of any tier can be certified here — the bottleneck is sample size, and
        # a "promising" candidate (helped more than it hurt) confirms the lead.
        n_fail = sum(1 for c in data if getattr(c.label, "value", None) == "fail")
        if self._baseline_repeats > 1 or self._candidate_repeats > 1:
            # Paired rates: the best any candidate can do is lift every case's
            # rate to 1.0 -> the e-value of those differences is the ceiling.
            from evalvitals.stats.evalue import evalue_bounded_mean

            diffs = [
                1.0 - r for c in data
                for r in [self._baseline_rates.get(c.id)] if r is not None
            ]
            ceiling = evalue_bounded_mean(diffs) if diffs else 1.0
        else:
            ceiling = evalue_bernoulli(n_fail, n_fail, p0=0.5) if n_fail > 0 else 1.0
        promising = [v for v in executed if (v.n_fixed - v.n_broken) > 0 and not v.reject]
        if ceiling < 1.0 / self._alpha:
            need = self._min_failures_for_power()
            evidence = ""
            if promising:
                best = max(
                    promising,
                    key=lambda v: (v.n_fixed - v.n_broken, v.effect or 0.0),
                )
                evidence = (
                    f" {best.candidate.name!r} already helps net "
                    f"{best.n_fixed - best.n_broken} case(s);"
                )
            return {
                "recommend_tier": None,
                "action": "gather_more_failures",
                "reason": (
                    f"underpowered by design: only {n_fail} failure case(s) — even a "
                    f"perfect fix tops out at e={ceiling:.1f} (< {1.0 / self._alpha:.0f} "
                    f"needed).{evidence} collect >= {need} failing "
                    "cases and re-validate before escalating the tier."
                ),
            }

        rec = self._recommend(routed_tiers, model=model)
        if promising and rec is not None:
            best = max(promising, key=lambda v: (v.n_fixed - v.n_broken, v.effect or 0.0))
            eff = f"{best.effect:+.3f}" if best.effect is not None else "n/a"
            ev = f"{best.e_value:.1f}" if best.e_value is not None else "n/a"
            rec["promising"] = {
                "candidate": best.candidate.name,
                "n_fixed": best.n_fixed,
                "n_broken": best.n_broken,
                "effect": best.effect,
                "e_value": best.e_value,
            }
            rec["reason"] += (
                f" (note: {best.candidate.name!r} is INCONCLUSIVE, not refuted — "
                f"{best.n_fixed} fixed / {best.n_broken} broken, effect {eff}, "
                f"e={ev} < {1.0 / self._alpha:.0f}; re-validating it on more or cleaner "
                "pairs (baseline_repeats>1 to drop sampling-unstable cases, or a larger "
                "confirm split) is a cheaper next step than escalating the tier)"
            )
        solo = [v for v in attempted if v.verdict == "model_independent"]
        if solo and rec is not None:
            rec["reason"] += (
                f" (note: {len(solo)} candidate(s) solved the task without the model "
                "and were not counted as repairs: "
                + ", ".join(v.candidate.name for v in solo[:3])
                + ")"
            )
        if never_ran and rec is not None:
            rec["reason"] += (
                f" (caveat: {len(never_ran)} candidate(s) never executed: "
                + ", ".join(v.candidate.name for v in never_ran[:3])
                + ")"
            )
        return rec

    def _min_failures_for_power(self) -> int:
        """Smallest n where a flawless fix could clear the e-value gate."""
        threshold = 1.0 / self._alpha
        for n in range(1, 200):
            if evalue_bernoulli(n, n, p0=0.5) >= threshold:
                return n
        return 200

    @staticmethod
    def _refine_signal(
        attempted: "list[FixValidation]", data: "CaseBatch"
    ) -> "dict[str, Any] | None":
        """Heterogeneity feedback for re-diagnosis (defect 3).

        A candidate that repairs one subset while breaking another is evidence
        the failure mode is *not* homogeneous: the right next move is to split
        the population by sub-mechanism and re-diagnose, not to keep proposing
        whole-population transforms.  Surface the partition so the loop (or a
        human) can re-scope the hypothesis.
        """
        split = [v for v in attempted if v.n_fixed > 0 and v.n_broken > 0]
        if not split:
            return None
        v = max(split, key=lambda x: min(x.n_fixed, x.n_broken))
        return {
            "kind": "heterogeneous_failure_mode",
            "candidate": v.candidate.name,
            "helped_cases": list(v.fixed_cases),
            "hurt_cases": list(v.broken_cases),
            "message": (
                f"{v.candidate.name!r} repaired {v.n_fixed} case(s) but broke "
                f"{v.n_broken} — the failure mode is likely subset-specific. "
                "Re-diagnose: what distinguishes the helped cases from the hurt "
                "ones, and gate the fix on that predicate."
            ),
        }

    def _validation_subset(self, data: "CaseBatch") -> "CaseBatch":
        """Label-stratified, deterministic subset for candidate validation."""
        cap = self.max_validation_cases
        if not cap or len(data) <= cap:
            return data
        import random as _random

        from evalvitals.core.case import CaseBatch, Label

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

    def _propose(
        self,
        hypotheses: "list[Hypothesis]",
        data: "CaseBatch",
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        context: "FixContext | None" = None,
    ) -> "list[FixCandidate]":
        context = context or FixContext()
        hyp_lines = (
            "\n".join(
                f"- [{getattr(h, 'predicted_failure_mode', '')}] {getattr(h, 'statement', h)}"
                for h in hypotheses
            )
            or "- (no verified hypotheses; failures are unexplained)"
        )
        if context.hypotheses_note:
            hyp_lines = f"({context.hypotheses_note.strip()})\n{hyp_lines}"
        # Full examples (prompt + the model's own output + expected answer)
        # come from cases the proposer may see in full — the loop's EXPLORE
        # split. Without such cases the examples are drawn from the validation
        # batch itself and the expected answer is withheld (see FixContext).
        if context.example_cases is not None:
            examples = _format_examples(context.example_cases, with_gold=True)
        else:
            examples = _format_examples(data, with_gold=False)
        # A video frame is visual content too -- a video-only case batch (no
        # .image ever set, only .video) must not silently read as "no images"
        # and lock every image-gated L1/L2/L3 candidate out. This was found
        # via a real run (musicavqa_videollama2): has_images was False for
        # every case despite the task being audio-VISUAL QA, so no visual
        # candidate at any tier was ever structurally eligible.
        has_images = any(
            getattr(getattr(case, "inputs", None), "image", None) is not None
            or getattr(getattr(case, "inputs", None), "video", None) is not None
            for case in data
        )
        has_audio = any(
            getattr(getattr(case, "inputs", None), "audio", None) is not None for case in data
        )
        tasks = {
            str((getattr(case, "metadata", {}) or {}).get("task", "")) for case in data
        }
        binary_hallucination_supported, _, _ = _binary_hallucination_direction(data)
        # Text-only batches: the image-tool catalog is noise for the judge and
        # an invitation to burn a candidate on a structural no-op.
        catalog = catalog_text() if has_images else _TEXT_ONLY_CATALOG_NOTE
        floor_names = self._floor_names(has_images=has_images, tasks=tasks)
        context_block = self._context_block(
            context, data, model, floor_names=floor_names
        )

        candidates: "list[FixCandidate]" = []
        # A code-only run is a pre-registered autonomous-repair experiment.
        # Do not spend three judge calls inventing L0/L1/declarative/L3
        # candidates that the allowlist will discard afterwards; apart from
        # latency and quota waste, those calls can fail before the requested
        # coding agent is ever reached.
        code_only = self._candidate_allowlist == frozenset({"coded_pipeline"})
        if not code_only and self.max_tier >= FixTier.L0_RUNTIME_CONFIG:
            candidates += self._l0_candidates(data, prior_names, model=model)
        if (
            not code_only
            and self.max_tier >= FixTier.L1_PROMPT
            and not self._paper_methods_only
        ):
            candidates += self._l1_candidates(
                hyp_lines,
                examples,
                prior_text,
                prior_names,
                has_images=has_images,
                tasks=tasks,
                binary_hallucination_supported=binary_hallucination_supported,
                context_block=context_block,
            )
        if not code_only and self.max_tier >= FixTier.L2_SCAFFOLD:
            candidates += self._l2_candidates(
                hyp_lines,
                examples,
                prior_text,
                prior_names,
                has_images=has_images,
                model=model,
                tasks=tasks,
                context_block=context_block,
                catalog=catalog,
            )
            # The floor: always-tested defaults, on top of the judge's ideas.
            present = {c.name for c in candidates}
            for name in floor_names:
                if name in present or name in prior_names:
                    continue
                spec = self._default_spec(name)
                if spec is not None:
                    candidates.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD, name=name, kind="spec",
                            source="floor", payload=spec.to_dict(),
                        )
                    )
            if self.codegen_available:
                candidates += self._l2_coded_candidate(
                    hyp_lines, examples, model, prior_text,
                    context_block=context_block, catalog=catalog,
                    text_only=not has_images,
                )
        if not code_only and self.max_tier >= FixTier.L3A_INTERNALS_READ:
            candidates += self._l3_candidates(
                hyp_lines,
                model,
                prior_text,
                prior_names,
                has_images=has_images,
                has_audio=has_audio,
                tasks=tasks,
                binary_hallucination_supported=binary_hallucination_supported,
            )
        if not code_only and self.max_tier >= FixTier.L4_PARAMETERS:
            candidates += self._l4_candidates(hyp_lines)
        if self._candidate_allowlist is not None:
            candidates = [c for c in candidates if c.name in self._candidate_allowlist]
        self._enforce_generation_floor(candidates)
        return candidates

    # -- proposer context ---------------------------------------------------

    def _floor_names(self, *, has_images: bool, tasks: "set[str] | None") -> "tuple[str, ...]":
        """Which built-in defaults are ALWAYS in the family for this batch.

        Text-only batches get the configured floor. Image batches keep their
        existing ladder (image transforms first; self_refine/self_consistency
        only for reasoning-shaped tasks — see ``_l2_candidates``), so the
        floor applies there only for those reasoning tasks.
        """
        if not self._floor_candidates:
            return ()
        reasoning_task = bool(
            tasks and tasks & {"multiple_choice", "exact_or_numeric", "vqa_consensus"}
        )
        if has_images and not reasoning_task:
            return ()
        return tuple(self._floor_candidates)

    def _default_spec(self, name: str) -> "PipelineSpec | None":
        """Built-in default L2 specs by name (the floor and the fallback)."""
        if name == "self_consistency_5":
            # Vote at a stochastic temperature: at T=0 five samples are one
            # sample. Inherit the baseline temperature when it already samples.
            base_t = self._baseline_generation_kwargs.get("temperature")
            try:
                base_t = float(base_t) if base_t is not None else None
            except (TypeError, ValueError):
                base_t = None
            temperature = base_t if (base_t is not None and base_t > 0.0) else 0.7
            return PipelineSpec(
                name="self_consistency_5",
                prompt_template="{prompt}",
                n_samples=5,
                generation_kwargs={"temperature": temperature},
            )
        if name == "self_refine":
            return PipelineSpec(name="self_refine", prompt_template="{prompt}",
                                strategy="self_refine")
        if name == "least_to_most":
            return PipelineSpec(name="least_to_most", prompt_template="{prompt}",
                                strategy="least_to_most")
        if name == "chain_of_verification":
            return PipelineSpec(name="chain_of_verification", prompt_template="{prompt}",
                                strategy="chain_of_verification")
        return None

    def _resolve_max_tokens_floor(self, model: "Model | None", data: "CaseBatch") -> "int | None":
        """The baseline decode budget: explicit > model attribute > case metadata."""
        raw = self._baseline_generation_kwargs.get("max_tokens")
        try:
            if raw is not None and int(raw) > 0:
                return int(raw)
        except (TypeError, ValueError):
            pass
        attr = getattr(model, "max_tokens", None)
        if isinstance(attr, int) and not isinstance(attr, bool) and attr > 0:
            return attr
        caps: "list[int]" = []
        for case in data:
            meta = getattr(case, "metadata", {}) or {}
            config = meta.get("generation_config") or {}
            try:
                cap = int(config.get("max_tokens"))
            except (AttributeError, TypeError, ValueError):
                continue
            if cap > 0:
                caps.append(cap)
        return max(caps) if caps else None

    def _enforce_generation_floor(self, candidates: "list[FixCandidate]") -> None:
        """Raise any candidate ``max_tokens`` below the baseline budget to it.

        A scaffold may give the model MORE decode room than the baseline had,
        never less: a lower cap truncates the chain of thought and the answer
        never appears, which the scorer reads as a wrong answer — a decoding
        artefact that says nothing about the candidate's idea. The judge's
        proposal is kept in ``payload["generation_kwargs_proposed"]`` for the
        record; the applied value is what ``PipelineSpec.from_dict`` reads.
        """
        floor = self._max_tokens_floor
        if not floor:
            return
        for candidate in candidates:
            payload = candidate.payload
            if not isinstance(payload, dict) or candidate.kind != "spec":
                continue
            gk = payload.get("generation_kwargs")
            if not isinstance(gk, dict):
                continue
            try:
                proposed = int(gk.get("max_tokens"))
            except (TypeError, ValueError):
                continue
            if proposed < floor:
                payload.setdefault("generation_kwargs_proposed", dict(gk))
                gk["max_tokens"] = int(floor)
                logger.info(
                    "FixAgent: %s proposed max_tokens=%d below the baseline budget %d; "
                    "raised to the floor (a candidate may not decode with less room "
                    "than the baseline)", candidate.name, proposed, floor,
                )

    def _context_block(
        self,
        context: FixContext,
        data: "CaseBatch",
        model: "Model | None",
        *,
        floor_names: "tuple[str, ...]" = (),
    ) -> str:
        """Render task / scoring / decoding / evidence / refuted notes for the proposer."""
        lines: "list[str]" = []
        if context.task_note:
            lines += ["TASK:", f"  {context.task_note.strip()}"]
        scoring = context.scoring_note or self._scoring_note
        if scoring:
            lines += ["HOW OUTPUTS ARE SCORED (the final answer must be recoverable this way):",
                      f"  {scoring.strip()}"]
        decoding = dict(self._baseline_generation_kwargs)
        decoding.update(context.baseline_decoding or {})
        floor = self._max_tokens_floor
        outputs = [str(getattr(c, "observed", "") or "") for c in data]
        lengths = sorted(len(o) for o in outputs if o)
        median_chars = lengths[len(lengths) // 2] if lengths else None
        if decoding or floor or median_chars:
            parts = []
            if floor:
                parts.append(f"max_tokens={floor}")
            for key in ("temperature", "top_p"):
                if key in decoding:
                    parts.append(f"{key}={decoding[key]}")
            note = "BASELINE DECODING: " + (", ".join(parts) if parts else "(unknown)")
            if median_chars:
                note += f"; median baseline output ≈ {median_chars} chars"
            lines.append(note)
            if floor:
                lines.append(
                    "  A candidate's max_tokens is a FLOOR-RAISED value: anything below "
                    f"{floor} is raised to {floor} (a candidate may give the model more "
                    "room, never less — the model needs its chain of thought), so do not "
                    "try to save tokens; only raise max_tokens if the baseline truncates."
                )
        if context.evidence:
            lines += ["DIAGNOSTIC EVIDENCE (what M2 statistics / M5 tests / exploration "
                      "established — read-only, steer WHAT to try):", context.evidence.rstrip()]
        if context.refuted:
            lines += ["REFUTED BY AN INTERVENTION EXPERIMENT (M4) — do NOT build a fix on these:"]
            lines += [f"  - {r}" for r in context.refuted]
        if floor_names:
            lines.append(
                "ALREADY IN THE TEST FAMILY (validated alongside your proposals — do not "
                "re-propose plain versions of these): " + ", ".join(
                    f"{n} ({_FLOOR_DESCRIPTIONS.get(n, 'built-in default')})" for n in floor_names
                )
            )
        if not lines:
            return ""
        return "\n" + "\n".join(lines) + "\n"
    def _l0_candidates(
        self,
        data: "CaseBatch",
        prior_names: "frozenset[str]" = frozenset(),
        *,
        model: "Model | None" = None,
    ) -> "list[FixCandidate]":
        """Propose a bounded decoding repair only from recorded telemetry.

        A short answer is not proof of truncation.  We require a backend to
        record ``metadata['finish_reason'] == 'length'`` and the baseline
        ``metadata['generation_config']['max_tokens']`` for at least one
        failing case. This makes the candidate useful for any OpenAI-style or
        local backend while preventing prompt-specific guesswork.
        """
        caps: "list[int]" = []
        policy_caps: "list[int]" = []
        for case in data:
            if getattr(getattr(case, "label", None), "value", None) != "fail":
                continue
            meta = getattr(case, "metadata", {}) or {}
            if str(meta.get("finish_reason", "")).lower() != "length":
                continue
            config = meta.get("generation_config") or {}
            try:
                cap = int(config.get("max_tokens"))
            except (AttributeError, TypeError, ValueError):
                continue
            if 1 <= cap < 8192:
                caps.append(cap)
            policy = meta.get("generation_policy") or {}
            try:
                policy_cap = int(policy.get("max_tokens_cap"))
            except (AttributeError, TypeError, ValueError):
                continue
            if cap < policy_cap <= 8192:
                policy_caps.append(policy_cap)
        out: "list[FixCandidate]" = []
        # A single, auditable policy: use the deployment's explicit safe cap
        # when present; otherwise double the observed cap. Use the maximum seen
        # cap so a mixed batch never *reduces* any case's decode budget.
        if caps and "increase_max_tokens" not in prior_names:
            old_cap = max(caps)
            new_cap = max(policy_caps) if policy_caps else min(8192, old_cap * 2)
            if new_cap > old_cap:
                spec = PipelineSpec(
                    name="increase_max_tokens",
                    generation_kwargs={"max_tokens": new_cap},
                )
                out.append(
                    FixCandidate(
                        tier=FixTier.L0_RUNTIME_CONFIG,
                        name=spec.name,
                        kind="spec",
                        source="telemetry",
                        payload=spec.to_dict(),
                    )
                )

        # VCD (Leng et al., CVPR 2024) is a decoding-space repair for binary
        # visual-grounding tasks. It contrasts first-token logits from the
        # original and a diffusion-distorted image. Do not offer it for open
        # generation: applying a first-token approximation there would not be
        # the paper's method. Backends opt in explicitly via ``generate_vcd``.
        from evalvitals.core.capability import Capability

        tasks = {str((getattr(case, "metadata", {}) or {}).get("task", "")) for case in data}
        supports_logprobs = bool(
            model is not None and Capability.LOGPROBS in getattr(model, "capabilities", frozenset())
        )
        supports_vcd = supports_logprobs and callable(getattr(model, "generate_vcd", None))
        # VCD distorts an IMAGE; a multimodal backend exposes generate_vcd even
        # when this batch is audio-only (caught live: audiocaps_hallucination
        # proposed both VCD candidates, each ran as not_executed). Require a
        # visual input on at least one case, like every other visual tier.
        has_visual = any(
            getattr(getattr(case, "inputs", None), "image", None) is not None
            or getattr(getattr(case, "inputs", None), "video", None) is not None
            for case in data
        )
        supports_vcd = supports_vcd and has_visual
        hallucination_direction_supported, _, _ = _binary_hallucination_direction(data)
        if (
            tasks == {"yes_no"}
            and supports_vcd
            and hallucination_direction_supported
            and "vcd_diffusion_noise" not in prior_names
        ):
            # VCD appendix A fixes POPE's total diffusion steps at 999
            # (MME/LLaVA-Bench use 500), with alpha=1 and beta=0.1.
            vcd_payload = {
                "alpha": 1.0,
                "beta": 0.1,
                "noise_step": 999,
            }
            out.append(
                FixCandidate(
                    tier=FixTier.L0_RUNTIME_CONFIG,
                    name="vcd_diffusion_noise",
                    kind="vcd",
                    source="paper_default",
                    payload=vcd_payload,
                )
            )
        # Gated sibling (defect 3's refine_signal, operationalised): VCD is a
        # *suppressive* repair -- it is the right direction only on cases
        # where the baseline asserted an object the image doesn't support
        # (false-Yes). Proposing this alongside the ungated candidate lets a
        # near-cancelling whole-slice result (helps false-Yes cases, hurts
        # false-No ones -- exactly the "heterogeneous_failure_mode" pattern
        # every POPE report already surfaces) resolve into a real, narrower
        # fix instead of a null. The predicate reads each case's own
        # already-recorded baseline expected/observed -- never a selection or
        # confirmation outcome -- so this is a candidate design choice, not a
        # post-hoc tuning of which cases to report.
        if (
            tasks == {"yes_no"}
            and supports_vcd
            and "vcd_diffusion_noise_gated_false_yes" not in prior_names
        ):
            out.append(
                FixCandidate(
                    tier=FixTier.L0_RUNTIME_CONFIG,
                    name="vcd_diffusion_noise_gated_false_yes",
                    kind="vcd",
                    source="conditional_default",
                    payload={"alpha": 1.0, "beta": 0.1, "noise_step": 999},
                    predicate=_false_yes_predicate,
                )
            )
        # AAD (Hsu et al. 2025, arXiv:2506.07233) is VCD's same shape applied
        # to audio instead of an image: contrasts real-audio decoding against
        # the identical prompt with the waveform silenced, at every step. No
        # internals read (no attention weights, no layer stability) -- just
        # two generate()-compatible forward passes and a LogitsProcessor, the
        # same cost/risk class as VCD, so it belongs at L0 next to it, not
        # gated through the L3a judge-selected paper-method catalog.
        paper_fidelity_early = getattr(model, "paper_method_fidelity", None)
        aad_fidelity = (
            paper_fidelity_early("aad") if callable(paper_fidelity_early) else "unavailable"
        )
        supports_aad = callable(getattr(model, "generate_aad", None))
        if (
            tasks == {"yes_no"}
            and supports_aad
            and aad_fidelity == "native_silence_contrast"
            and hallucination_direction_supported
            and "aad_silence_contrast" not in prior_names
        ):
            out.append(
                FixCandidate(
                    tier=FixTier.L0_RUNTIME_CONFIG,
                    name="aad_silence_contrast",
                    kind="aad",
                    source="paper_default",
                    payload={"alpha": 0.5},
                )
            )
        # Gated sibling, same reasoning as vcd_diffusion_noise_gated_false_yes
        # above: AAD is suppressive (promotes tokens whose probability rises
        # WITH audio, i.e. demotes an audio-ungrounded over-affirmation), so
        # it is the right direction only on false-Yes cases.
        if (
            tasks == {"yes_no"}
            and supports_aad
            and aad_fidelity == "native_silence_contrast"
            and "aad_silence_contrast_gated_false_yes" not in prior_names
        ):
            out.append(
                FixCandidate(
                    tier=FixTier.L0_RUNTIME_CONFIG,
                    name="aad_silence_contrast_gated_false_yes",
                    kind="aad",
                    source="conditional_default",
                    payload={"alpha": 0.5},
                    predicate=_false_yes_predicate,
                )
            )
        # ICD (Wang et al., ACL 2024) has the same binary, token-level
        # admission requirements but its negative condition is an instruction
        # disturbance rather than a corrupted image.  A backend can expose an
        # architecture-native method (e.g. an InstructBLIP Q-Former) or an
        # explicitly labelled architecture-adapted implementation.
        paper_fidelity = getattr(model, "paper_method_fidelity", None)
        icd_fidelity = paper_fidelity("icd") if callable(paper_fidelity) else "unavailable"
        if (
            tasks == {"yes_no"}
            and supports_logprobs
            and hallucination_direction_supported
            and callable(getattr(model, "generate_instruction_cd", None))
            and (
                icd_fidelity in {"exact", "native_binary_specialization"}
                or (icd_fidelity == "adapted" and self._allow_adapted_paper_methods)
            )
            and "icd_instruction_disturbance" not in prior_names
        ):
            out.append(
                FixCandidate(
                    tier=FixTier.L0_RUNTIME_CONFIG,
                    name="icd_instruction_disturbance",
                    kind="icd",
                    source="paper_default",
                    payload={"alpha": 1.0, "beta": 0.1, "qformer_mode": "normal"},
                )
            )
            # The official ICD POPE runner evaluates both Q-Former conditions:
            # disturbance alone and disturbance concatenated with the question.
            # The latter exists only on the native Q-Former architecture; a
            # decoder-prefix approximation would be a new, ungrounded method.
            if (
                icd_fidelity in {"exact", "native_binary_specialization"}
                and "icd_instruction_disturbance_question" not in prior_names
            ):
                out.append(
                    FixCandidate(
                        tier=FixTier.L0_RUNTIME_CONFIG,
                        name="icd_instruction_disturbance_question",
                        kind="icd",
                        source="paper_default",
                        payload={"alpha": 1.0, "beta": 0.1, "qformer_mode": "question"},
                    )
                )
        # Gated sibling, same rationale as VCD's above: ICD is also
        # suppressive, so restrict it to the per-case false-Yes subset rather
        # than requiring the whole slice to be false-Yes dominant.
        if (
            tasks == {"yes_no"}
            and supports_logprobs
            and callable(getattr(model, "generate_instruction_cd", None))
            and (
                icd_fidelity in {"exact", "native_binary_specialization"}
                or (icd_fidelity == "adapted" and self._allow_adapted_paper_methods)
            )
            and "icd_instruction_disturbance_gated_false_yes" not in prior_names
        ):
            out.append(
                FixCandidate(
                    tier=FixTier.L0_RUNTIME_CONFIG,
                    name="icd_instruction_disturbance_gated_false_yes",
                    kind="icd",
                    source="conditional_default",
                    payload={"alpha": 1.0, "beta": 0.1, "qformer_mode": "normal"},
                    predicate=_false_yes_predicate,
                )
            )
        return out

    @staticmethod
    def _signature(candidate: FixCandidate) -> "tuple[str, str, str]":
        """Identity of a candidate, to skip re-validating an identical one.

        Coded pipelines carry fresh source each round, so they never collide;
        templates / specs / primitives dedup on their defining payload. The
        candidate's ``name`` is always part of the signature: two candidates
        can share a kind and payload while differing only in ``predicate``
        (for example a paper method and its per-case-gated sibling, see
        ``_false_yes_predicate``) -- that is a different candidate, not a
        duplicate, and must not be silently dropped by the round's dedup set.
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

    def _l1_candidates(
        self,
        hyp_lines: str,
        examples: str,
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        *,
        has_images: bool = False,
        tasks: "set[str] | None" = None,
        binary_hallucination_supported: bool = True,
        context_block: str = "",
    ) -> "list[FixCandidate]":
        proposals = self._ask_judge(
            _L1_PROMPT.format(
                hypotheses=hyp_lines, examples=examples, k=self.max_judge_candidates,
                context=context_block,
            )
            + prior_text
        )
        out: "list[FixCandidate]" = []
        # Some permissive judges return an L2 pipeline for the L1 request
        # because both prompts include the same examples. Do not silently turn
        # that into an identity L1 candidate (and an unnecessary e-BH test).
        structural_keys = {"image_ops", "generation_kwargs", "n_samples", "strategy"}
        has_structural_proposal = False
        for p in proposals:
            template = str(p.get("prompt_template", ""))
            name = str(p.get("name", "")).strip()
            if structural_keys.intersection(p):
                has_structural_proposal = True
                continue
            if name and "{prompt}" in template:
                out.append(
                    FixCandidate(
                        tier=FixTier.L1_PROMPT,
                        name=name,
                        kind="template",
                        payload={"prompt_template": template},
                    )
                )
        # Image tasks always receive one conservative, declarative grounding
        # control. A judge can overfit a diagnosis split with a narrow prompt;
        # this candidate establishes whether simply forcing visual evidence
        # before world knowledge helps, and it is selected/validated by the
        # same held-out procedure as every other repair.
        if has_images and not has_structural_proposal and "visual_grounding" not in prior_names:
            out.insert(
                0,
                FixCandidate(
                    tier=FixTier.L1_PROMPT,
                    name="visual_grounding",
                    kind="template",
                    source="default",
                    payload={
                        "prompt_template": (
                            "Inspect the image carefully for {failure_axis}. Work from visible "
                            "evidence, then give only the final answer requested.\n\n{prompt}"
                        )
                    },
                ),
            )
        # Dual of the direction gate that withholds VCD/ICD/OPERA/PAI/IFCD on
        # a false-No-dominant slice (_binary_hallucination_direction): those
        # methods are all suppressive (push away from asserting an object),
        # which is the wrong direction for under-claiming. This is the
        # opposite-direction lever within our own framework -- a prompt that
        # asks the model to accept partial/ambiguous visual evidence rather
        # than requiring certainty before answering Yes. It is scoped to the
        # same batch-level diagnosis signal (never a specific case's outcome)
        # and only proposed for binary tasks where the direction is not
        # false-Yes-dominant, so it is never offered alongside (and diluting)
        # the already-validated false-Yes-side candidates.
        if (
            has_images
            and tasks == {"yes_no"}
            and not binary_hallucination_supported
            and "assertive_grounding" not in prior_names
        ):
            out.append(
                FixCandidate(
                    tier=FixTier.L1_PROMPT,
                    name="assertive_grounding",
                    kind="template",
                    source="default",
                    payload={
                        "prompt_template": (
                            "Inspect the image for {failure_axis}. If there is plausible visual "
                            "evidence for the object or attribute in the question -- even if "
                            "partial, small, or ambiguous -- answer Yes. Only answer No if you "
                            "are confident no such evidence is present anywhere in the "
                            "image.\n\n{prompt}"
                        )
                    },
                )
            )
        if not out and not has_structural_proposal and "attend_carefully" not in prior_names:
            # The judge gave nothing usable: one conservative default. Worded
            # for the modality — a text-only batch must not be told to
            # "examine the image".
            template = (
                "Examine the image carefully, including small, subtle and "
                "low-contrast regions, before answering. {prompt}"
                if has_images else
                "Work through this carefully step by step, re-read the question "
                "before committing, and double-check the final answer against the "
                "question's own wording before answering. {prompt}"
            )
            out = [
                FixCandidate(
                    tier=FixTier.L1_PROMPT,
                    name="attend_carefully",
                    kind="template",
                    source="default",
                    payload={"prompt_template": template},
                )
            ]
        return out[: self.max_judge_candidates]

    def _l2_candidates(
        self,
        hyp_lines: str,
        examples: str,
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        *,
        has_images: bool = False,
        model: "Model | None" = None,
        tasks: "set[str] | None" = None,
        context_block: str = "",
        catalog: "str | None" = None,
    ) -> "list[FixCandidate]":
        if catalog is None:
            catalog = catalog_text() if has_images else _TEXT_ONLY_CATALOG_NOTE
        proposals = (
            []
            if self._paper_methods_only
            else self._ask_judge(
                _L2_PROMPT.format(
                    hypotheses=hyp_lines,
                    examples=examples,
                    k=self.max_judge_candidates,
                    catalog=catalog,
                    context=context_block,
                )
                + prior_text
            )
        )
        out: "list[FixCandidate]" = []
        for p in proposals:
            spec = PipelineSpec.from_dict(p) if isinstance(p, dict) else None
            if spec is not None:
                out.append(
                    FixCandidate(tier=FixTier.L2_SCAFFOLD, name=spec.name, payload=spec.to_dict())
                )
        if not out:
            image_defaults = [
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="answer_bbox_crop",
                    source="default",
                    payload=PipelineSpec(
                        name="answer_bbox_crop",
                        image_ops=[
                            {
                                "tool": "crop_case_bbox",
                                "params": {
                                    "bbox_key": "answer_bbox_xyxy_norm",
                                    "padding": 0.40,
                                    "min_size_frac": 0.12,
                                    "sharpen_factor": 3.0,
                                    "contrast_factor": 1.4,
                                },
                            },
                        ],
                        prompt_template=(
                            "The image may have been cropped and enhanced around "
                            "the visual region that contains the answer. Read the "
                            "visible text or number carefully, then answer the "
                            "question. {prompt}"
                        ),
                    ).to_dict(),
                ),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="annotate_horizontal_band_count",
                    source="default",
                    payload=PipelineSpec(
                        name="annotate_horizontal_band_count",
                        image_ops=[
                            {
                                "tool": "annotate_horizontal_band_count",
                                "params": {
                                    "min_delta": 18.0,
                                    "color_delta": 35.0,
                                    "min_count": 8,
                                },
                            },
                        ],
                        prompt_template=(
                            "A visual counting overlay may have been added to the "
                            "image. If a COUNT value is visible, use that value. "
                            "{prompt}"
                        ),
                    ).to_dict(),
                ),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="separate_horizontal_bands",
                    source="default",
                    payload=PipelineSpec(
                        name="separate_horizontal_bands",
                        image_ops=[
                            {
                                "tool": "separate_horizontal_bands",
                                "params": {"min_delta": 18.0, "color_delta": 35.0},
                            },
                        ],
                        prompt_template=(
                            "The image has been preprocessed so adjacent colored "
                            "horizontal bands, if present, are separated by gray "
                            "gaps. Count every colored band. {prompt}"
                        ),
                    ).to_dict(),
                ),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="salient_crop",
                    source="default",
                    payload=PipelineSpec(
                        name="salient_crop",
                        image_ops=[
                            {
                                "tool": "crop_salient_region",
                                "params": {"padding": 0.04, "min_delta": 18.0},
                            },
                        ],
                    ).to_dict(),
                ),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="zoom_equalize",
                    source="default",
                    payload=PipelineSpec(
                        name="zoom_equalize",
                        image_ops=[
                            {"tool": "zoom_center", "params": {"factor": 1.6}},
                            {"tool": "equalize", "params": {}},
                        ],
                    ).to_dict(),
                ),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="upscale_sharpen",
                    source="default",
                    payload=PipelineSpec(
                        name="upscale_sharpen",
                        image_ops=[
                            {"tool": "upscale", "params": {"factor": 2.0}},
                            {"tool": "sharpen", "params": {"factor": 2.0}},
                        ],
                    ).to_dict(),
                ),
            ]
            text_defaults = [
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="self_refine",
                    source="default",
                    payload=PipelineSpec(
                        name="self_refine",
                        prompt_template="{prompt}",
                        strategy="self_refine",
                    ).to_dict(),
                ),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name="least_to_most",
                    source="default",
                    payload=PipelineSpec(
                        name="least_to_most",
                        prompt_template="{prompt}",
                        strategy="least_to_most",
                    ).to_dict(),
                ),
            ]
            # ``crop_case_bbox`` is only applicable when an upstream evaluator
            # supplies a bbox. For ordinary image cases, start with transforms
            # that actually touch the image instead of burning a candidate on a
            # structural no-op.
            defaults = image_defaults if has_images else text_defaults
            # self_refine/least_to_most were only ever offered for text-only
            # cases, even though run_pipeline already threads the case image
            # through every call of a multi-call strategy (fix_tools.py) --
            # nothing about them is text-specific. On a binary/grounding task
            # (yes_no) the image transforms above are the right first lever;
            # on a genuine multi-step reasoning task (multiple_choice,
            # exact_or_numeric -- MMMU, ChartQA) a purely visual transform
            # cannot fix a reasoning error the model makes after it has
            # already seen the image correctly, so offer self_refine there
            # too, prioritised ahead of the sharpen/crop family for that task
            # shape.
            # Positive allowlist, not "anything that isn't yes_no": unknown or
            # missing task metadata (common in unit tests and some non-VLM
            # integrations) must not silently opt into this branch.
            reasoning_task = bool(
                tasks and tasks & {"multiple_choice", "exact_or_numeric", "vqa_consensus"}
            )
            if has_images and reasoning_task:
                # self_refine (deterministic, single path, critique-then-revise)
                # and self_consistency (stochastic, multiple independent paths,
                # majority vote) target different failure shapes: self_refine
                # helps when the model's first pass missed something a second
                # look would catch; self_consistency helps when greedy decoding
                # is stuck on one answer but the model's distribution actually
                # has support elsewhere. Majority-vote aggregation already
                # exists in run_pipeline (fix_tools.py, n_samples > 1) but had
                # never been registered as a default candidate for any task.
                reasoning_defaults = []
                if "self_refine" not in prior_names:
                    reasoning_defaults.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD,
                            name="self_refine",
                            source="default",
                            payload=PipelineSpec(
                                name="self_refine",
                                prompt_template="{prompt}",
                                strategy="self_refine",
                            ).to_dict(),
                        )
                    )
                if "self_consistency_5" not in prior_names:
                    reasoning_defaults.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD,
                            name="self_consistency_5",
                            source="default",
                            payload=PipelineSpec(
                                name="self_consistency_5",
                                prompt_template="{prompt}",
                                n_samples=5,
                                generation_kwargs={"do_sample": True, "temperature": 0.7},
                            ).to_dict(),
                        )
                    )
                defaults = reasoning_defaults + defaults
            if has_images:
                if (
                    model is not None
                    and callable(getattr(model, "generate_detector_visual_search", None))
                    and "detector_visual_search_consensus" not in prior_names
                ):
                    defaults.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD,
                            name="detector_visual_search_consensus",
                            kind="detector_visual_search",
                            source="paper_inspired_black_box",
                            payload={"decision": "unanimous_crop_override"},
                        )
                    )
                # V* (Wu et al.) establishes that question-guided visual
                # search can repair failures caused by a missed local detail.
                # An opt-in backend performs locate -> crop -> re-ask without
                # access to labels.  It is deliberately distinct from the
                # trained SEAL reproduction: endpoint users get a testable
                # method-family control, never a false equivalence claim.
                if (
                    model is not None
                    and callable(getattr(model, "generate_visual_search", None))
                    and "guided_visual_search_consensus" not in prior_names
                ):
                    defaults.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD,
                            name="guided_visual_search_consensus",
                            kind="visual_search",
                            source="paper_inspired_black_box",
                            # SEAL is a search/query mechanism, not a single
                            # arbitrary crop. Preserve the full scene and give the
                            # answer pass three increasingly contextual views of
                            # the controller-localised target.
                            payload={
                                "min_side": 0.10,
                                "max_side": 0.50,
                                "scales": [0.10, 0.25, 0.50],
                                "decision": "unanimous_crop_override",
                            },
                        )
                    )
                preferred = [
                    "self_refine",
                    "self_consistency_5",
                    "detector_visual_search_consensus",
                    "guided_visual_search_consensus",
                    "salient_crop",
                    "upscale_sharpen",
                    "zoom_equalize",
                    "answer_bbox_crop",
                    "annotate_horizontal_band_count",
                    "separate_horizontal_bands",
                ]
                defaults.sort(key=lambda candidate: preferred.index(candidate.name))
            out = [c for c in defaults if c.name not in prior_names]
        if self._paper_methods_only:
            out = [candidate for candidate in out if candidate.source.startswith("paper_")]
        return out[: self.max_judge_candidates]

    def _l2_coded_candidate(
        self,
        hyp_lines: str,
        examples: str,
        model: "Model",
        prior_text: str = "",
        *,
        context_block: str = "",
        catalog: "str | None" = None,
        text_only: bool = False,
    ) -> "list[FixCandidate]":
        """The coding agent writes a brand-new pipeline (CLI first, judge fallback).

        Allocates its trial up front (not after dedup, unlike declarative
        candidates) because the CLI agent needs a workdir *now* to write
        ``pipeline.py`` into — coded candidates dedup on code text, which is
        ~never identical run-to-run, so the orphan-folder risk this would
        otherwise create is negligible.
        """
        from evalvitals.core.capability import Capability
        from evalvitals.eval_agent.stages.fix_pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        trial = (
            self._run_context.new_trial("fixes", "coded_pipeline")
            if self._run_context is not None
            else None
        )
        enable_attend = (
            self.max_tier >= FixTier.L3A_INTERNALS_READ
            and Capability.ATTENTION in getattr(model, "capabilities", frozenset())
        )
        attend_hint = (
            (
                "\n- A function  model_attend(case_id, prompt=None) -> "
                '{"grid": [[float,...],...], "shape": [H, W]}  is ALSO defined: '
                "the model's attention heatmap over image patches (read-only "
                "internals). Use it e.g. to find where the model looks, then "
                "crop_region there and re-ask."
            )
            if enable_attend
            else ""
        )
        code, source, prompt, raw = "", "", "", ""
        if catalog is None:
            catalog = _TEXT_ONLY_CATALOG_NOTE if text_only else catalog_text()
        selection_guidance = self._code_selection_guidance(prior_text)
        base = dict(
            hypotheses=hyp_lines,
            examples=examples,
            catalog=catalog,
            cases_file=CASES_FILENAME,
            marker=RESULT_MARKER,
            attend_hint=attend_hint,
            context=context_block,
            selection_guidance=selection_guidance,
        )
        if self._cli_config is not None and self._cli_config.provider != "llm":
            prompt = (
                _L2_CODE_PROMPT.format(fences_hint=", written to a file named pipeline.py", **base)
                + prior_text
            )
            code, raw = self._write_code_cli(prompt, trial)
            source = f"cli:{self._cli_config.provider}"
        if not code.strip() and self._judge is not None:
            prompt = (
                _L2_CODE_PROMPT.format(fences_hint=" inside a ```python code block", **base)
                + prior_text
            )
            try:
                raw = str(self._judge.generate(prompt))
            except Exception as exc:
                logger.warning("FixAgent: code-writing judge call failed: %s", exc)
                raw = ""
            code = _extract_code(raw)
            # Syntax gate: a judge that answered in prose must not become a
            # "coded pipeline" candidate (the CLI path returns real files).
            if code.strip():
                import ast

                try:
                    ast.parse(code)
                except SyntaxError:
                    logger.warning("FixAgent: judge code failed to parse; dropped")
                    code = ""
            source = "judge"
        self._emit_codegen(
            "coded_pipeline", prompt, source, code, raw, ok=bool(code.strip()), trial=trial
        )
        if not code.strip():
            return []
        tier = FixTier.L3A_INTERNALS_READ if enable_attend else FixTier.L2_SCAFFOLD
        return [
            FixCandidate(
                tier=tier,
                name="coded_pipeline",
                kind="code",
                payload={
                    "code": code,
                    "enable_attend": enable_attend,
                    "text_only": bool(text_only),
                    # Prompt instructions are advisory; the host bridge also
                    # enforces the selection rule without seeing gold labels.
                    "consensus_min_support": 2 if self.max_repair_rounds > 1 else 3,
                    "max_calls_per_case": 4,
                },
                source=source,
                trial=trial,
            )
        ]

    def _code_selection_guidance(self, prior_text: str = "") -> str:
        """Tell codegen whether this is discovery, revision, or one-shot use."""
        if prior_text:
            return (
                "- This is a FEEDBACK-DRIVEN EXPLORE revision. Use the prior "
                "helped/hurt prompts and implementation below as training "
                "feedback. Gate the revised fix on a prompt/task subtype that "
                "actually benefited and return the direct baseline elsewhere. "
                "If the prior attempt repaired zero cases, abandon its "
                "override mechanism instead of merely retuning it."
            )
        if self.max_repair_rounds > 1:
            return (
                "- This is EXPLORE round 1, used to learn which task subtypes "
                "benefit before a later candidate is frozen. Treat the direct "
                "answer as the baseline and keep it on ties, but you may use "
                "a controlled 2-of-3 alternative consensus so the paired "
                "helped/hurt feedback is informative. Never hard-code answers "
                "or compute the benchmark task outside the model."
            )
        return (
            "- Treat the ORIGINAL direct answer as the safety baseline. Keep "
            "it unless all 3 independent enhanced/reasoned passes agree on a "
            "different answer and none supports the baseline. Do not use "
            "unconditional majority replacement."
        )

    def _write_code_cli(self, prompt: str, trial: "Trial | None" = None) -> "tuple[str, str]":
        from pathlib import Path

        from evalvitals.agent_runtime.codegen import CodegenRunner

        workdir = Path(self._workdir(trial))
        result = CodegenRunner(self._cli_config).write_code(  # type: ignore[arg-type]
            prompt,
            workdir=workdir,
            timeout_sec=self._cli_config.timeout_sec,  # type: ignore[union-attr]
            preferred_filenames=("pipeline.py",),
        )
        self._last_usage = result.usage
        return result.code, result.raw_output

    def _workdir(self, trial: "Trial | None" = None) -> str:
        """Sandbox workdir for a coded run.

        With a *trial* (a ``RunContext`` is in play), each candidate gets its
        own durable ``<trial>/workspace/`` — no cross-attempt overwriting.
        Without one (legacy / no ``RunContext``), falls back to a single
        shared sandbox for the agent's lifetime, exactly as before.
        """
        if trial is not None:
            return str(trial.workspace)
        if self._sandbox is None:
            from evalvitals.agent_runtime.sandbox import ExperimentSandbox

            workdir = (
                self._run_context.new_workdir("fix") if self._run_context is not None else None
            )
            self._sandbox = ExperimentSandbox(workdir=workdir)
        return str(self._sandbox.workdir)

    @staticmethod
    def _format_prior(
        attempts: "list[FixValidation]", data: "CaseBatch | None" = None
    ) -> str:
        """Format failed prior attempts as a context block for judge prompts.

        Beyond "try a different mechanism", this surfaces the *partition* a
        prior candidate induced (helped vs hurt) so the next proposal can scope
        the fix instead of blindly transforming the whole population — a
        candidate that helps one subset and breaks another is asking to be
        gated by a predicate, not replaced (defect 3).
        """
        prompt_by_id = {
            case.id: str(getattr(getattr(case, "inputs", None), "prompt", ""))
            for case in (data or [])
        }
        items = []
        implementations = []
        heterogeneous = []
        for v in attempts:
            c = v.candidate
            if c.kind == "finetune_spec":
                continue
            effect = f"effect={v.effect:+.2f}" if v.effect is not None else "did not execute"
            broken = f", broke {v.broken_cases[:3]}" if v.broken_cases else ""
            trunc = (
                f"; {v.n_truncated} model call(s) hit the decode cap — its breaks are "
                "truncation, not the idea: give the model MORE room, never less"
                if v.n_truncated else ""
            )
            helped_prompts = [
                prompt_by_id.get(case_id, "")[:180]
                for case_id in v.fixed_cases[:8]
            ]
            hurt_prompts = [
                prompt_by_id.get(case_id, "")[:180]
                for case_id in v.broken_cases[:8]
            ]
            items.append(
                f"- [{c.tier.label}/{c.kind}] {c.name}: "
                f"{v.n_fixed} fixed / {v.n_broken} broken ({effect}{broken}{trunc}); "
                f"helped prompts={helped_prompts}; hurt prompts={hurt_prompts}"
            )
            code = c.payload.get("code") if c.kind == "code" else None
            if isinstance(code, str) and code.strip():
                implementations.append(
                    f"PREVIOUS IMPLEMENTATION ({c.name}):\n{code[:3000]}"
                )
            if v.n_fixed > 0 and v.n_broken > 0:
                heterogeneous.append(
                    f"  '{c.name}' HELPED {v.fixed_cases[:4]} but HURT "
                    f"{v.broken_cases[:4]} — these two groups differ; either gate "
                    "the fix so it only applies to the helped group, or target the "
                    f"mechanism that separates them. HELPED PROMPTS={helped_prompts}; "
                    f"HURT PROMPTS={hurt_prompts}."
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

    def _emit_codegen(
        self,
        name: str,
        prompt: str,
        source: str,
        code: str,
        raw: str,
        *,
        ok: bool,
        trial: "Trial | None" = None,
    ) -> None:
        """Log one codegen attempt; persist its prompt/code/thinking.

        With a *trial*, the files live under ``trial.root`` (alongside the
        rest of that attempt's record) instead of the run-global ``tools/`` —
        only a lean event (paths via ``extra["trial_root"]``) goes through
        :meth:`RunLogger.log_tool_codegen`, with no duplicate file copy.
        """
        extra = (
            {"cli_usage": self._last_usage}
            if source.startswith("cli:") and self._last_usage
            else None
        )
        if trial is not None:
            if prompt:
                trial.write(f"{name}_prompt.txt", prompt)
            if code:
                trial.write(f"{name}_code.py", code)
            if raw:
                trial.write(f"{name}_agent_thinking.txt", raw)
            extra = {**(extra or {}), "trial_root": str(trial.root)}
            prompt, code, raw = "", "", ""
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_tool_codegen(
                module="fix_pipeline",
                name=name,
                need="L2 coded repair pipeline",
                source=source,
                ok=ok,
                code=code,
                prompt=prompt,
                raw_output=raw,
                error="" if ok else "no code produced",
                extra=extra,
            )
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_tool_codegen failed: %s", exc)

    def _l3_candidates(
        self,
        hyp_lines: str,
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        *,
        has_images: bool = False,
        has_audio: bool = False,
        tasks: "set[str] | None" = None,
        binary_hallucination_supported: bool = True,
    ) -> "list[FixCandidate]":
        """Judge-parameterised configs of the pre-audited internals primitives."""
        out: "list[FixCandidate]" = []

        def finalize(options: "list[FixCandidate]") -> "list[FixCandidate]":
            # A frozen experiment may request a later catalogued candidate.
            # Apply that allowlist before the ordinary proposal cap; otherwise
            # an unrelated earlier default can silently erase the requested
            # paper route before ``_propose`` gets a chance to filter it.
            if self._candidate_allowlist is not None:
                options = [c for c in options if c.name in self._candidate_allowlist]
            return options[: self.max_judge_candidates]
        # Paper-method routes (OPERA/ViCrop/IFCD/PAI/TCD): each targets ONE
        # named failure mechanism, not "any failure this model/task shape can
        # exhibit". The condition below for each is STRUCTURAL eligibility
        # only -- can it physically run at all (capability, modality, task
        # shape, paper_method_fidelity, tier ceiling, not already tried)?
        # Whether its mechanism actually matches what was diagnosed is a
        # judgment call, not a fact you can `in`-check off the hypothesis
        # string -- so it is delegated to the judge below, over the catalog
        # of only the structurally-eligible candidates. No judge configured
        # -> _ask_judge returns [] -> no paper-method candidate is proposed;
        # there is no keyword fallback (a substring match is not a decision,
        # it is a hardcoded stand-in for one -- that was the actual gap here,
        # not that TCD specifically lacked a keyword list PAI/OPERA had).
        paper_fidelity = getattr(model, "paper_method_fidelity", None)
        opera_fidelity = paper_fidelity("opera") if callable(paper_fidelity) else "unavailable"
        vicrop_fidelity = paper_fidelity("vicrop") if callable(paper_fidelity) else "unavailable"
        pai_fidelity = paper_fidelity("pai") if callable(paper_fidelity) else "unavailable"
        # IFCD needs a trained TruthX representation editor. The available
        # public Vicuna artifact is useful for a controlled transfer trial,
        # but it is not IFCD's MSCOCO-trained editor, so it is opt-in through
        # ``allow_adapted_paper_methods`` and never passed off as native.
        ifcd_fidelity = paper_fidelity("ifcd") if callable(paper_fidelity) else "unavailable"
        # TCD (Li et al. 2026, arXiv:2604.15383) is a decoding-time repair for
        # unified audio-language models -- not a POPE-style yes_no task, so
        # scoped separately from the VCD/ICD/OPERA binary-hallucination
        # candidates. Needs ATTENTION (the decoder's audio-attention ratio
        # drives both the stability score and the per-step gate) on top of
        # the audio encoder's own hidden states, so it belongs at L3a like
        # OPERA, not L0 like VCD.
        tcd_fidelity = paper_fidelity("tcd") if callable(paper_fidelity) else "unavailable"

        _eligible: "list[tuple[str, FixCandidate, str]]" = []
        if (
            has_images
            and tasks == {"yes_no"}
            and binary_hallucination_supported
            and callable(getattr(model, "generate_opera_binary", None))
            and opera_fidelity == "native_binary_specialization"
            and "opera_overtrust_binary" not in prior_names
        ):
            _eligible.append((
                "opera_overtrust_binary",
                FixCandidate(
                    tier=FixTier.L3A_INTERNALS_READ,
                    name="opera_overtrust_binary",
                    kind="opera",
                    source="paper_default_binary_specialization",
                    payload={"num_attn_candidates": 5, "penalty_weight": 1.0},
                ),
                "OPERA: on binary yes/no questions, penalises next-token candidates "
                "that neglect image attention during decoding -- targets object/"
                "attribute hallucination caused by language priors overriding visual "
                "evidence (POPE-style over-trust).",
            ))
        # ViCrop (MLLMs Know Where to Look, ICLR 2025) is a read-only,
        # architecture-native paper route: task/general attention ratio,
        # adaptive crop, and an original+crop answer.  It must not be proposed
        # for a model whose vision/attention contract differs from LLaVA.
        if (
            has_images
            and callable(getattr(model, "generate_vicrop", None))
            and vicrop_fidelity == "native_selector_specialization"
            and "vicrop_relative_attention" not in prior_names
        ):
            _eligible.append((
                "vicrop_relative_attention",
                FixCandidate(
                    tier=FixTier.L3A_INTERNALS_READ,
                    name="vicrop_relative_attention",
                    kind="vicrop",
                    source="paper_default",
                    payload={"layer": 14},
                ),
                "ViCrop: uses attention to locate and crop the relevant image region "
                "before re-answering -- targets SMALL or LOCAL visual detail missed at "
                "the model's native resolution (tiny text, small objects, fine detail), "
                "not general hallucination and not a knowledge gap.",
            ))
        # This is a label-free deployment guard for transferring ViCrop to a
        # new local-detail benchmark, not a claim that the paper used it.
        if (
            has_images
            and callable(getattr(model, "generate_vicrop_consensus", None))
            and vicrop_fidelity == "native_selector_specialization"
            and "vicrop_consensus_guard" not in prior_names
        ):
            _eligible.append((
                "vicrop_consensus_guard",
                FixCandidate(
                    tier=FixTier.L3A_INTERNALS_READ,
                    name="vicrop_consensus_guard",
                    kind="vicrop_consensus",
                    source="safety_guard",
                    payload={"layer": 14},
                ),
                "ViCrop (consensus-guarded): the same small/local visual-detail "
                "crop-and-reanswer repair as vicrop_relative_attention, but only "
                "applies the cropped answer when it agrees with the original -- same "
                "target mechanism, a safety variant, not a different mechanism.",
            ))
        if (
            self.max_tier >= FixTier.L3B_INTERNALS_WRITE
            and has_images
            and tasks == {"yes_no"}
            and binary_hallucination_supported
            and callable(getattr(model, "generate_ifcd", None))
            and ifcd_fidelity == "adapted_truthx_artifact"
            and self._allow_adapted_paper_methods
            and "ifcd_truthx_contrast" not in prior_names
        ):
            _eligible.append((
                "ifcd_truthx_contrast",
                FixCandidate(
                    tier=FixTier.L3B_INTERNALS_WRITE,
                    name="ifcd_truthx_contrast",
                    kind="ifcd",
                    source="paper_adapted_truthx_artifact",
                    payload={"alpha": 0.1, "beta": 0.1, "edit_strength": 0.5, "top_layers": 15},
                ),
                "IFCD: on binary yes/no questions, contrasts internal representations "
                "against a trained truthfulness-editing direction -- targets the same "
                "object/attribute hallucination (language priors overriding visual "
                "evidence) as OPERA, via representation editing instead of "
                "decoding-time attention.",
            ))
        # PAI (ECCV 2024) is a distinct LLaVA mechanism: it changes the
        # image-attention logits while decoding, so it is an L3b intervention.
        # The native executor pairs the paper's attention branch with its
        # classifier-free-guidance cache; the source's pinned LLaVA stack is
        # still recorded as an architecture specialization.
        if (
            self.max_tier >= FixTier.L3B_INTERNALS_WRITE
            and has_images
            and (tasks != {"yes_no"} or binary_hallucination_supported)
            and callable(getattr(model, "generate_pai", None))
            and pai_fidelity == "native_attention_cfg_specialization"
            and "pai_image_attention" not in prior_names
        ):
            _eligible.append((
                "pai_image_attention",
                FixCandidate(
                    tier=FixTier.L3B_INTERNALS_WRITE,
                    name="pai_image_attention",
                    kind="pai",
                    source="paper_default_attention_cfg",
                    payload={
                        "alpha": 0.2,
                        "guidance_scale": 2.0,
                        "start_layer": 2,
                        "end_layer": 32,
                    },
                ),
                "PAI: amplifies image-attention logits during decoding via "
                "classifier-free guidance -- targets the same hallucination / "
                "language-prior-override mechanism as OPERA/IFCD, for open-ended "
                "(not just yes/no) tasks.",
            ))
        if (
            has_audio
            and tasks == {"multiple_choice"}
            and callable(getattr(model, "generate_tcd", None))
            and callable(getattr(model, "generate_tcd_baseline", None))
            and (
                tcd_fidelity == "native_layer_matched_stability"
                or (
                    tcd_fidelity == "adapted_truncated_layer_stability"
                    and self._allow_adapted_paper_methods
                )
            )
            and "tcd_temporal_blur" not in prior_names
        ):
            # No payload tuning: Table 6's own framing is "a single default
            # configuration... requires little tuning" -- the blur window and
            # update scale are already per-example adaptive (Eq. 5-6), so an
            # empty payload runs generate_tcd() at TCDHyperparams() defaults
            # rather than inventing a sweep the paper itself doesn't do.
            _eligible.append((
                "tcd_temporal_blur",
                FixCandidate(
                    tier=FixTier.L3A_INTERNALS_READ,
                    name="tcd_temporal_blur",
                    kind="tcd",
                    source="paper_default",
                    payload={},
                ),
                "TCD: contrasts decoding against a temporally-blurred version of the "
                "audio -- targets under-weighting of TRANSIENT, fine-grained acoustic "
                "detail (brief sounds, precise event timing/counting, telling multiple "
                "speakers apart) in favour of temporally-smooth context or language "
                "priors, on audio multiple-choice questions. Does NOT address a flat "
                "audio-perception knowledge gap (the answer is never in the model's "
                "sample pool at all) or a positional/letter-choice bias unrelated to "
                "audio content.",
            ))

        if _eligible:
            by_name = {name: cand for name, cand, _desc in _eligible}
            catalog_lines = "\n".join(f"- {name}: {desc}" for name, _cand, desc in _eligible)
            picked: "set[str]" = set()
            for p in self._ask_judge(
                _PAPER_METHOD_PROMPT.format(
                    hypotheses=hyp_lines, catalog=catalog_lines, k=self.max_judge_candidates,
                )
                + prior_text
            ):
                name = str(p.get("name", ""))
                if name in by_name and name not in picked:
                    picked.add(name)
                    out.append(by_name[name])

        catalog = primitives_catalog_text(model, self.max_tier)
        if not catalog:
            if not out:
                logger.info("FixAgent: no L3 primitive is available for %r", model)
            return finalize(out)
        for p in self._ask_judge(
            _L3_PROMPT.format(hypotheses=hyp_lines, catalog=catalog, k=self.max_judge_candidates)
            + prior_text
        ):
            prim = INTERNALS_PRIMITIVES.get(str(p.get("primitive", "")))
            if prim is None or prim.tier > self.max_tier or not prim.available(model):
                continue
            out.append(
                FixCandidate(
                    tier=prim.tier,
                    name=prim.name,
                    kind="primitive",
                    payload={"primitive": prim.name, "params": dict(p.get("params") or {})},
                )
            )
        if not out:
            # Internals-WRITE defaults only; reads (L3a) are authored by the
            # coded pipeline against model_attend(), not proposed as primitives.
            defaults = {
                "visual_embedding_boost": {"gamma": 1.5},
            }
            for name, params in defaults.items():
                if name in prior_names:
                    continue
                prim = INTERNALS_PRIMITIVES[name]
                if prim.tier <= self.max_tier and prim.available(model):
                    out.append(
                        FixCandidate(
                            tier=prim.tier,
                            name=name,
                            kind="primitive",
                            source="default",
                            payload={"primitive": name, "params": params},
                        )
                    )
        return finalize(out)

    def _l4_candidates(self, hyp_lines: str) -> "list[FixCandidate]":
        """L4 recipe — recorded for the escalation decision; executor is TODO."""
        spec: "FinetuneSpec | None" = None
        if self._judge is not None:
            raw = self._ask_judge_object(_L4_PROMPT.format(hypotheses=hyp_lines))
            if raw:
                spec = FinetuneSpec(
                    dataset_recipe=str(raw.get("dataset_recipe", "")),
                    method=str(raw.get("method", "lora")),
                    target=str(raw.get("target", "llm")),
                    eval_protocol=str(raw.get("eval_protocol", ""))
                    or FinetuneSpec("").eval_protocol,
                    rationale=str(raw.get("rationale", "")),
                )
        if spec is None or not spec.dataset_recipe:
            spec = FinetuneSpec(
                dataset_recipe="TODO: synthesise training data generalising the "
                "verified failure mechanism",
                rationale="default skeleton — no judge recipe available",
            )
        return [
            FixCandidate(
                tier=FixTier.L4_PARAMETERS,
                name="finetune_recipe",
                kind="finetune_spec",
                payload=spec.to_dict(),
                source="judge" if self._judge is not None else "default",
            )
        ]

    def _ask_judge_object(self, prompt: str) -> "dict[str, Any]":
        """Single-JSON-object variant of :meth:`_ask_judge`."""
        if self._judge is None:
            return {}
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            return {}
        match = re.search(
            r"\{.*\}", re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL), flags=re.DOTALL
        )
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _ask_judge(self, prompt: str) -> "list[dict[str, Any]]":
        if self._judge is None:
            return []
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            return []
        match = re.search(
            r"\[.*\]", re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL), flags=re.DOTALL
        )
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("FixAgent: unparseable judge proposal; using defaults")
            return []
        return [p for p in parsed if isinstance(p, dict)] if isinstance(parsed, list) else []

    # -- strategy compilation + validation --------------------------------

    def _baseline(
        self, model: "Model", data: "CaseBatch"
    ) -> "tuple[dict[str, Optional[bool]], set[str]]":
        """Measure the unmodified baseline as a per-case PASS RATE.

        Returns ``(modal_scores, unstable_ids)`` for callers that think in
        booleans, and stores the rates in ``self._baseline_rates`` /
        ``self._baseline_n`` for the paired-rates test.

        * ``baseline_repeats == 1`` — the frozen ``observed`` output is the
          baseline (one sample; rate 0/1). Re-generating it would make a
          supposedly paired comparison depend on endpoint non-determinism and
          waste a call per case. Cases without an ``observed`` are generated
          once.
        * ``baseline_repeats == k > 1`` — the frozen sample counts as sample 1
          and ``k-1`` fresh samples are drawn (all ``k`` fresh when there is no
          frozen one), so each case gets a rate in ``{0, 1/k, ..., 1}``. A case
          with ``0 < rate < 1`` is *unstable* (its baseline answer is a coin);
          it is REPORTED, not dropped: the paired-rates test weighs it by how
          much a candidate moves its rate, which is the honest accounting for
          a stochastic model — dropping it removed exactly the cases a
          variance-reduction scaffold repairs.
        """
        k = max(1, int(self._baseline_repeats))
        samples: "dict[str, list[bool]]" = {c.id: [] for c in data}
        need: "list[tuple[Any, int]]" = []
        for case in data:
            observed = getattr(case, "observed", None)
            if observed is not None:
                s0 = score_to_bool(self._score(case, str(observed)))
                if s0 is not None:
                    samples[case.id].append(bool(s0))
            fresh = k - len(samples[case.id])
            if fresh > 0:
                need.append((case, fresh))
        if need:
            self._sample_baseline(model, need, samples)
        scores: "dict[str, Optional[bool]]" = {}
        rates: "dict[str, Optional[float]]" = {}
        counts: "dict[str, int]" = {}
        unstable: "set[str]" = set()
        for cid, obs in samples.items():
            counts[cid] = len(obs)
            if not obs:
                scores[cid] = None
                rates[cid] = None
                continue
            rate = sum(1 for o in obs if o) / len(obs)
            rates[cid] = rate
            scores[cid] = rate >= 0.5  # modal; ties -> True
            if 0.0 < rate < 1.0:
                unstable.add(cid)
        self._baseline_rates = rates
        self._baseline_n = counts
        return scores, unstable

    def _sample_baseline(
        self,
        model: "Model",
        need: "list[tuple[Any, int]]",
        samples: "dict[str, list[bool]]",
    ) -> None:
        """Draw the missing fresh baseline samples (threaded like candidates)."""
        jobs = [case for case, n in need for _ in range(n)]

        def one(case: Any) -> "tuple[str, Optional[bool]]":
            try:
                output = str(model.generate(case.inputs))
            except Exception as exc:
                logger.debug("FixAgent: baseline generate failed on %s: %s", case.id, exc)
                return case.id, None
            return case.id, score_to_bool(self._score(case, output))

        if self._concurrency > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                results = list(pool.map(one, jobs))
        else:
            results = [one(case) for case in jobs]
        for cid, s in results:
            if s is not None:
                samples[cid].append(bool(s))

    def _baseline_fresh(
        self, model: "Model", data: "CaseBatch"
    ) -> "tuple[dict[str, Optional[bool]], set[str]]":
        """Generate a baseline ignoring any frozen observation (k fresh samples)."""
        k = max(1, int(self._baseline_repeats))
        samples: "dict[str, list[bool]]" = {c.id: [] for c in data}
        self._sample_baseline(model, [(c, k) for c in data], samples)
        scores: "dict[str, Optional[bool]]" = {}
        unstable: "set[str]" = set()
        for cid, obs in samples.items():
            if not obs:
                scores[cid] = None
                continue
            rate = sum(1 for o in obs if o) / len(obs)
            scores[cid] = rate >= 0.5
            if 0.0 < rate < 1.0:
                unstable.add(cid)
        return scores, unstable

    def _candidate_rates(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch"
    ) -> "tuple[dict[str, Optional[float]], int]":
        """Per-case candidate PASS RATE over ``candidate_repeats`` passes.

        Coded pipelines / internals primitives / fine-tune recipes run once
        (they are expensive and vote internally when they want to); template
        and spec candidates are repeated ``candidate_repeats`` times. Returns
        ``(rates, n_passes)``; a case is ``None`` when every pass was unscorable.
        """
        m = int(self._candidate_repeats) if candidate.kind in ("template", "spec") else 1
        m = max(1, m)
        tallies: "dict[str, list[bool]]" = {c.id: [] for c in data}
        for _ in range(m):
            scores = self._candidate_scores(candidate, model, data)
            for cid, s in scores.items():
                b = score_to_bool(s)
                if b is not None:
                    tallies.setdefault(cid, []).append(bool(b))
        rates = {
            cid: (sum(1 for o in obs if o) / len(obs) if obs else None)
            for cid, obs in tallies.items()
        }
        return rates, m

    def _applies(self, candidate: FixCandidate, case: "FailureCase") -> bool:
        """Whether *candidate* is applicable to *case* (defect 1).

        An explicit predicate wins.  Otherwise applicability is structural: a
        prompt template that is the identity and image ops that leave the image
        unchanged mean the candidate never touches the case, so it must not be
        credited or blamed for it.  Coded/primitive/finetune candidates run
        their own per-case logic, so they are treated as universally applicable.
        """
        if candidate.predicate is not None:
            try:
                return bool(candidate.predicate(case))
            except Exception as exc:
                logger.debug("FixAgent: predicate failed on %s: %s", case.id, exc)
                return True
        if candidate.kind == "template":
            return str(candidate.payload.get("prompt_template", "{prompt}")).strip() != "{prompt}"
        if candidate.kind == "spec":
            spec = PipelineSpec.from_dict(candidate.payload)
            if spec is None:
                return True
            try:
                return spec_changes_input(spec, case)
            except Exception as exc:
                logger.debug("FixAgent: applicability check failed on %s: %s", case.id, exc)
                return True
        return True

    def _strategy(
        self, candidate: FixCandidate
    ) -> "Callable[[Model, FailureCase], Optional[bool]]":
        """Compile a candidate to a per-case success function (ab_runner shape)."""
        if candidate.kind == "vcd":

            def vcd(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_vcd = getattr(model, "generate_vcd")
                    output = generate_vcd(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("VCD generation failed on %s: %s", case.id, exc)
                    return None

            return vcd
        if candidate.kind == "aad":

            def aad(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_aad = getattr(model, "generate_aad")
                    output = generate_aad(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("AAD generation failed on %s: %s", case.id, exc)
                    return None

            return aad
        if candidate.kind == "icd":

            def icd(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_icd = getattr(model, "generate_instruction_cd")
                    output = generate_icd(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("ICD generation failed on %s: %s", case.id, exc)
                    return None

            return icd
        if candidate.kind == "vicrop":

            def vicrop(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_vicrop = getattr(model, "generate_vicrop")
                    output = generate_vicrop(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("ViCrop generation failed on %s: %s", case.id, exc)
                    return None

            return vicrop
        if candidate.kind == "vicrop_consensus":

            def vicrop_consensus(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_vicrop = getattr(model, "generate_vicrop_consensus")
                    output = generate_vicrop(
                        case.inputs,
                        baseline_answer=str(getattr(case, "observed", "")),
                        **candidate.payload,
                    )
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("ViCrop consensus generation failed on %s: %s", case.id, exc)
                    return None

            return vicrop_consensus
        if candidate.kind == "opera":

            def opera(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_opera = getattr(model, "generate_opera_binary")
                    output = generate_opera(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("OPERA binary generation failed on %s: %s", case.id, exc)
                    return None

            return opera
        if candidate.kind == "ifcd":

            def ifcd(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_ifcd = getattr(model, "generate_ifcd")
                    output = generate_ifcd(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("IFCD generation failed on %s: %s", case.id, exc)
                    return None

            return ifcd
        if candidate.kind == "pai":

            def pai(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_pai = getattr(model, "generate_pai")
                    output = generate_pai(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("PAI generation failed on %s: %s", case.id, exc)
                    return None

            return pai
        if candidate.kind == "tcd":

            def tcd(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    generate_tcd = getattr(model, "generate_tcd")
                    output = generate_tcd(case.inputs, **candidate.payload)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("TCD generation failed on %s: %s", case.id, exc)
                    return None

            return tcd
        if candidate.kind == "visual_search":

            def visual_search(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    search = getattr(model, "generate_visual_search")
                    output = search(
                        case.inputs,
                        baseline_answer=getattr(case, "observed", None),
                        **candidate.payload,
                    )
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("Guided visual search failed on %s: %s", case.id, exc)
                    return None

            return visual_search
        if candidate.kind == "detector_visual_search":

            def detector_visual_search(model: "Model", case: "FailureCase") -> "Optional[bool]":
                try:
                    search = getattr(model, "generate_detector_visual_search")
                    output = search(
                        case.inputs,
                        baseline_answer=getattr(case, "observed", None),
                        **candidate.payload,
                    )
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug("Detector visual search failed on %s: %s", case.id, exc)
                    return None

            return detector_visual_search
        if candidate.kind == "template":
            template = candidate.payload["prompt_template"]

            def l1(model: "Model", case: "FailureCase") -> "Optional[bool]":
                inp = case.inputs
                metadata = getattr(case, "metadata", {}) or {}
                template_context = {str(key): value for key, value in metadata.items()}
                template_context["prompt"] = str(getattr(inp, "prompt", ""))
                template_context.setdefault("failure_axis", "the relevant visual evidence")
                # Inside the try, not before it: rendering the template is as
                # capable of failing as generating from it, and a single bad
                # case must score None rather than abort the whole validation.
                try:
                    # dataclasses.replace, not a bare Inputs(prompt=..., image=...):
                    # that silently dropped .video/.audio, so every L1 candidate was
                    # unconditionally inapplicable (generate() raising on the
                    # missing required modality field, caught below, scored as
                    # None for every case) on any non-image FailureCase.
                    new_inputs = dataclasses.replace(
                        inp, prompt=safe_format(template, template_context)
                    )
                    output = str(model.generate(new_inputs))
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, output))
                except Exception:
                    return None

            return l1

        spec = PipelineSpec.from_dict(candidate.payload)
        if spec is None:  # already validated at proposal time; belt and braces
            return lambda model, case: None

        def declarative(model: "Model", case: "FailureCase") -> "Optional[bool]":
            capture: "dict[str, Any]" = {}
            result = run_pipeline(model, case, spec, self._score, capture=capture)
            winner = capture.get("winner")
            if winner is None and capture.get("outputs"):
                winner = capture["outputs"][0]
            self._record_output(case.id, winner)
            return result

        return declarative

    def _candidate_scores(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch"
    ) -> "dict[str, Optional[bool]]":
        """Per-case success of one candidate (batch path for coded pipelines)."""
        if candidate.kind == "primitive":
            prim = INTERNALS_PRIMITIVES[candidate.payload["primitive"]]
            return prim.run(model, data, self._score, candidate.payload.get("params"))
        if candidate.kind == "code":
            result = self._run_coded(candidate, model, data)
            for cid, output in result.outputs.items():
                self._record_output(cid, output)
            if result.ok:
                self._frozen_model_control(candidate, data)
            return score_outputs(result, data, self._score)
        if candidate.kind == "finetune_spec":
            result = run_lora_repair(model, self._finetune_pool, data, candidate.payload, self._score)
            if isinstance(candidate.payload, dict):
                candidate.payload["exec_error"] = "" if result.ok else result.error
            return result.scores
        strategy = self._strategy(candidate)
        cases = list(data)

        def guarded(case: "FailureCase") -> "tuple[str, Optional[bool]]":
            # One case's failure (a template that cannot render, an adapter
            # error) scores None for THAT case; it must never abort the whole
            # candidate — let alone the fix stage.
            try:
                return case.id, strategy(model, case)
            except Exception as exc:
                logger.warning("FixAgent: %s failed on case %s: %s", candidate.name, case.id, exc)
                return case.id, None

        if self._concurrency > 1 and len(cases) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                results = list(pool.map(guarded, cases))
            return dict(results)
        return dict(guarded(case) for case in cases)

    def _record_output(self, case_id: str, output: Any) -> None:
        """Remember what a candidate produced for *case_id* (see FixValidation.outputs)."""
        if output is None:
            return
        self._captured[str(case_id)] = str(output)

    def _run_coded(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch"
    ) -> "CodedPipelineResult":
        """Run a coded candidate; a failed run gets ONE coder repair round.

        The execution error (strict-bridge message, timeout, traceback tail) is
        fed back verbatim — the coder fixes its own contract violation instead
        of the candidate silently dying.  The final error, if any, is stashed
        in ``payload["exec_error"]`` for honest escalation accounting.
        """
        workdir = self._workdir(candidate.trial)
        result = run_coded_pipeline(
            candidate.payload["code"],
            model,
            data,
            workdir=workdir,
            timeout_sec=self._exec_timeout_sec,
            enable_attend=bool(candidate.payload.get("enable_attend")),
            max_tokens_floor=self._max_tokens_floor,
            concurrency=self._concurrency,
            consensus_min_support=int(candidate.payload.get("consensus_min_support", 0)),
            max_calls_per_case=int(candidate.payload.get("max_calls_per_case", 0)),
        )
        if not result.ok and self.codegen_available:
            logger.warning("FixAgent: coded pipeline failed (%s) — one repair round", result.error)
            repaired, source, raw = self._repair_code(candidate, result.error)
            self._emit_codegen(
                "coded_pipeline_repair",
                self._last_repair_prompt,
                source,
                repaired,
                raw,
                ok=bool(repaired.strip()),
                trial=candidate.trial,
            )
            if repaired.strip():
                candidate.payload["code"] = repaired
                result = run_coded_pipeline(
                    repaired,
                    model,
                    data,
                    workdir=workdir,
                    timeout_sec=self._exec_timeout_sec,
                    enable_attend=bool(candidate.payload.get("enable_attend")),
                    max_tokens_floor=self._max_tokens_floor,
                    concurrency=self._concurrency,
                    consensus_min_support=int(
                        candidate.payload.get("consensus_min_support", 0)
                    ),
                    max_calls_per_case=int(candidate.payload.get("max_calls_per_case", 0)),
                )
        candidate.payload["exec_error"] = "" if result.ok else result.error
        candidate.payload["selection_guard"] = {
            "min_support": int(candidate.payload.get("consensus_min_support", 0)),
            "n_guarded": result.n_guarded,
            "guarded_ids": result.guarded_ids,
        }
        if not result.ok:
            logger.warning("FixAgent: coded pipeline produced no result: %s", result.error)
        return result

    def _frozen_model_control(self, candidate: FixCandidate, data: "CaseBatch") -> None:
        """Re-run a coded candidate with the model frozen; record what it still
        gets right in ``payload["frozen_model_control"]``.

        ``_validate`` reads ``solved`` and drops those failing cases from the
        paired test: with every model call answered by the model's own recorded
        output, a repair can only have come from the code. A control that does
        not complete (the pipeline crashes or spins without a live model) is
        evidence the pipeline needs the model — nothing is discounted, and the
        reason is kept.
        """
        from pathlib import Path

        from evalvitals.eval_agent.stages.fix_pipeline import frozen_model_control

        workdir = Path(self._workdir(candidate.trial)) / "frozen_model_control"
        ctrl = frozen_model_control(
            candidate.payload["code"], data, workdir=workdir,
            timeout_sec=self._exec_timeout_sec,
            consensus_min_support=int(candidate.payload.get("consensus_min_support", 0)),
            max_calls_per_case=int(candidate.payload.get("max_calls_per_case", 0)),
        )
        solved: "list[str]" = []
        if ctrl.ok:
            for cid, ok in score_outputs(ctrl, data, self._score).items():
                if ok is True:
                    solved.append(cid)
        candidate.payload["frozen_model_control"] = {
            "ok": ctrl.ok,
            "error": ctrl.error,
            "n_calls": ctrl.n_calls,
            "solved": solved,
        }
        if solved:
            logger.info(
                "FixAgent: frozen-model control — %s still solves %d/%d case(s) with the "
                "model held at its recorded answers", candidate.name, len(solved), len(data),
            )

    def _repair_code(self, candidate: FixCandidate, error: str) -> "tuple[str, str, str]":
        """Ask the coder to fix its failed pipeline; returns (code, source, raw)."""
        from evalvitals.eval_agent.stages.fix_pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        attend_clause = (
            " and model_attend(case_id, prompt=None)"
            if candidate.payload.get("enable_attend")
            else ""
        )
        base = _REPAIR_PROMPT_BODY.format(
            error=error[:600],
            code=str(candidate.payload.get("code", ""))[:4000],
            attend_clause=attend_clause,
            catalog=(_TEXT_ONLY_CATALOG_NOTE if candidate.payload.get("text_only")
                     else catalog_text()),
            cases_file=CASES_FILENAME,
            marker=RESULT_MARKER,
        )
        code, source, raw = "", "", ""
        if self._cli_config is not None and self._cli_config.provider != "llm":
            self._last_repair_prompt = (
                base + "\nWrite the corrected code to a file named pipeline.py."
            )
            code, raw = self._write_code_cli(self._last_repair_prompt, candidate.trial)
            source = f"cli:{self._cli_config.provider}"
        if not code.strip() and self._judge is not None:
            self._last_repair_prompt = (
                base + "\nReturn ONLY the corrected Python code inside a ```python code block."
            )
            try:
                raw = str(self._judge.generate(self._last_repair_prompt))
            except Exception as exc:
                logger.warning("FixAgent: repair judge call failed: %s", exc)
                raw = ""
            code = _extract_code(raw)
            if code.strip():
                import ast

                try:
                    ast.parse(code)
                except SyntaxError:
                    code = ""
            source = "judge"
        return code, source, raw

    def _validate(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        baseline: "dict[str, Optional[bool]]",
        unstable: "set[str] | None" = None,
    ) -> FixValidation:
        v = FixValidation(candidate=candidate)
        unstable = unstable or set()
        rates_mode = self._baseline_repeats > 1 or self._candidate_repeats > 1
        v.noise_model = "paired_rates" if rates_mode else "mcnemar"
        self._captured = {}
        truncated_before = _truncated_count(model)
        if rates_mode:
            cand_rates, m = self._candidate_rates(candidate, model, data)
            v.n_candidate_samples = m
            scores: "dict[str, Optional[bool]]" = {
                cid: (None if r is None else r >= 0.5) for cid, r in cand_rates.items()
            }
        else:
            scores = self._candidate_scores(candidate, model, data)
            cand_rates = {cid: (None if score_to_bool(sc) is None else float(bool(score_to_bool(sc))))
                          for cid, sc in scores.items()}
        truncated_after = _truncated_count(model)
        if truncated_before is not None and truncated_after is not None:
            v.n_truncated = max(0, truncated_after - truncated_before)
        v.outputs = dict(self._captured)
        self._captured = {}
        if isinstance(candidate.payload, dict):
            v.exec_error = str(candidate.payload.get("exec_error", "") or "")
        # Baseline rates: measured by _baseline (frozen sample + k-1 fresh);
        # when a caller hands in bare booleans (tests, external drivers) they
        # are 0/1 rates.
        base_rates: "dict[str, Optional[float]]" = {}
        for case in data:
            r = self._baseline_rates.get(case.id) if self._baseline_rates else None
            if r is None:
                b = score_to_bool(baseline.get(case.id))
                r = None if b is None else float(b)
            base_rates[case.id] = r
        if self._baseline_n:
            v.n_baseline_samples = max([1] + [int(n) for n in self._baseline_n.values()])

        n_fail = sum(1 for c in data if getattr(c.label, "value", None) == "fail")
        applicable_fail = 0
        base_vec: "list[float]" = []
        cand_vec: "list[float]" = []
        control = (candidate.payload.get("frozen_model_control") or {}
                   if isinstance(candidate.payload, dict) else {})
        solved_without_model = set(control.get("solved") or [])
        for case in data:
            rb = base_rates.get(case.id)
            rc = cand_rates.get(case.id)
            if rb is None or rc is None:
                continue
            b = rb >= 0.5  # modal baseline; ties -> True (as _baseline)
            c = rc >= 0.5
            # A failing case the pipeline also gets right with the model frozen
            # to its recorded answer was repaired by the code, not the model:
            # not a fix of the model, not a regression either — out of the test.
            if not b and case.id in solved_without_model:
                v.n_model_independent += 1
                continue
            if case.id in unstable:
                # One sample per arm (mcnemar): the case's baseline is a coin
                # and a flip would masquerade as a fix/regression -> held out.
                # Paired rates: it is REPORTED and stays in, weighed by how far
                # the candidate moves its rate.
                v.n_unstable += 1
                if not rates_mode:
                    continue
            # Applicability (defect 1): the safety/coverage test runs only on
            # cases the candidate actually touches.
            if not self._applies(candidate, case):
                continue
            is_fail = getattr(case.label, "value", None) == "fail"
            if is_fail:
                applicable_fail += 1
            base_vec.append(rb if rates_mode else float(b))
            cand_vec.append(rc if rates_mode else float(c))
            if not b and c:
                v.n_fixed += 1
                v.fixed_cases.append(case.id)
            elif b and not c:
                v.n_broken += 1
                v.broken_cases.append(case.id)
        v.n_pairs = len(base_vec)
        v.n_baseline_correct = sum(1 for x in base_vec if x >= 0.5)
        v.n_candidate_correct = sum(1 for x in cand_vec if x >= 0.5)
        v.n_applicable = v.n_pairs
        v.coverage = (applicable_fail / n_fail) if n_fail else None
        if v.n_pairs:
            v.baseline_rate = sum(base_vec) / v.n_pairs
            v.candidate_rate = sum(cand_vec) / v.n_pairs
        if v.n_pairs == 0 and v.n_model_independent:
            v.verdict = "model_independent"
            v.summary = (
                f"model-independent: all {v.n_model_independent} failing case(s) it "
                "repairs it also repairs with the model frozen to its recorded answers "
                "(frozen-model control) — the pipeline solves the task itself; no "
                "model-attributable pair to test"
            )
            return v
        if v.n_pairs == 0:
            v.verdict = "not_executed"
            v.summary = (
                f"never executed: {v.exec_error}"
                if v.exec_error
                else "no applicable scorable pair — candidate unvalidatable"
            )
            return v
        try:
            if rates_mode:
                stat = compare_paired_rates(base_vec, cand_vec, alpha=self._alpha)
                v.e_value_regression = stat.details.get("e_value_regression")
            else:
                stat = compare([x >= 0.5 for x in base_vec], [x >= 0.5 for x in cand_vec],
                               paired=True, alpha=self._alpha)
        except Exception as exc:
            v.verdict = "not_executed"
            v.summary = f"stats failed: {exc}"
            return v
        v.effect = stat.effect
        v.reject = bool(stat.reject)
        v.e_value = stat.e_value
        # Fixed = the paired test rejects with a net-positive effect: the
        # candidate repairs significantly more cases than it breaks.
        v.fixed = v.reject and (v.effect or 0.0) > 0
        v.verdict = self._verdict(v)
        cov = "" if v.coverage is None else f", coverage={v.coverage:.0%}"
        if rates_mode:
            noise = (f", {v.n_unstable} unstable weighed (k={v.n_baseline_samples} baseline"
                     f"/{v.n_candidate_samples} candidate samples)" if v.n_unstable else
                     f", k={v.n_baseline_samples}/{v.n_candidate_samples} samples")
        else:
            noise = f", {v.n_unstable} unstable dropped" if v.n_unstable else ""
        solo = (f", {v.n_model_independent} model-independent excluded"
                if v.n_model_independent else "")
        trunc = (f", {v.n_truncated} call(s) hit the decode cap"
                 if v.n_truncated else "")
        v.summary = f"{stat.summary()} [{v.verdict}{cov}{noise}{solo}{trunc}]"
        return v

    @staticmethod
    def _verdict(v: FixValidation) -> str:
        """Coarse triage label (defect 4): why a candidate did/didn't pass."""
        if v.fixed:
            return "fixed"
        if v.reject and (v.effect or 0.0) < 0:
            return "regressed"  # significantly worse
        net = v.n_fixed - v.n_broken
        if net > 0:
            return "partial"  # helped more than hurt, not significant
        if v.n_broken > v.n_fixed:
            return "unsafe"  # breaks more than it fixes
        return "no_effect"

    # -- recommendation + logging ------------------------------------------

    def _recommend(
        self,
        routed: "list[FixTier]",
        *,
        model: "Model | None" = None,
        reason_prefix: str = "",
    ) -> "dict[str, Any] | None":
        above = sorted(t for t in routed if t > self.max_tier)
        if above:
            target = above[0]
            reason = (
                f"verified hypotheses route to {target.label} "
                f"({target.describe()}), beyond the allowed {self.max_tier.label}"
            )
        elif self.max_tier < FixTier.L4_PARAMETERS:
            target = FixTier(self.max_tier + 1)
            reason = (
                f"no candidate within {self.max_tier.label} validated; the next "
                f"intervention space is {target.describe()}"
            )
        else:
            return None  # already at L4 — nothing above to recommend
        skipped: "list[str]" = []
        while (
            model is not None
            and target
            in {
                FixTier.L3A_INTERNALS_READ,
                FixTier.L3B_INTERNALS_WRITE,
            }
            and not self._tier_available(target, model)
        ):
            skipped.append(target.label)
            if target >= FixTier.L4_PARAMETERS:
                return None
            target = FixTier(target + 1)
        if skipped:
            reason += (
                f"; skipped unsupported tier(s) {', '.join(skipped)} for this model "
                f"and routed to {target.describe()}"
            )
        if reason_prefix:
            reason = f"{reason_prefix}; {reason}"
        return {"recommend_tier": target.label, "reason": reason}

    @staticmethod
    def _tier_available(tier: FixTier, model: "Model") -> bool:
        """Whether an invasive tier has a usable executor for this model."""
        if tier == FixTier.L3A_INTERNALS_READ:
            from evalvitals.core.capability import Capability

            return Capability.ATTENTION in getattr(model, "capabilities", frozenset())
        if tier == FixTier.L3B_INTERNALS_WRITE:
            return any(
                primitive.tier == tier and primitive.available(model)
                for primitive in INTERNALS_PRIMITIVES.values()
            )
        return True

    def _emit(self, outcome: FixOutcome) -> None:
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_fix(outcome)
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_fix failed: %s", exc)
