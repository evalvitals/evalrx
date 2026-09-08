"""Fix module — tiered, validated repair attempts after the diagnosis loop.

Design (intervention-space tiers, see :mod:`fix_tiers`): the allowed tier is
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
* **L2 declarative** — catalog-tool pipelines (:mod:`fix_tools`): cheap,
  deterministic, validated first.
* **L2 coded** — the coding agent (CLI agent first, judge fallback) writes a
  brand-new pipeline as Python: multiple model calls per case, branching on
  intermediate outputs — only the model itself is unchanged.  The code runs
  sandboxed with bridged model access (:mod:`fix_pipeline`); labels and
  rubrics never reach it, so it cannot cheat by echoing gold answers.
* **L3a** — internals read (:mod:`fix_internals`): contrastive decoders read
  and combine logits/attention from paired forward passes. Coded pipelines
  are L3a only when their source actually calls the bridged
  ``model_attend()``; merely exposing that optional API does not promote an
  otherwise black-box L2 scaffold.
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
rate differences (:func:`evalrx.stats.compare_paired_rates`) — a
stochastic model's flaky cases are weighed by how far the candidate moves
them, neither dropped nor mistaken for repairs.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from evalrx.analyzers.perturbation.prompt_contrast import _default_score
from evalrx.eval_agent.hypothesis import hypothesis_id
from evalrx.eval_agent.prompts.fix_agent import (
    _L1_PROMPT,
    _L2_CODE_PROMPT,
    _L2_PROMPT,
    _L3_PROMPT,
    _L4_PROMPT,
    _REPAIR_CATALOG_PROMPT,
    _REPAIR_PROMPT_BODY,
)
from evalrx.eval_agent.stages.fix_internals import (
    INTERNALS_PRIMITIVES,
    FinetuneSpec,
    primitives_catalog_text,
    run_lora_repair,
)
from evalrx.eval_agent.stages.fix_pipeline import (
    CodedPipelineResult,
    run_coded_pipeline,
    score_outputs,
)
from evalrx.eval_agent.stages.fix_tiers import FixTier, parse_tier, route_min_tier
from evalrx.eval_agent.stages.fix_tools import (
    PipelineSpec,
    _safe_generation_kwargs,
    catalog_text,
    run_pipeline,
    safe_format,
    score_to_bool,
    spec_changes_input,
)
from evalrx.eval_agent.stages.probe_generator import _extract_code
from evalrx.eval_agent.stages.repair_catalog import (
    discover_methods,
    method_names,
    supports_tier,
)
from evalrx.stats import compare, compare_paired_rates
from evalrx.stats.ebh import ebh
from evalrx.stats.evalue import evalue_bernoulli

if TYPE_CHECKING:
    from evalrx.agent_runtime.cli_types import CliAgentConfig
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model
    from evalrx.eval_agent.hypothesis import Hypothesis
    from evalrx.eval_agent.run_context import Trial

logger = logging.getLogger(__name__)


_MAX_JUDGE_CANDIDATES = 3
#: How many FAIL / PASS cases the proposer sees (prompt and the model's
#: baseline output; expected/gold answers are always withheld). Before this the
#: judge saw
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
    n_fail: int = _EXAMPLE_FAILS,
    n_pass: int = _EXAMPLE_PASSES,
) -> str:
    """Render prompts and baseline outputs; never expose expected/gold answers."""
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
        blocks.append("\n".join(lines))
    header = "(expected answers withheld; scored externally on disjoint cases)"
    return header + "\n\n" + "\n\n".join(blocks)

def _code_calls_name(code: str, name: str) -> bool:
    """Return whether executable source directly calls a named model bridge."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def _code_copies_example(code: str, examples: str, *, min_words: int = 8) -> bool:
    """Reject generated programs that memorize an example ID or prompt phrase."""
    try:
        constants = [
            node.value
            for node in ast.walk(ast.parse(str(code or "")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
    except SyntaxError:
        return False
    literal_text = "\n".join(constants).lower()
    ids = re.findall(r"^###\s+(?:FAIL|PASS)\s+case\s+(\S+)", examples, re.MULTILINE)
    if any(case_id.lower() in literal_text for case_id in ids):
        return True
    prompts = re.findall(r"^PROMPT:\s*(.+)$", examples, re.MULTILINE)
    literal_words = re.findall(r"[a-z0-9]+", literal_text)
    literal_ngrams = {
        tuple(literal_words[i : i + min_words])
        for i in range(max(0, len(literal_words) - min_words + 1))
    }
    for prompt in prompts:
        words = re.findall(r"[a-z0-9]+", prompt.lower())
        if any(
            tuple(words[i : i + min_words]) in literal_ngrams
            for i in range(max(0, len(words) - min_words + 1))
        ):
            return True
    return False


def _code_redefines_model_bridge(code: str) -> bool:
    """Reject user pipelines that shadow host-provided model bridge functions."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False
    bridge_names = {"model_generate", "model_attend"}
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in bridge_names
        for node in ast.walk(tree)
    )


def _explicit_multiple_choice_answer(value: Any) -> str:
    """Extract an explicit A-D commitment without mining unfinished prose."""
    text = str(value or "").strip().upper()
    marked = re.findall(
        r"(?:FINAL(?:\s+ANSWER)?|ANSWER|CHOICE|OPTION)\s*(?::|=|\-|\bIS\b)\s*"
        r"\(?([A-D])\)?\b",
        text,
    )
    if marked:
        return marked[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final = lines[-1] if lines else text
    bare = re.fullmatch(r"\(?([A-D])\)?[.!]?", final)
    return bare.group(1) if bare else ""


def _malformed_choice_predicate(case: Any) -> bool:
    """Gold-free gate for re-asking only malformed multiple-choice outputs."""
    task = str((getattr(case, "metadata", {}) or {}).get("task", ""))
    return task == "multiple_choice_letter" and not _explicit_multiple_choice_answer(
        getattr(case, "observed", None)
    )


_CHART_ARITHMETIC_RE = re.compile(
    r"\b(?:ratio|difference|sum|total|average|percent(?:age)?|how many|"
    r"more than|less than|times|add(?:ing|ed)?|subtract(?:ing|ed)?)\b",
    flags=re.IGNORECASE,
)


def _chart_arithmetic_predicate(case: Any) -> bool:
    """Gold-free gate for chart questions that require an explicit operation."""
    metadata = getattr(case, "metadata", {}) or {}
    inputs = getattr(case, "inputs", None)
    return (
        str(metadata.get("task", "")) == "exact_or_numeric"
        and getattr(inputs, "image", None) is not None
        and bool(_CHART_ARITHMETIC_RE.search(str(getattr(inputs, "prompt", ""))))
    )


def _chart_case_predicate(case: Any) -> bool:
    """Gold-free gate for image-backed exact/numeric chart questions."""
    metadata = getattr(case, "metadata", {}) or {}
    inputs = getattr(case, "inputs", None)
    return (
        str(metadata.get("task", "")) == "exact_or_numeric"
        and getattr(inputs, "image", None) is not None
    )


_CHART_COUNT_EXTRACT_RE = re.compile(
    r"\b(?:how many|number of)\b|"
    r"\bvalue of the (?:gray|grey|red|blue|green|yellow|orange|purple|"
    r"black|white|pink|brown) bar\b",
    flags=re.IGNORECASE,
)


def _chart_count_extract_predicate(case: Any) -> bool:
    """Gold-free gate for chart counting and explicit coloured-bar lookup."""
    return _chart_case_predicate(case) and bool(
        _CHART_COUNT_EXTRACT_RE.search(
            str(getattr(getattr(case, "inputs", None), "prompt", ""))
        )
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


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
                     :data:`_BUILTIN_DESCRIPTIONS` for the host's own.  Empty
                     is allowed and means "nothing readable to say" -- readers
                     are shown a blank rather than the slug in title case.
        kind:        ``"template"`` (L1) | ``"spec"`` (L2 declarative) |
                     ``"code"`` (agent-written pipeline) |
                     ``"registered_repair"`` (runtime-discovered executor) |
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


#: What each of the host's own repairs does, in one sentence, for a reader who
#: has never heard of this model or this field.  It lives here rather than in
#: the frontend because only this module knows what these candidates actually
#: do; a consumer handed ``vcd_diffusion_noise`` can title-case it and nothing
#: more, which produces a label that looks explained and is not.
#:
#: Judge-invented candidates are not in this table -- they carry their own
#: sentence from ``what_it_does`` in the proposal.  A name in neither place
#: resolves to "" and is rendered blank, on purpose.
_BUILTIN_DESCRIPTIONS: "dict[str, str]" = {
    "aad_silence_contrast":
        "Compares the model's answer with what it says when the sound is replaced by "
        "silence, and keeps only the part the sound itself explains.",
    "aad_silence_contrast_gated_false_yes":
        "Compares the answer with what the model says when the sound is replaced by "
        "silence, applied only where it said yes to something the audio does not support.",
    "annotate_horizontal_band_count":
        "Counts the bars in the chart and writes that number onto the picture before asking.",
    "answer_bbox_crop":
        "Crops the picture down to the region the answer is about before asking.",
    "assertive_grounding":
        "Tells the model to answer from what is actually in the picture rather than from "
        "what it expects to be there.",
    "attend_carefully":
        "Adds an instruction to examine the input closely before answering.",
    "chain_of_verification":
        "Has the model draft an answer, then check it with its own follow-up questions "
        "before committing.",
    "coded_pipeline":
        "Runs a short program the agent wrote that re-asks the model several different "
        "ways and keeps the answer those attempts agree on.",
    "detector_visual_search_consensus":
        "Uses an object detector to pick regions to look at, asks about each one, and "
        "keeps the answer the regions agree on.",
    "finetune_recipe":
        "Writes down a retraining plan for a person to run later. Nothing is changed "
        "automatically.",
    "guided_visual_search_consensus":
        "Uses the question to choose which parts of the picture to zoom into, then keeps "
        "the answer those views agree on.",
    "icd_instruction_disturbance":
        "Compares the answer with what the model says under a deliberately misleading "
        "instruction, and discounts whatever the misleading version produced too.",
    "icd_instruction_disturbance_gated_false_yes":
        "Discounts whatever a deliberately misleading instruction also produced, applied "
        "only where the model said yes to something the picture does not support.",
    "icd_instruction_disturbance_question":
        "Attaches a deliberately misleading instruction to the question itself, and "
        "discounts whatever the model says in both versions.",
    "ifcd_truthx_contrast":
        "Nudges the model's internal state toward the pattern it shows when it is being "
        "truthful, and compares that with the untouched run.",
    "increase_max_tokens":
        "Gives the model more room to write, so answers are not cut off part-way.",
    "least_to_most":
        "Breaks the question into smaller steps and has the model work through them in order.",
    "opera_overtrust_binary":
        "Stops the model leaning too hard on a few words it has already written when it "
        "makes a yes-or-no call.",
    "pai_image_attention":
        "Makes the model weigh the picture more heavily and its own prior expectations less.",
    "salient_crop":
        "Crops the picture to its most eye-catching region before asking.",
    "self_consistency_5":
        "Asks the same question five times and keeps the answer the model gives most often.",
    "self_refine":
        "Has the model criticise its own first answer and then rewrite it.",
    "separate_horizontal_bands":
        "Splits the chart into separate bars so each one can be read on its own.",
    "tcd_temporal_blur":
        "Compares the answer with what the model says when the timing in the clip is "
        "smeared out, and keeps the part real timing explains.",
    "upscale_sharpen":
        "Enlarges and sharpens the picture before asking.",
    "vcd_diffusion_noise":
        "Compares the answer with what the model says when the picture is replaced by "
        "noise, and discounts whatever it would have said without seeing anything.",
    "vcd_diffusion_noise_gated_false_yes":
        "Discounts whatever the model would say without seeing the picture, applied only "
        "where it said yes to something the picture does not support.",
    "vicrop_consensus_guard":
        "Zooms into the region the model was already looking at, and changes the answer "
        "only when the zoomed views agree.",
    "vicrop_relative_attention":
        "Finds the region the model was already looking at and zooms into it before asking again.",
    "visual_grounding":
        "Tells the model to read the answer off the picture first and to fall back on "
        "general knowledge only after that.",
    "zoom_equalize":
        "Zooms in and evens out the brightness so faint details become visible.",
}


def _judge_description(proposal: "Any") -> str:
    """The proposal's own ``what_it_does``, cleaned, or "".

    Judges that ignore the field, or answer it with the snake_case name in
    title case, contribute nothing a reader could not already see -- both come
    back empty rather than as a sentence that only looks like one.
    """
    if not isinstance(proposal, dict):
        return ""
    text = " ".join(str(proposal.get("what_it_does", "") or "").split()).strip()
    if not text:
        return ""
    name = str(proposal.get("name", "") or "").strip()
    if name and text.lower().rstrip(".") == name.replace("_", " ").lower():
        return ""
    return text[:300]


def _code_description(code: str) -> str:
    """The ``# WHAT_IT_DOES:`` header the codegen prompt asks for.

    The prompt asks for one line and shows a one-line example, but a model that
    wraps it anyway should not lose the second half of its own sentence, so
    immediately following comment lines are folded in until the code starts.
    """
    lines = code.splitlines()[:12]
    for index, line in enumerate(lines):
        if not line.strip().upper().startswith("# WHAT_IT_DOES:"):
            continue
        parts = [line.strip().split(":", 1)[1]]
        for follow in lines[index + 1:]:
            stripped = follow.strip()
            if not stripped.startswith("#"):
                break
            body = stripped.lstrip("#").strip()
            if not body or body.upper().startswith("WHAT_IT_DOES"):
                break
            parts.append(body)
        return " ".join(" ".join(parts).split()).strip()[:300]
    return ""


def plain_description(candidate: "FixCandidate") -> str:
    """One sentence for a reader, or "" when there is honestly nothing to say.

    Never falls back to the slug.  ``audio_evidence_then_answer`` rendered as
    "Audio Evidence Then Answer" reads like an explanation the run never
    produced, and a blank is the more honest signal that it did not.
    """
    described = (candidate.description or "").strip()
    return described or _BUILTIN_DESCRIPTIONS.get(candidate.name, "")


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
                          See :func:`~.fix_internals.run_lora_repair`.
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
        prewritten_code: str = "",
        candidate_model: "Model | None" = None,
        deployed_spec: "dict[str, Any] | None" = None,
    ) -> None:
        if verbose:
            # Surfaces this module's own logger.info()/.warning() calls (tier
            # routing, candidate generation, validation verdicts) — see
            # VLDiagnoseLoop's verbose= for the same convenience one layer up.
            from evalrx.logging_utils import enable_console_logging

            enable_console_logging()

        self._judge = judge
        self._finetune_pool = finetune_pool
        self.max_tier = parse_tier(max_tier)
        # Normally a fixed-ceiling run considers every cheaper tier. The loop's
        # auto-escalation path sets this transiently so each ladder station
        # evaluates only the newly opened intervention space.
        self.min_tier = parse_tier(min_tier) if min_tier is not None else None
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
        self._last_raw_stream = ""
        self._last_usage: dict | None = None
        self._baseline_generation_kwargs = dict(baseline_generation_kwargs or {})
        self._concurrency = max(1, int(concurrency))
        self._scoring_note = str(scoring_note or "")
        self._floor_candidates = (
            tuple(str(n) for n in floor_candidates) if floor_candidates else ()
        )
        self._prewritten_code = str(prewritten_code or "")
        # Winner-as-new-baseline (spec deploys, recursive rounds): when the
        # caller runs the whole loop against a DEPLOYED pipeline handle
        # (fix_tools.SpecPipelineModel), `candidate_model` is the raw model —
        # baseline arms keep measuring the deployed pipeline while proposals
        # and candidate arms run on the raw handle, so every candidate is a
        # full REPLACEMENT pipeline paired against the deployed one.
        # `deployed_spec` (the deployed PipelineSpec as a dict) is shown to
        # the proposer so its candidates are edits of a known incumbent
        # rather than blind wraps.
        self._candidate_model = candidate_model
        self._deployed_spec = dict(deployed_spec) if deployed_spec else None
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
        the M2/M4/explore evidence, M5-refuted hypotheses, the scoring rule
        and the baseline decoding budget.  ``proposal_data``, when supplied,
        is the discovery partition available to the repair author; ``data``
        remains untouched confirmation data and permits only one round.
        """
        outcome = FixOutcome(max_tier=self.max_tier)
        # Winner-as-new-baseline: `model` as handed in is what the baseline
        # arm must measure (the deployed pipeline, when one is configured);
        # proposals and candidate arms run on the raw handle so a candidate
        # REPLACES the deployed pipeline instead of nesting inside it.
        baseline_model = model
        if self._candidate_model is not None:
            model = self._candidate_model
        self._max_tokens_floor = self._resolve_max_tokens_floor(model, data)
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
        baseline, unstable = self._baseline(baseline_model, data)
        if not any(v is not None for v in baseline.values()):
            logger.warning("FixAgent: no scorable case (no rubrics); nothing to validate")
            outcome.recommendation = self._recommend(
                routed_tiers,
                model=model,
                data=data,
                reason_prefix=("no case carries a scoring rubric, so no fix can be validated"),
            )
            self._emit(outcome)
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
        if self.max_tier <= FixTier.L2_SCAFFOLD:
            if self._baseline_repeats > 1 or self._candidate_repeats > 1:
                repairable = any(
                    r is not None and r < 1.0 for r in self._baseline_rates.values()
                )
            else:
                repairable = any(v is False for v in baseline.values())
        if not repairable:
            n_scorable = sum(1 for v in baseline.values() if v is not None)
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
                    f"(< {1.0 / self._alpha:.0f} needed) — collect failing cases instead of "
                    "spending the candidate budget."
                ),
            }
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
                logger.info(
                    "FixAgent: validating tier=%s candidate=%s kind=%s source=%s",
                    candidate.tier.label,
                    candidate.name,
                    candidate.kind,
                    candidate.source,
                )
                validation = self._validate(candidate, model, data, baseline, unstable)
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
        # Same handle split as propose_and_validate: baseline arm = deployed
        # pipeline (when configured), candidate arm = raw model.
        baseline_model = model
        if self._candidate_model is not None:
            model = self._candidate_model
        self._max_tokens_floor = self._resolve_max_tokens_floor(model, data)
        self._enforce_generation_floor([candidate])
        baseline, unstable = self._baseline(baseline_model, data)
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
            from evalrx.stats.evalue import evalue_bounded_mean

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

        rec = self._recommend(routed_tiers, model=model, data=data)
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
        # A repair proposer never receives expected/gold answers.  Examples
        # contain only prompts and the model's recorded outputs; correctness is
        # evaluated outside the agent on disjoint EXPLORE/CONFIRM partitions.
        if context.example_cases is not None:
            examples = _format_examples(context.example_cases)
        else:
            examples = _format_examples(data)
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
        # Text-only batches: the image-tool catalog is noise for the judge and
        # an invitation to burn a candidate on a structural no-op.
        catalog = catalog_text() if has_images else _TEXT_ONLY_CATALOG_NOTE
        floor_names = self._floor_names(has_images=has_images, tasks=tasks)
        context_block = self._context_block(
            context, data, model, floor_names=floor_names
        )
        # Recursive rounds only (see _edit_note): the L1/L2 proposers may also
        # EDIT the previously deployed template instead of wrapping it. The
        # coded-pipeline path is left out — its cases_file carries rendered
        # prompts only, so {original_prompt} means nothing to written code.
        proposal_context = context_block + self._edit_note(data)

        candidates: "list[FixCandidate]" = []
        # Pre-registered conditional repair for audio/other A-D tasks whose
        # baseline never committed to an option.  It gates only on the
        # observable output contract (never gold/correctness), so clean
        # answers are preserved while malformed prose is re-asked and voted.
        # Keeping it built-in also avoids asking codegen to rediscover this
        # simple, high-frequency failure mode and then implement brittle
        # option parsing from scratch.
        malformed_name = "malformed_choice_consensus"
        malformed_enabled = (
            tasks == {"multiple_choice_letter"}
            and malformed_name not in prior_names
            and (
                self._candidate_allowlist is None
                or malformed_name in self._candidate_allowlist
            )
            and self.max_tier >= FixTier.L2_SCAFFOLD
        )
        if malformed_enabled:
            spec = PipelineSpec(
                name=malformed_name,
                prompt_template=(
                    "{prompt}\n\nListen to the audio evidence carefully and decide silently. "
                    "Do not explain or repeat the choices. Reply with exactly one line: "
                    "FINAL: X, where X is A, B, C, or D."
                ),
                n_samples=3,
                generation_kwargs={"do_sample": True, "temperature": 0.35, "top_p": 0.9},
                output_key_pattern=r"(?:FINAL(?:\s+ANSWER)?|ANSWER)\s*:\s*\(?([A-D])\)?",
            )
            candidates.append(
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name=malformed_name,
                    kind="spec",
                    source="conditional_default",
                    payload=spec.to_dict(),
                    predicate=_malformed_choice_predicate,
                )
            )
        chart_name = "chart_arithmetic_verify"
        chart_enabled = (
            "exact_or_numeric" in tasks
            and has_images
            and chart_name not in prior_names
            and self._candidate_allowlist is not None
            and chart_name in self._candidate_allowlist
            and self.max_tier >= FixTier.L2_SCAFFOLD
        )
        if chart_enabled:
            spec = PipelineSpec(
                name=chart_name,
                prompt_template=(
                    "Treat this as a chart measurement problem. First identify every "
                    "legend/category and plotted value needed by the question, respecting "
                    "the axis scale. Then perform the requested operation and independently "
                    "check the arithmetic. Do all work internally and return only the short "
                    "answer requested by the original question.\n\n{prompt}"
                ),
                strategy="chain_of_verification",
            )
            candidates.append(
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name=chart_name,
                    kind="spec",
                    source="conditional_default",
                    payload=spec.to_dict(),
                    predicate=_chart_arithmetic_predicate,
                )
            )
        consensus_name = "chart_verified_consensus"
        consensus_enabled = (
            "exact_or_numeric" in tasks
            and has_images
            and consensus_name not in prior_names
            and self._candidate_allowlist is not None
            and consensus_name in self._candidate_allowlist
            and self.max_tier >= FixTier.L2_SCAFFOLD
        )
        if consensus_enabled:
            spec = PipelineSpec(
                name=consensus_name,
                prompt_template=(
                    "Read the chart as measured evidence: identify the relevant labels, "
                    "legend entries, marks, and axis scale; derive the requested answer; "
                    "then check it independently. Work internally and return only the "
                    "short answer requested.\n\n{prompt}"
                ),
                strategy="chain_of_verification",
                n_samples=3,
                generation_kwargs={"temperature": 0.35, "top_p": 0.9},
                baseline_override_min_support=3,
            )
            candidates.append(
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD,
                    name=consensus_name,
                    kind="spec",
                    source="conditional_default",
                    payload=spec.to_dict(),
                    predicate=_chart_case_predicate,
                )
            )
        count_name = "upscale_count_extract"
        count_enabled = (
            "exact_or_numeric" in tasks
            and has_images
            and count_name not in prior_names
            and self._candidate_allowlist is not None
            and count_name in self._candidate_allowlist
            and self.max_tier >= FixTier.L2_SCAFFOLD
        )
        if count_enabled:
            count_spec = self._default_spec("upscale_sharpen")
            if count_spec is not None:
                count_spec.name = count_name
                candidates.append(
                    FixCandidate(
                        tier=FixTier.L2_SCAFFOLD,
                        name=count_name,
                        kind="spec",
                        source="conditional_default",
                        payload=count_spec.to_dict(),
                        predicate=_chart_count_extract_predicate,
                    )
                )
        # An explicit allowlist is a pre-registration request. Built-in L2
        # defaults must be materialised before any judge proposal/family-size
        # truncation; filtering only afterwards could silently leave the run
        # with attempted=0 even though the requested candidate exists.
        requested_defaults = {
            "self_consistency_5", "self_refine", "least_to_most",
            "chain_of_verification", "upscale_sharpen", "zoom_equalize",
        }
        if (
            self._candidate_allowlist is not None
            and len(self._candidate_allowlist) == 1
            and self.max_tier >= FixTier.L2_SCAFFOLD
        ):
            requested = next(iter(self._candidate_allowlist))
            if requested in requested_defaults and requested not in prior_names:
                requested_spec = self._default_spec(requested)
                if requested_spec is not None:
                    candidates.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD,
                            name=requested,
                            kind="spec",
                            source="pre_registered_default",
                            payload=requested_spec.to_dict(),
                        )
                    )
        # A code-only run is a pre-registered autonomous-repair experiment.
        # Do not spend three judge calls inventing L0/L1/declarative/L3
        # candidates that the allowlist will discard afterwards; apart from
        # latency and quota waste, those calls can fail before the requested
        # coding agent is ever reached.
        code_only = self._candidate_allowlist == frozenset({"coded_pipeline"})
        catalog_method_names = method_names()
        preregistered_only = self._candidate_allowlist in {
            frozenset({malformed_name}), frozenset({chart_name}),
            frozenset({consensus_name}),
            frozenset({count_name}),
            *(frozenset({name}) for name in requested_defaults),
        }
        catalog_method_only = self._candidate_allowlist in {
            frozenset({name}) for name in catalog_method_names
        }
        skip_lower_tiers = preregistered_only or catalog_method_only
        if (
            not code_only
            and not skip_lower_tiers
            and self.max_tier >= FixTier.L0_RUNTIME_CONFIG
            and (self.min_tier is None or self.min_tier <= FixTier.L0_RUNTIME_CONFIG)
        ):
            candidates += self._runtime_candidates(data, prior_names)
        if (
            not code_only
            and not skip_lower_tiers
            and self.max_tier >= FixTier.L1_PROMPT
            and (self.min_tier is None or self.min_tier <= FixTier.L1_PROMPT)
            and not self._paper_methods_only
        ):
            candidates += self._l1_candidates(
                hyp_lines,
                examples,
                prior_text,
                prior_names,
                has_images=has_images,
                tasks=tasks,
                context_block=proposal_context,
            )
        if (
            not code_only
            and not skip_lower_tiers
            and self.max_tier >= FixTier.L2_SCAFFOLD
            and (self.min_tier is None or self.min_tier <= FixTier.L2_SCAFFOLD)
        ):
            candidates += self._l2_candidates(
                hyp_lines,
                examples,
                prior_text,
                prior_names,
                has_images=has_images,
                model=model,
                tasks=tasks,
                context_block=proposal_context,
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
        if (not skip_lower_tiers and self.max_tier >= FixTier.L2_SCAFFOLD
                and (self.min_tier is None or self.min_tier <= FixTier.L2_SCAFFOLD)
                and self.codegen_available):
            # The coder-written pipeline is the ONE candidate a code-only run
            # exists to field, so it sits outside the ``not code_only`` gate
            # (nested inside it, --code-only proposed nothing at all).
            candidates += self._l2_coded_candidate(
                hyp_lines, examples, model, prior_text,
                context_block=context_block, catalog=catalog,
                text_only=not has_images,
            )
        if (
            not code_only
            and not preregistered_only
            and self.max_tier >= FixTier.L2_SCAFFOLD
            and (self.min_tier is None or self.min_tier <= FixTier.L3B_INTERNALS_WRITE)
        ):
            # The declarative catalog contains tool-assisted L2 as well as
            # internals-aware L3 repairs.
            # Discover it at the current ladder station; the catalog's max-tier
            # gate and the final min-tier filter keep each method in its tier.
            candidates += self._l3_candidates(
                hyp_lines,
                model,
                prior_text,
                prior_names,
                has_images=has_images,
                has_audio=has_audio,
                tasks=tasks,
            )
        if (
            not code_only
            and not skip_lower_tiers
            and self.max_tier >= FixTier.L4_PARAMETERS
            and (self.min_tier is None or self.min_tier <= FixTier.L4_PARAMETERS)
        ):
            candidates += self._l4_candidates(hyp_lines)
        if self.min_tier is not None:
            candidates = [c for c in candidates if c.tier >= self.min_tier]
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
        if name == "upscale_sharpen":
            return PipelineSpec(
                name="upscale_sharpen",
                image_ops=[
                    {"tool": "upscale", "params": {"factor": 2.0}},
                    {"tool": "sharpen", "params": {"factor": 2.0}},
                ],
            )
        if name == "zoom_equalize":
            return PipelineSpec(
                name="zoom_equalize",
                image_ops=[
                    {"tool": "zoom_center", "params": {"factor": 1.6}},
                    {"tool": "equalize", "params": {}},
                ],
            )
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
        record; the applied value is what ``PipelineSpec.from_dict`` (spec) or
        the template runner (template) reads.
        """
        floor = self._max_tokens_floor
        if not floor:
            return
        for candidate in candidates:
            payload = candidate.payload
            if not isinstance(payload, dict) or candidate.kind not in ("spec", "template"):
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
            lines += ["DIAGNOSTIC EVIDENCE (what M2 statistics / M4 tests / exploration "
                      "established — read-only, steer WHAT to try):", context.evidence.rstrip()]
        if context.refuted:
            lines += ["REFUTED BY AN INTERVENTION EXPERIMENT (M5) — do NOT build a fix on these:"]
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

    @staticmethod
    def _deployed_template(data: "CaseBatch") -> "str | None":
        """The template a previous repair round already deployed, if coherent.

        Recursive-round manifests record the winning template in every row's
        ``metadata.recursive_stack[-1]["template"]`` alongside the pristine
        ``metadata.original_prompt`` (the round-0 text).  Returns the template
        only when EVERY case carries the same one plus its original prompt —
        anything less means there is no single deployed template to edit.
        """
        tpl: "str | None" = None
        for case in data:
            md = getattr(case, "metadata", {}) or {}
            stack = md.get("recursive_stack")
            entry = stack[-1] if isinstance(stack, list) and stack else None
            text = entry.get("template") if isinstance(entry, dict) else None
            if not text or not md.get("original_prompt"):
                return None
            if tpl is None:
                tpl = str(text)
            elif str(text) != tpl:
                return None
        return tpl

    def _edit_note(self, data: "CaseBatch") -> str:
        """Proposer context inviting EDIT candidates on recursive rounds.

        Only rendered when the batch carries a coherent deployed template (see
        :meth:`_deployed_template`); default runs get "" and are unchanged.
        An edit candidate is an ordinary template/spec proposal whose template
        is written against ``{original_prompt}`` — the L1 closure and the spec
        runner both fill that placeholder from case metadata, so it REPLACES
        the deployed template instead of wrapping it.

        A SPEC deploy (``deployed_spec`` + ``candidate_model``: the baseline
        handle runs the previous winner's whole pipeline) is different: rows
        stay pristine, so templates use ``{prompt}`` as usual, and EVERY
        candidate is validated as a full pipeline paired against the deployed
        one — the note shows the incumbent spec and asks for revisions of it.
        """
        if self._deployed_spec is not None:
            spec_json = json.dumps(self._deployed_spec, indent=2, default=str)
            return (
                "\nDEPLOYED PIPELINE (the baseline your candidates are paired "
                "against is NOT the raw model: it is the already-deployed pipeline "
                "below — a previous repair round's validated winner. Every case's "
                "recorded baseline output came from running it):\n"
                "<<<\n" + spec_json + "\n>>>\n"
                "Your candidates run as full pipelines that REPLACE this one, so "
                "beating the baseline means beating this pipeline, not the plain "
                "model.  Prefer minimal revisions of it — keep what its validation "
                "proved, change the ONE part the failure evidence indicts (its "
                "template wording, its sampling, its scaffold) — over unrelated "
                "fresh ideas that discard its confirmed gains.\n"
            )
        deployed = self._deployed_template(data)
        if not deployed:
            return ""
        return (
            "\nDEPLOYED TEMPLATE (a previous repair round already rewrote every case "
            "prompt: each case's {prompt} value IS the rendered output of the template "
            "below, and the pristine pre-rewrite text is available as the placeholder "
            "{original_prompt}):\n"
            "<<<\n" + deployed.rstrip() + "\n>>>\n"
            "In addition to your other strategies, propose 1-2 EDIT candidates: a "
            "minimal revision of the deployed template — change, tighten, or remove "
            "the ONE clause the failure evidence indicts, and keep what already "
            "works.  Write an edit against {original_prompt} (and do NOT also "
            "include {prompt}): it REPLACES the deployed template instead of "
            "wrapping it, so the model sees one coherent instruction rather than "
            "an override fighting the text above it.\n"
        )

    def _runtime_candidates(
        self,
        data: "CaseBatch",
        prior_names: "frozenset[str]" = frozenset(),
    ) -> "list[FixCandidate]":
        """Propose a decode-budget repair only from recorded telemetry."""
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
        return out

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

    def _l1_candidates(
        self,
        hyp_lines: str,
        examples: str,
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        *,
        has_images: bool = False,
        tasks: "set[str] | None" = None,
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
        structural_keys = {"image_ops", "n_samples", "strategy"}
        has_structural_proposal = False
        for p in proposals:
            template = str(p.get("prompt_template", ""))
            name = str(p.get("name", "")).strip()
            if structural_keys.intersection(p):
                has_structural_proposal = True
                continue
            # An EDIT candidate (recursive rounds, see _edit_note) replaces the
            # deployed template by rendering against {original_prompt} instead
            # of wrapping the already-rewritten {prompt}.
            if name and ("{prompt}" in template or "{original_prompt}" in template):
                payload: "dict[str, Any]" = {"prompt_template": template}
                # L1 stays prompt-only except for decode ROOM: a template that
                # asks for intermediate work must be able to finish (at a 64-
                # token vlm budget every such rewrite truncated and scored as a
                # regression). max_tokens is floor-enforced later; sampler
                # controls (temperature/top_p) remain L2-only and are dropped.
                max_tokens = _safe_generation_kwargs(p.get("generation_kwargs")).get("max_tokens")
                if max_tokens:
                    payload["generation_kwargs"] = {"max_tokens": max_tokens}
                out.append(
                    FixCandidate(
                        tier=FixTier.L1_PROMPT,
                        name=name,
                        kind="template",
                        description=_judge_description(p),
                        payload=payload,
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
                    FixCandidate(
                        tier=FixTier.L2_SCAFFOLD, name=spec.name,
                        description=_judge_description(p), payload=spec.to_dict(),
                    )
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
        from evalrx.core.capability import Capability
        from evalrx.eval_agent.stages.fix_pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        trial = (
            self._run_context.new_trial("fixes", "coded_pipeline")
            if self._run_context is not None
            else None
        )
        attend_available = (
            self.max_tier >= FixTier.L3A_INTERNALS_READ
            and Capability.ATTENTION in getattr(model, "capabilities", frozenset())
        )
        attend_hint = (
            (
                "\n- A function  model_attend(case_id, prompt=None) -> "
                '{"grid": [[float,...],...], "shape": [H, W]}  is ALSO defined: '
                "the model's attention heatmap over image patches (read-only "
                "internals). Use it e.g. to find where the model looks, then "
                "crop_region there and re-ask. This forward pass counts as ONE "
                "of the per-case model-call budget below."
            )
            if attend_available
            else ""
        )
        code, source, prompt, raw = "", "", "", ""
        if catalog is None:
            catalog = _TEXT_ONLY_CATALOG_NOTE if text_only else catalog_text()
        selection_guidance = self._code_selection_guidance(prior_text)
        # Explore rounds learn from a 2-of-3 consensus; a one-shot candidate
        # must have all three enhanced passes agree. The prompt states the
        # same number the host bridge enforces.
        min_support = 2 if self.max_repair_rounds > 1 else 3
        base = dict(
            hypotheses=hyp_lines,
            examples=examples,
            catalog=catalog,
            cases_file=CASES_FILENAME,
            marker=RESULT_MARKER,
            attend_hint=attend_hint,
            context=context_block,
            selection_guidance=selection_guidance,
            min_support=min_support,
        )
        if self._prewritten_code.strip():
            code = self._prewritten_code
            source = "prewritten"
            prompt = "Frozen prewritten coded pipeline supplied by the caller."
        elif self._cli_config is not None and self._cli_config.provider != "llm":
            prompt = (
                _L2_CODE_PROMPT.format(fences_hint=", written to a file named pipeline.py", **base)
                + prior_text
            )
            code, raw = self._write_code_cli(prompt, trial)
            source = f"cli:{self._cli_config.provider}"
            if _code_copies_example(code, examples) or _code_redefines_model_bridge(code):
                logger.warning("FixAgent: generated code violated anti-memorization/bridge rules; dropped")
                code = ""
        if not code.strip() and self._judge is not None:
            self._last_raw_stream = ""
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
        if _code_copies_example(code, examples) or _code_redefines_model_bridge(code):
            logger.warning("FixAgent: generated code violated anti-memorization/bridge rules; dropped")
            code = ""
        self._emit_codegen(
            "coded_pipeline", prompt, source, code, raw, ok=bool(code.strip()), trial=trial
        )
        if not code.strip():
            return []
        # Classify by the intervention the generated program actually uses,
        # not by an optional bridge merely being advertised in its prompt.
        # This prevents black-box multi-call/image-tool scaffolds from being
        # reported as L3a just because the model happens to expose attention.
        enable_attend = attend_available and _code_calls_name(code, "model_attend")
        tier = FixTier.L3A_INTERNALS_READ if enable_attend else FixTier.L2_SCAFFOLD
        return [
            FixCandidate(
                tier=tier,
                name="coded_pipeline",
                kind="code",
                description=_code_description(code),
                payload={
                    "code": code,
                    "enable_attend": enable_attend,
                    "text_only": bool(text_only),
                    # Prompt instructions are advisory; the host bridge also
                    # enforces the selection rule without seeing gold labels
                    # (anchored on each case's recorded baseline_output).
                    "consensus_min_support": min_support,
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
                "aggregate fixed/broken counts and implementation below as "
                "training feedback. Raw case IDs, prompts, and answers are "
                "deliberately withheld: revise the general mechanism, never "
                "match a benchmark item or phrase. Keep baseline_output outside "
                "a label-free, generally applicable gate. If the prior attempt "
                "repaired zero cases, abandon its override mechanism instead of "
                "merely retuning it."
            )
        if self.max_repair_rounds > 1:
            return (
                "- This is EXPLORE round 1, used to learn which task subtypes "
                "benefit before a later candidate is frozen. Treat "
                "baseline_output as the baseline answer and keep it on ties, "
                "but you may use a controlled 2-of-3 alternative consensus "
                "(two DISTINCT enhanced calls agreeing on the same different "
                "answer) so the paired helped/hurt feedback is informative. "
                "Never hard-code answers or compute the benchmark task "
                "outside the model."
            )
        return (
            "- Treat baseline_output (the ORIGINAL direct answer) as the "
            "safety baseline. Keep it unless all 3 independent enhanced/"
            "reasoned passes agree on a different answer and none supports "
            "the baseline. Do not use unconditional majority replacement."
        )

    def _write_code_cli(self, prompt: str, trial: "Trial | None" = None) -> "tuple[str, str]":
        from pathlib import Path

        from evalrx.agent_runtime.codegen import CodegenRunner

        workdir = Path(self._workdir(trial))
        result = CodegenRunner(self._cli_config).write_code(  # type: ignore[arg-type]
            prompt,
            workdir=workdir,
            timeout_sec=self._cli_config.timeout_sec,  # type: ignore[union-attr]
            preferred_filenames=("pipeline.py",),
        )
        self._last_usage = result.usage
        self._last_raw_stream = ""
        if result.raw_stream_path:
            try:
                self._last_raw_stream = (workdir / result.raw_stream_path).read_text(encoding="utf-8")
            except OSError:
                pass
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
            from evalrx.agent_runtime.sandbox import ExperimentSandbox

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
        inline = bool(getattr(self.run_logger, "inline_text_artifacts", False))
        if trial is not None and not inline:
            if prompt:
                trial.write(f"{name}_prompt.txt", prompt)
            if code:
                trial.write(f"{name}_code.py", code)
            if raw:
                trial.write(f"{name}_agent_thinking.txt", raw)
            if self._last_raw_stream:
                trial.write(f"{name}_agent_raw_stream.txt", self._last_raw_stream)
            extra = {**(extra or {}), "trial_root": str(trial.root)}
            prompt, code, raw = "", "", ""
        elif trial is not None:
            # V2 keeps the exact prompt/code/response in M5/log.json.  The
            # trial workspace is ephemeral execution state and is snapshotted
            # separately by log_experiment/log_fix.
            extra = {**(extra or {}), "trial_root": str(trial.root)}
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
                raw_stream=self._last_raw_stream,
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
    ) -> "list[FixCandidate]":
        """Select registered repair capabilities and internals primitives.

        This layer is intentionally method-agnostic: executor names, fidelity
        gates, task compatibility, defaults, and evidence-facing descriptions
        live in :mod:`repair_catalog`.  The judge sees only structurally
        executable entries and performs mechanism matching against diagnosis.
        """
        out: "list[FixCandidate]" = []

        def finalize(options: "list[FixCandidate]") -> "list[FixCandidate]":
            if self._candidate_allowlist is not None:
                options = [c for c in options if c.name in self._candidate_allowlist]
            return options[: self.max_judge_candidates]

        discovered = discover_methods(
            model,
            max_tier=self.max_tier,
            has_images=has_images,
            has_audio=has_audio,
            tasks=tasks or set(),
            allow_adapted=self._allow_adapted_paper_methods,
            prior_names=prior_names,
        )
        if discovered:
            by_name = {method.name: method for method in discovered}
            catalog_lines = "\n".join(
                f"- {method.name} [{method.tier.label}]: {method.description}"
                for method in discovered
            )
            requested = (
                next(iter(self._candidate_allowlist))
                if self._candidate_allowlist is not None
                and len(self._candidate_allowlist) == 1
                else None
            )
            selected_names: list[str]
            if requested in by_name:
                # A one-name allowlist is a frozen experiment. Structural
                # discovery still gates it, but mechanism matching is already
                # pre-registered and cannot be vetoed by a fresh judge call.
                selected_names = [requested]
            else:
                selected_names = []
                for proposal in self._ask_judge(
                    _REPAIR_CATALOG_PROMPT.format(
                        hypotheses=hyp_lines,
                        catalog=catalog_lines,
                        k=self.max_judge_candidates,
                    )
                    + prior_text
                ):
                    name = str(proposal.get("name", ""))
                    if name in by_name and name not in selected_names:
                        selected_names.append(name)
            for name in selected_names:
                method = by_name[name]
                out.append(
                    FixCandidate(
                        tier=method.tier,
                        name=method.name,
                        kind="registered_repair",
                        source=method.source,
                        payload={
                            "executor": method.executor,
                            "baseline_executor": method.baseline_executor,
                            "kwargs": dict(method.payload),
                            "pass_baseline_answer": method.pass_baseline_answer,
                        },
                    )
                )

        # Generic pre-audited internals-write primitives share the same
        # evidence-driven selection path but are registered separately because
        # they are host hooks rather than model-provided executor capabilities.
        catalog = primitives_catalog_text(model, self.max_tier)
        if catalog:
            for proposal in self._ask_judge(
                _L3_PROMPT.format(
                    hypotheses=hyp_lines,
                    catalog=catalog,
                    k=self.max_judge_candidates,
                )
                + prior_text
            ):
                primitive = INTERNALS_PRIMITIVES.get(str(proposal.get("primitive", "")))
                if (
                    primitive is None
                    or primitive.tier > self.max_tier
                    or not primitive.available(model)
                ):
                    continue
                out.append(
                    FixCandidate(
                        tier=primitive.tier,
                        name=primitive.name,
                        kind="primitive",
                        payload={
                            "primitive": primitive.name,
                            "params": dict(proposal.get("params") or {}),
                        },
                    )
                )
        if not out and catalog:
            # Registry defaults are a fallback only when the judge returned no
            # executable choice; no paper method is special-cased here.
            for primitive in INTERNALS_PRIMITIVES.values():
                if (
                    primitive.name not in prior_names
                    and primitive.tier <= self.max_tier
                    and primitive.available(model)
                ):
                    out.append(
                        FixCandidate(
                            tier=primitive.tier,
                            name=primitive.name,
                            kind="primitive",
                            source="default",
                            payload={"primitive": primitive.name, "params": {}},
                        )
                    )
        if not out and not catalog and not discovered:
            logger.info("FixAgent: no L3 repair capability is available for %r", model)
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
        raw = ""
        error = None
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            error = str(exc)
            self._log_model_exchange("fix_judge_object", prompt, raw, error=error)
            return {}
        self._log_model_exchange("fix_judge_object", prompt, raw)
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
        raw = ""
        error = None
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            error = str(exc)
            self._log_model_exchange("fix_judge_list", prompt, raw, error=error)
            return []
        self._log_model_exchange("fix_judge_list", prompt, raw)
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

    def _log_model_exchange(
        self, operation: str, prompt: str, output: str, *, error: "str | None" = None,
    ) -> None:
        """Best-effort detailed V2 model-call logging without changing V1."""
        fn = getattr(self.run_logger, "log_model_exchange", None)
        if not callable(fn):
            return
        try:
            fn(
                "M5", role="fix_judge", operation=operation,
                inputs=prompt, output=output, error=error,
            )
        except Exception as exc:  # logging must never break repair
            logger.debug("FixAgent: model exchange logging failed: %s", exc)

    def _log_target_exchange(
        self,
        operation: str,
        inputs: Any,
        output: Any,
        *,
        case_id: str,
        error: "str | None" = None,
        duration_sec: "float | None" = None,
        metadata: "dict[str, Any] | None" = None,
    ) -> None:
        fn = getattr(self.run_logger, "log_model_exchange", None)
        if not callable(fn):
            return
        try:
            fn(
                "M5", role="target_model", operation=operation,
                inputs=inputs, output=output, error=error, duration_sec=duration_sec,
                metadata={"case_id": case_id, **(metadata or {})},
            )
        except Exception as exc:
            logger.debug("FixAgent: target exchange logging failed: %s", exc)

    def _log_bridge_exchange(self, record: "dict[str, Any]") -> None:
        response = record.get("response") or {}
        self._log_target_exchange(
            "coded_pipeline_bridge", record.get("request"), response.get("output"),
            case_id=str(record.get("case_id") or ""), error=response.get("error"),
            duration_sec=record.get("duration_sec"),
            metadata={
                "replayed_from_recorded_baseline": bool(
                    record.get("replayed_from_recorded_baseline")
                )
            },
        )

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
            started = time.perf_counter()
            try:
                output = str(model.generate(case.inputs))
            except Exception as exc:
                logger.debug("FixAgent: baseline generate failed on %s: %s", case.id, exc)
                self._log_target_exchange(
                    "fresh_baseline", case.inputs, None, case_id=case.id,
                    error=str(exc), duration_sec=time.perf_counter() - started,
                )
                return case.id, None
            self._log_target_exchange(
                "fresh_baseline", case.inputs, output, case_id=case.id,
                duration_sec=time.perf_counter() - started,
            )
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
        if candidate.kind == "registered_repair":

            def registered_repair(
                model: "Model", case: "FailureCase"
            ) -> "Optional[bool]":
                executor_name = str(candidate.payload.get("executor", ""))
                kwargs = dict(candidate.payload.get("kwargs") or {})
                if candidate.payload.get("pass_baseline_answer"):
                    kwargs["baseline_answer"] = str(getattr(case, "observed", ""))
                try:
                    executor = getattr(model, executor_name)
                    output = executor(case.inputs, **kwargs)
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    logger.debug(
                        "Registered repair %s failed on %s: %s",
                        candidate.name,
                        case.id,
                        exc,
                    )
                    return None

            return registered_repair
        if candidate.kind == "visual_search":

            def visual_search(model: "Model", case: "FailureCase") -> "Optional[bool]":
                started = time.perf_counter()
                try:
                    search = getattr(model, "generate_visual_search")
                    output = search(
                        case.inputs,
                        baseline_answer=getattr(case, "observed", None),
                        **candidate.payload,
                    )
                    self._log_target_exchange(
                        "guided_visual_search", case.inputs, output, case_id=case.id,
                        duration_sec=time.perf_counter() - started,
                        metadata={"candidate": candidate.name, "parameters": candidate.payload},
                    )
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    self._log_target_exchange(
                        "guided_visual_search", case.inputs, None, case_id=case.id,
                        error=str(exc), duration_sec=time.perf_counter() - started,
                        metadata={"candidate": candidate.name, "parameters": candidate.payload},
                    )
                    logger.debug("Guided visual search failed on %s: %s", case.id, exc)
                    return None

            return visual_search
        if candidate.kind == "detector_visual_search":

            def detector_visual_search(model: "Model", case: "FailureCase") -> "Optional[bool]":
                started = time.perf_counter()
                try:
                    search = getattr(model, "generate_detector_visual_search")
                    output = search(
                        case.inputs,
                        baseline_answer=getattr(case, "observed", None),
                        **candidate.payload,
                    )
                    self._log_target_exchange(
                        "detector_visual_search", case.inputs, output, case_id=case.id,
                        duration_sec=time.perf_counter() - started,
                        metadata={"candidate": candidate.name, "parameters": candidate.payload},
                    )
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, str(output)))
                except Exception as exc:
                    self._log_target_exchange(
                        "detector_visual_search", case.inputs, None, case_id=case.id,
                        error=str(exc), duration_sec=time.perf_counter() - started,
                        metadata={"candidate": candidate.name, "parameters": candidate.payload},
                    )
                    logger.debug("Detector visual search failed on %s: %s", case.id, exc)
                    return None

            return detector_visual_search
        if candidate.kind == "template":
            template = candidate.payload["prompt_template"]
            gen_kwargs = _safe_generation_kwargs(candidate.payload.get("generation_kwargs"))

            def l1(model: "Model", case: "FailureCase") -> "Optional[bool]":
                inp = case.inputs
                metadata = getattr(case, "metadata", {}) or {}
                template_context = {str(key): value for key, value in metadata.items()}
                template_context["prompt"] = str(getattr(inp, "prompt", ""))
                template_context.setdefault("failure_axis", "the relevant visual evidence")
                # Inside the try, not before it: rendering the template is as
                # capable of failing as generating from it, and a single bad
                # case must score None rather than abort the whole validation.
                started = time.perf_counter()
                try:
                    # dataclasses.replace, not a bare Inputs(prompt=..., image=...):
                    # that silently dropped .video/.audio, so every L1 candidate was
                    # unconditionally inapplicable (generate() raising on the
                    # missing required modality field, caught below, scored as
                    # None for every case) on any non-image FailureCase.
                    new_inputs = dataclasses.replace(
                        inp, prompt=safe_format(template, template_context)
                    )
                    output = str(model.generate(new_inputs, **gen_kwargs))
                    self._log_target_exchange(
                        "template_candidate", new_inputs, output, case_id=case.id,
                        duration_sec=time.perf_counter() - started,
                        metadata={"generation_kwargs": gen_kwargs, "candidate": candidate.name},
                    )
                    self._record_output(case.id, output)
                    return score_to_bool(self._score(case, output))
                except Exception as exc:
                    self._log_target_exchange(
                        "template_candidate", locals().get("new_inputs", inp), None,
                        case_id=case.id, error=str(exc),
                        duration_sec=time.perf_counter() - started,
                        metadata={"generation_kwargs": gen_kwargs, "candidate": candidate.name},
                    )
                    return None

            return l1

        spec = PipelineSpec.from_dict(candidate.payload)
        if spec is None:  # already validated at proposal time; belt and braces
            return lambda model, case: None

        def declarative(model: "Model", case: "FailureCase") -> "Optional[bool]":
            capture: "dict[str, Any]" = {}
            result = run_pipeline(
                model, case, spec, self._score, capture=capture,
                call_logger=lambda record: self._log_target_exchange(
                    "declarative_pipeline", record.get("inputs"), record.get("output"),
                    case_id=case.id, error=record.get("error"),
                    duration_sec=record.get("duration_sec"),
                    metadata={
                        "candidate": candidate.name,
                        "generation_kwargs": record.get("generation_kwargs") or {},
                    },
                ),
            )
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
            if getattr(self.run_logger, "preserve_full_model_io", False):
                from evalrx.eval_agent.model_instrumentation import InstrumentedModel

                model = InstrumentedModel(
                    model, self.run_logger, stage="M5",
                    cycle=int(getattr(self.run_logger, "current_cycle", -1)),
                    analyzer=f"fix_primitive:{candidate.name}",
                    case_prompts={case.inputs.prompt: case.id for case in data},
                    batch_case_ids=[case.id for case in data],
                )
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
            if getattr(self.run_logger, "preserve_full_model_io", False):
                from evalrx.eval_agent.model_instrumentation import InstrumentedModel

                model = InstrumentedModel(
                    model, self.run_logger, stage="M5",
                    cycle=int(getattr(self.run_logger, "current_cycle", -1)),
                    analyzer=f"fix_finetune:{candidate.name}",
                    case_prompts={case.inputs.prompt: case.id for case in data},
                    batch_case_ids=[case.id for case in data],
                )
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
                # Do not execute a conditional repair outside its registered
                # population. Validation already excludes those cases, but
                # generating for them wastes compute and can advance a
                # stochastic model before the applicable cases are sampled.
                if not self._applies(candidate, case):
                    return case.id, None
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

    def _registered_baseline_scores(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch"
    ) -> "dict[str, Optional[bool]] | None":
        """Run a registered repair's matched control arm, when declared.

        A decoding repair may change both sampling and logits. Comparing it to
        Stage 0's greedy output would confound those changes; this control
        keeps the paired test about the intervention itself.
        """
        if candidate.kind != "registered_repair":
            return None
        executor_name = str(candidate.payload.get("baseline_executor", "") or "")
        if not executor_name:
            return None
        executor = getattr(model, executor_name, None)
        if not callable(executor):
            return {case.id: None for case in data}

        def guarded(case: "FailureCase") -> "tuple[str, Optional[bool]]":
            try:
                output = executor(case.inputs)
                return case.id, score_to_bool(self._score(case, str(output)))
            except Exception as exc:
                logger.warning(
                    "FixAgent: matched baseline %s failed on case %s: %s",
                    executor_name,
                    case.id,
                    exc,
                )
                return case.id, None

        cases = list(data)
        if self._concurrency > 1 and len(cases) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                return dict(pool.map(guarded, cases))
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
            call_logger=self._log_bridge_exchange,
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
                    call_logger=self._log_bridge_exchange,
                )
        candidate.payload["exec_error"] = "" if result.ok else result.error
        candidate.payload["selection_guard"] = {
            "min_support": int(candidate.payload.get("consensus_min_support", 0)),
            "n_guarded": result.n_guarded,
            "guarded_ids": result.guarded_ids,
            # Audit trail for the anchor semantics: how many cases were
            # anchored on their recorded baseline because the pipeline made
            # no direct call, how many plain direct calls were answered from
            # the record, and which cases had nothing to anchor on.
            "n_anchored_from_recorded": result.n_anchored_from_recorded,
            "n_replayed": result.n_replayed,
            "unanchored_ids": result.unanchored_ids,
        }
        if result.unanchored_ids:
            logger.warning(
                "FixAgent: %d case(s) excluded from the coded pipeline result — no "
                "recorded baseline and no direct model_generate(case_id) call to "
                "anchor the selection guard on", len(result.unanchored_ids),
            )
        if candidate.trial is not None:
            # Persist the audit trail next to this attempt's prompt and code:
            # the payload itself is never logged, and "did the guard anchor on
            # the record, did it revert anything" is the first question a
            # reviewer asks of a coded round.
            candidate.trial.write(
                "coded_pipeline_result.json",
                json.dumps({
                    "ok": result.ok,
                    "exec_error": "" if result.ok else result.error,
                    "n_calls": result.n_calls,
                    "n_outputs": len(result.outputs),
                    **candidate.payload["selection_guard"],
                }, indent=1),
            )
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

        from evalrx.eval_agent.stages.fix_pipeline import frozen_model_control

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
        if candidate.trial is not None:
            candidate.trial.write(
                "frozen_model_control.json",
                json.dumps(candidate.payload["frozen_model_control"], indent=1),
            )
        if solved:
            logger.info(
                "FixAgent: frozen-model control — %s still solves %d/%d case(s) with the "
                "model held at its recorded answers", candidate.name, len(solved), len(data),
            )

    def _repair_code(self, candidate: FixCandidate, error: str) -> "tuple[str, str, str]":
        """Ask the coder to fix its failed pipeline; returns (code, source, raw)."""
        from pathlib import Path

        from evalrx.eval_agent.stages.fix_pipeline import (
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
            # The repair round restates the selection contract in full: a
            # repair that only echoes the execution error used to re-violate
            # the selection rule it was never told about.
            min_support=int(candidate.payload.get("consensus_min_support", 0) or 0),
            selection_guidance=self._code_selection_guidance(),
        )
        code, source, raw = "", "", ""
        if self._cli_config is not None and self._cli_config.provider != "llm":
            # The execution host consumes/removes pipeline.py. Restore the
            # user's source before asking the CLI coder to edit it, otherwise
            # an agent can mistake fix_pipeline_exec.py for the target.
            Path(self._workdir(candidate.trial), "pipeline.py").write_text(
                str(candidate.payload.get("code", "")), encoding="utf-8"
            )
            self._last_repair_prompt = (
                base + "\nWrite the corrected code to a file named pipeline.py."
            )
            code, raw = self._write_code_cli(self._last_repair_prompt, candidate.trial)
            source = f"cli:{self._cli_config.provider}"
        if not code.strip() and self._judge is not None:
            self._last_raw_stream = ""
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
        if _code_redefines_model_bridge(code):
            logger.warning("FixAgent: repaired code redefined a host model bridge; dropped")
            code = ""
        return code, source, raw

    def _validate(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        baseline: "dict[str, Optional[bool]]",
        unstable: "set[str] | None" = None,
    ) -> FixValidation:
        v = FixValidation(candidate=candidate, e_threshold=1.0 / self._alpha)
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
        matched_baseline = self._registered_baseline_scores(candidate, model, data)
        base_rates: "dict[str, Optional[float]]" = {}
        for case in data:
            if matched_baseline is not None:
                b = score_to_bool(matched_baseline.get(case.id))
                r = None if b is None else float(b)
            else:
                r = self._baseline_rates.get(case.id) if self._baseline_rates else None
                if r is None:
                    b = score_to_bool(baseline.get(case.id))
                    r = None if b is None else float(b)
            base_rates[case.id] = r
        if matched_baseline is not None:
            v.n_baseline_samples = 1
        elif self._baseline_n:
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
        data: "CaseBatch | None" = None,
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
            and not self._tier_available(target, model, data=data)
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

    def _tier_available(
        self,
        tier: FixTier,
        model: "Model",
        *,
        data: "CaseBatch | None" = None,
    ) -> bool:
        """Whether an invasive tier has an executor compatible with this task."""
        if data is not None and tier in {
            FixTier.L3A_INTERNALS_READ,
            FixTier.L3B_INTERNALS_WRITE,
        }:
            has_images = any(
                getattr(c.inputs, "image", None) is not None
                or getattr(c.inputs, "video", None) is not None
                for c in data
            )
            has_audio = any(getattr(c.inputs, "audio", None) is not None for c in data)
            tasks = {str((getattr(c, "metadata", {}) or {}).get("task", "")) for c in data}
            registered = any(
                method.tier == tier
                for method in discover_methods(
                    model,
                    max_tier=tier,
                    has_images=has_images,
                    has_audio=has_audio,
                    tasks=tasks,
                    allow_adapted=self._allow_adapted_paper_methods,
                )
            )
            if registered:
                return True
            if tier == FixTier.L3A_INTERNALS_READ:
                from evalrx.core.capability import Capability

                return bool(
                    self._allow_codegen
                    and Capability.ATTENTION in getattr(model, "capabilities", frozenset())
                )
            return bool(
                has_images
                and any(
                    primitive.tier == tier and primitive.available(model)
                    for primitive in INTERNALS_PRIMITIVES.values()
                )
            )
        if tier == FixTier.L3A_INTERNALS_READ:
            from evalrx.core.capability import Capability

            return (
                Capability.ATTENTION in getattr(model, "capabilities", frozenset())
                or supports_tier(model, tier)
            )
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
