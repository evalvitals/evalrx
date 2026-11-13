"""Candidate proposal for the fix stage.

:class:`FixProposer` turns hypotheses and example cases into repair
candidates for every allowed tier: telemetry-driven runtime fixes (L0), judge
prompts (L1), declarative pipelines and the built-in defaults (L2), the coded
pipeline (via :class:`~.codegen.PipelineCoder`), catalog repairs
(L3) and fine-tune recipes (L4). The proposer never sees gold answers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, cast

from evalrx.eval_agent.prompts.fix_agent import (
    _L1_PROMPT,
    _L2_PROMPT,
    _L3_PROMPT,
    _L4_PROMPT,
    _REPAIR_CATALOG_PROMPT,
)
from evalrx.eval_agent.stages.fix.codegen import PipelineCoder
from evalrx.eval_agent.stages.fix.descriptions import judge_description
from evalrx.eval_agent.stages.fix.internals import (
    INTERNALS_PRIMITIVES,
    FinetuneSpec,
    primitives_catalog_text,
)
from evalrx.eval_agent.stages.fix.recorder import FixRecorder
from evalrx.eval_agent.stages.fix.repair_catalog import discover_methods, method_names
from evalrx.eval_agent.stages.fix.tiers import FixTier
from evalrx.eval_agent.stages.fix.tools import (
    TEXT_ONLY_CATALOG_NOTE,
    PipelineSpec,
    _safe_generation_kwargs,
    catalog_text,
)
from evalrx.eval_agent.stages.fix.types import FixCandidate, FixContext

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Judge, Model
    from evalrx.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)

MAX_JUDGE_CANDIDATES = 3


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


def _self_consistency_5(baseline_temperature: Any) -> PipelineSpec:
    # Vote at a stochastic temperature: at T=0 five samples are one sample.
    # Inherit the baseline temperature when it already samples.
    try:
        base_t = float(baseline_temperature) if baseline_temperature is not None else None
    except (TypeError, ValueError):
        base_t = None
    temperature = base_t if (base_t is not None and base_t > 0.0) else 0.7
    return PipelineSpec(
        name="self_consistency_5",
        prompt_template="{prompt}",
        n_samples=5,
        generation_kwargs={"temperature": temperature},
    )


#: Built-in L2 specs by name: what the floor and a single-name pre-registered
#: allowlist resolve to, and the base of some conditional defaults. Each
#: builder takes the baseline's sampling temperature. The judge-less fallback
#: list in ``_l2_candidates`` is separate (its self_consistency_5 samples with
#: do_sample at 0.7), and the reader-facing sentence lives in
#: ``descriptions.BUILTIN_DESCRIPTIONS``.
BUILTIN_SPECS: "dict[str, Callable[[Any], PipelineSpec]]" = {
    "self_consistency_5": _self_consistency_5,
    "self_refine": lambda _t: PipelineSpec(
        name="self_refine", prompt_template="{prompt}", strategy="self_refine"
    ),
    "least_to_most": lambda _t: PipelineSpec(
        name="least_to_most", prompt_template="{prompt}", strategy="least_to_most"
    ),
    "chain_of_verification": lambda _t: PipelineSpec(
        name="chain_of_verification", prompt_template="{prompt}",
        strategy="chain_of_verification",
    ),
    "upscale_sharpen": lambda _t: PipelineSpec(
        name="upscale_sharpen",
        image_ops=[
            {"tool": "upscale", "params": {"factor": 2.0}},
            {"tool": "sharpen", "params": {"factor": 2.0}},
        ],
    ),
    "zoom_equalize": lambda _t: PipelineSpec(
        name="zoom_equalize",
        image_ops=[
            {"tool": "zoom_center", "params": {"factor": 1.6}},
            {"tool": "equalize", "params": {}},
        ],
    ),
}


def builtin_spec(name: str, *, baseline_temperature: Any = None) -> "PipelineSpec | None":
    """A fresh built-in L2 spec by name, or ``None`` for an unknown name."""
    build = BUILTIN_SPECS.get(name)
    return build(baseline_temperature) if build is not None else None


@dataclass(frozen=True)
class ConditionalDefault:
    """A built-in L2 repair applied only to the cases its gold-free gate selects.

    ``qualifies(tasks, has_images)`` decides whether the batch can use it at
    all; ``predicate`` picks the cases it is applied to (and judged on).
    ``opt_in`` defaults are proposed only when named in the candidate
    allowlist; the others whenever the batch qualifies and no allowlist
    excludes them. Adding a conditional repair is one entry in
    :data:`CONDITIONAL_DEFAULTS` (plus its sentence in ``BUILTIN_DESCRIPTIONS``).
    """

    name: str
    qualifies: "Callable[[set[str], bool], bool]"
    predicate: "Callable[[Any], bool]"
    spec: "Callable[[], PipelineSpec]"
    opt_in: bool = True


def _chart_batch(tasks: "set[str]", has_images: bool) -> bool:
    return "exact_or_numeric" in tasks and has_images


#: Proposed in this order, ahead of every judge proposal.
CONDITIONAL_DEFAULTS: "tuple[ConditionalDefault, ...]" = (
    # For audio/other A-D tasks whose baseline never committed to an option.
    # It gates only on the observable output contract (never gold or
    # correctness), so clean answers are preserved while malformed prose is
    # re-asked and voted. Built in so codegen need not rediscover this simple,
    # high-frequency failure and implement brittle option parsing from scratch.
    ConditionalDefault(
        name="malformed_choice_consensus",
        qualifies=lambda tasks, has_images: tasks == {"multiple_choice_letter"},
        predicate=_malformed_choice_predicate,
        spec=lambda: PipelineSpec(
            name="malformed_choice_consensus",
            prompt_template=(
                "{prompt}\n\nListen to the audio evidence carefully and decide silently. "
                "Do not explain or repeat the choices. Reply with exactly one line: "
                "FINAL: X, where X is A, B, C, or D."
            ),
            n_samples=3,
            generation_kwargs={"do_sample": True, "temperature": 0.35, "top_p": 0.9},
            output_key_pattern=r"(?:FINAL(?:\s+ANSWER)?|ANSWER)\s*:\s*\(?([A-D])\)?",
        ),
        opt_in=False,
    ),
    ConditionalDefault(
        name="chart_arithmetic_verify",
        qualifies=_chart_batch,
        predicate=_chart_arithmetic_predicate,
        spec=lambda: PipelineSpec(
            name="chart_arithmetic_verify",
            prompt_template=(
                "Treat this as a chart measurement problem. First identify every "
                "legend/category and plotted value needed by the question, respecting "
                "the axis scale. Then perform the requested operation and independently "
                "check the arithmetic. Do all work internally and return only the short "
                "answer requested by the original question.\n\n{prompt}"
            ),
            strategy="chain_of_verification",
        ),
    ),
    ConditionalDefault(
        name="chart_verified_consensus",
        qualifies=_chart_batch,
        predicate=_chart_case_predicate,
        spec=lambda: PipelineSpec(
            name="chart_verified_consensus",
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
        ),
    ),
    ConditionalDefault(
        name="upscale_count_extract",
        qualifies=_chart_batch,
        predicate=_chart_count_extract_predicate,
        # A BUILTIN_SPECS key, so never None.
        spec=lambda: cast(PipelineSpec, builtin_spec("upscale_sharpen")),
    ),
)


def enforce_generation_floor(candidates: "list[FixCandidate]", floor: "int | None") -> None:
    """Raise any candidate ``max_tokens`` below the baseline budget to it.

    A scaffold may give the model MORE decode room than the baseline had,
    never less: a lower cap truncates the chain of thought and the answer
    never appears, which the scorer reads as a wrong answer — a decoding
    artefact that says nothing about the candidate's idea. The judge's
    proposal is kept in ``payload["generation_kwargs_proposed"]`` for the
    record; the applied value is what ``PipelineSpec.from_dict`` (spec) or
    the template runner (template) reads.
    """
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
            proposed_raw: Any = gk.get("max_tokens")  # may be absent or non-numeric
            proposed = int(proposed_raw)
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


class FixProposer:
    """Proposes repair candidates for one fix agent.

    Holds the proposal settings; the agent passes the per-call tier window
    and the decode floor to :meth:`propose`.
    """

    def __init__(
        self,
        *,
        judge: "Judge | None",
        recorder: FixRecorder,
        coder: PipelineCoder,
        max_judge_candidates: int = MAX_JUDGE_CANDIDATES,
        allow_adapted_paper_methods: bool = False,
        paper_methods_only: bool = False,
        candidate_allowlist: "Iterable[str] | None" = None,
        baseline_generation_kwargs: "dict[str, Any] | None" = None,
        scoring_note: str = "",
        floor_candidates: "Iterable[str] | None" = ("self_consistency_5",),
        deployed_spec: "dict[str, Any] | None" = None,
    ) -> None:
        self._judge = judge
        self.recorder = recorder
        self.coder = coder
        self.max_judge_candidates = max(1, int(max_judge_candidates))
        self._allow_adapted_paper_methods = bool(allow_adapted_paper_methods)
        self._paper_methods_only = bool(paper_methods_only)
        self._candidate_allowlist = (
            frozenset(str(name) for name in candidate_allowlist)
            if candidate_allowlist is not None
            else None
        )
        self._baseline_generation_kwargs = dict(baseline_generation_kwargs or {})
        self._scoring_note = str(scoring_note or "")
        self._floor_candidates = (
            tuple(str(n) for n in floor_candidates) if floor_candidates else ()
        )
        self._deployed_spec = dict(deployed_spec) if deployed_spec else None

    def propose(
        self,
        hypotheses: "list[Hypothesis]",
        data: "CaseBatch",
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        context: "FixContext | None" = None,
        *,
        max_tier: FixTier,
        min_tier: "FixTier | None" = None,
        max_tokens_floor: "int | None" = None,
    ) -> "list[FixCandidate]":
        """Candidates for every allowed tier, in validation order."""
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
        catalog = catalog_text() if has_images else TEXT_ONLY_CATALOG_NOTE
        floor_names = self._floor_names(has_images=has_images, tasks=tasks)
        context_block = self._context_block(
            context, data, model, floor_names=floor_names, max_tokens_floor=max_tokens_floor
        )
        # Recursive rounds only (see _edit_note): the L1/L2 proposers may also
        # EDIT the previously deployed template instead of wrapping it. The
        # coded-pipeline path is left out — its cases_file carries rendered
        # prompts only, so {original_prompt} means nothing to written code.
        proposal_context = context_block + self._edit_note(data)

        candidates: "list[FixCandidate]" = []
        allowlist = self._candidate_allowlist
        if max_tier >= FixTier.L2_SCAFFOLD:
            for default in CONDITIONAL_DEFAULTS:
                wanted = (
                    allowlist is not None and default.name in allowlist
                    if default.opt_in
                    else allowlist is None or default.name in allowlist
                )
                if (
                    not wanted
                    or default.name in prior_names
                    or not default.qualifies(tasks, has_images)
                ):
                    continue
                spec = default.spec()
                spec.name = default.name
                candidates.append(
                    FixCandidate(
                        tier=FixTier.L2_SCAFFOLD,
                        name=default.name,
                        kind="spec",
                        source="conditional_default",
                        payload=spec.to_dict(),
                        predicate=default.predicate,
                    )
                )
        # An explicit allowlist is a pre-registration request. Built-in L2
        # defaults must be materialised before any judge proposal/family-size
        # truncation; filtering only afterwards could silently leave the run
        # with attempted=0 even though the requested candidate exists.
        requested_defaults = set(BUILTIN_SPECS)
        if (
            self._candidate_allowlist is not None
            and len(self._candidate_allowlist) == 1
            and max_tier >= FixTier.L2_SCAFFOLD
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
            frozenset({name})
            for name in [*(d.name for d in CONDITIONAL_DEFAULTS), *requested_defaults]
        }
        catalog_method_only = self._candidate_allowlist in {
            frozenset({name}) for name in catalog_method_names
        }
        skip_lower_tiers = preregistered_only or catalog_method_only
        if (
            not code_only
            and not skip_lower_tiers
            and max_tier >= FixTier.L0_RUNTIME_CONFIG
            and (min_tier is None or min_tier <= FixTier.L0_RUNTIME_CONFIG)
        ):
            candidates += self._runtime_candidates(data, prior_names)
        if (
            not code_only
            and not skip_lower_tiers
            and max_tier >= FixTier.L1_PROMPT
            and (min_tier is None or min_tier <= FixTier.L1_PROMPT)
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
            and max_tier >= FixTier.L2_SCAFFOLD
            and (min_tier is None or min_tier <= FixTier.L2_SCAFFOLD)
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
                floor_spec = self._default_spec(name)
                if floor_spec is not None:
                    candidates.append(
                        FixCandidate(
                            tier=FixTier.L2_SCAFFOLD, name=name, kind="spec",
                            source="floor", payload=floor_spec.to_dict(),
                        )
                    )
        if (not skip_lower_tiers and max_tier >= FixTier.L2_SCAFFOLD
                and (min_tier is None or min_tier <= FixTier.L2_SCAFFOLD)
                and self.coder.available):
            # The coder-written pipeline is the ONE candidate a code-only run
            # exists to field, so it sits outside the ``not code_only`` gate
            # (nested inside it, --code-only proposed nothing at all).
            candidates += self.coder.propose(
                hyp_lines, examples, model, prior_text, max_tier=max_tier,
                context_block=context_block, catalog=catalog,
                text_only=not has_images,
            )
        if (
            not code_only
            and not preregistered_only
            and max_tier >= FixTier.L2_SCAFFOLD
            and (min_tier is None or min_tier <= FixTier.L3B_INTERNALS_WRITE)
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
                max_tier=max_tier,
                has_images=has_images,
                has_audio=has_audio,
                tasks=tasks,
            )
        if (
            not code_only
            and not skip_lower_tiers
            and max_tier >= FixTier.L4_PARAMETERS
            and (min_tier is None or min_tier <= FixTier.L4_PARAMETERS)
        ):
            candidates += self._l4_candidates(hyp_lines)
        if min_tier is not None:
            candidates = [c for c in candidates if c.tier >= min_tier]
        if self._candidate_allowlist is not None:
            candidates = [c for c in candidates if c.name in self._candidate_allowlist]
        enforce_generation_floor(candidates, max_tokens_floor)
        return candidates

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
        """A built-in L2 spec by name, sampling at the baseline's temperature."""
        return builtin_spec(
            name, baseline_temperature=self._baseline_generation_kwargs.get("temperature")
        )

    def resolve_max_tokens_floor(self, model: "Model | None", data: "CaseBatch") -> "int | None":
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
            config: Any = meta.get("generation_config") or {}
            try:
                cap = int(config.get("max_tokens"))
            except (AttributeError, TypeError, ValueError):
                continue
            if cap > 0:
                caps.append(cap)
        return max(caps) if caps else None

    def _context_block(
        self,
        context: FixContext,
        data: "CaseBatch",
        model: "Model | None",
        *,
        floor_names: "tuple[str, ...]" = (),
        max_tokens_floor: "int | None" = None,
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
        floor = max_tokens_floor
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
            config: Any = meta.get("generation_config") or {}
            try:
                cap = int(config.get("max_tokens"))
            except (AttributeError, TypeError, ValueError):
                continue
            if 1 <= cap < 8192:
                caps.append(cap)
            policy: Any = meta.get("generation_policy") or {}
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
                        description=judge_description(p),
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
            catalog = catalog_text() if has_images else TEXT_ONLY_CATALOG_NOTE
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
                        description=judge_description(p), payload=spec.to_dict(),
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
            # through every call of a multi-call strategy (tools.py) --
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
                # exists in run_pipeline (tools.py, n_samples > 1) but had
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

    def _l3_candidates(
        self,
        hyp_lines: str,
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
        *,
        max_tier: FixTier,
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
            max_tier=max_tier,
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
        catalog = primitives_catalog_text(model, max_tier)
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
                    or primitive.tier > max_tier
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
                    and primitive.tier <= max_tier
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
            self.recorder.judge_exchange("fix_judge_object", prompt, raw, error=error)
            return {}
        self.recorder.judge_exchange("fix_judge_object", prompt, raw)
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
            self.recorder.judge_exchange("fix_judge_list", prompt, raw, error=error)
            return []
        self.recorder.judge_exchange("fix_judge_list", prompt, raw)
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
