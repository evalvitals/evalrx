"""Candidate validation: measure the baseline, run a candidate, pair the two.

:class:`CandidateValidator` executes every candidate kind on a validation
batch and runs the paired test against the unmodified baseline
(:mod:`evalrx.stats` — McNemar + e-value, or paired rates under repeats).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.eval_agent.stages.fix.codegen import PipelineCoder
from evalrx.eval_agent.stages.fix.internals import INTERNALS_PRIMITIVES, run_lora_repair
from evalrx.eval_agent.stages.fix.pipeline import score_outputs
from evalrx.eval_agent.stages.fix.recorder import FixRecorder
from evalrx.eval_agent.stages.fix.tools import (
    PipelineSpec,
    _safe_generation_kwargs,
    run_pipeline,
    safe_format,
    score_to_bool,
    spec_changes_input,
)
from evalrx.eval_agent.stages.fix.types import FixCandidate, FixValidation, truncated_count
from evalrx.eval_agent.stages.fix.verdict import verdict
from evalrx.stats import compare, compare_paired_rates

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

logger = logging.getLogger(__name__)


@dataclass
class Baseline:
    """The unmodified model's record on a validation batch.

    ``scores`` are modal pass/fail per case, ``rates`` the per-case pass rate,
    ``counts`` the samples behind each rate, ``unstable`` the cases whose
    samples disagree.
    """

    scores: "dict[str, Optional[bool]]"
    rates: "dict[str, Optional[float]]" = field(default_factory=dict)
    counts: "dict[str, int]" = field(default_factory=dict)
    unstable: "set[str]" = field(default_factory=set)

    @classmethod
    def from_scores(
        cls, scores: "dict[str, Optional[bool]]", unstable: "set[str] | None" = None
    ) -> "Baseline":
        """A one-sample baseline from bare pass/fail booleans (0/1 rates)."""
        return cls(scores=dict(scores), unstable=set(unstable or ()))


def applies(candidate: FixCandidate, case: "FailureCase") -> bool:
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


class CandidateValidator:
    """Runs candidates and baselines for one fix agent.

    Owns the per-candidate output capture; the agent hands it everything else
    (alpha, the decode floor) per call.
    """

    def __init__(
        self,
        *,
        score: "Callable[[FailureCase, str], Optional[bool]]",
        recorder: FixRecorder,
        coder: PipelineCoder,
        progress: "Callable[..., None]",
        concurrency: int = 1,
        baseline_repeats: int = 1,
        candidate_repeats: int = 1,
        finetune_pool: "Any | None" = None,
    ) -> None:
        self.score = score
        self.recorder = recorder
        self.coder = coder
        self._progress = progress
        self.concurrency = concurrency
        self.baseline_repeats = baseline_repeats
        self.candidate_repeats = candidate_repeats
        self.finetune_pool = finetune_pool
        # case id -> the final output that was scored, filled while a
        # candidate runs and moved onto its FixValidation.
        self._captured: "dict[str, str]" = {}

    @property
    def rates_mode(self) -> bool:
        """Whether validation pairs per-case pass rates instead of single samples."""
        return self.baseline_repeats > 1 or self.candidate_repeats > 1

    def baseline(self, model: "Model", data: "CaseBatch") -> Baseline:
        """Measure the unmodified baseline as a per-case PASS RATE.

        Returns the modal per-case scores, the per-case rates and sample
        counts (for the paired-rates test) and the unstable case ids.

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
        k = max(1, int(self.baseline_repeats))
        samples: "dict[str, list[bool]]" = {c.id: [] for c in data}
        need: "list[tuple[Any, int]]" = []
        for case in data:
            observed = getattr(case, "observed", None)
            if observed is not None:
                s0 = score_to_bool(self.score(case, str(observed)))
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
        return Baseline(scores=scores, rates=rates, counts=counts, unstable=unstable)

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
                self.recorder.target_exchange(
                    "fresh_baseline", case.inputs, None, case_id=case.id,
                    error=str(exc), duration_sec=time.perf_counter() - started,
                )
                return case.id, None
            self.recorder.target_exchange(
                "fresh_baseline", case.inputs, output, case_id=case.id,
                duration_sec=time.perf_counter() - started,
            )
            return case.id, score_to_bool(self.score(case, output))

        if self.concurrency > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                results = list(pool.map(one, jobs))
        else:
            results = [one(case) for case in jobs]
        for cid, s in results:
            if s is not None:
                samples[cid].append(bool(s))

    def matched_baseline(
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
                return case.id, score_to_bool(self.score(case, str(output)))
            except Exception as exc:
                logger.warning(
                    "FixAgent: matched baseline %s failed on case %s: %s",
                    executor_name,
                    case.id,
                    exc,
                )
                return case.id, None

        cases = list(data)
        if self.concurrency > 1 and len(cases) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                return dict(pool.map(guarded, cases))
        return dict(guarded(case) for case in cases)

    # -- per-candidate-kind execution ----------------------------------------
    #
    # A candidate kind either runs per case (a strategy: one model call chain
    # per case, threaded by scores()) or as a batch (its own executor over the
    # whole batch). Adding a kind is one method plus one entry below.

    def strategy(
        self, candidate: FixCandidate
    ) -> "Callable[[Model, FailureCase], Optional[bool]]":
        """Compile a candidate to a per-case success function (ab_runner shape)."""
        build = {
            "registered_repair": self._registered_repair_strategy,
            "visual_search": self._visual_search_strategy,
            "detector_visual_search": self._visual_search_strategy,
            "template": self._template_strategy,
        }.get(candidate.kind, self._spec_strategy)
        return build(candidate)

    def scores(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        *,
        max_tokens_floor: "int | None" = None,
    ) -> "dict[str, Optional[bool]]":
        """Per-case success of one candidate on *data*."""
        batch = {
            "primitive": self._run_primitive,
            "code": self._run_code,
            "finetune_spec": self._run_finetune,
        }.get(candidate.kind)
        if batch is not None:
            return batch(candidate, model, data, max_tokens_floor=max_tokens_floor)
        strategy = self.strategy(candidate)
        cases = list(data)
        from threading import Lock

        progress_lock = Lock()
        completed = 0

        def guarded(case: "FailureCase") -> "tuple[str, Optional[bool]]":
            nonlocal completed
            # One case's failure (a template that cannot render, an adapter
            # error) scores None for THAT case; it must never abort the whole
            # candidate — let alone the fix stage.
            try:
                # Do not execute a conditional repair outside its registered
                # population. Validation already excludes those cases, but
                # generating for them wastes compute and can advance a
                # stochastic model before the applicable cases are sampled.
                if not applies(candidate, case):
                    return case.id, None
                return case.id, strategy(model, case)
            except Exception as exc:
                logger.warning("FixAgent: %s failed on case %s: %s", candidate.name, case.id, exc)
                return case.id, None
            finally:
                with progress_lock:
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == len(cases):
                        self._progress("candidate_cases", candidate=candidate.name,
                                       completed=completed, total=len(cases))

        if self.concurrency > 1 and len(cases) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                results = list(pool.map(guarded, cases))
            return dict(results)
        return dict(guarded(case) for case in cases)

    def _registered_repair_strategy(
        self, candidate: FixCandidate
    ) -> "Callable[[Model, FailureCase], Optional[bool]]":
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
                return score_to_bool(self.score(case, str(output)))
            except Exception as exc:
                logger.debug(
                    "Registered repair %s failed on %s: %s",
                    candidate.name,
                    case.id,
                    exc,
                )
                return None

        return registered_repair

    #: kind -> (model method, logged operation, log label)
    _VISUAL_SEARCH = {
        "visual_search": ("generate_visual_search", "guided_visual_search",
                          "Guided visual search"),
        "detector_visual_search": ("generate_detector_visual_search",
                                   "detector_visual_search", "Detector visual search"),
    }

    def _visual_search_strategy(
        self, candidate: FixCandidate
    ) -> "Callable[[Model, FailureCase], Optional[bool]]":
        method, operation, label = self._VISUAL_SEARCH[candidate.kind]

        def visual_search(model: "Model", case: "FailureCase") -> "Optional[bool]":
            started = time.perf_counter()
            try:
                search = getattr(model, method)
                output = search(
                    case.inputs,
                    baseline_answer=getattr(case, "observed", None),
                    **candidate.payload,
                )
                self.recorder.target_exchange(
                    operation, case.inputs, output, case_id=case.id,
                    duration_sec=time.perf_counter() - started,
                    metadata={"candidate": candidate.name, "parameters": candidate.payload},
                )
                self._record_output(case.id, output)
                return score_to_bool(self.score(case, str(output)))
            except Exception as exc:
                self.recorder.target_exchange(
                    operation, case.inputs, None, case_id=case.id,
                    error=str(exc), duration_sec=time.perf_counter() - started,
                    metadata={"candidate": candidate.name, "parameters": candidate.payload},
                )
                logger.debug("%s failed on %s: %s", label, case.id, exc)
                return None

        return visual_search

    def _template_strategy(
        self, candidate: FixCandidate
    ) -> "Callable[[Model, FailureCase], Optional[bool]]":
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
                self.recorder.target_exchange(
                    "template_candidate", new_inputs, output, case_id=case.id,
                    duration_sec=time.perf_counter() - started,
                    metadata={"generation_kwargs": gen_kwargs, "candidate": candidate.name},
                )
                self._record_output(case.id, output)
                return score_to_bool(self.score(case, output))
            except Exception as exc:
                self.recorder.target_exchange(
                    "template_candidate", locals().get("new_inputs", inp), None,
                    case_id=case.id, error=str(exc),
                    duration_sec=time.perf_counter() - started,
                    metadata={"generation_kwargs": gen_kwargs, "candidate": candidate.name},
                )
                return None

        return l1

    def _spec_strategy(
        self, candidate: FixCandidate
    ) -> "Callable[[Model, FailureCase], Optional[bool]]":
        spec = PipelineSpec.from_dict(candidate.payload)
        if spec is None:  # already validated at proposal time; belt and braces
            return lambda model, case: None

        def declarative(model: "Model", case: "FailureCase") -> "Optional[bool]":
            capture: "dict[str, Any]" = {}
            result = run_pipeline(
                model, case, spec, self.score, capture=capture,
                call_logger=lambda record: self.recorder.target_exchange(
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

    def _run_primitive(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch", **_: Any
    ) -> "dict[str, Optional[bool]]":
        model = self.recorder.instrument(model, data, f"fix_primitive:{candidate.name}")
        prim = INTERNALS_PRIMITIVES[candidate.payload["primitive"]]
        return prim.run(model, data, self.score, candidate.payload.get("params"))

    def _run_code(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        *,
        max_tokens_floor: "int | None" = None,
    ) -> "dict[str, Optional[bool]]":
        result = self.coder.run(candidate, model, data, max_tokens_floor=max_tokens_floor)
        for cid, output in result.outputs.items():
            self._record_output(cid, output)
        if result.ok:
            self.coder.frozen_model_control(candidate, data)
        return score_outputs(result, data, self.score)

    def _run_finetune(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch", **_: Any
    ) -> "dict[str, Optional[bool]]":
        model = self.recorder.instrument(model, data, f"fix_finetune:{candidate.name}")
        result = run_lora_repair(model, self.finetune_pool, data, candidate.payload, self.score)
        if isinstance(candidate.payload, dict):
            candidate.payload["exec_error"] = "" if result.ok else result.error
        return result.scores

    def rates(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        *,
        max_tokens_floor: "int | None" = None,
    ) -> "tuple[dict[str, Optional[float]], int]":
        """Per-case candidate PASS RATE over ``candidate_repeats`` passes.

        Coded pipelines / internals primitives / fine-tune recipes run once
        (they are expensive and vote internally when they want to); template
        and spec candidates are repeated ``candidate_repeats`` times. Returns
        ``(rates, n_passes)``; a case is ``None`` when every pass was unscorable.
        """
        m = int(self.candidate_repeats) if candidate.kind in ("template", "spec") else 1
        m = max(1, m)
        tallies: "dict[str, list[bool]]" = {c.id: [] for c in data}
        for _ in range(m):
            scores = self.scores(candidate, model, data, max_tokens_floor=max_tokens_floor)
            for cid, s in scores.items():
                b = score_to_bool(s)
                if b is not None:
                    tallies.setdefault(cid, []).append(bool(b))
        rates = {
            cid: (sum(1 for o in obs if o) / len(obs) if obs else None)
            for cid, obs in tallies.items()
        }
        return rates, m

    def validate(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        baseline: Baseline,
        *,
        alpha: float,
        max_tokens_floor: "int | None" = None,
    ) -> FixValidation:
        """Run *candidate* on *data* and pair it against *baseline*.

        One sample per arm is McNemar + the Bernoulli-mixture e-value; with
        repeats each case is a pass rate and the test is the betting e-value on
        the paired rate differences.
        """
        v = FixValidation(candidate=candidate, e_threshold=1.0 / alpha)
        unstable = baseline.unstable
        rates_mode = self.rates_mode
        v.noise_model = "paired_rates" if rates_mode else "mcnemar"
        self._captured = {}
        truncated_before = truncated_count(model)
        if rates_mode:
            cand_rates, m = self.rates(
                candidate, model, data, max_tokens_floor=max_tokens_floor
            )
            v.n_candidate_samples = m
            scores: "dict[str, Optional[bool]]" = {
                cid: (None if r is None else r >= 0.5) for cid, r in cand_rates.items()
            }
        else:
            scores = self.scores(candidate, model, data, max_tokens_floor=max_tokens_floor)
            cand_rates = {cid: (None if score_to_bool(sc) is None else float(bool(score_to_bool(sc))))
                          for cid, sc in scores.items()}
        truncated_after = truncated_count(model)
        if truncated_before is not None and truncated_after is not None:
            v.n_truncated = max(0, truncated_after - truncated_before)
        v.outputs = dict(self._captured)
        self._captured = {}
        if isinstance(candidate.payload, dict):
            v.exec_error = str(candidate.payload.get("exec_error", "") or "")
        # Baseline rates: measured by baseline() (frozen sample + k-1 fresh);
        # a Baseline built from bare booleans carries 0/1 rates.
        matched_baseline = self.matched_baseline(candidate, model, data)
        base_rates: "dict[str, Optional[float]]" = {}
        for case in data:
            if matched_baseline is not None:
                b = score_to_bool(matched_baseline.get(case.id))
                r = None if b is None else float(b)
            else:
                r = baseline.rates.get(case.id)
                if r is None:
                    b = score_to_bool(baseline.scores.get(case.id))
                    r = None if b is None else float(b)
            base_rates[case.id] = r
        if matched_baseline is not None:
            v.n_baseline_samples = 1
        elif baseline.counts:
            v.n_baseline_samples = max([1] + [int(n) for n in baseline.counts.values()])

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
            b = rb >= 0.5  # modal baseline; ties -> True (as baseline())
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
            if not applies(candidate, case):
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
                stat = compare_paired_rates(base_vec, cand_vec, alpha=alpha)
                v.e_value_regression = stat.details.get("e_value_regression")
            else:
                stat = compare([x >= 0.5 for x in base_vec], [x >= 0.5 for x in cand_vec],
                               paired=True, alpha=alpha)
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
        v.verdict = verdict(v)
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

    def _record_output(self, case_id: str, output: Any) -> None:
        """Remember what a candidate produced for *case_id* (see FixValidation.outputs)."""
        if output is None:
            return
        self._captured[str(case_id)] = str(output)
