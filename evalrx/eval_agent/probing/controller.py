"""MVP-2: the triage battery, nested expansion and route follow-ups chosen by fixed rules.

A run goes through three phases:

1. **Battery.** The battery runs on a matched sample of EXPLORE. Its steps,
   in order:

   1. ``answer_extraction_audit``: re-reads the Stage-0 outputs (no model call).
   2. ``termination_audit`` without continuation: re-reads the outputs too.
   3. ``modality_ablation(mode="swap")``: the N check.
   4. ``attention_knockout(target="prediction")``: the first N-layer measurement.
   5. ``self_consistency``: re-samples each case (U).
   6. ``prompt_contrast`` with the question kept verbatim in neutral wrappers (U).
   7. ``prompt_contrast`` with an instruction not to default to "no" (yes/no
      tasks only), descriptive S.
   8. ``embed_lens(layer=0.9)``: is the case's own question word near the
      media tokens late in the stack (S2).
   9. ``patchscope(target="prediction", media_layers=3)``: how early the
      model's own answer was readable, and is the question word read off the
      media tokens (S3, S2).

   Steps with a target read the model's own answer, never gold: their
   readings are compared between FAIL cases and PASS controls, and a FAIL
   output is not gold by definition.

2. **Expansion.** While an evidence gap holds that more cases can close, the
   controller decides to expand. The sample then doubles along the sampler's
   rank prefixes, and the battery runs again only where it has to:

   * every step's per-case rows are independent of the batch (the lenses tell
     question words from a fixed reference, the tuned lens is fit on fixed
     EXPLORE cases, the causal trace seeds noise per case), so steps run on the
     new cases alone and the old rows stay current;
   * the descriptive steps no route reads (``embed_lens``, ``patchscope``) are
     ``deferred``: nothing the expansion rule looks at depends on them, so they
     run once the expansions have settled.

   Steps run in chunks of ``chunk`` cases (FAIL and PASS alternating), with the
   budget checked before each chunk on the cost measured so far.

3. **Follow-ups.** Each active route proposes its next layer's candidate (see
   :mod:`~evalrx.eval_agent.probing.rules`), and the candidate runs on the
   route's FAIL cases plus matched PASS controls.

Every decision is a trace event that names its rule, the condition it read,
and the candidates it passed over and why. The run stops when:

* no route has a candidate left;
* ``max_actions`` decisions (expansions + follow-ups) have been made; or
* the budget is spent, or the next chunk's estimate exceeds what is left (a
  step's own measured cost per case, else the most expensive case measured in
  this run; there is no per-analyzer cost prior).

The rules scheduler calls no judge and names no hypothesis. The scored
scheduler (MVP-3) may take an explainer: the judge then proposes competing
explanations once the sample has settled, and their predictions give each
candidate its D (:mod:`.explanations`); no explanation is a finding. CONFIRM
never enters. Outputs land in ``out_dir``::

    split_manifest.json  triage_report[.scored].json  triage_report[.scored].md
    selection_trace.jsonl  cost_ledger.jsonl  observations/<action_id>.json
    explanations/<key>.json   (with an explainer: each judge call's prompt, reply and cost)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from evalrx.analyzers.perturbation.prompt_contrast import _DEFAULT_STRATEGIES as _PROMPT_STRATEGIES
from evalrx.eval_agent.probing.actions import (
    FAILED,
    OK,
    SKIPPED,
    PlanStep,
    ProbeAction,
    canonical,
)
from evalrx.eval_agent.probing.evidence import StepEvidence
from evalrx.eval_agent.probing.executor import execute, gate
from evalrx.eval_agent.probing.explanations import (
    CARDS,
    EXPLAIN_VERSION,
    Explainer,
    ExplanationSet,
    Hypothesis,
    Metric,
    carry_over,
    check,
    decidability,
    disagreement,
    label_bound,
    outcome,
    restates_route,
)
from evalrx.eval_agent.probing.explanations import spread as reading_spread
from evalrx.eval_agent.probing.ledger import Ledger
from evalrx.eval_agent.probing.rules import (
    LAYER_READOUT,
    LAYERS,
    MEDIA_ATTENTION_CUT,
    ROUTE_ORDER,
    RULES_VERSION,
    Layer,
    Proposal,
    active_routes,
    candidate_steps,
    expansion_gaps,
    matched_controls,
    next_for_route,
    next_size,
    public,
    route_fail_ids,
    route_states,
)
from evalrx.eval_agent.probing.sampler import matched_sample, rank
from evalrx.eval_agent.probing.scoring import (
    SCORING_VERSION,
    Scored,
    Weights,
    availability,
    axis_of,
    choose,
    coverage_gain,
    coverage_units,
    public_units,
    step_cost,
    value,
)
from evalrx.eval_agent.probing.triage import (
    ROUTE_STEPS,
    follow_up_reading,
    follow_up_values,
    summarize,
    triage_cases,
)

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

POLICY_VERSION = "mvp2-0"
SCHEDULERS = ("rules", "scored")

#: Neutral wrappers around the question, kept verbatim (with its output
#: contract): the meaning cannot change, only the surface. A wrapper whose
#: answer is unreadable is an invalid control, not an unstable answer.
REWORD_STRATEGIES = {
    "baseline": "{prompt}",
    "question_frame": "Question: {prompt}\nAnswer:",
    "instruction_frame": "Answer the following question.\n\n{prompt}",
}
#: A changed decision criterion, not a rewording: tells a stable "no" that
#: moves when the model is told not to default to it from one that does not.
THRESHOLD_STRATEGIES = {"baseline": "{prompt}", "sensitive": _PROMPT_STRATEGIES["sensitive"]}

#: Cases of EXPLORE (in the sampler's rank order) the tuned lens is fit on: fixed, so a
#: case's readout does not depend on the sample it is read with.
FIT_CASES = 32

#: What a FAIL case on each route has shown, as the judge is told.
ROUTE_MEANING = {
    "H": "the output lost the answer: an extraction, truncation or format check fired",
    "N": "the answer did not change when the media was swapped for media on which the correct answer differs",
    "U": "re-sampling at the host's settings mostly gave another answer, or a neutral rewording moved it",
    "S": "a stable error: the output, media-swap and stability checks all ran and none fired",
    "X": "a required check could not run on the case",
}

BATTERY: tuple[PlanStep, ...] = (
    PlanStep("answer_extraction_audit", {"reask": False}, "H",
             "re-read Stage-0 outputs for grading and extraction problems"),
    PlanStep("termination_audit", {"continue_non_clean": False}, "H",
             "classify how each Stage-0 output ended, without continuing it"),
    PlanStep("modality_ablation", {"mode": "swap", "donor_rule": "counterfactual"}, "N",
             "re-ask with media on which the same question's gold differs from the model's answer "
             "(else any different media); an answer that should change but does not is the N signal",
             calls_per_case={"generate": 2}),
    PlanStep("attention_knockout", {"target": "prediction"}, "N1",
             "cut text-to-media attention per layer band; margin change of the model's own first answer token",
             fidelity_check="edit", calls_per_case={"forward": 5}, evidence=MEDIA_ATTENTION_CUT),
    PlanStep("self_consistency", {"n": 5}, "U",
             "re-sample each case at the host's frozen sampling settings; another answer sampled "
             "more often than the Stage-0 answer is the U signal",
             calls_per_case={"generate": 5}, per_case=True, keep_artifacts=("samples",)),
    PlanStep("prompt_contrast", {"strategies": REWORD_STRATEGIES}, "U",
             "re-ask with the question verbatim inside two neutral wrappers; an answer that "
             "moves is the U signal",
             calls_per_case={"generate": 3}, key="prompt_reword", keep_artifacts=("outputs",)),
    PlanStep("prompt_contrast", {"strategies": THRESHOLD_STRATEGIES}, "S",
             "re-ask with an instruction not to default to a negative answer: which errors move, "
             "and which correct answers it breaks",
             calls_per_case={"generate": 2}, key="prompt_threshold", keep_artifacts=("outputs",),
             task_kinds=("yes_no",)),
    PlanStep("embed_lens", {"layer": 0.9}, "S2",
             "nearest vocabulary words of the media tokens late in the stack: how close the "
             "case's own question word gets",
             calls_per_case={"forward": 1}, deferred=True),
    PlanStep("patchscope", {"target": "prediction", "media_layers": 3}, "S3",
             "decode the answer position at every layer and the media tokens at three depths: "
             "how early the model's own answer was readable, and whether the question word is "
             "read off the media",
             fidelity_check="patch", calls_per_case={"forward": 1}, deferred=True, evidence=LAYER_READOUT),
)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonical(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def analyzer_extras(analyzer_kwargs: "dict[str, dict[str, Any]] | None", discovery: "CaseBatch", seed: int, *,
                    donor_pool: "CaseBatch | None" = None) -> dict[str, dict[str, Any]]:
    """Per-analyzer constructor extras: the host's, plus what is fixed on DISCOVERY.

    The lenses tell question words (and embed_lens recurring words) from a fixed reference,
    DISCOVERY in the sampler's rank order, and the tuned lens is fit on the first ``FIT_CASES``
    of it, so a case's row never depends on the sample it is read with; a validation run on
    other cases uses the same frozen artefacts. Swap donors come from the partition the cases
    are drawn from (*donor_pool*, default DISCOVERY).
    """
    from evalrx.core.case import CaseBatch

    extras = {k: dict(v) for k, v in (analyzer_kwargs or {}).items()}
    extras.setdefault("modality_ablation", {}).setdefault("donor_pool", donor_pool if donor_pool is not None
                                                          else discovery)
    ranked = CaseBatch(sorted(discovery, key=lambda c: rank(seed, str(c.id))))
    for name in ("embed_lens", "patchscope"):
        extras.setdefault(name, {}).setdefault("reference", ranked)
    extras.setdefault("tuned_lens", {}).setdefault("fit_cases", CaseBatch(list(ranked)[:FIT_CASES]))
    return extras


def tool_version(analyzer: Any) -> str:
    """Content hash of the analyzer's module and the shared ``analyzers/_internals``.

    Part of every action identity, so changing how a tool measures (or checks
    its own fidelity) makes a resumed run re-execute it instead of reading
    back an observation the old code produced.
    """
    import evalrx.analyzers._internals as internals

    paths = [Path(inspect.getsourcefile(type(analyzer)) or "")]
    paths += sorted(Path(internals.__file__).parent.glob("*.py"))
    h = hashlib.sha1()
    for path in paths:
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def output_cap(model_identity: dict[str, Any]) -> "int | None":
    """The generation cap the host set, per call or in the runtime (``None`` = unknown here)."""
    for generation in (model_identity.get("generation") or {}, model_identity.get("runtime_generation") or {}):
        for key in ("max_new_tokens", "max_output_tokens", "max_tokens"):
            if generation.get(key) is not None:
                return int(generation[key])
    return None


class _Run:
    """The state one invocation carries: evidence per step, the executions, the ledger."""

    def __init__(self, model: Any, explore: "CaseBatch", ledger: Ledger, *, context: dict[str, Any],
                 extras: dict[str, dict[str, Any]], task_kind: "str | None", budget_sec: float,
                 price_fn: Any, run_logger: Any, say: Callable[[str], None],
                 per_call_generation: bool = True, budget_usd: float = 0.0, chunk: int = 0) -> None:
        self.model, self.ledger, self.context, self.extras = model, ledger, context, extras
        self.per_call_generation, self.budget_usd, self.chunk = per_call_generation, budget_usd, chunk
        self.task_kind, self.budget_sec, self.price_fn, self.run_logger, self.say = (
            task_kind, budget_sec, price_fn, run_logger, say)
        self.by_id = {str(c.id): c for c in explore}
        self.order = {str(c.id): i for i, c in enumerate(explore)}
        self.evidence: dict[str, StepEvidence] = {}
        self.executions: list[dict[str, Any]] = []
        self.blocked: dict[str, str] = {}
        self.judge_cost: dict[str, float] = {}

    def batch(self, ids: "list[str]") -> "CaseBatch":
        from evalrx.core.case import CaseBatch

        return CaseBatch([self.by_id[i] for i in sorted(set(ids), key=self.order.__getitem__)])

    def config(self, step: PlanStep) -> dict[str, Any]:
        """The constructor arguments the step actually runs with: on a backend that drops
        per-call generation settings it gets none, so its identity says what ran."""
        config = {**step.config, **self.extras.get(step.analyzer, {})}
        if not self.per_call_generation:
            config.pop("gen_kwargs", None)
        return config

    def output_cap(self, step: PlanStep) -> "int | None":
        """The output cap *step* runs under: its own per-call setting, else the host's."""
        asked = (self.config(step).get("gen_kwargs") or {}).get("max_new_tokens")
        return int(asked) if asked is not None else output_cap(self.context.get("model") or {})

    def build(self, step: PlanStep) -> Any:
        from evalrx.core.registry import registry

        return registry.analyzers.get(step.analyzer)(**self.config(step))

    def gate(self, step: PlanStep, ids: "list[str]") -> "tuple[str, str] | None":
        """Why *step* cannot run on *ids* here — ``(status, reason)`` — before anything is spent."""
        if step.task_kinds and self.task_kind not in step.task_kinds:
            return "not_applicable", f"for task kinds {list(step.task_kinds)}; this task is {self.task_kind!r}"
        cap = self.output_cap(step)
        if step.min_output_tokens and cap is not None and cap < step.min_output_tokens:
            return "not_applicable", (f"the output cap of {cap} tokens is below the {step.min_output_tokens} "
                                      f"this step needs ({step.caveat or 'text before the answer'})")
        try:
            analyzer = self.build(step)
        except Exception as exc:  # noqa: BLE001 - a candidate that cannot be built says so
            return "could_not_build", f"{type(exc).__name__}: {exc}"
        why = gate(analyzer, self.model, self.batch(ids))
        return ("unsupported", why) if why else None

    def estimate_sec(self, step: PlanStep, n_cases: int) -> "float | None":
        """Seconds a budget check assumes: the step's own cost per case, else the most
        expensive case measured in this run (``None`` before anything was measured)."""
        cost, _ = step_cost(step, n_cases, self.evidence, "wall_sec", unmeasured="max")
        return round(cost, 1) if cost is not None else None

    def estimate_usd(self, step: PlanStep, n_cases: int) -> "float | None":
        cost, _ = step_cost(step, n_cases, self.evidence, "usd", unmeasured="max")
        return round(cost, 6) if cost is not None else None

    def chunks(self, step: PlanStep, ids: "list[str]") -> "list[list[str]]":
        """*ids* in the order a step runs them: a ``reuse`` step in chunks of ``self.chunk``
        cases with FAIL and PASS alternating (so every chunk can be compared with its
        controls); any other step, or no chunk size, in one piece."""
        if not step.reuse or self.chunk <= 0 or len(ids) <= self.chunk:
            return [list(ids)] if ids else []
        def is_fail(cid: str) -> bool:
            return getattr(getattr(self.by_id[cid], "label", None), "value", None) == "fail"

        fails = [c for c in ids if is_fail(c)]
        passes = [c for c in ids if not is_fail(c)]
        order = [c for pair in zip(fails, passes) for c in pair]
        order += fails[len(passes):] + passes[len(fails):]
        return [order[i:i + self.chunk] for i in range(0, len(order), self.chunk)]

    def charged(self, axis: str) -> float:
        """What this run's evidence cost when each observation was first executed (a read-back
        at its original cost, once), plus its judge calls: the same for a resumed run and for
        a second scheduler on a shared ledger as for the run that executed it."""
        seen: dict[str, float] = {}
        for ev in self.evidence.values():
            for obs in ev.chunks:
                seen.setdefault(obs.action_id, float((obs.cost or {}).get(axis) or 0.0))
        return sum(seen.values()) + float(self.judge_cost.get(axis) or 0.0)

    def over_budget(self, estimate: "float | None", estimate_usd: "float | None" = None) -> "str | None":
        """Why an action with these estimates may not start (``None`` = it may)."""
        if self.budget_sec > 0:
            spent = self.charged("wall_sec")
            if spent >= self.budget_sec:
                return f"wall-clock budget {self.budget_sec}s spent"
            if estimate is not None and spent + estimate > self.budget_sec:
                return f"estimated {estimate}s exceeds the {round(self.budget_sec - spent, 1)}s left"
        if self.budget_usd > 0:
            spent = self.charged("usd")
            if spent >= self.budget_usd:
                return f"USD budget {self.budget_usd} spent"
            if estimate_usd is not None and spent + estimate_usd > self.budget_usd:
                return f"estimated USD {estimate_usd} exceeds the {round(self.budget_usd - spent, 6)} left"
        return None

    def _identity(self, step: PlanStep) -> tuple[Any, str, dict[str, Any], str]:
        """``(analyzer, tool version, identity context, step identity)``; raises if it cannot be built."""
        analyzer = self.build(step)
        version = tool_version(analyzer)
        context = {**self.context, "tool_version": version, "per_case": step.per_case,
                   "keep_artifacts": list(step.keep_artifacts)}
        template = ProbeAction(analyzer=step.analyzer, config=self.config(step), case_ids=(),
                               route=step.route, purpose=step.purpose)
        return analyzer, version, context, template.step_identity(context)

    def adopt_for(self, step: PlanStep, target: "list[str]", *, kind: str, revision: int,
                  rule_id: "str | None" = None, route: "str | None" = None) -> bool:
        """Take up earlier chunks of *step* on *target* before a decision is made about it;
        ``True`` when something was taken up (the decision is then re-made)."""
        if not step.reuse:
            return False
        try:
            _, _, _, step_id = self._identity(step)
        except Exception:  # noqa: BLE001 - run_step records the construction failure
            return False
        ev = self.evidence.setdefault(step.name, StepEvidence(step))
        todo = ev.todo(target)
        if not todo or self.over_budget(self.estimate_sec(step, len(todo)), self.estimate_usd(step, len(todo))):
            return False   # what a fresh run could not afford is not read back either: run_step records it
        n = len(ev.chunks)
        self._adopt(ev, step_id, target, {"kind": kind, "revision": revision, "key": step.name,
                                          "analyzer": step.analyzer, "rule_id": rule_id, "route": route})
        return len(ev.chunks) > n

    def _log(self, record: dict[str, Any]) -> None:
        record["n"] = len(self.executions)
        self.executions.append(record)

    def _adopt(self, ev: StepEvidence, step_id: str, target: "list[str]",
               base: dict[str, Any]) -> "dict[str, Any] | None":
        """Take up chunks of the same step identity measured before (by this run or an earlier
        one) inside *target*, on cases no chunk covers yet; the last such execution record."""
        record = None
        known = {o.action_id for o in ev.chunks}
        wanted = set(target)
        for obs in self.ledger.chunks_for(step_id):
            ids = set(obs.planned_ids)
            # only chunks inside the target: one that also covers other cases would bring
            # evidence on them that no decision of this run asked for
            if obs.action_id in known or not ids or not ids <= wanted or ids & ev.covered():
                continue
            ev.add(obs)
            row = {**base, "n_cases": len(ids & wanted), "action_id": obs.action_id, "step_id": step_id}
            self.ledger.trace({**row, "event": "reused", "status": obs.status, "attempt": obs.attempt,
                               "note": "same step identity on other cases; the step's rows do not depend on "
                                       "the batch, so these cases are read from it and not re-run"})
            record = {**row, "status": obs.status, "reused": True, "fidelity": obs.fidelity,
                      "cost": obs.cost, "error": obs.error}
            self._log(record)
            self.say(f"[triage] {ev.step.name} ({row['n_cases']} cases): reused chunk ({obs.status})")
        return record

    def run_step(self, step: PlanStep, target: "list[str]", *, kind: str, revision: int,
                 rule_id: "str | None" = None, route: "str | None" = None,
                 budget_checked: bool = False) -> "dict[str, Any] | None":
        """Bring *step*'s evidence up to date on *target*; the last execution record, or
        ``None`` when every case is already covered.

        A ``reuse`` step runs in chunks (:meth:`chunks`) with the budget checked before
        each one on the cost measured so far, so an action overshoots by at most one
        chunk; a refusal by the step's own check stops the remaining chunks."""
        ev = self.evidence.setdefault(step.name, StepEvidence(step))
        base = {"kind": kind, "revision": revision, "key": step.name, "analyzer": step.analyzer,
                "rule_id": rule_id, "route": route}
        if step.task_kinds and self.task_kind not in step.task_kinds:
            if any(e["key"] == step.name and e["status"] == "not_applicable" for e in self.executions):
                return None
            reason = f"step is for task kinds {list(step.task_kinds)}; this task is {self.task_kind!r}"
            self.ledger.trace({**base, "event": "not_applicable", "reason": reason})
            record = {**base, "status": "not_applicable", "reused": False, "reason": reason}
            self._log(record)
            self.say(f"[triage] {step.name}: not applicable ({self.task_kind})")
            return record
        config = self.config(step)
        try:
            analyzer, version, context, step_id = self._identity(step)
        except Exception as exc:  # noqa: BLE001 - a bad config is an outcome, not a crash
            if not ev.todo(target):
                return None
            reason = f"could not construct: {type(exc).__name__}: {exc}"
            self.ledger.trace({**base, "event": "skipped", "status": "failed", "reason": reason})
            self.blocked[step.name] = reason
            record = {**base, "status": "failed", "reused": False, "error": str(exc)}
            self._log(record)
            return record
        todo = ev.todo(target)
        if not todo:
            return None
        record = None
        for part in self.chunks(step, todo):
            out = self._run_part(step, part, ev, base, analyzer, version, context, step_id, config,
                                 kind=kind, revision=revision, rule_id=rule_id, budget_checked=budget_checked)
            record = out if out is not None else record
            if out is not None and out["status"] == FAILED and not out.get("reused"):
                # an execution that raised costs time and would likely raise again on the rest
                # (an unsupported chunk costs nothing: the rest still get their own verdict)
                self.blocked.setdefault(step.name, f"failed: {out.get('error') or ''}".strip())
                break
            if (out is not None and out["status"] == SKIPPED) or ev.refused() is not None:
                break
        return record

    def _run_part(self, step: PlanStep, target: "list[str]", ev: StepEvidence, base: dict[str, Any],
                  analyzer: Any, version: str, context: dict[str, Any], step_id: str, config: dict[str, Any], *,
                  kind: str, revision: int, rule_id: "str | None", budget_checked: bool) -> "dict[str, Any] | None":
        """One case set of :meth:`run_step`: read back what exists, else execute."""
        todo = ev.todo(target)
        if not todo:
            return None
        # Checked as if the cases ran fresh, before anything is read back: the budget is
        # charged with the evidence read, so a resumed run (or a second scheduler on the
        # same ledger) must skip what the run that executed it could not afford.
        estimate = self.estimate_sec(step, len(todo))
        why = None if budget_checked else self.over_budget(estimate, self.estimate_usd(step, len(todo)))
        if why is not None:
            self.ledger.trace({**base, "event": "skipped", "status": SKIPPED, "reason": why, "n_cases": len(todo)})
            self.blocked[step.name] = why
            record = {**base, "status": SKIPPED, "reused": False, "reason": why, "n_cases": len(todo)}
            self._log(record)
            self.say(f"[triage] {step.name}: skipped ({why})")
            return record
        record = self._adopt(ev, step_id, target, base) if step.reuse else None

        todo = ev.todo(target)
        if not todo:
            return record
        cases = self.batch(todo)
        action = ProbeAction(
            analyzer=step.analyzer, config=config, case_ids=tuple(str(c.id) for c in cases),
            route=step.route, purpose=step.purpose, source=rule_id or kind,
            reserve={k: v * len(cases) for k, v in step.calls_per_case.items()},
        )
        action_id = action.identity(context)
        base = {**base, "n_cases": len(cases), "action_id": action_id, "step_id": step_id}

        done = self.ledger.settled(action_id)
        if done is not None:
            self.ledger.trace({**base, "event": "reused", "status": done.status, "attempt": done.attempt,
                               "note": "terminal observation with the same identity; read back, not re-run"})
            ev.add(done)
            record = {**base, "status": done.status, "reused": True, "fidelity": done.fidelity,
                      "cost": done.cost, "error": done.error}
            self._log(record)
            self.say(f"[triage] {step.name} ({len(cases)} cases): reused ({done.status})")
            return record

        estimate = self.estimate_sec(step, len(cases))
        prior = self.ledger.attempts(action_id)
        attempt = prior + 1
        self.ledger.trace({
            **base, "event": "planned", "attempt": attempt, "action": action.to_dict(),
            "params": canonical(analyzer.get_params()), "tool_version": version,
            "fidelity_check": step.fidelity_check, "reuse": step.reuse, "estimate_sec": estimate,
            **({"previous_attempts_uncertain": prior} if prior else {}),
        })
        self.say(f"[triage] {step.name}: running on {len(cases)} cases ({kind}, revision {revision})")
        obs = execute(action, action_id, attempt, analyzer, self.model, cases,
                      fidelity_check=step.fidelity_check, per_case=step.per_case,
                      keep_artifacts=step.keep_artifacts, run_logger=self.run_logger)
        if obs.cost.get("api_usage") is not None:
            obs.cost["usd"] = self.price_fn(obs.cost["api_usage"]) if self.price_fn is not None else None
        self.ledger.save(obs, step_id=step_id)
        self.ledger.cost({"kind": "action", "action_id": action_id, "analyzer": step.analyzer,
                          "attempt": attempt, "reserved_calls": action.reserve, "estimate_sec": estimate,
                          "baseline_reused": obs.status == OK and not obs.cost.get("calls"),
                          "judge_calls": 0, **obs.cost})
        self.ledger.trace({**base, "event": "settled", "attempt": attempt, "status": obs.status,
                           "fidelity": obs.fidelity.get("status"), "n_measured": len(obs.measured_ids),
                           "n_missing": len(obs.missing), "error_type": obs.error_type, "error": obs.error})
        ev.add(obs)
        record = {**base, "status": obs.status, "reused": False, "fidelity": obs.fidelity, "cost": obs.cost,
                  "error": obs.error, "n_measured": len(obs.measured_ids), "missing": obs.missing,
                  "estimate_sec": estimate}
        self._log(record)
        self.say(f"[triage] {step.name}: {obs.status} (fidelity={obs.fidelity.get('status')}, "
                 f"{obs.cost.get('wall_sec')}s, calls={obs.cost.get('calls')})")
        return record


def run_triage(
    model: Any,
    explore: "CaseBatch",
    *,
    out_dir: "str | Path",
    model_identity: dict[str, Any],
    n_fail: int = 8,
    max_fail: "int | None" = None,
    max_actions: int = 12,
    min_route_fail: int = 3,
    min_cohort: int = 8,
    seed: int = 0,
    stratum: str = "gold",
    host_assumed_finish: bool = False,
    answer_contract: bool = False,
    n_needs_counterfactual: bool = False,
    task_kind: "str | None" = None,
    answer_fn: "Callable[[Any, str], str | None] | None" = None,
    agree_fn: "Callable[[Any, str, str], bool] | None" = None,
    analyzer_kwargs: "dict[str, dict[str, Any]] | None" = None,
    price_fn: "Callable[[dict[str, Any] | None], float | None] | None" = None,
    battery: "tuple[PlanStep, ...]" = BATTERY,
    layers: "tuple[Layer, ...]" = LAYERS,
    budget_sec: float = 0.0,
    stage0_cost: "dict[str, Any] | None" = None,
    partition: "dict[str, Any] | None" = None,
    per_call_generation: bool = True,
    scheduler: str = "rules",
    weights: "Weights | None" = None,
    explore_share: float = 0.1,
    explore_from: str = "targets",
    min_gain: float = 0.01,
    min_cover: int = 8,
    budget_usd: float = 0.0,
    chunk: int = 8,
    explainer: "Explainer | None" = None,
    run_logger: Any = None,
    progress: "Callable[[str], None] | None" = print,
) -> dict[str, Any]:
    """Run the battery, the rule-chosen expansions and follow-ups on *explore*; return the report.

    Args:
        model:               the model under test.
        explore:             Stage-0-labelled EXPLORE cases (CONFIRM never enters here).
        out_dir:             where the trace, ledger, observations and report go.
        model_identity:      spec / weights / backend / dtype / generation settings;
                             part of every action identity, so a different model
                             never reuses an observation.
        n_fail:              FAIL cases in the first sample (PASS controls are matched to them).
        max_fail:            the most FAIL cases expansion may reach (``None`` = *n_fail*:
                             no expansion).
        max_actions:         decisions after the battery (expansions + follow-ups); the scored
                             scheduler's further chunks of a step it already took do not count.
        min_route_fail:      FAIL cases a route needs before it proposes follow-ups.
        min_cohort:          S cohort / stable-PASS size per stratum below which the
                             expansion rule counts a gap.
        seed:                rank seed of the nested sampler.
        stratum:             ``"gold"`` (match PASS on the gold answer) or ``"none"``.
        host_assumed_finish: the host, not the backend, set ``finish_reason``.
        answer_contract:     the task declares an answer tag or output contract.
        n_needs_counterfactual: few-answer task (yes/no, letters): a swap only tests N
                             with a counterfactual donor. Donors are drawn from the
                             whole of *explore*, never from outside it.
        task_kind:           the task's kind; steps limited to other kinds are
                             recorded as ``not_applicable`` and not run.
        answer_fn:           ``(case, output) -> answer key`` (``""`` = no readable
                             answer, ``None`` = the task has no categorical answer).
        agree_fn:            ``(case, candidate, reference) -> bool``: when two keys are
                             one answer, as the task grades (default: equal keys); every
                             answer comparison of the checks and readings uses it.
        analyzer_kwargs:     per-analyzer constructor extras the host supplies
                             (e.g. the task's ``score_fn``, the sampling settings).
        price_fn:            ``api_usage -> USD`` (``None`` = no price on record).
        budget_sec:          wall-clock cap (0 = none) on the evidence the run reads,
                             each observation at the cost of its first execution (so a
                             resumed run, or a second scheduler on the same ledger, is
                             charged what the evidence cost), plus its judge calls. A
                             running action is not interrupted; one whose estimate (from
                             the same step's earlier seconds per case) exceeds what is
                             left is skipped.
        stage0_cost:         the shared baseline cost, logged once and kept separate.
        partition:           how EXPLORE/CONFIRM were split (recorded, not used).
        per_call_generation: the backend honours per-call generation settings (an
                             OpenAI-compatible endpoint drops them): when it does
                             not, steps run without their ``gen_kwargs`` and are gated
                             on the host's output cap.
        scheduler:           ``"rules"`` (MVP-2: rounds over the routes' soft layer order)
                             or ``"scored"`` (MVP-3: the candidate with the highest
                             U = q·[D+T+C−R]₊/(c+ε), see :mod:`.scoring`). Both share
                             the ledger; the scored report is ``triage_report.scored``.
        weights, explore_share, min_gain, min_cover: the scored scheduler's weights,
                             exploration's share of the evidence spending, value floor
                             and per-unit coverage target.
        explore_from:        ``"targets"`` (exploration takes inactive routes' next candidates
                             and other routes' representatives, see :mod:`.scoring`) or
                             ``"candidates"`` (an ablation: the candidates other than the top).
        budget_usd:          USD cap (0 = none), charged like ``budget_sec`` and checked
                             with the step's earlier USD per case.
        chunk:               cases per chunk of a ``reuse`` step (FAIL and PASS alternating;
                             0 = one piece). The budget is checked before each chunk on the
                             cost measured so far, so an action overshoots by at most one
                             chunk; the scored scheduler decides one chunk at a time.
        explainer:           the scored scheduler's source of competing explanations
                             (:mod:`.explanations`): asked once the sample has settled and
                             again when the evidence has thinned a route's live explanations
                             (at most ``explainer.max_asks`` times, and only while the largest
                             call so far fits the budget); its predictions give each
                             candidate its D. ``None`` = D stays 0.
        run_logger:          optional RunLoggerV2 for per-call model logs.
    """
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}; got {scheduler!r}")
    if explainer is not None and scheduler != "scored":
        raise ValueError("explanations give D to the scored scheduler; the rules scheduler does not read them")
    if explore_from not in ("targets", "candidates"):
        raise ValueError(f"explore_from must be 'targets' or 'candidates'; got {explore_from!r}")
    weights = weights or Weights()
    started = time.monotonic()
    say = progress or (lambda _msg: None)
    out = Path(out_dir)
    ledger = Ledger(out)
    extras = analyzer_extras(analyzer_kwargs, explore, seed)
    if bound := label_bound((*battery, *candidate_steps(layers)), extras):
        raise ValueError(f"steps {bound} read the gold answer (target='gold'): their readings are compared "
                         "between FAIL and PASS, where that restates the label; use target='prediction'")
    max_fail = max(int(n_fail), int(max_fail or n_fail))
    n_available = sum(1 for c in explore if getattr(getattr(c, "label", None), "value", None) == "fail")
    run = _Run(model, explore, ledger, context={"policy": POLICY_VERSION, "model": model_identity},
               extras=extras, task_kind=task_kind, budget_sec=budget_sec, price_fn=price_fn,
               run_logger=run_logger, say=say, per_call_generation=per_call_generation, budget_usd=budget_usd,
               chunk=chunk)
    rank_of = partial(rank, seed)
    triage = partial(triage_cases, host_assumed_finish=host_assumed_finish, answer_contract=answer_contract,
                     n_needs_counterfactual=n_needs_counterfactual, answer_fn=answer_fn, agree_fn=agree_fn,
                     stratum=stratum)

    sample, draw = matched_sample(explore, n_fail, seed=seed, stratum=stratum)
    draws = [draw]

    def manifest() -> None:
        _write_json(out / "split_manifest.json", {
            "policy": POLICY_VERSION, "partition": partition or {}, "n_explore": len(explore),
            "sample": draw.to_dict(), "revisions": [d.to_dict() for d in draws],
        })

    manifest()
    if stage0_cost:
        # Every invocation's Stage 0 is spent compute (the runner does not cache
        # it), even when every action is then read back from an earlier run.
        n_prior = sum(1 for e in ledger.cost_events() if e.get("kind") == "stage0")
        ledger.cost({"kind": "stage0", "shared": True, "invocation": n_prior + 1, **stage0_cost})
    say(f"[triage] sample: {len(draw.fail_ids)} FAIL + {len(draw.pass_ids)} PASS "
        f"(stratum={stratum}, shortfall={draw.shortfall or 'none'})")

    # A step whose rows depend on the batch would have to re-run on every larger
    # sample. When the routes do not read it, nothing the expansion rule looks at
    # depends on it either, so it runs once, on the sample the expansions settle on.
    deferred = [s for s in battery if s.deferred or (not s.reuse and s.name not in ROUTE_STEPS)]
    immediate = [s for s in battery if s not in deferred]
    revision = 0
    for step in immediate:
        run.run_step(step, list(draw.case_ids), kind="battery", revision=revision)

    def settle() -> None:
        for step in deferred:
            run.run_step(step, list(draw.case_ids), kind="battery", revision=revision)

    decisions: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    followed: list[tuple[str, str, str]] = []   # (route, rule_id, step key), in execution order
    stop: "dict[str, Any] | None" = None
    expansion_note: "dict[str, Any] | None" = None
    round_no = 0

    def decide(record: dict[str, Any]) -> None:
        record = {"event": "decision", "n": len(decisions), "round": round_no, "revision": revision,
                  "scheduler": scheduler, "rules_version": RULES_VERSION, **record}
        ledger.trace(record)
        decisions.append(record)

    spending = ("expand", "follow_up", "extend", "battery")

    def n_spent() -> int:  # every decision that spends
        return sum(1 for d in decisions if d["kind"] in spending)

    def n_actions() -> int:
        """What ``max_actions`` caps: expansions, follow-ups and scheduled battery steps. The scored
        scheduler's further chunks of a step it already took (``extend``) do not count: chunking a
        step must not use up the cap a whole step used before."""
        kinds = ("expand", "follow_up", "battery") if scheduler == "scored" else spending
        return sum(1 for d in decisions if d["kind"] in kinds)

    def expansion_plan() -> "tuple[Any, Any, list[str], float | None] | None":
        """The next nested sample, its new cases and the battery's estimated seconds; ``None`` at the cap."""
        size = next_size(len(draw.fail_ids), max_fail, n_available)
        if size is None:
            return None
        bigger, bigger_draw = matched_sample(explore, size, seed=seed, stratum=stratum)
        if not (set(draw.fail_ids) <= set(bigger_draw.fail_ids) and set(draw.pass_ids) <= set(bigger_draw.pass_ids)):
            raise RuntimeError("nested sampler returned a draw that does not contain the previous one")
        new_ids = [cid for cid in bigger_draw.case_ids if cid not in set(draw.case_ids)]
        estimates = [run.estimate_sec(s, len(new_ids) if s.reuse else len(bigger_draw.case_ids))
                     for s in immediate if not (s.task_kinds and task_kind not in s.task_kinds)]
        estimate = round(sum(e for e in estimates if e is not None), 1) if any(
            e is not None for e in estimates) else None
        return bigger, bigger_draw, new_ids, estimate

    def not_possible(gaps: list[dict[str, Any]]) -> dict[str, Any]:
        return {"state": "not_possible", "gaps": gaps,
                "reason": (f"the sample has the cap of {max_fail} FAIL cases" if len(draw.fail_ids) >= max_fail
                           else f"EXPLORE has no FAIL case beyond the {n_available} drawn")}

    def do_expand(plan: Any, gaps: list[dict[str, Any]], extra: dict[str, Any]) -> None:
        nonlocal sample, draw, revision
        bigger, bigger_draw, new_ids, estimate = plan
        decide({"kind": "expand", "rule_id": "E1", "from_fail": len(draw.fail_ids),
                "to_fail": len(bigger_draw.fail_ids), "n_new_cases": len(new_ids), "gaps": gaps,
                "estimate_sec": estimate, **extra})
        say(f"[triage] expand {len(draw.fail_ids)} -> {len(bigger_draw.fail_ids)} FAIL "
            f"({len(new_ids)} new cases; gaps: {[g['gap'] for g in gaps]})")
        sample, draw = bigger, bigger_draw
        draws.append(draw)
        revision += 1
        manifest()
        ledger.trace({"event": "revision", "revision": revision, "n_fail": len(draw.fail_ids),
                      "n_pass": len(draw.pass_ids), "n_new_cases": len(new_ids), "scheduler": scheduler})
        for step in immediate:
            run.run_step(step, list(draw.case_ids), kind="expansion", revision=revision, rule_id="E1",
                         budget_checked=True)

    by_rule = {x.rule_id: x for x in layers}
    step_layers: dict[tuple[str, str], Layer] = {}
    for x in layers:
        for name in (*x.served_by, *(c.name for c in x.candidates)):
            step_layers.setdefault((x.route, name), x)

    class Readings:
        """The current evidence read per case, once per decision. Called as ``(route, step key,
        reading) -> outcome``: the reading's FAIL-minus-control difference on the route's current
        FAIL cases (the layer's filter applied) and their PASS controls."""

        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records
            self.groups: dict[Any, tuple[list[str], list[str]]] = {}
            self.values: dict[tuple[str, str], dict[str, Any]] = {}
            self.spreads: dict[tuple[str, str], "float | None"] = {}

        def value(self, key: str, cid: str) -> dict[str, Any]:
            if (key, cid) not in self.values:
                self.values[(key, cid)] = follow_up_values(run.evidence[key], run.by_id[cid], answer_fn, agree_fn)
            return self.values[(key, cid)]

        def cases(self, route: str, key: str) -> tuple[list[str], list[str]]:
            """The route's FAIL cases for *key* (its layer's filter applied) and their PASS controls."""
            layer = step_layers.get((route, key))
            flt = layer.case_filter if layer is not None else None
            if (route, flt) not in self.groups:
                fail_ids = route_fail_ids(self.records, route, flt)
                self.groups[(route, flt)] = (fail_ids, matched_controls(self.records, fail_ids, rank_of)[0])
            return self.groups[(route, flt)]

        def __call__(self, route: str, key: str, metric: Metric) -> dict[str, Any]:
            ev = run.evidence.get(key)
            if ev is None or not ev.chunks:
                return {"n": [0, 0], "observed": "not_measured"}
            fail_ids, controls = self.cases(route, key)
            return outcome({cid: self.value(key, cid) for cid in (*fail_ids, *controls)}, fail_ids, controls, metric)

        def spread(self, key: str, metric: Metric) -> "float | None":
            """The reading's spread within FAIL and within PASS cases, over every case the step
            measured in this run (``None`` before it measured one)."""
            if (key, metric.field) not in self.spreads:
                ev = run.evidence.get(key)
                groups: dict[str, list[float]] = {"fail": [], "pass": []}
                for cid in sorted(ev.covered()) if ev is not None else ():
                    v = self.value(key, cid)
                    if v["state"] == "measured" and metric.field in v["values"]:
                        label = getattr(getattr(run.by_id[cid], "label", None), "value", None)
                        groups["fail" if label == "fail" else "pass"].append(float(v["values"][metric.field]))
                self.spreads[(key, metric.field)] = reading_spread(list(groups.values()), metric.kind)
            return self.spreads[(key, metric.field)]

        def decidable(self, route: str, key: str) -> Callable[[str, str, str], dict[str, Any]]:
            """``(reading, direction a, direction b) -> decidability`` of a conflict on *route*'s cases."""
            fail_ids, controls = self.cases(route, key)
            n = min(len(fail_ids), len(controls))
            metrics = {m.field: m for m in CARDS.get(key, ())}
            return lambda field, a, b: decidability(a, b, n, metrics[field], self.spread(key, metrics[field]))

    outcomes_for = Readings
    steps_by_name = {s.name: s for s in (*battery, *candidate_steps(layers))}

    def alternatives(st: dict[str, Any]) -> list[tuple[PlanStep, float, str]]:
        """``(step, R, the step it is an alternative to)`` for a layer's other candidates that can
        still run on the route: those of an answered layer (to the tool that answered it), and of
        an open one (to its current candidate). R is 0.5 when the alternative gives the kind of
        evidence that tool gives, 0 when it gives another kind (see :mod:`.scoring`)."""
        if st["state"] == "done":
            base = st.get("via") or ""
        elif st["state"] in ("open", "extend"):
            base = st["step"].name
        else:
            return []
        first = steps_by_name.get(base)
        target = [*st["_fail_ids"], *st["_control_ids"]]
        out = []
        for step in by_rule[st["rule_id"]].candidates:
            if step.name == base or step.name in run.blocked:
                continue
            ev = run.evidence.get(step.name)
            if ev is not None and (ev.refused() is not None or not ev.todo(target)):
                continue
            if run.gate(step, target) is not None:
                continue
            same = first is not None and step.evidence_kind == first.evidence_kind
            out.append((step, 0.5 if same else 0.0, base))
        return out

    Option = tuple[int, dict[str, Any], PlanStep, float, str]   # (layer index, state, step, R, alternative to)

    def runnable(states: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[Option]], dict[str, list[Option]]]:
        """Per route, what the scored scheduler may run (each open or extending layer's current
        candidate, with R = 0, and the alternatives of its answered layers), and what the judge
        is offered: those, and the alternatives of the open layers, which become candidates once
        their layer is answered."""
        pool: dict[str, list[Option]] = {}
        offer: dict[str, list[Option]] = {}
        for route, sts in states.items():
            rows, extra = [], []
            for li, st in enumerate(sts):
                alts = [(li, st, step, R, base) for step, R, base in alternatives(st)]
                if st["state"] in ("open", "extend"):
                    rows.append((li, st, st["step"], 0.0, ""))
                    extra += alts
                else:
                    rows += alts
            pool[route], offer[route] = rows, rows + extra
        return pool, offer

    explored_on: set[tuple[str, str]] = set()   # (route, step key) an exploration decision started

    def exploration_targets(records: list[dict[str, Any]], states: dict[str, list[dict[str, Any]]], growing: bool,
                            axis: str, readings: Readings
                            ) -> tuple[list[Scored], list[Scored], list[dict[str, Any]], list[dict[str, Any]]]:
        """What only exploration takes (gain 0), the completions of what it found something with, and
        what the budget filtered out of each.

        A target is priced at the conservative estimate a budget check uses (exploration's share
        is a budget). Completing a target exploration started on a route is exploration too, and
        counts against its share, until the cases measured so far show a reading outside its band
        (the verdicts' 95% interval): the step has found something, and its completion is then an
        incomplete step that T values like any other.

        Targets: the first open layer's candidate of every route with FAIL cases too few to be
        active (a low-frequency problem the rules do not pursue), and on every active route the
        representative of each other route's layer it does not own (outside its method family),
        when it has readings to compare with the PASS controls (a card: the H re-ask and
        continuation only mean something where their check fired) and the layer does not narrow
        its cases."""
        rows = []
        for row in active_routes(records, min_route_fail):
            if row["active"]:
                continue
            sts = route_states(row["route"], records, run.evidence, run.gate, rank_of, layers=layers,
                               blocked=run.blocked)
            for li, st in enumerate(sts):
                if st["state"] in ("open", "extend") and not st.get("battery"):
                    rows.append((row["route"], li, by_rule[st["rule_id"]], st["step"], st["_fail_ids"],
                                 st["_control_ids"], f"a route too small to be active ({row['n_fail']} FAIL)"))
                    break
        for route, sts in states.items():
            own = {n for st in sts for n in (*by_rule[st["rule_id"]].served_by,
                                             *(c.name for c in by_rule[st["rule_id"]].candidates))}
            fail_ids = route_fail_ids(records, route)
            controls = matched_controls(records, fail_ids, rank_of)[0]
            for layer in layers:
                if layer.route != route and layer.case_filter is None and layer.candidates \
                        and layer.candidates[0].name not in own and layer.candidates[0].name in CARDS:
                    rows.append((route, 99, layer, layer.candidates[0], fail_ids, controls,
                                 f"outside the route's method family (the representative of {layer.rule_id})"))
        targets: list[Scored] = []
        completions: list[Scored] = []
        held: list[dict[str, Any]] = []
        held_targets: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for route, li, layer, step, fail_ids, controls, why in rows:
            if (route, step.name) in seen or step.name in run.blocked or not step.reuse:
                continue
            seen.add((route, step.name))
            target = [*fail_ids, *controls]
            ev = run.evidence.get(step.name)
            if ev is not None and ev.refused() is not None:
                continue
            todo = ev.todo(target) if ev is not None else list(target)
            if not todo or run.gate(step, target) is not None:
                continue
            part = run.chunks(step, todo)[0]
            fail_left = set(todo) & set(fail_ids)
            share = len(set(part) & fail_left) / len(fail_left) if fail_left else len(part) / len(todo)
            started = (route, step.name) in explored_on
            found = [f"{m.field} {o['observed']}" for m in CARDS.get(step.name, ())
                     if (o := readings(route, step.name, m)).get("observed") in ("up", "down")] if started else []
            if started:
                why = (f"completing an exploration that found {', '.join(found)}" if found else
                       f"completing an exploration ({why}); no reading outside its band yet")
            cost, basis = step_cost(step, len(part), run.evidence, axis, unmeasured="typical" if found else "max")
            cand = Scored(kind="extend" if started else "follow_up", key=step.name, route=route,
                          rule_id=layer.rule_id, order=(ROUTE_ORDER.index(route), li), n_new_cases=len(part),
                          T=share if found else 0.0, q=availability(step.analyzer, run.evidence), cost=cost,
                          basis=basis, gaps=[f"incomplete:{route}:{step.name}"] if found else [], note=why,
                          payload=Proposal(layer=layer, step=step, kind="explore", fail_ids=fail_ids,
                                           control_ids=controls, todo=part, tried=[], before=[],
                                           target_override=part))
            over = run.over_budget(run.estimate_sec(step, len(part)), run.estimate_usd(step, len(part)))
            if over is not None:
                (held if found else held_targets).append({"key": step.name, "route": route,
                                                          "rule_id": layer.rule_id, "reason": over})
                continue
            (completions if found else targets).append(value(cand, weights, axis))
        return targets, completions, held, held_targets

    def route_view(records: list[dict[str, Any]], gaps: list[dict[str, Any]]
                   ) -> tuple[dict[str, list[dict[str, Any]]], bool]:
        """Every active route's layer states (pending battery steps first), and whether an
        expansion can still follow."""
        sample_ids = list(draw.case_ids)
        growing = bool(gaps) and expansion_note is None and \
            next_size(len(draw.fail_ids), max_fail, n_available) is not None
        pending = {s.name: (s, sample_ids) for s in deferred if s.name not in run.blocked and (
            s.name not in run.evidence or (run.evidence[s.name].refused() is None
                                           and run.evidence[s.name].todo(sample_ids)))}
        rows = [r for r in active_routes(records, min_route_fail) if r["active"]]
        # Decisions read only this scheduler's own evidence and price every candidate as if
        # it ran fresh: chunks in the ledger (an earlier invocation, the other scheduler)
        # save compute when the chosen step runs, never change what is chosen.
        return {r["route"]: route_states(r["route"], records, run.evidence, run.gate, rank_of, layers=layers,
                                         blocked=run.blocked, pending=pending) for r in rows}, growing

    explanation_sets: list[ExplanationSet] = []

    def explanation_request(records: list[dict[str, Any]], options: dict[str, list[Any]],
                            outcome_of: Callable[[str, str, Metric], dict[str, Any]]
                            ) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
        """What the judge sees, and per route the steps (with their readings) it may predict, in a
        seeded order: each open layer's current candidate, and every layer's other candidates as
        alternatives (to the tool that answered it, or to its current candidate). An alternative
        runs once its layer is answered if its D beats its redundancy, so its predictions are
        asked for up front. Then every other carded step measured on the route (``already_run``):
        a prediction on it is checked on new cases by the validator, and gives no D."""
        offered: dict[str, dict[str, list[str]]] = {}
        routes: dict[str, Any] = {}
        for route, rows in options.items():
            to_run: list[dict[str, Any]] = []
            for _, st, step, _, alternative_to in rows:
                layer = by_rule[st["rule_id"]]
                # a reading the route is cut on cannot tell explanations of that route apart
                if (step.name in CARDS and not restates_route(route, step.name)
                        and all(c["step"] != step.name for c in to_run)):
                    to_run.append({"step": step.name, "tool": step.analyzer, "layer": layer.rule_id,
                                   "question": layer.question, "what": step.purpose,
                                   **({"alternative_to": alternative_to} if alternative_to else {}),
                                   **({"caveat": step.caveat} if step.caveat else {}),
                                   "readings": [m.card() for m in CARDS[step.name]]})
            measured: dict[str, Any] = {}
            for key in sorted(CARDS):
                rows = {m.field: {k: v for k, v in o.items() if k != "observed" or v != "not_measured"}
                        for m in CARDS[key] if (o := outcome_of(route, key, m))["n"] != [0, 0]}
                if rows:
                    measured[key] = rows
            ran = [{"step": key, "readings": [m.card() for m in CARDS[key]]} for key in measured
                   if not restates_route(route, key) and all(c["step"] != key for c in to_run)]
            if not to_run and not ran:
                continue
            random.Random(f"{seed}:{revision}:{route}").shuffle(to_run)
            offered[route] = {c["step"]: [r["field"] for r in c["readings"]] for c in [*to_run, *ran]}
            fail_ids = route_fail_ids(records, route)
            by_id = {r["case_id"]: r for r in records}
            routes[route] = {"meaning": ROUTE_MEANING.get(route, ""), "n_fail": len(fail_ids),
                             "fail_by_stratum": dict(sorted(Counter(by_id[c]["stratum"] for c in fail_ids).items())),
                             "measured": measured, "to_run": to_run, "already_run": ran}
        previous = []
        if explanation_sets and explanation_sets[-1].hypotheses:
            prev = explanation_sets[-1]
            for hid, row in check(prev.hypotheses, outcome_of).items():
                h = next(x for x in prev.hypotheses if x.id == hid)
                previous.append({"id": hid, "route": h.route, "claim": h.claim, "state": row["state"],
                                 "verdicts": [{"step": p["step"], "reading": p["reading"], "predicted": p["predicted"],
                                               "observed": p["outcome"].get("observed"), "verdict": p["verdict"]}
                                              for p in row["predictions"]]})
        context = {
            "task": {"kind": task_kind, "controls_matched_on": stratum},
            "model_under_test": {k: model_identity.get(k) for k in ("spec", "backend")},
            "sample": {"revision": revision, "n_fail": len(draw.fail_ids), "n_pass": len(draw.pass_ids)},
            "checks": {"meaning": "FAIL vs PASS cases on which each route's check fired (fired/tested), "
                                  "with the 95% interval of the difference",
                       **(revisions[-1]["signals"] if revisions else {})},
            "measured_meaning": "per reading: FAIL mean, control mean, n [FAIL, control], diff = FAIL - "
                                "control, its 95% interval, and the direction the whole interval lies in",
            "routes": routes, "previous": previous,
        }
        return context, offered

    def judge_estimate(axis: str) -> "float | None":
        costs = [float(es.cost[axis]) for es in explanation_sets if es.cost.get(axis) is not None]
        return max(costs) if costs else None

    def n_asked() -> int:
        return sum(1 for es in explanation_sets if es.source != "not_asked")

    def ask_reason(options: dict[str, list[Any]], growing: bool, verdicts: dict[str, Any],
                   current: "ExplanationSet | None") -> "str | None":
        """Why the judge is asked now (``None`` = it is not).

        Explanations are about the routes, and an expansion revises the routes: the
        judge is first asked once no expansion can follow (the sample has settled,
        or the next one is over budget), whatever is left to run: the tools already
        run on a route are explained too. It is asked again when the evidence has
        decided a prediction since the last ask and a route with tools left that the
        judge can be asked about (as ``explanation_request`` offers them) has fewer
        than two live explanations: only a tool still to run brings evidence."""
        if growing:
            plan = expansion_plan()
            if plan is not None and run.over_budget(plan[3]) is None:
                return None
        # a judge call is charged like an action: asked only while its estimate fits, the
        # largest call so far (the first is unknown: a run overshoots by at most one call)
        if run.over_budget(judge_estimate("wall_sec"), judge_estimate("usd")) is not None:
            return None
        if current is None:
            return "the sample has settled: no expansion can follow"
        with_cards = sorted(route for route, rows in options.items()
                            if any(row[2].name in CARDS and not restates_route(route, row[2].name) for row in rows))
        new = _decided(verdicts) - current.decided_at_ask
        thin = [route for route in with_cards if sum(1 for h in current.hypotheses if h.route == route
                                                     and verdicts[h.id]["state"] == "live") < 2]
        if not new or not thin:
            return None
        return (f"the evidence decided {len(new)} prediction(s) since the last ask, and route(s) {thin} "
                f"with tools left have fewer than two live explanations")

    def explain(records: list[dict[str, Any]], options: dict[str, list[Any]],
                outcome_of: Callable[[str, str, Metric], dict[str, Any]], reason: str) -> ExplanationSet:
        context, offered = explanation_request(records, options, outcome_of)
        if not offered:
            es = ExplanationSet(revision, "", offered=offered, source="not_asked",
                                error="no active route has a tool with readings to predict")
        else:
            assert explainer is not None
            es = explainer.propose(out, revision, context, offered)
            es.seen = {route: sorted(r["measured"]) for route, r in context["routes"].items()}
            for axis in ("wall_sec", "usd"):
                if es.cost.get(axis) is not None:
                    run.judge_cost[axis] = run.judge_cost.get(axis, 0.0) + float(es.cost[axis])
            if es.source in ("judge", "failed"):
                ledger.cost({"kind": "judge", "key": es.key, "revision": revision, "source": es.source,
                             **es.cost})
        es.reason = reason
        if explanation_sets:
            # what is still live stays: a re-ask (or a failed one) adds explanations, it does not replace them
            prev = explanation_sets[-1]
            states = check(prev.hypotheses, outcome_of)
            kept = [h for h in prev.hypotheses if states[h.id]["state"] == "live"]
            taken = {h.id for x in explanation_sets for h in x.hypotheses}
            es.hypotheses, extra = carry_over(kept, es.hypotheses, taken)
            es.dropped, es.carried = es.dropped + extra, [h.id for h in kept]
        ledger.trace({"event": "explanations", "revision": revision, "key": es.key, "source": es.source,
                      "reason": reason, "error": es.error, "by_route": dict(Counter(h.route for h in es.hypotheses)),
                      "dropped": es.dropped, "cost": es.cost})
        say(f"[triage] explanations (revision {revision}): {es.source}, "
            f"{dict(Counter(h.route for h in es.hypotheses)) or 'none'}" + (f" ({es.error})" if es.error else ""))
        explanation_sets.append(es)
        return es

    def scored_pool(records: list[dict[str, Any]], gaps: list[dict[str, Any]],
                    states: dict[str, list[dict[str, Any]]], options: dict[str, list[Any]], growing: bool,
                    live: list[Hypothesis], readings: Readings) -> tuple[list[Scored], list[dict[str, Any]], str]:
        """Every feasible candidate with its value, what the budget filtered out, and the cost axis."""
        nonlocal expansion_note
        # A batch-dependent step's reading is superseded by any later expansion, and no
        # route reads it: while an expansion can still follow, running it is wasted. Its
        # layer waits for it (a battery step) rather than falling back to another tool.
        units = coverage_units(states, records, run.evidence, layers, min_cover=min_cover)
        axis = axis_of(run.evidence)
        by_id = {r["case_id"]: r for r in records}
        pool: list[Scored] = []
        filtered: list[dict[str, Any]] = []
        for route, rows in options.items():
            for li, st, step, R, alternative_to in rows:
                layer = by_rule[st["rule_id"]]
                if growing and not step.reuse:
                    filtered.append({"key": step.name, "route": route, "rule_id": layer.rule_id,
                                     "reason": "batch-dependent: a pending expansion would supersede it"})
                    continue
                battery_step = bool(st.get("battery")) and not alternative_to
                target = st["target"] if battery_step else [*st["_fail_ids"], *st["_control_ids"]]
                ev = run.evidence.get(step.name)
                todo = ev.todo(target) if ev is not None else list(target)
                if not todo:      # the same observation exists: R = 1, not a candidate
                    continue
                # a reuse step runs its next chunk; T and D are about the step on the whole route,
                # so the chunk earns them for the share of the route's remaining FAIL cases it covers
                part = run.chunks(step, todo)[0]
                fail_left = set(todo) & set(st["_fail_ids"])
                fail_new = set(part) & fail_left
                share = len(fail_new) / len(fail_left) if fail_left else len(part) / len(todo)
                C, touched = coverage_gain(layer.rule_id, fail_new, units)
                # a step already measured on part of the route's FAIL cases (a smaller sample, earlier
                # chunks); a battery step on part of the sample. Shared PASS controls do not count:
                # describe-first measured for S has not started N's question
                started = step.reuse and ev is not None and bool(
                    ev.covered() & set(target if battery_step else st["_fail_ids"]))
                T, gap_ids = 0.0, []
                if started:
                    T, gap_ids = share, [f"incomplete:{layer.rule_id}:{step.name}"]
                elif layer.route == "H" and any(by_id[c]["H"]["disagreements"] for c in fail_new):
                    T, gap_ids = share, [f"contradiction:{layer.rule_id}"]
                kind = "extend" if started else ("battery" if battery_step else "follow_up")
                D, d_detail = (disagreement(route, step.name, live, readings.decidable(route, step.name))
                               if live else (0.0, {}))
                cost, basis = step_cost(step, len(part), run.evidence, axis)
                cand = Scored(kind=kind, key=step.name, route=route, rule_id=layer.rule_id,
                              order=(ROUTE_ORDER.index(route), li), n_new_cases=len(part), D=D * share, T=T, C=C,
                              R=R * share, d_detail=d_detail, q=availability(step.analyzer, run.evidence), cost=cost,
                              basis=basis, alternative_to=alternative_to, gaps=gap_ids, units=touched,
                              payload=Proposal(layer=layer, step=step,
                                               kind="alternative" if alternative_to else st["state"],
                                               fail_ids=st["_fail_ids"], control_ids=st["_control_ids"], todo=part,
                                               tried=st.get("tried", []), before=[], target_override=part))
                why = run.over_budget(run.estimate_sec(step, len(part)), run.estimate_usd(step, len(part)))
                if why is not None:
                    filtered.append({"key": step.name, "route": route, "rule_id": layer.rule_id, "reason": why})
                    continue
                pool.append(value(cand, weights, axis))
        if gaps and expansion_note is None:
            plan = expansion_plan()
            if plan is None:
                expansion_note = not_possible(gaps)
                decide({"kind": "expand_not_possible", "rule_id": "E1", **expansion_note})
            else:
                steps = [s for s in immediate if not (s.task_kinds and task_kind not in s.task_kinds)]
                qs = [availability(s.analyzer, run.evidence) for s in steps
                      if any(o.status != "unsupported" for o in run.evidence[s.name].chunks)]
                costs = [step_cost(s, len(plan[2]), run.evidence, axis)[0] for s in steps]
                cand = Scored(kind="expand", key="expand", rule_id="E1", order=(-1, 0), n_new_cases=len(plan[2]),
                              T=0.5, q=sum(qs) / len(qs) if qs else 0.5,
                              cost=sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None,
                              gaps=[f"{g['gap']}:{g.get('route') or g.get('stratum')}" for g in gaps], payload=plan)
                why = run.over_budget(plan[3])
                if why is not None:
                    filtered.append({"key": "expand", "rule_id": "E1", "reason": why})
                else:
                    pool.append(value(cand, weights, axis))
        return pool, filtered, axis

    explored: dict[str, float] = {}   # what exploration decisions spent, per cost axis
    if scheduler == "scored":
        current: "ExplanationSet | None" = None
        while stop is None:
            records = triage(sample, run.evidence)
            summary = summarize(records)
            gaps = expansion_gaps(summary, records, min_cohort=min_cohort)
            if not revisions or revisions[-1]["revision"] != revision:
                revisions.append(_revision_row(revision, draw, summary, records, gaps))
            states, growing = route_view(records, gaps)
            options, offer = runnable(states)
            outcome_of = outcomes_for(records)
            # explanations a prediction already missed on leave the live set: D is recomputed
            # from the evidence at every decision, without asking the judge again
            verdicts = check(current.hypotheses, outcome_of) if current is not None else {}
            if explainer is not None and n_asked() < explainer.max_asks:
                reason = ask_reason(offer, growing, verdicts, current)
                if reason is not None:
                    current = explain(records, offer, outcome_of, reason)
                    verdicts = check(current.hypotheses, outcome_of)
                    current.decided_at_ask = _decided(verdicts)
            live = [h for h in (current.hypotheses if current is not None else []) if verdicts[h.id]["state"] == "live"]
            pool, filtered, axis = scored_pool(records, gaps, states, options, growing, live, outcome_of)
            targets: "list[Scored] | None" = None
            held_targets: list[dict[str, Any]] = []
            if explore_from == "targets":
                targets, completions, held, held_targets = exploration_targets(records, states, growing, axis,
                                                                               outcome_of)
                pool, filtered = pool + completions, filtered + held
            # exploration's share is of the evidence: the judge's calls are not an opportunity to explore
            spent = run.charged(axis) - float(run.judge_cost.get(axis) or 0.0)
            pick, why = choose(pool, decision_no=n_spent(), seed=seed, explore_share=explore_share,
                               explored=explored.get(axis, 0.0), spent=spent, min_gain=min_gain, targets=targets)
            if pick is None:
                over = [f for f in filtered if not f["reason"].startswith("batch-dependent")]
                # a candidate the budget filtered out binds before the gain floor does
                stop = {"reason": "budget" if over else "below_threshold" if why != "none" else "no_candidate",
                        "detail": ((f"{len(over)} candidate(s) over budget; the rest " +
                                    (f"below the gain floor {min_gain}" if pool else "none"))
                                   if over else f"no candidate reaches the gain floor {min_gain}" if why != "none"
                                   else "no feasible candidate"),
                        "best_gain": max((c.gain for c in pool), default=None), "filtered": filtered,
                        "candidates": [c.row() for c in sorted(pool, key=lambda c: (-c.U, c.order))]}
                break
            if pick.kind != "extend" and n_actions() >= max_actions:
                stop = {"reason": "max_actions", "max_actions": max_actions,
                        "pending": f"{pick.route}:{pick.rule_id}:{pick.key}"}
                break
            ranked = sorted(pool, key=lambda c: (-c.U, c.order))
            skipped = [c.rule_id for c in pool if why == "best" and pick.route and c.route == pick.route
                       and c.order < pick.order]
            extra = {"choice": why, "scoring_version": SCORING_VERSION, "axis": axis,
                     "U": round(pick.U, 6), "gain": round(pick.gain, 6),
                     "rationale": _scored_rationale(pick, why, explored.get(axis, 0.0), spent, explore_share),
                     **({"alternative_to": pick.alternative_to} if pick.alternative_to else {}),
                     **({"explore_target": pick.note} if why == "explore" and pick.note else {}),
                     "candidates": [c.row() for c in ranked], "filtered": filtered,
                     **({"explore_targets": [c.row() for c in targets], "explore_filtered": held_targets}
                        if targets else {}),
                     **({"explanations": {"key": current.key, "revision": current.revision,
                                          "states": {hid: v["state"] for hid, v in verdicts.items()}}}
                        if current is not None else {}),
                     **({"soft_order_exception": f"{pick.rule_id} taken before open layer(s) {skipped} of route "
                                                 f"{pick.route}: higher value"} if skipped else {})}
            before = {a: run.charged(a) for a in ("wall_sec", "usd")}

            def account() -> None:   # what an exploration decision spent counts against its share
                if why == "explore":
                    for a in before:
                        explored[a] = explored.get(a, 0.0) + run.charged(a) - before[a]

            if pick.kind == "expand":
                do_expand(pick.payload, gaps, extra)
                account()
                round_no += 1
                continue
            proposal = pick.payload
            decide({"kind": pick.kind, "rule_id": pick.rule_id, "route": pick.route, "key": pick.key,
                    "analyzer": proposal.step.analyzer, "question": proposal.layer.question,
                    "n_route_fail": len(proposal.fail_ids), "n_new_cases": pick.n_new_cases, **extra})
            record = run.run_step(proposal.step, proposal.target_ids,
                                  kind="battery" if pick.kind == "battery" else "follow_up", revision=revision,
                                  rule_id=pick.rule_id, route=pick.route)
            account()
            if record is None:
                run.blocked.setdefault(proposal.step.name, "nothing left to run")
            elif record["status"] == SKIPPED:
                stop = {"reason": "budget", "detail": record.get("reason")}
                break
            if why == "explore" and targets is not None and pick in targets:
                explored_on.add((pick.route, pick.key))
            if pick.kind != "battery" and (pick.route, pick.rule_id, pick.key) not in followed:
                followed.append((pick.route, pick.rule_id, pick.key))
            round_no += 1

    while stop is None:
        records = triage(sample, run.evidence)
        summary = summarize(records)
        gaps = expansion_gaps(summary, records, min_cohort=min_cohort)
        if not revisions or revisions[-1]["revision"] != revision:
            revisions.append(_revision_row(revision, draw, summary, records, gaps))
        if gaps and expansion_note is None:
            plan = expansion_plan()
            if plan is not None:
                if n_actions() >= max_actions:
                    stop = {"reason": "max_actions", "max_actions": max_actions, "pending": "expansion"}
                    break
                why = run.over_budget(plan[3])
                if why is not None:
                    expansion_note = {"state": "skipped", "reason": why, "gaps": gaps}
                    decide({"kind": "expand_skipped", "rule_id": "E1", "reason": why, "gaps": gaps,
                            "estimate_sec": plan[3]})
                    continue
                do_expand(plan, gaps, {"rationale": "an evidence gap a larger nested sample can close; expansion "
                                                    "precedes the routes' candidates because it revises the routes"})
                continue
            expansion_note = not_possible(gaps)
            decide({"kind": "expand_not_possible", "rule_id": "E1", **expansion_note})

        settle()  # the sample is final: the deferred battery steps run once, before any follow-up
        records = triage(sample, run.evidence)
        ran = 0
        for row in active_routes(records, min_route_fail):
            if not row["active"]:
                continue
            route = row["route"]
            proposal = next_for_route(route, records, run.evidence, run.gate, rank_of, layers=layers,
                                      blocked=run.blocked)
            if proposal is None:
                continue
            if n_actions() >= max_actions:
                stop = {"reason": "max_actions", "max_actions": max_actions,
                        "pending": f"{route}:{proposal.layer.rule_id}:{proposal.step.name}"}
                break
            # chunks of the same step measured before (an earlier invocation, a smaller sample)
            # are read back first, so the decision states what it actually runs
            adopted = run.adopt_for(proposal.step, proposal.target_ids, kind="follow_up", revision=revision,
                                    rule_id=proposal.layer.rule_id, route=route)
            todo = run.evidence[proposal.step.name].todo(proposal.target_ids)
            decide({"kind": "follow_up", "rule_id": proposal.layer.rule_id, "route": route,
                    "proposal": proposal.kind, "key": proposal.step.name, "analyzer": proposal.step.analyzer,
                    "question": proposal.layer.question, "n_route_fail": len(proposal.fail_ids),
                    "n_controls": len(proposal.control_ids), "n_new_cases": len(todo),
                    "n_read_back": len(proposal.todo) - len(todo),
                    "passed_over": proposal.tried, "earlier_layers": proposal.before,
                    "rationale": (f"route {route} active with {row['n_fail']} FAIL cases; "
                                  + ("extend the layer's step to the route's new cases" if proposal.kind == "extend"
                                     else "first open layer in the soft order"))})
            record = run.run_step(proposal.step, proposal.target_ids, kind="follow_up", revision=revision,
                                  rule_id=proposal.layer.rule_id, route=route)
            if record is None and not adopted:  # defensive: an open layer always has cases to run
                run.blocked.setdefault(proposal.step.name, "nothing left to run")
            elif record is not None and record["status"] == SKIPPED:
                stop = {"reason": "budget", "detail": record.get("reason")}
                break
            if (route, proposal.layer.rule_id, proposal.step.name) not in followed:
                followed.append((route, proposal.layer.rule_id, proposal.step.name))
            ran += 1
        round_no += 1
        if stop is None and not ran:
            stop = {"reason": "no_candidate", "detail": "every active route's layers are done or exhausted"}

    settle()  # a stop before the sample settled (max_actions with an expansion pending)
    records = triage(sample, run.evidence)
    summary = summarize(records)
    ledger.trace({"event": "stop", "round": round_no, "revision": revision, "scheduler": scheduler, **stop})
    say(f"[triage] stop: {stop['reason']} after {n_actions()} actions")
    final_gaps = expansion_gaps(summary, records, min_cohort=min_cohort)
    n_expansions = sum(1 for d in decisions if d["kind"] == "expand")
    final_states = {row["route"]: route_states(row["route"], records, run.evidence, run.gate, rank_of, layers=layers,
                                               blocked=run.blocked)
                    for row in active_routes(records, min_route_fail) if row["active"]}
    coverage = public_units(coverage_units(final_states, records, run.evidence, layers, min_cover=min_cover))
    final_outcome = outcomes_for(records)
    outcomes = {route: {key: {m.field: final_outcome(route, key, m) for m in CARDS[key]}
                        for key in sorted(CARDS) if run.evidence.get(key) is not None and run.evidence[key].chunks}
                for route in final_states}
    executed_wall = sum(float((e.get("cost") or {}).get("wall_sec") or 0.0) for e in run.executions
                        if not e.get("reused"))
    asked = [es for es in explanation_sets if es.source in ("judge", "failed")]
    judge_wall = sum(float(es.cost.get("wall_sec") or 0.0) for es in asked)
    report = {
        "policy": POLICY_VERSION,
        "scheduler": scheduler,
        **({"scoring_version": SCORING_VERSION, "weights": weights.__dict__, "explore_share": explore_share,
            "explore_from": explore_from,
            "explored": {a: round(v, 6) for a, v in explored.items()},
            "min_gain": min_gain, "min_cover": min_cover} if scheduler == "scored" else {}),
        "rules_version": RULES_VERSION,
        "model": model_identity,
        "sample": draw.to_dict(),
        "settings": {"host_assumed_finish": host_assumed_finish, "answer_contract": answer_contract,
                     "n_needs_counterfactual": n_needs_counterfactual, "budget_sec": budget_sec,
                     "task_kind": task_kind, "answer_fn": canonical(answer_fn), "agree_fn": canonical(agree_fn),
                     "n_fail": n_fail,
                     "max_fail": max_fail, "max_actions": max_actions, "min_route_fail": min_route_fail,
                     "min_cohort": min_cohort, "per_call_generation": per_call_generation, "chunk": chunk},
        "executions": run.executions,
        "battery": _battery_state(battery, run, draw),
        "revisions": revisions,
        "expansion": {"n_expansions": n_expansions, "final_gaps": final_gaps,
                      **(expansion_note or {"state": ("closed" if n_expansions else "no_gap") if not final_gaps
                                            else "pending"})},
        "decisions": decisions,
        "stop": stop,
        "cost": {**_cost_summary(run.executions, stage0_cost), "evidence": _evidence_cost(run),
                 "judge": _judge_cost(explanation_sets),
                 "charged": {"wall_sec": round(run.charged("wall_sec"), 3),
                             **({"usd": round(run.charged("usd"), 6)} if budget_usd > 0 else {}),
                             "_note": "evidence at its first execution's cost plus judge calls: what the "
                                      "budgets are checked against"},
                 "controller_sec": round(time.monotonic() - started - executed_wall - judge_wall, 3)},
        "coverage": coverage,
        "outcomes": outcomes,
        **({"explanations": _explanations_report(explanation_sets, explainer, final_outcome)}
           if explainer is not None else {}),
        "summary": summary,
        "follow_ups": _follow_ups(followed, records, run, sample, rank_of, layers, answer_fn, agree_fn),
        "routes": _route_report(records, run, rank_of, layers, min_route_fail),
        "checks": _acceptance(battery, layers, run, draw, ledger),
        "cases": records,
    }
    name = report_name(scheduler, explainer is not None)
    _write_json(out / f"{name}.json", report)
    (out / f"{name}.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _scored_rationale(pick: Scored, why: str, explored: float, spent: float, share: float) -> str:
    """One line saying why the scored scheduler took *pick*."""
    terms = f"U {pick.U:.4g} = q {pick.q:.2f} x [D {pick.D:.3f} + T {pick.T:.3f} + C {pick.C:.3f} - R {pick.R:.3f}]"
    terms += f" / cost {pick.cost:.4g}" if pick.cost is not None else " / cost unknown"
    if why == "explore":
        target = f" of {pick.note}" if pick.note else ""
        return (f"exploration{target} ({explored:.4g} of the {spent:.4g} spent on evidence so far went to "
                f"exploration; share {share}): " + terms)
    alt = f"; an alternative to {pick.alternative_to}" if pick.alternative_to else ""
    return f"highest value: {terms}{alt}"


def report_name(scheduler: str, explained: bool = False) -> str:
    """``triage_report`` (rules), ``triage_report.scored``, or ``triage_report.scored_d`` (with D):
    the arms of a comparison can share one probing directory and its ledger."""
    if scheduler == "rules":
        return "triage_report"
    return f"triage_report.{scheduler}" + ("_d" if explained else "")


#: The MVP-0 entry point, now the same controller (no expansion unless ``max_fail`` is given).
run_triage_mvp0 = run_triage


def _revision_row(revision: int, draw: Any, summary: dict[str, Any], records: list[dict[str, Any]],
                  gaps: list[dict[str, Any]]) -> dict[str, Any]:
    routes = Counter(tag for r in records if r["label"] == "fail" for tag in r["routes"])
    return {
        "revision": revision, "n_fail": len(draw.fail_ids), "n_pass": len(draw.pass_ids),
        "signals": {tag: {"fail": f"{summary[tag]['fail']['n_fired']}/{summary[tag]['fail']['n_tested']}",
                          "pass": f"{summary[tag]['pass']['n_fired']}/{summary[tag]['pass']['n_tested']}",
                          "routing_signal": summary[tag]["routing_signal"], "ci95": summary[tag]["signal_ci95"]}
                    for tag in ("H_label_free", "N", "U")},
        "fail_routes": dict(sorted(routes.items())),
        "gaps": gaps,
    }


def _evidence_cost(run: _Run) -> dict[str, Any]:
    """What the evidence this report reads cost when each observation was first executed
    (a read-back counts at its original cost, once): comparable across schedulers that
    share a ledger."""
    seen: dict[str, dict[str, Any]] = {}
    for ev in run.evidence.values():
        for obs in ev.chunks:
            seen.setdefault(obs.action_id, obs.cost or {})
    costs = list(seen.values())
    calls: dict[str, int] = {}
    for c in costs:
        for k, v in (c.get("calls") or {}).items():
            calls[k] = calls.get(k, 0) + int(v)
    priced = [c for c in costs if c.get("api_usage") is not None]
    out: dict[str, Any] = {"n_observations": len(costs), "calls": calls,
                           "wall_sec": round(sum(float(c.get("wall_sec") or 0.0) for c in costs), 3)}
    if priced:
        usd = [c.get("usd") for c in priced]
        out["usd"] = round(sum(usd), 6) if all(v is not None for v in usd) else None
    return out


def _decided(verdicts: dict[str, Any]) -> set[tuple[str, str, str]]:
    """The primary predictions the evidence has decided (hit or miss), as (explanation, step, reading)."""
    return {(hid, p["step"], p["reading"]) for hid, row in verdicts.items() for p in row["predictions"]
            if p["verdict"] in ("hit", "miss") and p.get("primary", True)}


def _judge_cost(sets: list[ExplanationSet]) -> dict[str, Any]:
    """Judge calls this invocation made, and the replies it read back (spent by the run that asked)."""
    asked = [es for es in sets if es.source in ("judge", "failed")]
    usd = [es.cost.get("usd") for es in asked if "api_usage" in es.cost]
    return {"n_calls": len(asked), "n_failed": sum(1 for es in asked if es.source == "failed"),
            "n_read_back": sum(1 for es in sets if es.source == "read_back"),
            "wall_sec": round(sum(float(es.cost.get("wall_sec") or 0.0) for es in asked), 3),
            **({"usd": round(sum(usd), 6) if all(v is not None for v in usd) else None} if usd else {})}


def _explanations_report(sets: list[ExplanationSet], explainer: Any,
                         outcome_of: Callable[[str, str, Metric], dict[str, Any]]) -> dict[str, Any]:
    """Every explanation set with each prediction checked on the final evidence, and the tally."""
    rows, tally = [], Counter()
    for es in sets:
        final = check(es.hypotheses, outcome_of)
        for hid, h in final.items():
            route = hid.split(".", 1)[0]
            tally[h["state"]] += 1
            for p in h["predictions"]:
                p["seen_when_predicted"] = p["step"] in es.seen.get(route, [])
                tally[f"prediction_{p['verdict']}"] += 1
                if not p["seen_when_predicted"]:
                    tally[f"unseen_prediction_{p['verdict']}"] += 1
        rows.append({**es.to_dict(), "final": final})
    return {"version": EXPLAIN_VERSION, "judge": explainer.identity, "max_per_route": explainer.max_per_route,
            "max_asks": explainer.max_asks,
            "sets": rows, "tally": dict(sorted(tally.items())),
            "_note": ("predictions are checked by code on the final evidence: hit = the 95% interval of FAIL - "
                      "control lies in the predicted region, miss = outside it; an explanation with a miss is "
                      "contradicted. A prediction on a step whose readings the judge already saw in part "
                      "(seen_when_predicted) is not out of sample; unseen_* counts the others")}


def _battery_state(battery: "tuple[PlanStep, ...]", run: _Run, draw: Any) -> list[dict[str, Any]]:
    """Per battery step: which observations the current sample is read from."""
    out = []
    for step in battery:
        ev = run.evidence.get(step.name)
        if ev is None or not ev.chunks:
            status = next((e["status"] for e in run.executions if e["key"] == step.name), "not run")
            out.append({"key": step.name, "analyzer": step.analyzer, "reuse": step.reuse, "status": status})
            continue
        used = Counter(o.action_id for cid in draw.case_ids if (o := ev.for_case(cid)) is not None)
        out.append({"key": step.name, "analyzer": step.analyzer, "reuse": step.reuse,
                    **({"reuse_note": step.reuse_note} if step.reuse_note else {}),
                    "statuses": ev.statuses_on(draw.case_ids), "read_from": dict(used),
                    "n_chunks": len(ev.chunks),
                    "covers_sample": all(ev.for_case(cid) is not None for cid in draw.case_ids)})
    return out


def _follow_ups(followed: list[tuple[str, str, str]], records: list[dict[str, Any]], run: _Run, sample: Any,
                rank_of: Callable[[str], Any], layers: "tuple[Layer, ...]",
                answer_fn: Any, agree_fn: Any) -> list[dict[str, Any]]:
    """Descriptive readings of every follow-up that ran, on the route's current cases."""
    by_rule = {x.rule_id: x for x in layers}
    out = []
    for route, rule_id, key in followed:
        layer, ev = by_rule[rule_id], run.evidence[key]
        fail_ids = route_fail_ids(records, route, layer.case_filter)
        controls, shortfall = matched_controls(records, fail_ids, rank_of)
        chunks = [o for o in ev.chunks if set(o.planned_ids) & set(fail_ids)]
        out.append({
            "route": route, "rule_id": rule_id, "key": key, "analyzer": ev.step.analyzer,
            "config": canonical(ev.step.config), "question": layer.question,
            "statuses": ev.statuses_on([*fail_ids, *controls]),
            "fidelity": sorted({str((o.fidelity or {}).get("status")) for o in chunks}),
            "n_route_fail": len(fail_ids), "n_controls": len(controls), "control_shortfall": shortfall,
            "not_run_on": len(ev.todo([*fail_ids, *controls])) if ev.step.reuse else None,
            **({"caveat": ev.step.caveat} if ev.step.caveat else {}),
            "reading": follow_up_reading(ev, sample, fail_ids, controls, answer_fn, agree_fn),
        })
    return out


def _route_report(records: list[dict[str, Any]], run: _Run, rank_of: Callable[[str], Any],
                  layers: "tuple[Layer, ...]", min_route_fail: int) -> list[dict[str, Any]]:
    """Every route with FAIL cases: active or not, and each layer's state when the run stopped."""
    out = []
    for row in active_routes(records, min_route_fail):
        states = route_states(row["route"], records, run.evidence, run.gate, rank_of, layers=layers,
                              blocked=run.blocked)
        out.append({**row, "layers": [public(s) for s in states]})
    return out


def _cost_summary(executions: list[dict[str, Any]], stage0_cost: "dict[str, Any] | None") -> dict[str, Any]:
    """This invocation's spend: Stage 0, the actions it executed, and (apart) what it read back.

    Token and USD totals only cover actions whose provider reported usage; a
    missing price leaves ``usd`` unknown rather than counting it as zero.
    """
    def total(rows: list[dict[str, Any]]) -> dict[str, Any]:
        costs = [r.get("cost") or {} for r in rows]
        usage = [c["api_usage"] for c in costs if c.get("api_usage") is not None]
        usd = [c.get("usd") for c in costs if c.get("api_usage") is not None]
        calls: dict[str, int] = {}
        for c in costs:
            for k, v in (c.get("calls") or {}).items():
                calls[k] = calls.get(k, 0) + int(v)
        out: dict[str, Any] = {"n_actions": len(rows), "calls": calls,
                               "wall_sec": round(sum(float(c.get("wall_sec") or 0.0) for c in costs), 3)}
        if usage:
            out["api_usage"] = {k: sum(int(u.get(k, 0)) for u in usage) for k in usage[0]}
            out["usd"] = round(sum(usd), 6) if all(v is not None for v in usd) else None
        return out

    ran = [e for e in executions if e.get("cost") is not None and not e.get("reused")]
    reused = [e for e in executions if e.get("reused")]
    by_kind = {kind: total([e for e in ran if e["kind"] == kind]) for kind in ("battery", "expansion", "follow_up")}
    return {"stage0": stage0_cost or {}, "actions_run": total(ran), "actions_run_by_kind": by_kind,
            "actions_read_back": {**total(reused), "_note": "spent by the run that executed them"}}


def _acceptance(battery: "tuple[PlanStep, ...]", layers: "tuple[Layer, ...]", run: _Run, draw: Any,
                ledger: Ledger) -> dict[str, Any]:
    """The acceptance properties that can be checked from the run's own records."""
    allowed = {s.analyzer for s in battery} | {s.analyzer for s in candidate_steps(layers)}
    traced = {e.get("analyzer") for e in ledger.trace_events() if e.get("event") in ("planned", "settled")}
    disjoint, one_batch, configs = {}, {}, {}
    for key, ev in run.evidence.items():
        if not ev.chunks:
            continue
        counts = Counter(cid for o in ev.chunks for cid in o.planned_ids)
        if ev.step.reuse:
            disjoint[key] = all(n == 1 for n in counts.values())
        else:
            read = {o.action_id for cid in draw.case_ids if (o := ev.for_case(cid)) is not None}
            if key in {s.name for s in battery}:
                one_batch[key] = len(read) == 1 and set(ev.chunks[-1].planned_ids) == set(draw.case_ids)
        planned = run.config(ev.step)
        configs[key] = all(o.params.get(k) == canonical(planned[k]) for o in ev.chunks
                           for k in ev.step.config if k in planned)
    costs = ledger.cost_events()
    observations = [o for ev in run.evidence.values() for o in ev.chunks]
    return {
        "only_rule_actions": traced <= allowed,
        "battery_covers_sample": {b["key"]: b.get("covers_sample") for b in _battery_state(battery, run, draw)
                                  if "covers_sample" in b},
        "reused_chunks_disjoint": disjoint,
        "batch_dependent_read_from_current_sample": one_batch,
        "configs_match_plan": configs,
        "stage0_cost_separate": any(e.get("kind") == "stage0" for e in costs),
        "no_zero_for_failure": all(o.findings is None for o in observations if o.status != OK),
        "fidelity_states": {key: sorted({str((o.fidelity or {}).get("status")) for o in ev.chunks})
                            for key, ev in run.evidence.items() if ev.chunks},
    }


def _why_not(report: dict[str, Any], key: str) -> str:
    rows = [e for e in report["executions"] if e["key"] == key]
    if not rows:
        return "not in the plan"
    step = rows[-1]
    reason = step.get("error") or step.get("reason") or ""
    return f"{step['status']}" + (f" ({reason})" if reason else "")


def _markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# Triage report ({report['policy']}, {report['rules_version']})", "",
        f"Model: `{report['model'].get('spec')}` on `{report['model'].get('backend')}`. "
        f"Sample: {len(report['sample']['fail_ids'])} FAIL + {len(report['sample']['pass_ids'])} PASS, "
        f"stratum `{report['sample']['stratum_rule']}`, shortfall {report['sample']['shortfall'] or 'none'}. "
        f"Stopped: {report['stop']['reason']}"
        + (f" ({report['stop'].get('detail') or report['stop'].get('pending')})"
           if report["stop"].get("detail") or report["stop"].get("pending") else "") + ".",
        "", "## Sample revisions", "",
        "| revision | FAIL | PASS | H (label-free) | N | U | FAIL routes | gaps |", "|---:|---:|---:|---|---|---|---|---|",
    ]
    for rev in report["revisions"]:
        sig = rev["signals"]
        cells = [f"{sig[t]['fail']} vs {sig[t]['pass']}, CI {sig[t]['ci95']}" for t in ("H_label_free", "N", "U")]
        gaps = "; ".join(f"{g['gap']} {g.get('route') or g.get('stratum')}" for g in rev["gaps"]) or "none"
        lines.append(f"| {rev['revision']} | {rev['n_fail']} | {rev['n_pass']} | {' | '.join(cells)} "
                     f"| {rev['fail_routes']} | {gaps} |")
    exp = report.get("expansion") or {}
    if exp.get("state") in ("skipped", "not_possible"):
        lines += ["", f"Expansion {exp['state']}: {exp['reason']}."]
    lines += ["", "## Decisions", "", "| # | round | kind | rule | route | step | new cases | rationale / passed over |",
              "|---:|---:|---|---|---|---|---:|---|"]
    for d in report["decisions"]:
        passed = "; ".join(f"{t['key']}: {t['status']}" + (f" ({t['reason']})" if t.get("reason") else "")
                           for t in d.get("passed_over") or [])
        text = d.get("rationale") or d.get("reason") or ""
        if passed:
            text = f"{text}. Passed over: {passed}"
        lines.append(f"| {d['n']} | {d['round']} | {d['kind']} | {d['rule_id']} | {d.get('route', '-')} "
                     f"| {d.get('key', '-')} | {d.get('n_new_cases', '-')} | {text} |")
    lines += ["", "## Actions", "",
              "| # | kind | rev | step | cases | status | fidelity | wall s | calls |", "|---:|---|---:|---|---:|---|---|---:|---|"]
    for e in report["executions"]:
        cost = e.get("cost") or {}
        name = e["key"] if e["key"] == e["analyzer"] else f"{e['analyzer']} ({e['key']})"
        lines.append(f"| {e['n']} | {e['kind']} | {e['revision']} | {name} | {e.get('n_cases', '-')} "
                     f"| {e['status']}{' (reused)' if e.get('reused') else ''} "
                     f"| {(e.get('fidelity') or {}).get('status', '-')} | {cost.get('wall_sec', '-')} "
                     f"| {cost.get('calls', '-')} |")
    cost = report.get("cost") or {}
    run, stage0 = cost.get("actions_run") or {}, cost.get("stage0") or {}
    by_kind = ", ".join(f"{k} {v['wall_sec']} s" for k, v in (cost.get("actions_run_by_kind") or {}).items()
                        if v["n_actions"])
    lines += ["", f"Cost of this invocation: Stage 0 {stage0.get('wall_sec', '-')} s, {stage0.get('calls', '-')}"
              + (f", {stage0['api_usage']}, USD {stage0.get('usd')}" if stage0.get("api_usage") else "")
              + f"; actions run {run.get('wall_sec', 0)} s ({by_kind or 'none'}), {run.get('calls', {})}"
              + (f", {run['api_usage']}, USD {run.get('usd')}" if run.get("api_usage") else "") + "."]
    lines += ["", "## Routes on the final sample", "",
              "| route | FAIL fired/tested | PASS fired/tested | routing signal | 95% CI of the difference |",
              "|---|---|---|---:|---|"]
    for tag in ("H_label_free", "N", "U"):
        f, p = s[tag]["fail"], s[tag]["pass"]
        lines.append(f"| {tag} | {f['n_fired']}/{f['n_tested']} | {p['n_fired']}/{p['n_tested']} "
                     f"| {s[tag]['routing_signal']} | {s[tag]['signal_ci95']} |")
    for part, rates in s["U_parts"].items():
        f, p = rates["fail"], rates["pass"]
        lines.append(f"| U: {part} | {f['n_fired']}/{f['n_tested']} | {p['n_fired']}/{p['n_tested']} | - | - |")
    lines += [f"| S (stable FAIL) / stable PASS | {s['S']['fail']} | {s['S']['stable_pass']} | - | - |",
              f"| X (a required check missing) | {s['X']['fail']} | {s['X']['pass']} | - | - |", "",
              "The routing signal is max(0, P(tag|FAIL) - P(tag|PASS)) on this matched sample: it steers "
              "later actions and is not an explained share of failures. S = a FAIL case on which H, N "
              "(with media) and both U parts ran and none fired; a missing check leaves it in X."]
    donors = s.get("N_donors") or {}
    if donors:
        lines += ["", "Swap donors (counterfactual = the same question's gold on the donor media differs "
                  "from the model's answer, so a correct answer has to change):", "",
                  "| label | counterfactual | content only | no swap | followed donor |", "|---|---:|---:|---:|---:|"]
        for label in ("fail", "pass"):
            d = donors[label]
            lines.append(f"| {label.upper()} | {d['counterfactual']} | {d['content']} | {d['no_swap']} "
                         f"| {d['counterfactual_followed_donor']}/{d['counterfactual']} |")
    split = s.get("N_by_base_answer")
    if split:
        lines += ["", "N by the unablated answer (unchanged after swap / tested):", "",
                  "| base answer | FAIL | PASS |", "|---|---|---|"]
        for answer, row in split.items():
            if answer.startswith("_"):
                continue
            lines.append(f"| {answer or '(empty)'} | {row['fail']['n_fired']}/{row['fail']['n_tested']} "
                         f"| {row['pass']['n_fired']}/{row['pass']['n_tested']} |")
        lines += ["", f"_{split['_note']}._"]
    ko = s["attention_knockout"]
    lines += ["", "## Knockout (descriptive)", ""]
    if not any(ko[label]["n_measured"] for label in ("fail", "pass")):
        lines.append(f"Not measured: {_why_not(report, 'attention_knockout')}.")
    else:
        lines += [f"Fidelity: {', '.join(ko['fidelity'])}.", "",
                  "| label | measured | mean ko_delta_full | mean ko_delta_late |", "|---|---:|---:|---:|"]
        for label in ("fail", "pass"):
            row = ko[label]
            lines.append(f"| {label.upper()} | {row['n_measured']} | {row['mean_ko_delta_full']} "
                         f"| {row['mean_ko_delta_late']} |")
        lines += ["", f"_{ko['_note']}._"]
    threshold = s.get("S_threshold")
    if threshold:
        lines += ["", "## Threshold instruction (S, behavioural)", "",
                  "| label | stratum | measured | answer changed | fixed | broken |", "|---|---|---:|---:|---:|---:|"]
        for g in threshold["groups"]:
            lines.append(f"| {str(g['label']).upper()} | {g['stratum']} | {g['n_measured']} | {g['n_changed']} "
                         f"| {g['n_fixed']} | {g['n_broken']} |")
        lines += ["", f"_{threshold['_note']}._"]
    readings = s.get("S_readings") or {}
    cohorts = [(name, stratum, row) for name, by in readings.items() if not name.startswith("_")
               for stratum, row in by.items()]
    measured = any(row["embed_lens"]["n"] or row["patchscope"]["n"] for _, _, row in cohorts)
    if cohorts and not measured:
        sizes = ", ".join(f"{name} {stratum} {row['n']}" for name, stratum, row in cohorts)
        lines += ["", "## S readings (descriptive)", "",
                  f"Not measured: embed_lens {_why_not(report, 'embed_lens')}; "
                  f"patchscope {_why_not(report, 'patchscope')}. Cohort sizes: {sizes}."]
    elif cohorts:
        lines += ["", "## S readings (descriptive)", "",
                  "| cohort | stratum | n | mean el_query_sim | answer first before last | mean peak / final prob "
                  "| mean late drop | question word on media |", "|---|---|---:|---:|---:|---|---:|---:|"]
        for name, stratum, row in cohorts:
            ps = row["patchscope"]
            lines.append(f"| {name} | {stratum} | {row['n']} | {row['embed_lens']['mean_query_sim']} "
                         f"| {ps['n_target_first_before_last']}/{ps['n']} | {ps['mean_peak_prob']} / "
                         f"{ps['mean_final_prob']} | {ps['mean_late_drop']} "
                         f"| {ps['n_media_query_hit']}/{ps['n_media_query_read']} |")
        lines += ["", f"_{readings['_note']}._"]
    if report["follow_ups"]:
        lines += ["", "## Follow-ups (descriptive)", ""]
        for fu in report["follow_ups"]:
            reading = fu["reading"]
            fail, ctrl = reading["route_fail"], reading["controls"]
            lines += [f"### {fu['route']} / {fu['rule_id']}: `{fu['analyzer']}` ({fu['key']})", "",
                      f"{fu['question']}. Status {fu['statuses']}, fidelity {fu['fidelity']}; "
                      f"measured {fail['n_measured']}/{fail['n']} route FAIL, {ctrl['n_measured']}/{ctrl['n']} "
                      f"PASS controls" + (f"; control shortfall {fu['control_shortfall']}" if fu["control_shortfall"] else "")
                      + (f"; not run on {fu['not_run_on']} current cases" if fu.get("not_run_on") else "") + "."]
            if fu.get("caveat"):
                lines += ["", f"Caveat: {fu['caveat']}."]
            fields = sorted(set(fail["means"]) | set(ctrl["means"]))
            if fields:
                lines += ["", "| reading | route FAIL mean (n) | PASS control mean (n) |", "|---|---:|---:|"]
                lines += [f"| {f} | {fail['means'].get(f)} ({fail['n_by_field'].get(f, 0)}) "
                          f"| {ctrl['means'].get(f)} ({ctrl['n_by_field'].get(f, 0)}) |" for f in fields]
            low = [f for f in fail["means"] if f.startswith("complied_") and (fail["means"][f] or 0.0) < 0.5]
            if low:
                lines += ["", f"Warning: on most route cases the model did not add the text {low[0][9:]} asks for "
                          f"(mean {low[0]} {fail['means'][low[0]]}): an unchanged answer there does not answer "
                          "the question."]
            untested = {**fail["untested"], **{f"(controls) {k}": v for k, v in ctrl["untested"].items()}}
            if untested:
                lines += ["", f"Untested: {untested}."]
            lines.append("")
        lines.append("_Means over the route's FAIL cases and their stratum-matched PASS controls; descriptive, "
                     "no test, no mechanism named._")
    expl = report.get("explanations")
    if expl:
        judge, judge_cost = expl["judge"], (report.get("cost") or {}).get("judge") or {}
        lines += ["", "## Competing explanations (D)", "",
                  f"Judge `{judge.get('provider')}` / `{judge.get('model')}`: {judge_cost.get('n_calls', 0)} call(s) "
                  f"({judge_cost.get('n_failed', 0)} failed), {judge_cost.get('wall_sec', 0)} s"
                  + (f", USD {judge_cost['usd']}" if judge_cost.get("usd") is not None else "")
                  + f"; {judge_cost.get('n_read_back', 0)} read back. Tally on the final evidence: {expl['tally']}."]
        for i, es in enumerate(expl["sets"]):
            lines += ["", f"### Ask {i + 1} (revision {es['revision']}): {es['source']}"
                      + (f" ({es['error']})" if es.get("error") else ""), "", f"Asked because {es['reason']}.", ""]
            if not es["hypotheses"]:
                lines.append("No explanation." + (f" Dropped: {es['dropped']}." if es["dropped"] else ""))
                continue
            if es.get("carried"):
                lines += [f"Kept from the last ask (still live): {', '.join(es['carried'])}.", ""]
            lines += ["| explanation | claim | state | predictions: predicted -> observed (verdict) |",
                      "|---|---|---|---|"]
            for h in es["hypotheses"]:
                final = es["final"][h["id"]]
                preds = "; ".join(("**primary** " if p.get("primary") else "")
                                  + f"{p['step']}.{p['reading']}: {p['predicted']} -> "
                                  f"{p['outcome'].get('observed')} ({p['verdict']}"
                                  + (", seen" if p.get("seen_when_predicted") else "") + ")"
                                  for p in final["predictions"]) or "none explicit"
                lines.append(f"| {h['id']} | {h['claim']} | {final['state']} | {preds} |")
            if es["dropped"]:
                lines += ["", f"Dropped from the reply: {es['dropped']}."]
        lines += ["", f"_{expl['_note']}._"]
    if report["routes"]:
        lines += ["", "## Route layers when the run stopped", "",
                  "| route | FAIL | active | layer | state | via / next | passed over |", "|---|---:|---|---|---|---|---|"]
        for row in report["routes"]:
            for st in row["layers"]:
                passed = "; ".join(f"{t['key']}: {t['status']}" for t in st.get("tried") or [])
                lines.append(f"| {row['route']} | {row['n_fail']} | {row['active']} | {st['rule_id']} | {st['state']} "
                             f"| {st.get('via') or st.get('key') or '-'} | {passed or '-'} |")
    lines += ["", "## Checks", ""]
    for name, held in report["checks"].items():
        lines.append(f"- `{name}`: {held}")
    lines.append("")
    return "\n".join(lines)
