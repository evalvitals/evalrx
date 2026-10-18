"""MVP-2: the triage battery, nested expansion and route follow-ups chosen by fixed rules.

A run goes through three phases:

1. **Battery.** The battery runs on a matched sample of EXPLORE. Its steps,
   in order:

   1. ``answer_extraction_audit``: re-reads the Stage-0 outputs (no model call).
   2. ``termination_audit`` without continuation: re-reads the outputs too.
   3. ``modality_ablation(mode="swap")``: the N check.
   4. ``attention_knockout(target="gold")``: the first N-layer measurement.
   5. ``self_consistency``: re-samples each case (U).
   6. ``prompt_contrast`` with the question kept verbatim in neutral wrappers (U).
   7. ``prompt_contrast`` with an instruction not to default to "no" (yes/no
      tasks only), descriptive S.
   8. ``embed_lens(layer=0.9)``: is the case's own question word near the
      media tokens late in the stack (S2).
   9. ``patchscope(target="gold", media_layers=3)``: was the gold answer
      readable before the output, and is the question word read off the
      media tokens (S3, S2).

2. **Expansion.** While an evidence gap holds that more cases can close, the
   controller decides to expand. The sample then doubles along the sampler's
   rank prefixes, and the battery runs again only where it has to:

   * steps whose per-case rows do not depend on the batch run on the new cases
     alone, and their old rows stay current;
   * steps that infer something from the batch (``embed_lens``, ``patchscope``)
     are not read by the routes, so nothing the expansion rule looks at depends
     on them: they run once, on the whole sample the expansions settle on.

3. **Follow-ups.** Each active route proposes its next layer's candidate (see
   :mod:`~evalrx.eval_agent.probing.rules`), and the candidate runs on the
   route's FAIL cases plus matched PASS controls.

Every decision is a trace event that names its rule, the condition it read,
and the candidates it passed over and why. The run stops when:

* no route has a candidate left;
* ``max_actions`` decisions (expansions + follow-ups) have been made; or
* the wall-clock budget is spent, or an action's estimate exceeds what is left.

No judge is called, no hypothesis is named, and CONFIRM never enters. Outputs
land in ``out_dir``::

    split_manifest.json  triage_report.json  triage_report.md
    selection_trace.jsonl  cost_ledger.jsonl  observations/<action_id>.json
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import Counter
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from evalrx.analyzers.perturbation.prompt_contrast import _DEFAULT_STRATEGIES as _PROMPT_STRATEGIES
from evalrx.eval_agent.probing.actions import OK, SKIPPED, PlanStep, ProbeAction, canonical
from evalrx.eval_agent.probing.evidence import StepEvidence
from evalrx.eval_agent.probing.executor import execute, gate
from evalrx.eval_agent.probing.ledger import Ledger
from evalrx.eval_agent.probing.rules import (
    LAYERS,
    RULES_VERSION,
    Layer,
    active_routes,
    candidate_steps,
    expansion_gaps,
    next_for_route,
    next_size,
    public,
    route_fail_ids,
    route_states,
)
from evalrx.eval_agent.probing.sampler import matched_sample, rank
from evalrx.eval_agent.probing.triage import ROUTE_STEPS, follow_up_reading, summarize, triage_cases

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

POLICY_VERSION = "mvp2-0"

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

_BATCH_WORDS = ("the question words (and recurring media words) are inferred from the batch's prompts "
                "and images")

BATTERY: tuple[PlanStep, ...] = (
    PlanStep("answer_extraction_audit", {"reask": False}, "H",
             "re-read Stage-0 outputs for grading and extraction problems"),
    PlanStep("termination_audit", {"continue_non_clean": False}, "H",
             "classify how each Stage-0 output ended, without continuing it"),
    PlanStep("modality_ablation", {"mode": "swap", "donor_rule": "counterfactual"}, "N",
             "re-ask with media on which the same question's gold differs from the model's answer "
             "(else any different media); an answer that should change but does not is the N signal",
             calls_per_case={"generate": 2}),
    PlanStep("attention_knockout", {"target": "gold"}, "N1",
             "cut text-to-media attention per layer band; margin change of the gold first token",
             fidelity_check="edit", calls_per_case={"forward": 5}),
    PlanStep("self_consistency", {"n": 5}, "U",
             "re-sample each case at the host's frozen sampling settings; an answer that differs "
             "from the Stage-0 answer is the U signal",
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
             calls_per_case={"forward": 1}, reuse=False, reuse_note=_BATCH_WORDS),
    PlanStep("patchscope", {"target": "gold", "media_layers": 3}, "S3",
             "decode the answer position at every layer and the media tokens at three depths: "
             "whether the gold answer was readable before the output, and whether the question "
             "word is read off the media",
             fidelity_check="patch", calls_per_case={"forward": 1}, reuse=False, reuse_note=_BATCH_WORDS),
)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonical(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
                 price_fn: Any, run_logger: Any, say: Callable[[str], None]) -> None:
        self.model, self.ledger, self.context, self.extras = model, ledger, context, extras
        self.task_kind, self.budget_sec, self.price_fn, self.run_logger, self.say = (
            task_kind, budget_sec, price_fn, run_logger, say)
        self.by_id = {str(c.id): c for c in explore}
        self.order = {str(c.id): i for i, c in enumerate(explore)}
        self.evidence: dict[str, StepEvidence] = {}
        self.executions: list[dict[str, Any]] = []
        self.blocked: dict[str, str] = {}

    def batch(self, ids: "list[str]") -> "CaseBatch":
        from evalrx.core.case import CaseBatch

        return CaseBatch([self.by_id[i] for i in sorted(set(ids), key=self.order.__getitem__)])

    def config(self, step: PlanStep) -> dict[str, Any]:
        return {**step.config, **self.extras.get(step.analyzer, {})}

    def build(self, step: PlanStep) -> Any:
        from evalrx.core.registry import registry

        return registry.analyzers.get(step.analyzer)(**self.config(step))

    def gate(self, step: PlanStep, ids: "list[str]") -> "tuple[str, str] | None":
        """Why *step* cannot run on *ids* here — ``(status, reason)`` — before anything is spent."""
        if step.task_kinds and self.task_kind not in step.task_kinds:
            return "not_applicable", f"for task kinds {list(step.task_kinds)}; this task is {self.task_kind!r}"
        cap = output_cap(self.context.get("model") or {})
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
        ev = self.evidence.get(step.name)
        per_case = ev.sec_per_case() if ev is not None else None
        return round(per_case * n_cases, 1) if per_case is not None else None

    def over_budget(self, estimate: "float | None") -> "str | None":
        if self.budget_sec <= 0:
            return None
        spent = self.ledger.spent_wall_sec()
        if spent >= self.budget_sec:
            return f"wall-clock budget {self.budget_sec}s spent"
        if estimate is not None and spent + estimate > self.budget_sec:
            return f"estimated {estimate}s exceeds the {round(self.budget_sec - spent, 1)}s left"
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
        one) on cases of *target* that no chunk covers yet; the last such execution record."""
        record = None
        known = {o.action_id for o in ev.chunks}
        wanted = set(target)
        for obs in self.ledger.chunks_for(step_id):
            ids = set(obs.planned_ids)
            if obs.action_id in known or not ids & wanted or ids & ev.covered():
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
        """Bring *step*'s evidence up to date on *target*; the execution record, or ``None``
        when every case is already covered."""
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
        why = None if budget_checked else self.over_budget(estimate)
        if why is not None:
            self.ledger.trace({**base, "event": "skipped", "status": SKIPPED, "reason": why})
            self.blocked[step.name] = why
            record = {**base, "status": SKIPPED, "reused": False, "reason": why}
            self._log(record)
            self.say(f"[triage] {step.name}: skipped ({why})")
            return record

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
    analyzer_kwargs: "dict[str, dict[str, Any]] | None" = None,
    price_fn: "Callable[[dict[str, Any] | None], float | None] | None" = None,
    battery: "tuple[PlanStep, ...]" = BATTERY,
    layers: "tuple[Layer, ...]" = LAYERS,
    budget_sec: float = 0.0,
    stage0_cost: "dict[str, Any] | None" = None,
    partition: "dict[str, Any] | None" = None,
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
        max_actions:         decisions after the battery (expansions + follow-ups).
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
        analyzer_kwargs:     per-analyzer constructor extras the host supplies
                             (e.g. the task's ``score_fn``, the sampling settings).
        price_fn:            ``api_usage -> USD`` (``None`` = no price on record).
        budget_sec:          wall-clock cap on executed actions (0 = none). A running
                             action is not interrupted; one whose estimate (from the
                             same step's earlier seconds per case) exceeds what is
                             left is skipped.
        stage0_cost:         the shared baseline cost, logged once and kept separate.
        partition:           how EXPLORE/CONFIRM were split (recorded, not used).
        run_logger:          optional RunLoggerV2 for per-call model logs.
    """
    say = progress or (lambda _msg: None)
    out = Path(out_dir)
    ledger = Ledger(out)
    extras = {k: dict(v) for k, v in (analyzer_kwargs or {}).items()}
    # swap donors come from the whole partition the sample was drawn from
    extras.setdefault("modality_ablation", {}).setdefault("donor_pool", explore)
    max_fail = max(int(n_fail), int(max_fail or n_fail))
    n_available = sum(1 for c in explore if getattr(getattr(c, "label", None), "value", None) == "fail")
    run = _Run(model, explore, ledger, context={"policy": POLICY_VERSION, "model": model_identity},
               extras=extras, task_kind=task_kind, budget_sec=budget_sec, price_fn=price_fn,
               run_logger=run_logger, say=say)
    rank_of = partial(rank, seed)
    triage = partial(triage_cases, host_assumed_finish=host_assumed_finish, answer_contract=answer_contract,
                     n_needs_counterfactual=n_needs_counterfactual, answer_fn=answer_fn)

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
    deferred = [s for s in battery if not s.reuse and s.name not in ROUTE_STEPS]
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
                  "rules_version": RULES_VERSION, **record}
        ledger.trace(record)
        decisions.append(record)

    def n_actions() -> int:  # decisions that spend: expansions and follow-ups
        return sum(1 for d in decisions if d["kind"] in ("expand", "follow_up"))

    while stop is None:
        records = triage(sample, run.evidence)
        summary = summarize(records)
        gaps = expansion_gaps(summary, records, min_cohort=min_cohort)
        if not revisions or revisions[-1]["revision"] != revision:
            revisions.append(_revision_row(revision, draw, summary, records, gaps))
        size = next_size(len(draw.fail_ids), max_fail, n_available)
        if gaps and size is not None and expansion_note is None:
            if n_actions() >= max_actions:
                stop = {"reason": "max_actions", "max_actions": max_actions, "pending": "expansion"}
                break
            bigger, bigger_draw = matched_sample(explore, size, seed=seed, stratum=stratum)
            if not (set(draw.fail_ids) <= set(bigger_draw.fail_ids) and set(draw.pass_ids) <= set(bigger_draw.pass_ids)):
                raise RuntimeError("nested sampler returned a draw that does not contain the previous one")
            new_ids = [cid for cid in bigger_draw.case_ids if cid not in set(draw.case_ids)]
            estimates = [run.estimate_sec(s, len(new_ids) if s.reuse else len(bigger_draw.case_ids))
                         for s in immediate if not (s.task_kinds and task_kind not in s.task_kinds)]
            estimate = round(sum(e for e in estimates if e is not None), 1) if any(
                e is not None for e in estimates) else None
            why = run.over_budget(estimate)
            if why is not None:
                expansion_note = {"state": "skipped", "reason": why, "gaps": gaps}
                decide({"kind": "expand_skipped", "rule_id": "E1", "reason": why, "gaps": gaps,
                        "estimate_sec": estimate})
                continue
            decide({"kind": "expand", "rule_id": "E1", "from_fail": len(draw.fail_ids), "to_fail": len(bigger_draw.fail_ids),
                    "n_new_cases": len(new_ids), "gaps": gaps, "estimate_sec": estimate,
                    "rationale": "an evidence gap a larger nested sample can close; expansion precedes the "
                                 "routes' candidates because it revises the routes"})
            say(f"[triage] expand {len(draw.fail_ids)} -> {len(bigger_draw.fail_ids)} FAIL "
                f"({len(new_ids)} new cases; gaps: {[g['gap'] for g in gaps]})")
            sample, draw = bigger, bigger_draw
            draws.append(draw)
            revision += 1
            manifest()
            ledger.trace({"event": "revision", "revision": revision, "n_fail": len(draw.fail_ids),
                          "n_pass": len(draw.pass_ids), "n_new_cases": len(new_ids)})
            for step in immediate:
                run.run_step(step, list(draw.case_ids), kind="expansion", revision=revision, rule_id="E1",
                             budget_checked=True)
            continue
        if gaps and expansion_note is None:
            expansion_note = {"state": "not_possible", "gaps": gaps,
                              "reason": (f"the sample has the cap of {max_fail} FAIL cases" if len(draw.fail_ids) >= max_fail
                                         else f"EXPLORE has no FAIL case beyond the {n_available} drawn")}
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
    ledger.trace({"event": "stop", "round": round_no, "revision": revision, **stop})
    say(f"[triage] stop: {stop['reason']} after {n_actions()} actions")
    final_gaps = expansion_gaps(summary, records, min_cohort=min_cohort)
    n_expansions = sum(1 for d in decisions if d["kind"] == "expand")
    report = {
        "policy": POLICY_VERSION,
        "rules_version": RULES_VERSION,
        "model": model_identity,
        "sample": draw.to_dict(),
        "settings": {"host_assumed_finish": host_assumed_finish, "answer_contract": answer_contract,
                     "n_needs_counterfactual": n_needs_counterfactual, "budget_sec": budget_sec,
                     "task_kind": task_kind, "answer_fn": canonical(answer_fn), "n_fail": n_fail,
                     "max_fail": max_fail, "max_actions": max_actions, "min_route_fail": min_route_fail,
                     "min_cohort": min_cohort},
        "executions": run.executions,
        "battery": _battery_state(battery, run, draw),
        "revisions": revisions,
        "expansion": {"n_expansions": n_expansions, "final_gaps": final_gaps,
                      **(expansion_note or {"state": ("closed" if n_expansions else "no_gap") if not final_gaps
                                            else "pending"})},
        "decisions": decisions,
        "stop": stop,
        "cost": _cost_summary(run.executions, stage0_cost),
        "summary": summary,
        "follow_ups": _follow_ups(followed, records, run, sample, rank_of, layers, answer_fn),
        "routes": _route_report(records, run, rank_of, layers, min_route_fail),
        "checks": _acceptance(battery, layers, run, draw, ledger),
        "cases": records,
    }
    _write_json(out / "triage_report.json", report)
    (out / "triage_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


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
                answer_fn: Any) -> list[dict[str, Any]]:
    """Descriptive readings of every follow-up that ran, on the route's current cases."""
    from evalrx.eval_agent.probing.rules import matched_controls

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
            "reading": follow_up_reading(ev, sample, fail_ids, controls, answer_fn),
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
        configs[key] = all(o.params.get(k) == canonical(v) for o in ev.chunks for k, v in ev.step.config.items())
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
                  "| cohort | stratum | n | mean el_query_sim | gold first before last | mean peak / final prob "
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
            untested = {**fail["untested"], **{f"(controls) {k}": v for k, v in ctrl["untested"].items()}}
            if untested:
                lines += ["", f"Untested: {untested}."]
            lines.append("")
        lines.append("_Means over the route's FAIL cases and their stratum-matched PASS controls; descriptive, "
                     "no test, no mechanism named._")
    if report["routes"]:
        lines += ["", "## Route layers when the run stopped", "",
                  "| route | FAIL | active | layer | state | via / next | passed over |", "|---|---:|---|---|---|---|---|"]
        for row in report["routes"]:
            for st in row["layers"]:
                passed = "; ".join(f"{t['key']}: {t['status']}" for t in st.get("tried") or [])
                lines.append(f"| {row['route']} | {row['n_fail']} | {row['active']} | {st['rule_id']} | {st['state']} "
                             f"| {st.get('via') or st.get('key') or '-'} | {passed or '-'} |")
    lines += ["", "## Checks", ""]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {value}")
    lines.append("")
    return "\n".join(lines)
