"""MVP-0: the fixed triage plan, executed strictly, with its trace and costs.

The plan is a fixed list of actions on one matched sample, in this order:

1. ``answer_extraction_audit`` (re-reads the Stage-0 outputs; no model call)
2. ``termination_audit`` without continuation (re-reads the outputs too)
3. ``modality_ablation(mode="swap")`` — the N triage check
4. ``attention_knockout(target="gold")`` — the first N-layer measurement, on
   white-box backends; elsewhere it is gated off as ``unsupported``
5. ``self_consistency`` — re-samples each case (U)
6. ``prompt_contrast`` with the question kept verbatim in neutral wrappers (U)
7. ``prompt_contrast`` with an instruction not to default to "no" (yes/no tasks
   only): which stable errors move under a changed decision criterion (S)
8. ``embed_lens(layer=0.9)`` — is the case's own question word close to the
   media tokens late in the stack (S2)
9. ``patchscope(target="gold", media_layers=3)`` — was the gold answer readable
   before the output, and is the question word read off the media tokens (S3, S2)

No judge is called and nothing is selected adaptively: MVP-0 establishes the
execution, metering, fidelity and trace contracts the adaptive controller
(MVP-3) will be built on. Outputs land in ``out_dir``::

    split_manifest.json  triage_report.json  triage_report.md
    selection_trace.jsonl  cost_ledger.jsonl  observations/<action_id>.json
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from evalrx.analyzers.perturbation.prompt_contrast import _DEFAULT_STRATEGIES as _PROMPT_STRATEGIES
from evalrx.eval_agent.probing.actions import OK, SKIPPED, ProbeAction, canonical
from evalrx.eval_agent.probing.executor import execute
from evalrx.eval_agent.probing.ledger import Ledger
from evalrx.eval_agent.probing.proposals import propose
from evalrx.eval_agent.probing.sampler import matched_sample
from evalrx.eval_agent.probing.triage import summarize, triage_cases

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

POLICY_VERSION = "mvp0-2"


@dataclass(frozen=True)
class PlanStep:
    """One row of the fixed plan: what to build, why, and what it may cost.

    ``key`` names the step when one analyzer serves two steps (default: the
    analyzer). ``per_case`` runs a one-case analyzer once per case;
    ``keep_artifacts`` keeps those result artifacts with the findings;
    ``task_kinds`` limits the step to those task kinds (empty = every kind).
    """

    analyzer: str
    config: dict[str, Any]
    route: str
    purpose: str
    fidelity_check: "str | None" = None          # "edit" = tap/linear self-check, "patch" = patch self-check
    calls_per_case: dict[str, int] = field(default_factory=dict)
    key: str = ""
    per_case: bool = False
    keep_artifacts: tuple[str, ...] = ()
    task_kinds: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.key or self.analyzer


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


MVP0_PLAN: tuple[PlanStep, ...] = (
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
             calls_per_case={"forward": 1}),
    PlanStep("patchscope", {"target": "gold", "media_layers": 3}, "S3",
             "decode the answer position at every layer and the media tokens at three depths: "
             "whether the gold answer was readable before the output, and whether the question "
             "word is read off the media",
             fidelity_check="patch", calls_per_case={"forward": 1}),
)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonical(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _build(step: PlanStep, extra: dict[str, Any]) -> Any:
    from evalrx.core.registry import registry

    return registry.analyzers.get(step.analyzer)(**{**step.config, **extra})


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


def run_triage_mvp0(
    model: Any,
    explore: "CaseBatch",
    *,
    out_dir: "str | Path",
    model_identity: dict[str, Any],
    n_fail: int = 8,
    seed: int = 0,
    stratum: str = "gold",
    host_assumed_finish: bool = False,
    answer_contract: bool = False,
    n_needs_counterfactual: bool = False,
    task_kind: "str | None" = None,
    answer_fn: "Callable[[Any, str], str | None] | None" = None,
    analyzer_kwargs: "dict[str, dict[str, Any]] | None" = None,
    price_fn: "Callable[[dict[str, Any] | None], float | None] | None" = None,
    plan: "tuple[PlanStep, ...]" = MVP0_PLAN,
    budget_sec: float = 0.0,
    stage0_cost: "dict[str, Any] | None" = None,
    partition: "dict[str, Any] | None" = None,
    run_logger: Any = None,
    progress: "Callable[[str], None] | None" = print,
) -> dict[str, Any]:
    """Run the fixed triage plan on a matched sample of *explore*; return the report.

    Args:
        model:               the model under test.
        explore:             Stage-0-labelled EXPLORE cases (CONFIRM never enters here).
        out_dir:             where the trace, ledger, observations and report go.
        model_identity:      spec / weights / backend / dtype / generation settings;
                             part of every action identity, so a different model
                             never reuses an observation.
        n_fail:              FAIL cases to draw (PASS controls are matched to them).
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
                             answer, ``None`` = the task has no categorical answer):
                             how the U checks tell whether an answer moved. Default:
                             a coarse yes/no-or-prefix key.
        analyzer_kwargs:     per-analyzer constructor extras the host supplies
                             (e.g. the task's ``score_fn``, the sampling settings);
                             recorded in the config.
        price_fn:            ``api_usage -> USD`` (``None`` = no price on record) for
                             API models; the cost ledger and report carry it.
        budget_sec:          wall-clock cap on executed actions (0 = none). An action
                             already running is not interrupted; later ones are skipped.
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
    sample, draw = matched_sample(explore, n_fail, seed=seed, stratum=stratum)
    _write_json(out / "split_manifest.json", {
        "policy": POLICY_VERSION, "partition": partition or {}, "n_explore": len(explore),
        "sample": draw.to_dict(),
    })
    if stage0_cost:
        # Every invocation's Stage 0 is spent compute (the runner does not cache
        # it), even when every action is then read back from an earlier run.
        n_prior = sum(1 for e in ledger.cost_events() if e.get("kind") == "stage0")
        ledger.cost({"kind": "stage0", "shared": True, "invocation": n_prior + 1, **stage0_cost})
    say(f"[triage] sample: {len(draw.fail_ids)} FAIL + {len(draw.pass_ids)} PASS "
        f"(stratum={stratum}, shortfall={draw.shortfall or 'none'})")

    context = {"policy": POLICY_VERSION, "model": model_identity}
    observations: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    for step_no, step in enumerate(plan):
        extra = extras.get(step.analyzer, {})
        action = ProbeAction(
            analyzer=step.analyzer, config={**step.config, **extra}, case_ids=draw.case_ids,
            route=step.route, purpose=step.purpose,
            reserve={k: v * len(draw.case_ids) for k, v in step.calls_per_case.items()},
        )
        if step.task_kinds and task_kind not in step.task_kinds:
            base = {"step": step_no, "key": step.name, "analyzer": step.analyzer}
            reason = f"step is for task kinds {list(step.task_kinds)}; this task is {task_kind!r}"
            ledger.trace({**base, "event": "not_applicable", "reason": reason})
            steps.append({**base, "status": "not_applicable", "reused": False, "reason": reason})
            say(f"[triage] {step_no} {step.name}: not applicable ({task_kind})")
            continue
        try:
            analyzer = _build(step, extra)
        except Exception as exc:  # noqa: BLE001 - a bad config is an outcome, not a crash
            base = {"step": step_no, "key": step.name, "action_id": action.identity(context),
                    "analyzer": step.analyzer}
            ledger.trace({**base, "event": "skipped", "status": "failed",
                          "reason": f"could not construct: {type(exc).__name__}: {exc}"})
            steps.append({**base, "status": "failed", "error": str(exc)})
            continue
        version = tool_version(analyzer)
        action_id = action.identity({**context, "tool_version": version,
                                     "per_case": step.per_case, "keep_artifacts": list(step.keep_artifacts)})
        base = {"step": step_no, "key": step.name, "action_id": action_id, "analyzer": step.analyzer}

        done = ledger.settled(action_id)
        if done is not None:
            ledger.trace({**base, "event": "reused", "status": done.status, "attempt": done.attempt,
                          "note": "terminal observation with the same identity; read back, not re-run"})
            observations[step.name] = done
            steps.append({**base, "status": done.status, "reused": True, "fidelity": done.fidelity,
                          "cost": done.cost, "error": done.error})
            say(f"[triage] {step_no} {step.name}: reused ({done.status})")
            continue

        if budget_sec > 0 and ledger.spent_wall_sec() >= budget_sec:
            ledger.trace({**base, "event": "skipped", "status": SKIPPED,
                          "reason": f"wall-clock budget {budget_sec}s spent"})
            steps.append({**base, "status": SKIPPED, "reason": "budget"})
            say(f"[triage] {step_no} {step.name}: skipped (budget)")
            continue

        prior = ledger.attempts(action_id)
        attempt = prior + 1
        ledger.trace({
            **base, "event": "planned", "attempt": attempt, "action": action.to_dict(),
            "params": canonical(analyzer.get_params()), "tool_version": version,
            "fidelity_check": step.fidelity_check,
            "rationale": "MVP-0 fixed order", "candidates": [s.name for s in plan],
            **({"previous_attempts_uncertain": prior} if prior else {}),
        })
        say(f"[triage] {step_no} {step.name}: running on {len(sample)} cases")
        obs = execute(action, action_id, attempt, analyzer, model, sample,
                      fidelity_check=step.fidelity_check, per_case=step.per_case,
                      keep_artifacts=step.keep_artifacts, run_logger=run_logger)
        if obs.cost.get("api_usage") is not None:
            obs.cost["usd"] = price_fn(obs.cost["api_usage"]) if price_fn is not None else None
        ledger.save(obs)
        ledger.cost({"kind": "action", "action_id": action_id, "analyzer": step.analyzer,
                     "attempt": attempt, "reserved_calls": action.reserve,
                     "baseline_reused": obs.status == OK and not obs.cost.get("calls"),
                     "judge_calls": 0, **obs.cost})
        ledger.trace({**base, "event": "settled", "attempt": attempt, "status": obs.status,
                      "fidelity": obs.fidelity.get("status"), "n_measured": len(obs.measured_ids),
                      "n_missing": len(obs.missing), "error_type": obs.error_type, "error": obs.error})
        observations[step.name] = obs
        steps.append({**base, "status": obs.status, "reused": False, "fidelity": obs.fidelity,
                      "cost": obs.cost, "error": obs.error, "n_measured": len(obs.measured_ids),
                      "missing": obs.missing})
        say(f"[triage] {step_no} {step.name}: {obs.status} "
            f"(fidelity={obs.fidelity.get('status')}, {obs.cost.get('wall_sec')}s, calls={obs.cost.get('calls')})")

    records = triage_cases(sample, observations, host_assumed_finish=host_assumed_finish,
                           answer_contract=answer_contract, n_needs_counterfactual=n_needs_counterfactual,
                           answer_fn=answer_fn)
    report = {
        "policy": POLICY_VERSION,
        "model": model_identity,
        "sample": draw.to_dict(),
        "settings": {"host_assumed_finish": host_assumed_finish, "answer_contract": answer_contract,
                     "n_needs_counterfactual": n_needs_counterfactual, "budget_sec": budget_sec,
                     "task_kind": task_kind, "answer_fn": canonical(answer_fn)},
        "steps": steps,
        "cost": _cost_summary(steps, stage0_cost),
        "summary": summarize(records),
        "next_candidates": propose(
            records, model, sample, task_kind=task_kind, extras=extras,
            executed={(s.analyzer, str(canonical({**s.config, **extras.get(s.analyzer, {})})))
                      for s in plan if s.name in observations}),
        "checks": _acceptance(plan, steps, observations, draw, ledger),
        "cases": records,
    }
    _write_json(out / "triage_report.json", report)
    (out / "triage_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _cost_summary(steps: list[dict[str, Any]], stage0_cost: "dict[str, Any] | None") -> dict[str, Any]:
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

    ran = [s for s in steps if s.get("status") not in ("not_applicable", SKIPPED) and not s.get("reused")]
    reused = [s for s in steps if s.get("reused")]
    return {"stage0": stage0_cost or {}, "actions_run": total(ran),
            "actions_read_back": {**total(reused), "_note": "spent by the run that executed them"}}


def _acceptance(plan: "tuple[PlanStep, ...]", steps: list[dict[str, Any]], observations: dict[str, Any],
                draw: Any, ledger: Ledger) -> dict[str, Any]:
    """The MVP-0 acceptance properties that can be checked from the run's own records."""
    planned = {s.analyzer for s in plan}
    traced = {e.get("analyzer") for e in ledger.trace_events() if e.get("event") in ("planned", "settled")}
    case_sets = {name: obs.planned_ids == list(draw.case_ids) and set(obs.measured_ids) <= set(draw.case_ids)
                 for name, obs in observations.items()}
    configs = {}
    for s in plan:
        obs = observations.get(s.name)
        if obs is not None:
            configs[s.name] = all(obs.params.get(k) == canonical(v) for k, v in s.config.items())
    costs = ledger.cost_events()
    return {
        "only_planned_actions": traced <= planned and len(steps) == len(plan),
        "case_sets_match_plan": case_sets,
        "configs_match_plan": configs,
        "stage0_cost_separate": any(e.get("kind") == "stage0" for e in costs),
        "no_zero_for_failure": all(obs.findings is None for obs in observations.values() if obs.status != OK),
        "fidelity_states": {name: (obs.fidelity or {}).get("status") for name, obs in observations.items()},
    }


def _why_not(report: dict[str, Any], key: str) -> str:
    step = next((st for st in report["steps"] if (st.get("key") or st["analyzer"]) == key), None)
    if step is None:
        return "not in the plan"
    reason = step.get("error") or step.get("reason") or ""
    return f"{step['status']}" + (f" ({reason})" if reason else "")


def _markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# Triage report ({report['policy']})", "",
        f"Model: `{report['model'].get('spec')}` on `{report['model'].get('backend')}`. "
        f"Sample: {len(report['sample']['fail_ids'])} FAIL + {len(report['sample']['pass_ids'])} PASS, "
        f"stratum `{report['sample']['stratum_rule']}`, shortfall {report['sample']['shortfall'] or 'none'}.",
        "", "## Actions", "",
        "| step | analyzer | status | fidelity | wall s | calls |", "|---|---|---|---|---:|---|",
    ]
    for st in report["steps"]:
        cost = st.get("cost") or {}
        name = st.get("key") or st["analyzer"]
        if name != st["analyzer"]:
            name = f"{st['analyzer']} ({name})"
        lines.append(f"| {st['step']} | {name} | {st['status']}{' (reused)' if st.get('reused') else ''} "
                     f"| {(st.get('fidelity') or {}).get('status', '-')} | {cost.get('wall_sec', '-')} "
                     f"| {cost.get('calls', '-')} |")
    cost = report.get("cost") or {}
    run, stage0 = cost.get("actions_run") or {}, cost.get("stage0") or {}
    lines += ["", f"Cost of this invocation: Stage 0 {stage0.get('wall_sec', '-')} s, {stage0.get('calls', '-')}"
              + (f", {stage0['api_usage']}, USD {stage0.get('usd')}" if stage0.get("api_usage") else "")
              + f"; actions run {run.get('wall_sec', 0)} s, {run.get('calls', {})}"
              + (f", {run['api_usage']}, USD {run.get('usd')}" if run.get("api_usage") else "") + "."]
    lines += ["", "## Routes on this sample", "",
              "| route | FAIL fired/tested | PASS fired/tested | routing signal |", "|---|---|---|---:|"]
    for tag in ("H_label_free", "N", "U"):
        f, p = s[tag]["fail"], s[tag]["pass"]
        lines.append(f"| {tag} | {f['n_fired']}/{f['n_tested']} | {p['n_fired']}/{p['n_tested']} "
                     f"| {s[tag]['routing_signal']} |")
    for part, rates in s["U_parts"].items():
        f, p = rates["fail"], rates["pass"]
        lines.append(f"| U: {part} | {f['n_fired']}/{f['n_tested']} | {p['n_fired']}/{p['n_tested']} | - |")
    lines += [f"| S (stable FAIL) / stable PASS | {s['S']['fail']} | {s['S']['stable_pass']} | - |",
              f"| X (a required check missing) | {s['X']['fail']} | {s['X']['pass']} | - |", "",
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
    proposals = report.get("next_candidates") or {}
    if proposals.get("candidates"):
        lines += ["", "## Next candidates (proposed by rule, not run)", "",
                  "| route | FAIL cases | analyzer | config | status | question / reason |", "|---|---:|---|---|---|---|"]
        for c in proposals["candidates"]:
            text = c.get("question") or ""
            if c.get("reason"):
                text = f"{text} ({c['reason']})" if text else c["reason"]
            lines.append(f"| {c['route']} | {c['n_fail_cases']} | {c['analyzer']} | {c.get('config', '')} "
                         f"| {c['status']} | {text} |")
        if proposals.get("x_untested"):
            lines += ["", f"Why X cases stay untested: {proposals['x_untested']}."]
        lines += ["", f"_{proposals['_note']}._"]
    lines += ["", "## Checks", ""]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {value}")
    lines.append("")
    return "\n".join(lines)
