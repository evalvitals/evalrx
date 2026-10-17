"""MVP-0: the fixed triage plan, executed strictly, with its trace and costs.

The plan is four actions on one matched sample, in this order:

1. ``answer_extraction_audit`` (re-reads the Stage-0 outputs; no model call)
2. ``termination_audit`` without continuation (re-reads the outputs too)
3. ``modality_ablation(mode="swap")`` — the N triage check
4. ``attention_knockout(target="gold")`` — the first N-layer measurement, on
   white-box backends; elsewhere it is gated off as ``unsupported``

No judge is called and nothing is selected adaptively: MVP-0 establishes the
execution, metering, fidelity and trace contracts the adaptive controller
(MVP-3) will be built on. Outputs land in ``out_dir``::

    split_manifest.json  triage_report.json  triage_report.md
    selection_trace.jsonl  cost_ledger.jsonl  observations/<action_id>.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from evalrx.eval_agent.probing.actions import OK, SKIPPED, ProbeAction, canonical
from evalrx.eval_agent.probing.executor import execute
from evalrx.eval_agent.probing.ledger import Ledger
from evalrx.eval_agent.probing.sampler import matched_sample
from evalrx.eval_agent.probing.triage import summarize, triage_cases

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

POLICY_VERSION = "mvp0-1"


@dataclass(frozen=True)
class PlanStep:
    """One row of the fixed plan: what to build, why, and what it may cost."""

    analyzer: str
    config: dict[str, Any]
    route: str
    purpose: str
    fidelity_check: "str | None" = None          # "edit" = tap/linear self-check applies
    calls_per_case: dict[str, int] = field(default_factory=dict)


MVP0_PLAN: tuple[PlanStep, ...] = (
    PlanStep("answer_extraction_audit", {"reask": False}, "H",
             "re-read Stage-0 outputs for grading and extraction problems"),
    PlanStep("termination_audit", {"continue_non_clean": False}, "H",
             "classify how each Stage-0 output ended, without continuing it"),
    PlanStep("modality_ablation", {"mode": "swap"}, "N",
             "re-ask with a different case's media; an unchanged answer is the N signal",
             calls_per_case={"generate": 2}),
    PlanStep("attention_knockout", {"target": "gold"}, "N1",
             "cut text-to-media attention per layer band; margin change of the gold first token",
             fidelity_check="edit", calls_per_case={"forward": 5}),
)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonical(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _build(step: PlanStep, extra: dict[str, Any]) -> Any:
    from evalrx.core.registry import registry

    return registry.analyzers.get(step.analyzer)(**{**step.config, **extra})


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
    analyzer_kwargs: "dict[str, dict[str, Any]] | None" = None,
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
        analyzer_kwargs:     per-analyzer constructor extras the host supplies
                             (e.g. the task's ``score_fn``); recorded in the config.
        budget_sec:          wall-clock cap on executed actions (0 = none). An action
                             already running is not interrupted; later ones are skipped.
        stage0_cost:         the shared baseline cost, logged once and kept separate.
        partition:           how EXPLORE/CONFIRM were split (recorded, not used).
        run_logger:          optional RunLoggerV2 for per-call model logs.
    """
    say = progress or (lambda _msg: None)
    out = Path(out_dir)
    ledger = Ledger(out)
    extras = analyzer_kwargs or {}
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
        action_id = action.identity(context)
        base = {"step": step_no, "action_id": action_id, "analyzer": step.analyzer}

        done = ledger.settled(action_id)
        if done is not None:
            ledger.trace({**base, "event": "reused", "status": done.status, "attempt": done.attempt,
                          "note": "terminal observation with the same identity; read back, not re-run"})
            observations[step.analyzer] = done
            steps.append({**base, "status": done.status, "reused": True, "fidelity": done.fidelity,
                          "cost": done.cost, "error": done.error})
            say(f"[triage] {step_no} {step.analyzer}: reused ({done.status})")
            continue

        if budget_sec > 0 and ledger.spent_wall_sec() >= budget_sec:
            ledger.trace({**base, "event": "skipped", "status": SKIPPED,
                          "reason": f"wall-clock budget {budget_sec}s spent"})
            steps.append({**base, "status": SKIPPED, "reason": "budget"})
            say(f"[triage] {step_no} {step.analyzer}: skipped (budget)")
            continue

        prior = ledger.attempts(action_id)
        attempt = prior + 1
        try:
            analyzer = _build(step, extra)
        except Exception as exc:  # noqa: BLE001 - a bad config is an outcome, not a crash
            ledger.trace({**base, "event": "skipped", "status": "failed",
                          "reason": f"could not construct: {type(exc).__name__}: {exc}"})
            steps.append({**base, "status": "failed", "error": str(exc)})
            continue
        ledger.trace({
            **base, "event": "planned", "attempt": attempt, "action": action.to_dict(),
            "params": canonical(analyzer.get_params()), "fidelity_check": step.fidelity_check,
            "rationale": "MVP-0 fixed order", "candidates": [s.analyzer for s in plan],
            **({"previous_attempts_uncertain": prior} if prior else {}),
        })
        say(f"[triage] {step_no} {step.analyzer}: running on {len(sample)} cases")
        obs = execute(action, action_id, attempt, analyzer, model, sample,
                      fidelity_check=step.fidelity_check, run_logger=run_logger)
        ledger.save(obs)
        ledger.cost({"kind": "action", "action_id": action_id, "analyzer": step.analyzer,
                     "attempt": attempt, "reserved_calls": action.reserve,
                     "baseline_reused": obs.status == OK and not obs.cost.get("calls"),
                     "judge_calls": 0, **obs.cost})
        ledger.trace({**base, "event": "settled", "attempt": attempt, "status": obs.status,
                      "fidelity": obs.fidelity.get("status"), "n_measured": len(obs.measured_ids),
                      "n_missing": len(obs.missing), "error_type": obs.error_type, "error": obs.error})
        observations[step.analyzer] = obs
        steps.append({**base, "status": obs.status, "reused": False, "fidelity": obs.fidelity,
                      "cost": obs.cost, "error": obs.error, "n_measured": len(obs.measured_ids),
                      "missing": obs.missing})
        say(f"[triage] {step_no} {step.analyzer}: {obs.status} "
            f"(fidelity={obs.fidelity.get('status')}, {obs.cost.get('wall_sec')}s, calls={obs.cost.get('calls')})")

    records = triage_cases(sample, observations, host_assumed_finish=host_assumed_finish,
                           answer_contract=answer_contract)
    report = {
        "policy": POLICY_VERSION,
        "model": model_identity,
        "sample": draw.to_dict(),
        "settings": {"host_assumed_finish": host_assumed_finish, "answer_contract": answer_contract,
                     "budget_sec": budget_sec},
        "steps": steps,
        "summary": summarize(records),
        "checks": _acceptance(plan, steps, observations, draw, ledger),
        "cases": records,
    }
    _write_json(out / "triage_report.json", report)
    (out / "triage_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _acceptance(plan: "tuple[PlanStep, ...]", steps: list[dict[str, Any]], observations: dict[str, Any],
                draw: Any, ledger: Ledger) -> dict[str, Any]:
    """The MVP-0 acceptance properties that can be checked from the run's own records."""
    planned = {s.analyzer for s in plan}
    traced = {e.get("analyzer") for e in ledger.trace_events() if e.get("event") in ("planned", "settled")}
    case_sets = {name: obs.planned_ids == list(draw.case_ids) and set(obs.measured_ids) <= set(draw.case_ids)
                 for name, obs in observations.items()}
    configs = {}
    for s in plan:
        obs = observations.get(s.analyzer)
        if obs is not None:
            configs[s.analyzer] = all(obs.params.get(k) == canonical(v) for k, v in s.config.items())
    costs = ledger.cost_events()
    return {
        "only_planned_actions": traced <= planned and len(steps) == len(plan),
        "case_sets_match_plan": case_sets,
        "configs_match_plan": configs,
        "stage0_cost_separate": any(e.get("kind") == "stage0" for e in costs),
        "no_zero_for_failure": all(obs.findings is None for obs in observations.values() if obs.status != OK),
        "fidelity_states": {name: (obs.fidelity or {}).get("status") for name, obs in observations.items()},
    }


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
        lines.append(f"| {st['step']} | {st['analyzer']} | {st['status']}{' (reused)' if st.get('reused') else ''} "
                     f"| {(st.get('fidelity') or {}).get('status', '-')} | {cost.get('wall_sec', '-')} "
                     f"| {cost.get('calls', '-')} |")
    lines += ["", "## Routes on this sample", "",
              "| route | FAIL fired/tested | PASS fired/tested | routing signal |", "|---|---|---|---:|"]
    for tag in ("H_label_free", "N"):
        f, p = s[tag]["fail"], s[tag]["pass"]
        lines.append(f"| {tag} | {f['n_fired']}/{f['n_tested']} | {p['n_fired']}/{p['n_tested']} "
                     f"| {s[tag]['routing_signal']} |")
    lines += [f"| X (no H/N signal) | {s['X']['fail']} | {s['X']['pass']} | - |", "",
              "The routing signal is max(0, P(tag|FAIL) - P(tag|PASS)) on this matched sample: it steers "
              "later actions and is not an explained share of failures. U was not run; S is not assigned."]
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
    lines += ["", "## Knockout (descriptive)", "", f"Fidelity: {', '.join(ko['fidelity'])}.", "",
              "| label | measured | mean ko_delta_full | mean ko_delta_late |", "|---|---:|---:|---:|"]
    for label in ("fail", "pass"):
        row = ko[label]
        lines.append(f"| {label.upper()} | {row['n_measured']} | {row['mean_ko_delta_full']} "
                     f"| {row['mean_ko_delta_late']} |")
    lines += ["", f"_{ko['_note']}._", "", "## Checks", ""]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {value}")
    lines.append("")
    return "\n".join(lines)
