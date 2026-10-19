"""MVP-3 scored scheduler: U(a) = q·[D + T + C − R]₊ / (c + ε), against the rules.

* q counts distinct observations (a read-back once), a refusal as a failure;
* c comes from the step's own cost per case, else measured per-call rates;
* C is the coverage gap an action removes, T the recorded gap it completes;
* the best U wins, ties go to the soft order, every k-th decision explores
  with a seed, and a run stops when no gain reaches the threshold;
* a scored run is deterministic on resume and comparable with the rules run
  that shares its ledger.
"""

from __future__ import annotations

import json

from evalrx.eval_agent.probing import BATTERY, LAYERS, run_triage
from evalrx.eval_agent.probing.actions import OK, UNFAITHFUL, Observation
from evalrx.eval_agent.probing.compare import compare
from evalrx.eval_agent.probing.evidence import StepEvidence
from evalrx.eval_agent.probing.rules import candidate_steps
from evalrx.eval_agent.probing.scoring import (
    Scored,
    availability,
    choose,
    coverage_gain,
    per_case_costs,
    step_cost,
)
from tests.test_eval_agent.test_probing_mvp2 import _kw, _pope_like, _Table


def _step(key):
    return {s.name: s for s in [*BATTERY, *candidate_steps(LAYERS)]}[key]


def _obs(aid, ids, status=OK, cost=None, fidelity=None):
    return Observation(action_id=aid, analyzer="x", attempt=1, status=status, params={}, planned_ids=list(ids),
                       measured_ids=list(ids), findings={"per_case": []} if status == OK else None,
                       fidelity=fidelity or {"status": "not_applicable"}, cost=cost or {})


def test_availability_counts_each_observation_once_and_a_refusal_as_a_failure():
    ko = _step("attention_knockout")
    ok = _obs("a", ["c1"])
    ev = {"attention_knockout": StepEvidence(ko, [ok, ok, _obs("b", ["c2"], status=UNFAITHFUL)])}
    assert availability("attention_knockout", ev) == (1 + 1) / (2 + 2)
    assert availability("gradcam", ev) == 0.5   # never tried


def test_costs_come_from_the_steps_own_measurement_else_the_runs_typical_or_worst_case():
    sc, ko = _step("self_consistency"), _step("attention_knockout")
    ev = {"self_consistency": StepEvidence(sc, [_obs("s", ["a", "b"], cost={"wall_sec": 10.0, "calls": {"generate": 10}})]),
          "attention_knockout": StepEvidence(ko, [_obs("k", ["a"], cost={"wall_sec": 4.0, "calls": {"forward": 8}}),
                                                  _obs("k2", ["b", "c"], cost={"wall_sec": 2.0, "calls": {"forward": 16}})]),
          "answer_extraction_audit": StepEvidence(_step("answer_extraction_audit"),
                                                  [_obs("x", ["a", "b"], cost={"wall_sec": 0.0, "calls": {}})])}
    assert sorted(per_case_costs(ev, "wall_sec")) == [1.0, 4.0, 5.0]   # no model call: not a cost sample
    assert step_cost(sc, 3, ev, "wall_sec") == (15.0, "own")                 # 5 s per case, its own
    assert step_cost(ko, 3, ev, "wall_sec") == (6.0, "own")                  # 6 s over 3 cases
    # never run: priced at the run's median case when ranking, its worst case in a budget check
    assert step_cost(_step("gradcam"), 2, ev, "wall_sec") == (8.0, "typical")
    assert step_cost(_step("gradcam"), 2, ev, "wall_sec", unmeasured="max") == (10.0, "max")
    assert step_cost(_step("causal_trace"), 2, {}, "wall_sec") == (None, "none")


def test_coverage_gain_is_the_gap_removed_over_all_units():
    units = {"S1:yes": {"rule_id": "S1", "stratum": "yes", "need": 2, "measured": 0, "gap": 1.0, "_ids": ["a", "b"]},
             "S1:no": {"rule_id": "S1", "stratum": "no", "need": 2, "measured": 1, "gap": 0.5, "_ids": ["c", "d"]},
             "U1:yes": {"rule_id": "U1", "stratum": "yes", "need": 1, "measured": 0, "gap": 1.0, "_ids": ["e"]},
             "N2:no": {"rule_id": "N2", "stratum": "no", "need": 1, "measured": 1, "gap": 0.0, "_ids": ["f"]}}
    gain, touched = coverage_gain("S1", {"a", "d"}, units)
    assert gain == (0.5 + 0.5) / 4 and touched == ["S1:yes", "S1:no"]   # one of two: 1 -> 0.5; 1 of 2 done -> 0
    assert coverage_gain("N2", {"f"}, units) == (0.0, [])


def test_choose_takes_the_best_value_breaks_ties_by_soft_order_explores_within_its_share_and_stops():
    def s(key, U, gain, order, cost=10.0):
        return Scored(kind="follow_up", key=key, U=U, gain=gain, order=order, cost=cost)

    pool = [s("late", 0.5, 0.2, (1, 2)), s("early", 0.5, 0.2, (1, 0)), s("cheap", 0.1, 0.3, (0, 0))]
    kw = dict(seed=0, explore_share=0.1, min_gain=0.01)
    # 10 s of exploration would be 10/(50+10) of the spending: over the share, so the best runs
    best, why = choose(pool, decision_no=0, explored=0.0, spent=50.0, **kw)
    assert (best.key, why) == ("early", "best")
    # at 90 s spent it fits (10 <= 0.1 * 100): a seeded draw among the others runs instead
    picked, why = choose(pool, decision_no=3, explored=0.0, spent=90.0, **kw)
    assert why == "explore" and picked.key != "early"
    assert choose(pool, decision_no=3, explored=0.0, spent=90.0, **kw)[0].key == picked.key   # seeded
    assert choose(pool, decision_no=3, explored=10.0, spent=100.0, **kw)[1] == "best"         # share used up
    assert choose(pool, decision_no=3, explored=0.0, spent=1e6, **{**kw, "explore_share": 0.0})[1] == "best"
    # below the gain floor every candidate is an exploration candidate; once none fits, the run stops
    low = [s("x", 0.0, 0.0, (0, 0)), s("y", 0.0, 0.0, (0, 1), cost=100.0)]
    assert choose(low, decision_no=0, explored=0.0, spent=90.0, **kw) == (low[0], "explore")
    assert choose(low, decision_no=0, explored=0.0, spent=50.0, **kw) == (None, "below_threshold")
    assert choose([], decision_no=0, explored=0.0, spent=0.0, **kw) == (None, "none")


def test_scored_run_is_traceable_deterministic_on_resume_and_comparable_with_the_rules(tmp_path):
    cases, table = _pope_like()
    out = tmp_path / "p"
    rules = run_triage(_Table(table), cases, **_kw(out, max_fail=8, min_route_fail=1))
    model = _Table(table)
    scored = run_triage(model, cases, **_kw(out, max_fail=8, min_route_fail=1, scheduler="scored"))
    assert scored["scheduler"] == "scored" and (out / "triage_report.scored.json").is_file()
    assert (out / "triage_report.json").is_file()     # the rules report is not overwritten
    for d in scored["decisions"]:
        if d["kind"] in ("expand", "follow_up", "extend", "battery"):
            top = d["candidates"][0]
            assert {"U", "gain", "T", "C", "R", "q", "cost"} <= set(top) and d["choice"] in ("best", "explore")
    # the battery is complete on the final sample whatever the order of choices
    assert all(scored["checks"]["battery_covers_sample"].values())
    assert scored["cost"]["evidence"]["n_observations"] > 0
    again = run_triage(_Table(table), cases, **_kw(out, max_fail=8, min_route_fail=1, scheduler="scored"))
    assert [(d["kind"], d.get("key")) for d in again["decisions"]] == [(d["kind"], d.get("key"))
                                                                      for d in scored["decisions"]]
    diff = compare(rules, scored)
    assert diff["same_sample"] and diff["schedulers"] == ["rules", "scored"]
    json.dumps(diff)


def test_scored_runs_a_batch_dependent_step_once_after_the_last_expansion(tmp_path):
    from evalrx.eval_agent.probing.actions import PlanStep
    from evalrx.eval_agent.probing.rules import Layer

    cases, table = _pope_like()
    batchy = PlanStep("prompt_contrast", {"strategies": {"baseline": "{prompt}", "q": "Q: {prompt}"}}, "Z",
                      "a reading inferred from the whole batch", key="batchy", reuse=False,
                      reuse_note="test: pretends to depend on the batch")
    fallback = PlanStep("prompt_contrast", {"strategies": {"baseline": "{prompt}", "f": "F: {prompt}"}}, "Z1",
                        "a batch-dependent fallback", key="batchy_fallback", reuse=False, reuse_note="test")
    layers = (*LAYERS, Layer("Z1", "S", "a batch-dependent S reading", (fallback,), served_by=("batchy",)))
    report = run_triage(_Table(table), cases, **_kw(tmp_path / "z", max_fail=8, min_route_fail=1,
                                                   scheduler="scored", battery=(*BATTERY, batchy), layers=layers))
    runs = [e for e in report["executions"] if e["key"] == "batchy"]
    last = report["revisions"][-1]["revision"]
    assert [(e["revision"], e["n_cases"]) for e in runs] == [(last, len(report["sample"]["case_ids"]))]
    assert last == 1  # it waited for the expansion, though cheap
    (chosen,) = [d for d in report["decisions"] if d.get("key") == "batchy"]
    assert chosen["kind"] == "battery" and chosen["revision"] == last  # a scored choice once the sample settled
    # its layer waited for it: the batch-dependent fallback never ran
    assert not [e for e in report["executions"] if e["key"] == "batchy_fallback"]
    held = [f for d in report["decisions"] for f in d.get("filtered") or [] if f["key"] in ("batchy", "batchy_fallback")]
    assert held and all(f["key"] == "batchy" and "supersede" in f["reason"] for f in held)
