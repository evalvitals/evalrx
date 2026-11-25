"""The scored scheduler: U(a) = q·[D + T + C − R]₊ / (c + ε), against the rules.

* q counts distinct observations (a read-back once), a refusal as a failure;
* c comes from the step's own cost per case, else measured per-call rates; with a cost
  table it comes from the table, and the decisions then do not depend on the timings;
* C is the coverage gap an action removes, T the recorded gap it completes;
* the best U wins, ties go to the soft order, every k-th decision explores
  with a seed, and a run stops when no gain reaches the threshold;
* a scored run is deterministic on resume and comparable with the rules run
  that shares its ledger.
"""

from __future__ import annotations

import json
import random
import types

import pytest

from evalrx.eval_agent.triage import BATTERY, LAYERS, CostTable, run_triage
from evalrx.eval_agent.triage.actions import OK, UNFAITHFUL, UNSUPPORTED, Observation
from evalrx.eval_agent.triage.evidence import StepEvidence
from evalrx.eval_agent.triage.rules import candidate_steps
from evalrx.eval_agent.triage.scoring import (
    Scored,
    availability,
    choose,
    coverage_gain,
    looked_enough,
    per_case_costs,
    priced,
    promising,
    step_cost,
)
from tests.test_eval_agent.test_triage_expansion import _kw, _pope_like, _Table


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
    # ran without calling the model: priced at what it measured (nothing), not as unmeasured --
    # else a budget check prices every audit of the next expansion at the run's worst case
    assert step_cost(_step("answer_extraction_audit"), 64, ev, "wall_sec", unmeasured="max") == (0.0, "own")
    unsupported = {"gradcam": StepEvidence(_step("gradcam"), [_obs("g", ["a"], status=UNSUPPORTED,
                                                                   cost={"wall_sec": 0.0, "calls": {}})])}
    assert step_cost(_step("gradcam"), 2, {**ev, **unsupported}, "wall_sec", unmeasured="max") == (10.0, "max")
    # never run: priced at the run's median case when ranking, its worst case in a budget check
    assert step_cost(_step("gradcam"), 2, ev, "wall_sec") == (8.0, "typical")
    assert step_cost(_step("gradcam"), 2, ev, "wall_sec", unmeasured="max") == (10.0, "max")
    assert step_cost(_step("causal_trace"), 2, {}, "wall_sec") == (None, "none")


def test_a_cost_table_prices_a_step_per_case_and_its_one_time_cost_until_the_step_has_run():
    table = CostTable.from_dict({"steps": {"tuned_lens": {"per_case": 0.5, "fixed": 300.0},
                                           "gradcam": {"per_case": 2.0}, "self_consistency": {"per_case": 6.0},
                                           "answer_extraction_audit": {"per_case": 0.0}}})
    tl, sc = _step("tuned_lens"), _step("self_consistency")
    # nothing measured is needed: the table prices a run's first action
    assert step_cost(tl, 8, {}, "wall_sec", table=table) == (304.0, "table+fixed")
    assert step_cost(sc, 3, {}, "wall_sec", table=table) == (18.0, "table")
    # a gate result is not a run of the step; a completed chunk is, and the fit is then paid
    gated = {"tuned_lens": StepEvidence(tl, [_obs("u", ["a"], status=UNSUPPORTED, cost={"wall_sec": 0.0, "calls": {}})])}
    assert step_cost(tl, 8, gated, "wall_sec", table=table) == (304.0, "table+fixed")
    ran = {"tuned_lens": StepEvidence(tl, [_obs("t", ["a", "b"], cost={"wall_sec": 999.0, "calls": {"forward": 80}})]),
           "self_consistency": StepEvidence(sc, [_obs("s", ["a", "b"], cost={"wall_sec": 1.0, "calls": {"generate": 10}}),
                                                 _obs("s", ["a", "b"], cost={"wall_sec": 1.0, "calls": {"generate": 10}})])}
    assert step_cost(tl, 8, ran, "wall_sec", table=table) == (4.0, "table")
    assert step_cost(sc, 3, ran, "wall_sec", table=table) == (18.0, "table")        # not its measured 0.5 s per case
    # a step the table does not list: the median of its model-calling prices, the maximum for a conservative check
    assert step_cost(_step("causal_trace"), 2, ran, "wall_sec", table=table) == (4.0, "table_typical")
    assert step_cost(_step("causal_trace"), 2, ran, "wall_sec", unmeasured="max", table=table) == (12.0, "table_max")
    # the evidence at the table's prices: each observation once, the one-time price once
    assert priced(ran, table) == 300.0 + 0.5 * 2 + 6.0 * 2
    assert priced(gated, table) == 0.0
    # a table on another axis leaves this axis to the measurements
    assert step_cost(sc, 3, ran, "usd", table=table) == (None, "none")


def test_a_cost_table_has_a_content_identity_and_refuses_rows_without_a_price(tmp_path):
    rows = {"gradcam": {"per_case": 2.0}, "tuned_lens": {"per_case": 0.5, "fixed": 300.0}}
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"axis": "wall_sec", "source": {"runs": ["a"]}, "steps": rows}))
    table = CostTable.load(path)
    assert table.steps == {"gradcam": (2.0, 0.0), "tuned_lens": (0.5, 300.0)}
    # the identity is of the prices, not of where they came from or how the file orders them
    assert table.sha1 == CostTable.from_dict({"steps": dict(reversed(rows.items())), "source": "elsewhere"}).sha1
    assert table.sha1 != CostTable.from_dict({"steps": {**rows, "gradcam": {"per_case": 2.5}}}).sha1
    assert table.identity()["source"] == {"runs": ["a"]}
    for bad in ({}, {"steps": {}}, {"steps": {"gradcam": {"fixed": 1.0}}}, {"steps": {"gradcam": {"per_case": -1}}},
                {"steps": {"gradcam": {"per_case": float("nan")}}}, {"axis": "tokens", "steps": rows}):
        with pytest.raises(ValueError):
            CostTable.from_dict(bad)


def _jittery_clock(monkeypatch, seed):
    """Every action of the run takes a different, seeded number of seconds."""
    from evalrx.eval_agent.triage import executor

    rng, now = random.Random(seed), [0.0]

    def monotonic():
        now[0] += rng.uniform(0.5, 30.0)
        return now[0]

    monkeypatch.setattr(executor, "time", types.SimpleNamespace(monotonic=monotonic))


def test_with_a_cost_table_the_decisions_do_not_depend_on_the_runs_timings(tmp_path, monkeypatch):
    cases, answers = _pope_like()
    table = CostTable.from_dict({"steps": {s.name: {"per_case": 1.0 + i % 4, "fixed": 40.0 if i == 2 else 0.0}
                                           for i, s in enumerate([*BATTERY, *candidate_steps(LAYERS)])}})
    # N has 2 FAIL cases, too few to be an active route: a target for exploration once its price fits the share
    kw = dict(max_fail=8, scheduler="scored", explore_share=0.3, cost_table=table)

    def run(name, seed, **over):
        _jittery_clock(monkeypatch, seed)
        return run_triage(_Table(answers), cases, **_kw(tmp_path / name, **{**kw, **over}))

    def decided(report):   # the whole decision records but the ids of the actions they started
        return json.dumps([{k: v for k, v in d.items() if k != "action_id"} for d in report["decisions"]],
                          sort_keys=True, default=str)

    a, b = run("a", 1), run("b", 2)
    assert a["cost"]["charged"]["wall_sec"] != b["cost"]["charged"]["wall_sec"]     # the two runs were timed differently
    assert decided(a) == decided(b)
    assert [(e["kind"], e["key"], e.get("route"), e.get("n_cases")) for e in a["executions"]] == \
        [(e["kind"], e["key"], e.get("route"), e.get("n_cases")) for e in b["executions"]]
    assert any(d.get("choice") == "explore" for d in a["decisions"])                # exploration ran, at the table's prices
    spending = [d for d in a["decisions"] if d.get("candidates")]
    assert spending and all(d["cost_basis"] == "table" and c["basis"].startswith("table")
                            for d in spending for c in [*d["candidates"], *d.get("explore_targets", [])]
                            if c["key"] != "expand")
    # the report says what the decisions were priced from, and how far the table is from the measurements
    assert a["cost_basis"] == "table" and a["cost_table"]["sha1"] == table.sha1
    assert a["cost"]["table"]["priced"] == b["cost"]["table"]["priced"]
    assert a["cost"]["table"]["steps"] and "cost table" in (tmp_path / "a" / "triage_report.scored.md").read_text()
    # without a table the same two clocks price the candidates differently
    c, d = run("c", 1, cost_table=None), run("d", 2, cost_table=None)
    assert c["cost_basis"] == "measured" and "cost_table" not in c
    first = lambda r: [x["cost"] for x in next(x for x in r["decisions"] if x.get("candidates"))["candidates"]]  # noqa: E731
    assert first(c) != first(d)


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
    targets = [s("t1", 0.0, 0.0, (2, 98)), s("t2", 0.0, 0.0, (2, 99))]
    kw = dict(seed=0, explore_share=0.1, min_gain=0.01)
    # 10 s of exploration would be 10/(50+10) of the spending: over the share, so the best runs
    best, why = choose(pool, decision_no=0, explored=0.0, spent=50.0, targets=targets, **kw)
    assert (best.key, why) == ("early", "best")
    # at 90 s spent it fits (10 <= 0.1 * 100): a seeded draw among the targets runs instead
    picked, why = choose(pool, decision_no=3, explored=0.0, spent=90.0, targets=targets, **kw)
    assert why == "explore" and picked in targets
    assert choose(pool, decision_no=3, explored=0.0, spent=90.0, targets=targets, **kw)[0].key == picked.key   # seeded
    assert choose(pool, decision_no=3, explored=10.0, spent=100.0, targets=targets, **kw)[1] == "best"   # share used up
    assert choose(pool, decision_no=3, explored=0.0, spent=1e6, targets=targets,
                  **{**kw, "explore_share": 0.0})[1] == "best"
    # exploration draws only from its targets: without one the best runs, and below the gain floor the run stops
    low = [s("x", 0.0, 0.0, (0, 0)), s("y", 0.0, 0.0, (0, 1), cost=100.0)]
    assert choose(pool, decision_no=3, explored=0.0, spent=90.0, targets=[], **kw)[1] == "best"
    assert choose(low, decision_no=0, explored=0.0, spent=90.0, targets=[], **kw) == (None, "below_threshold")
    assert choose([], decision_no=0, explored=0.0, spent=0.0, targets=[], **kw) == (None, "none")
    # a target that fits runs even with no candidate left; one that does not fit leaves the best
    target = s("t", 0.0, 0.0, (2, 99), cost=5.0)
    assert choose([], decision_no=0, explored=0.0, spent=90.0, targets=[target], **kw) == (target, "explore")
    assert choose(low, decision_no=0, explored=0.0, spent=90.0, targets=[target], **kw) == (target, "explore")
    assert choose(pool, decision_no=3, explored=0.0, spent=10.0, targets=[target], **kw)[1] == "best"


def test_choose_completes_an_exploration_already_begun_before_it_starts_another():
    def s(key, kind, cost, order):
        return Scored(kind=kind, key=key, U=0.0, gain=0.0, order=order, cost=cost)

    new = [s(f"new{i}", "follow_up", 5.0, (0, i)) for i in range(6)]
    begun = [s("begun_a", "extend", 5.0, (2, 98)), s("begun_b", "extend", 8.0, (2, 99))]
    kw = dict(seed=0, explore_share=0.1, min_gain=0.01, explored=0.0)
    # every target fits: the draw is among the two begun, whatever the decision
    picks = {choose([], decision_no=n, spent=1000.0, targets=new + begun, **kw)[0].key for n in range(40)}
    assert picks == {"begun_a", "begun_b"}
    # 6 s of exploration is the most that fits at 54 s spent: begun_b (8 s) does not, begun_a does
    assert {choose([], decision_no=n, spent=54.0, targets=new + begun, **kw)[0].key for n in range(40)} == {"begun_a"}
    # no begun exploration fits (both dearer than what is left of the share): a new target starts
    dear = [s("begun_a", "extend", 50.0, (2, 98)), s("begun_b", "extend", 80.0, (2, 99))]
    picks = [choose([], decision_no=n, spent=54.0, targets=new + dear, **kw) for n in range(40)]
    assert {why for _, why in picks} == {"explore"} and {p.key for p, _ in picks} <= {t.key for t in new}
    assert len({p.key for p, _ in picks}) > 1   # still a draw among the new targets


def test_an_exploration_has_looked_enough_when_no_reading_could_leave_its_band_on_the_route():
    def o(n, diff, half):
        return {"n": [n, n], "diff": diff, "ci95": [diff - half, diff + half], "observed": "undetermined"}

    # 8 of 40 FAIL cases, difference 0.02 +- 0.30, band 0.1: narrowed to the route the interval is
    # +- 0.30 * sqrt(8/40) = 0.134 around twice the difference (0.04): its low end -0.094 is not above 0.1
    flat = o(8, 0.02, 0.30)
    assert not promising(flat, 0.1, 40) and looked_enough([(flat, 0.1)], 40)
    # the same sample with a difference of 0.15: 0.30 - 0.134 = 0.166 clears the band, and so does a
    # negative one below it; as measured (stretch 1) 0.15 - 0.134 does not
    assert promising(o(8, 0.15, 0.30), 0.1, 40) and promising(o(8, -0.15, 0.30), 0.1, 40)
    assert not promising(o(8, 0.15, 0.30), 0.1, 40, stretch=1.0)
    assert not looked_enough([(flat, 0.1), (o(8, 0.15, 0.30), 0.1)], 40)   # one promising reading keeps it open
    # too few FAIL cases yet, or no interval yet: not looked at
    assert not looked_enough([(o(4, 0.02, 0.30), 0.1)], 40)
    assert promising({"n": [1, 1], "observed": "undetermined"}, 0.1, 40)
    assert not looked_enough([({"n": [9, 9], "observed": "undetermined"}, 0.1)], 40)
    # a route smaller than the minimum is looked at once it is measured whole; no reading, never closed
    assert looked_enough([(o(5, 0.0, 0.2), 0.1)], 5) and not looked_enough([(o(4, 0.0, 0.2), 0.1)], 5)
    assert not looked_enough([], 40)
    # a larger route narrows the projection: what was closed on 12 FAIL cases is open again on 400
    edge = o(8, 0.09, 0.40)   # 0.18 - 0.40 * sqrt(8/12) = -0.147 ; 0.18 - 0.40 * sqrt(8/400) = 0.123
    assert looked_enough([(edge, 0.1)], 12) and not looked_enough([(edge, 0.1)], 400)


def test_scored_run_is_traceable_deterministic_on_resume_and_shares_the_sample_with_the_rules(tmp_path):
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
    assert rules["sample"]["case_ids"] == scored["sample"]["case_ids"]   # the same nested sample, one ledger
    assert (rules["scheduler"], scored["scheduler"]) == ("rules", "scored")


def test_scored_runs_a_batch_dependent_step_once_after_the_last_expansion(tmp_path):
    from evalrx.eval_agent.triage.actions import PlanStep
    from evalrx.eval_agent.triage.rules import Layer

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
