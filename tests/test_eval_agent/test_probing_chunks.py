"""Chunked execution: no per-analyzer cost prior, spending bounded by what was measured.

* a ``reuse`` step runs in chunks with FAIL and PASS alternating, and every case
  reads the same as when the step ran in one piece;
* the budget is checked before each chunk on the cost measured so far, so an
  action whose cost nobody knew overshoots by at most one chunk;
* the scored scheduler prices a step it has not run at the run's typical case,
  and at its own measured cost once a chunk of it ran;
* further chunks of a step the scored scheduler already took do not use up
  ``max_actions``.
"""

from __future__ import annotations

import pytest

import evalrx.eval_agent.probing.controller as controller
from evalrx.eval_agent.probing import run_triage
from tests.test_eval_agent.test_probing_mvp2 import _kw, _pope_like, _strip, _Table

DESCRIBE_SEC = 10.0   # an expensive step whose cost no prior knows


@pytest.fixture
def metered(monkeypatch):
    """One second per case, ten for describe-first: deterministic costs a budget can bind on."""
    real = controller.execute

    def execute(action, *args, **kwargs):
        obs = real(action, *args, **kwargs)
        if obs.cost.get("calls"):
            per_case = DESCRIBE_SEC if "describe_first" in (action.config.get("strategies") or {}) else 1.0
            obs.cost["wall_sec"] = per_case * len(obs.planned_ids)
        return obs

    monkeypatch.setattr(controller, "execute", execute)


def _label(cases, cid):
    return next(c for c in cases if str(c.id) == cid).label.value


def test_a_reuse_step_runs_in_alternating_chunks_and_reads_the_same_as_in_one_piece(tmp_path):
    cases, table = _pope_like()
    whole_model, chunked_model = _Table(table), _Table(table)
    whole = run_triage(whole_model, cases, **_kw(tmp_path / "w", max_fail=8, min_route_fail=1, chunk=0))
    chunked = run_triage(chunked_model, cases, **_kw(tmp_path / "c", max_fail=8, min_route_fail=1, chunk=4))
    assert {r["case_id"]: _strip(r) for r in chunked["cases"]} == {r["case_id"]: _strip(r) for r in whole["cases"]}
    assert chunked_model.n_generate == whole_model.n_generate
    runs = [e for e in chunked["executions"] if e["key"] == "self_consistency" and not e["reused"]]
    assert len(runs) > 1 and all(e["n_cases"] <= 4 for e in runs)
    first = controller.Ledger(tmp_path / "c").settled(runs[0]["action_id"]).planned_ids
    assert {_label(cases, cid) for cid in first} == {"fail", "pass"}   # comparable from the first chunk
    assert chunked["settings"]["chunk"] == 4


def _budget_before_first_follow_up(tmp_path, scheduler):
    """The charge just before the first follow-up of an unbudgeted run, plus ten seconds: room
    for the follow-up at the worst case measured so far (one second per case), not at its real cost."""
    full = run_triage(_Table(_pope_like()[1]), _pope_like()[0],
                      **_kw(tmp_path / f"{scheduler}-full", max_fail=4, min_route_fail=1, chunk=4, scheduler=scheduler))
    spent = 0.0
    for e in full["executions"]:
        if e["kind"] == "follow_up":
            break
        if not e["reused"]:
            spent += float((e.get("cost") or {}).get("wall_sec") or 0.0)
    return spent + 10.0


@pytest.mark.parametrize("scheduler", ["rules", "scored"])
def test_an_unknown_expensive_step_overshoots_the_budget_by_at_most_one_chunk(tmp_path, metered, scheduler):
    budget = _budget_before_first_follow_up(tmp_path, scheduler)
    overshoot = {}
    for chunk in (0, 4):
        report = run_triage(_Table(_pope_like()[1]), _pope_like()[0],
                            **_kw(tmp_path / f"{scheduler}-{chunk}", max_fail=4, min_route_fail=1, chunk=chunk,
                                  scheduler=scheduler, budget_sec=budget))
        overshoot[chunk] = report["cost"]["charged"]["wall_sec"] - budget
        if chunk:   # the first chunk showed the real cost: the rest is not run, and the run stops there
            assert report["stop"]["reason"] == "budget"
            assert [e["key"] for e in report["executions"] if e["key"] == "describe_first" and e["status"] == "ok"] == [
                "describe_first"]
    assert 0 < overshoot[4] <= 4 * DESCRIBE_SEC < overshoot[0]


def test_the_scored_scheduler_prices_a_step_at_the_typical_case_until_a_chunk_of_it_ran(tmp_path, metered):
    report = run_triage(_Table(_pope_like()[1]), _pope_like()[0],
                        **_kw(tmp_path / "s", max_fail=4, min_route_fail=1, chunk=2, scheduler="scored"))
    rows = [(d["n"], c) for d in report["decisions"] for c in d.get("candidates") or [] if c["key"] == "describe_first"]
    first_n = rows[0][0]
    assert {c["basis"] for n, c in rows if n == first_n} == {"typical"}
    assert rows[0][1]["cost"] == pytest.approx(2 * 1.0)   # every case measured so far cost one second
    later = [c for n, c in rows if n > first_n]
    assert later and {c["basis"] for c in later} == {"own"}
    assert all(c["cost"] == pytest.approx(DESCRIBE_SEC * c["n_new_cases"]) for c in later)


def test_further_chunks_of_a_taken_step_do_not_use_up_max_actions(tmp_path):
    cases, table = _pope_like()
    report = run_triage(_Table(table), cases, **_kw(tmp_path / "m", max_fail=8, min_route_fail=1, chunk=2,
                                                    scheduler="scored", max_actions=2))   # an expansion and one step
    kinds = [d["kind"] for d in report["decisions"] if d["kind"] != "expand_not_possible"]
    (taken,) = [d for d in report["decisions"] if d["kind"] == "follow_up"]
    extends = [d for d in report["decisions"] if d["kind"] == "extend" and d["route"] == taken["route"]]
    assert kinds[:2] == ["expand", "follow_up"] and extends
    assert report["stop"]["reason"] == "max_actions" and report["stop"]["pending"] != (
        f"{taken['route']}:{taken['rule_id']}:{taken['key']}")
    route_cases = [r for r in report["routes"] if r["route"] == taken["route"]][0]
    layer = next(x for x in route_cases["layers"] if x["rule_id"] == taken["rule_id"])
    assert layer["state"] == "done"   # the step ran on all of its route's cases


def test_the_control_schedulers_run_over_the_scored_pool_and_never_explore(tmp_path):
    import pytest

    from evalrx.eval_agent.probing.controller import report_name
    from tests.test_eval_agent.test_probing_mvp2 import _kw, _pope_like, _Table

    cases, table = _pope_like()
    reports = {}
    for scheduler in ("random", "cheapest"):
        reports[scheduler] = run_triage(_Table(table), cases, **_kw(tmp_path / scheduler, max_fail=4, min_route_fail=1,
                                                                    chunk=2, scheduler=scheduler, explore_share=0.5))
    for scheduler, report in reports.items():
        assert report["scheduler"] == scheduler and report["explore_share"] == 0.0 and report["explored"] == {}
        chosen = [d for d in report["decisions"] if "choice" in d]
        assert chosen and {d["choice"] for d in chosen} == {scheduler}
        assert all(d["rationale"].startswith("control:") for d in chosen)
        assert report["stop"]["reason"] in ("below_threshold", "no_candidate", "max_actions", "budget")
        assert report_name(scheduler) == f"triage_report.{scheduler}"
    # the cheapest control takes the lowest-cost candidate at or above the gain floor at every decision
    floor = reports["cheapest"]["min_gain"]
    steps = [d for d in reports["cheapest"]["decisions"]
             if d.get("choice") == "cheapest" and d["kind"] in ("follow_up", "battery", "extend")]
    assert steps
    for d in steps:
        priced = [c for c in d["candidates"] if c["cost"] is not None and c["gain"] >= floor]
        taken = next(c for c in priced if c["key"] == d["key"] and c.get("route") == d.get("route"))
        assert taken["cost"] == min(c["cost"] for c in priced)
    with pytest.raises(ValueError):
        run_triage(_Table(table), cases, **_kw(tmp_path / "x", scheduler="random", explainer=object()))
