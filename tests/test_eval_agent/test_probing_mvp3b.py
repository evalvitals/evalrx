"""MVP-3b: D from competing explanations the judge proposes, checked by code.

* a prediction is a hit when the 95% interval of FAIL - control lies in its
  region, a miss when it lies outside, undetermined otherwise;
* D is the share of pairs of a route's live explanations whose predictions for
  the step conflict; ``unknown`` counts in the denominator only;
* the judge's reply keeps only offered steps, card readings and the four
  directions, and a duplicate explanation is dropped;
* the scored scheduler asks once the sample has settled and again when the
  evidence thins a route's live explanations, reads a stored reply back on
  resume, never shows the judge a case, and charges its calls to the budget.
"""

from __future__ import annotations

import json

import pytest

from evalrx.eval_agent.probing import run_triage
from evalrx.eval_agent.probing.explanations import (
    CARDS,
    Explainer,
    Hypothesis,
    Metric,
    check,
    difference,
    disagreement,
    observed,
    outcome,
    parse,
    verdict,
)
from tests.test_eval_agent.test_probing_mvp2 import _kw, _pope_like, _Table


def test_verdicts_follow_the_interval_against_the_predicted_region():
    assert observed([0.2, 0.4], 0.1) == "up" and observed([-0.05, 0.08], 0.1) == "same"
    assert observed([-0.4, -0.2], 0.1) == "down" and observed([-0.05, 0.3], 0.1) == "undetermined"
    assert verdict("up", [0.2, 0.4], 0.1) == "hit" and verdict("up", [-0.3, 0.1], 0.1) == "miss"
    assert verdict("up", [0.05, 0.3], 0.1) == "undetermined"
    assert verdict("same", [-0.05, 0.08], 0.1) == "hit" and verdict("same", [0.15, 0.3], 0.1) == "miss"
    assert verdict("down", [-0.05, 0.3], 0.1) == "miss" and verdict("unknown", [0.2, 0.4], 0.1) == "undetermined"
    assert verdict("up", None, 0.1) == "undetermined"
    diff, ci = difference([1.0, 1.2, 0.8, 1.0], [0.0, 0.2, -0.2, 0.0], "mean")
    assert diff == pytest.approx(1.0) and ci[0] > 0.5 and ci[1] < 1.5
    diff, ci = difference([1, 1, 1, 0], [0, 0, 0, 0], "rate")
    assert diff == 0.75 and ci[0] < 0.75 < ci[1]
    assert difference([1.0], [0.0, 1.0], "mean") is None


def test_outcome_reads_the_cases_with_the_field_and_says_when_nothing_was_measured():
    m = Metric("x", "a reading", 0.1)
    values = {"f1": {"state": "measured", "values": {"x": 1.0}}, "f2": {"state": "measured", "values": {"x": 1.0}},
              "f3": {"state": "untested", "reason": "failed"},
              "c1": {"state": "measured", "values": {"x": 0.0}}, "c2": {"state": "measured", "values": {"y": 3.0}}}
    o = outcome(values, ["f1", "f2", "f3"], ["c1", "c2"], m)
    assert o["n"] == [2, 1] and o["observed"] == "undetermined" and "ci95" not in o
    assert outcome({}, ["f1"], ["c1"], m)["observed"] == "not_measured"


def _h(hid, route, **preds):
    return Hypothesis(hid, route, f"claim {hid}", {k: dict(v) for k, v in preds.items()})


def test_disagreement_is_the_share_of_live_pairs_whose_predictions_conflict():
    a = _h("S.h1", "S", gradcam={"gc_top5_share": "up"})
    b = _h("S.h2", "S", gradcam={"gc_top5_share": "down"})
    c = _h("S.h3", "S", patchscope={"ps_late_drop": "up"})   # unknown on gradcam
    D, detail = disagreement("S", "gradcam", [a, b, c])
    assert D == pytest.approx(1 / 3) and detail["conflicting"] == [{"pair": ["S.h1", "S.h2"],
                                                                    "readings": ["gc_top5_share"]}]
    assert disagreement("S", "patchscope", [a, b, c])[0] == 0.0
    assert disagreement("S", "gradcam", [a])[0] == 0.0
    assert disagreement("N", "gradcam", [a, b])[0] == 0.0          # another route's explanations
    # a conflict counts by how decidable it is
    D, detail = disagreement("S", "gradcam", [a, b, c], lambda f, x, y: {"decidable": 0.25})
    assert D == pytest.approx(0.25 / 3) and detail["conflict_share"] == pytest.approx(1 / 3)


def test_decidability_asks_whether_the_routes_cases_can_settle_a_conflict():
    from evalrx.eval_agent.probing.explanations import (
        cases_needed,
        decidability,
        half_width,
        spread,
    )

    assert half_width(9) < 1.0 < half_width(8)       # under one spread from 9 cases a side
    # a true "up" one spread past the band contradicts "same" from 9 cases a side, whatever the band
    assert cases_needed("up", "same", 0.0) == cases_needed("up", "same", 2.0) == 9
    # a true "same" contradicts "up" only once the interval fits in the band: never with no band
    assert cases_needed("same", "up", 0.0) == float("inf")
    assert cases_needed("same", "up", 0.5) > cases_needed("same", "up", 1.0) == 9   # a band of one spread
    assert cases_needed("up", "down", 0.5) < cases_needed("up", "same", 0.5)
    mean = Metric("x", "a mean", 0.05)
    # unmeasured mean: only the "up" side can settle; three cases go a third of the way there
    tiny = decidability("up", "same", 3, mean, None)
    assert tiny["basis"] == "unmeasured" and tiny["needed"] == [9, None]
    assert tiny["decidable"] == pytest.approx(3 / 9 / 2, abs=1e-3)
    assert decidability("up", "same", 16, mean, None)["decidable"] == 0.5
    assert decidability("up", "same", 16, mean, 0.1)["decidable"] == pytest.approx((1 + 16 / 32) / 2)
    assert decidability("up", "same", 1, mean, 0.1)["decidable"] == 0.0
    assert decidability("up", "down", 16, mean, 0.0)["decidable"] == 1.0   # a constant reading
    # a share's spread is smoothed (0/16 still has one) and taken at its largest before it is measured
    assert spread([[0.0] * 16, [0.0] * 16], "rate") == pytest.approx(0.3)
    assert spread([[1.0, 2.0, 3.0], [5.0]], "mean") == pytest.approx(1.0)
    assert spread([[1.0]], "mean") is None
    share = Metric("y", "a share", 0.1, "rate")
    assert decidability("up", "same", 16, share, None)["needed"][1] > decidability(
        "up", "same", 16, share, 0.3)["needed"][1]


def test_check_contradicts_on_a_miss_and_calls_an_explanation_without_predictions_untestable():
    a = _h("S.h1", "S", gradcam={"gc_top5_share": "up"})
    b = _h("S.h2", "S", gradcam={"gc_top5_share": "same"})
    c = _h("S.h3", "S")
    seen = {"gc_top5_share": {"n": [17, 17], "ci95": [0.2, 0.4], "observed": "up"}}
    states = check([a, b, c], lambda route, step, m: seen.get(m.field, {"n": [0, 0], "observed": "not_measured"}))
    assert {k: v["state"] for k, v in states.items()} == {"S.h1": "live", "S.h2": "contradicted",
                                                         "S.h3": "untestable"}
    assert states["S.h1"]["predictions"][0]["verdict"] == "hit"


def test_parse_keeps_offered_steps_card_readings_directions_and_competing_explanations_with_primaries():
    offered = {"S": {"gradcam": ["gc_top5_share", "gc_entropy"]}, "N": {"attention_temperature": ["tt_slope"]}}
    reply = "```json\n" + json.dumps({"routes": {
        "S": [{"id": "h1", "claim": "concentrated off target", "primary": [["gradcam", "gc_top5_share"]],
               "predictions": {"gradcam": {"gc_top5_share": "↑", "gc_entropy": "sideways", "gc_margin": "up"},
                               "patchscope": {"ps_late_drop": "up"}}},
              {"id": "h2", "claim": "same as h1, reworded", "primary": [["gradcam", "gc_top5_share"]],
               "predictions": {"gradcam": {"gc_top5_share": "up"}}},
              {"id": "h3", "claim": "diffuse", "primary": [["gradcam", "gc_entropy"], ["gradcam", "gc_margin"]],
               "predictions": {"gradcam": {"gc_entropy": "up", "gc_top5_share": "down"}}},
              {"id": "h4", "claim": "no primary named", "predictions": {"gradcam": {"gc_entropy": "down"}}},
              {"id": "h5", "claim": "a third that competes", "primary": [["gradcam", "gc_top5_share"]],
               "predictions": {"gradcam": {"gc_top5_share": "same"}}}],
        "U": [{"id": "h1", "claim": "not offered"}],
        "N": [{"claim": ""}]}}) + "\n```"
    hyps, dropped = parse(reply, offered, max_per_route=2)
    assert [(h.id, h.predictions, h.primary) for h in hyps] == [
        ("S.h1", {"gradcam": {"gc_top5_share": "up"}}, (("gradcam", "gc_top5_share"),)),
        # told apart from h1 on h1's primary reading, which h3 also predicts
        ("S.h3", {"gradcam": {"gc_entropy": "up", "gc_top5_share": "down"}}, (("gradcam", "gc_entropy"),))]
    reasons = {d["what"]: d["reason"] for d in dropped}
    assert reasons["route U"] == "not offered" and reasons["N[0]"] == "no claim"
    assert reasons["S.h1 gradcam.gc_entropy"] == "direction 'sideways'"
    assert reasons["S.h1 gradcam.gc_margin"] == "reading not on the card"
    assert reasons["S.h1 patchscope"] == "step not offered on this route"
    assert "cannot be told apart from S.h1" in reasons["S.h2"]
    assert reasons["S.h3 primary ['gradcam', 'gc_margin']"] == "not one of its explicit predictions"
    assert "no primary prediction" in reasons["S.h4"] and "more than 2" in reasons["S.h5"]
    assert parse("no json here", offered, 3) == ([], [{"what": "reply", "reason": parse("no json here", offered,
                                                                                        3)[1][0]["reason"]}])
    assert parse(json.dumps({"x": 1}), offered, 3)[1] == [{"what": "reply", "reason": "no 'routes' object"}]


def test_only_primary_predictions_decide_and_conflict():
    a = Hypothesis("S.h1", "S", "a", {"gradcam": {"gc_top5_share": "up", "gc_entropy": "down"}},
                   primary=(("gradcam", "gc_top5_share"),))
    b = Hypothesis("S.h2", "S", "b", {"gradcam": {"gc_top5_share": "same", "gc_entropy": "up"}},
                   primary=(("gradcam", "gc_top5_share"),))
    seen = {"gc_top5_share": {"n": [17, 17], "ci95": [0.2, 0.4], "observed": "up"},
            "gc_entropy": {"n": [17, 17], "ci95": [0.2, 0.4], "observed": "up"}}
    states = check([a, b], lambda route, step, m: seen[m.field])
    # a's secondary gc_entropy misses, but only its primary decides; b's primary misses
    assert states["S.h1"]["state"] == "live" and states["S.h2"]["state"] == "contradicted"
    assert [(p["reading"], p["primary"], p["verdict"]) for p in states["S.h1"]["predictions"]] == [
        ("gc_entropy", False, "miss"), ("gc_top5_share", True, "hit")]
    # the pair conflicts on both readings, but D counts only the primary one
    assert disagreement("S", "gradcam", [a, b])[1]["conflicting"][0]["readings"] == ["gc_top5_share"]


def test_every_card_reading_names_its_band_and_kind():
    for key, metrics in CARDS.items():
        assert metrics and len({m.field for m in metrics}) == len(metrics), key
        assert all(m.band > 0 and m.kind in ("mean", "rate") and m.meaning for m in metrics), key


class _Judge:
    """Two explanations per route that disagree on the first reading of every offered step; they stand
    on the steps named in *primary* where offered, else on the first offered step (one to run first)."""

    def __init__(self, fail=False, primary=()):
        self.prompts, self.fail, self.primary = [], fail, set(primary)

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("timed out")
        ctx = json.loads(prompt[prompt.index("Evidence (JSON):") + len("Evidence (JSON):"):])
        routes = {}
        for route, r in ctx["routes"].items():
            steps = r["to_run"] + r.get("already_run", [])
            up = {c["step"]: {c["readings"][0]["field"]: "up"} for c in steps}
            same = {c["step"]: {c["readings"][0]["field"]: "same"} for c in steps}
            on = [c for c in steps if c["step"] in self.primary] or steps[:1]
            primary = [[c["step"], c["readings"][0]["field"]] for c in on[:2]]
            routes[route] = [{"id": "h1", "claim": "it moves", "predictions": up, "primary": primary},
                             {"id": "h2", "claim": "it does not", "predictions": same, "primary": primary}]
        return json.dumps({"routes": routes})


def test_explainer_reads_a_stored_reply_back_and_asks_again_after_a_failure(tmp_path):
    offered = {"S": {"gradcam": ["gc_top5_share"]}}
    context = {"routes": {"S": {"to_run": [{"step": "gradcam", "readings": [{"field": "gc_top5_share"}]}]}}}
    judge = _Judge()
    first = Explainer(judge, identity={"provider": "stub", "model": "a"}).propose(tmp_path, 0, context, offered)
    again = Explainer(judge, identity={"provider": "stub", "model": "a"}).propose(tmp_path, 0, context, offered)
    assert (first.source, again.source, len(judge.prompts)) == ("judge", "read_back", 1)
    assert [h.to_dict() for h in again.hypotheses] == [h.to_dict() for h in first.hypotheses]
    assert again.cost == first.cost and first.cost["calls"] == {"judge": 1}
    # another judge is another key
    Explainer(judge, identity={"provider": "stub", "model": "b"}).propose(tmp_path, 0, context, offered)
    assert len(judge.prompts) == 2
    broken = _Judge(fail=True)
    failed = Explainer(broken, identity={"provider": "stub", "model": "c"}).propose(tmp_path, 0, context, offered)
    assert failed.source == "failed" and "timed out" in failed.error and not failed.hypotheses
    Explainer(broken, identity={"provider": "stub", "model": "c"}).propose(tmp_path, 0, context, offered)
    assert len(broken.prompts) == 2   # a failure is not stored


def _explained(out, judge, **over):
    return run_triage(_Table(_pope_like()[1]), _pope_like()[0],
                      **_kw(out, max_fail=8, min_route_fail=1, scheduler="scored",
                            explainer=Explainer(judge, identity={"provider": "stub", "model": "stub"}), **over))


def test_scored_run_asks_once_the_sample_settles_and_resumes_without_asking(tmp_path):
    cases, _ = _pope_like()
    out = tmp_path / "p"
    judge = _Judge()
    report = _explained(out, judge)
    last = report["revisions"][-1]["revision"]
    (asked,) = report["explanations"]["sets"]
    assert len(judge.prompts) == 1 and asked["source"] == "judge" and asked["revision"] == last == 1
    assert "settled" in asked["reason"] and len(list((out / "explanations").glob("*.json"))) == 1
    # before the sample settled, D is 0: the routes the explanations are about still change
    for d in report["decisions"]:
        if d.get("candidates") and d["revision"] < last:
            assert all(c["D"] == 0.0 for c in d["candidates"]) and "explanations" not in d
    scored = [c for d in report["decisions"] for c in d.get("candidates") or [] if c["kind"] != "expand"]
    assert any(c["D"] > 0 and c["d_detail"]["conflict_share"] == 1.0 for c in scored)
    # each conflict counts by how decidable it is on the route's few cases: never in full here
    assert all(0 < x["decidable"] < 1 for c in scored for x in c["d_detail"].get("conflicting", []))
    # the judge never sees a case: no case id, no question text, no CONFIRM
    for p in judge.prompts:
        assert not any(str(c.id) in p for c in cases) and "Is there a" not in p and "confirm" not in p.lower()
    cost = report["cost"]
    assert cost["judge"]["n_calls"] == 1 and cost["charged"]["wall_sec"] >= cost["judge"]["wall_sec"]
    assert report["explanations"]["tally"]["live"] >= 2 and report["outcomes"]
    assert not (out / "triage_report.scored.json").exists()   # the D = 0 arm's report is not overwritten
    md = (out / "triage_report.scored_d.md").read_text()
    assert "## Competing explanations (D)" in md and "S.h1" in md

    again_judge = _Judge()
    again = _explained(out, again_judge)
    assert not again_judge.prompts and again["cost"]["judge"]["n_read_back"] == 1
    assert [(d["kind"], d.get("key")) for d in again["decisions"]] == [(d["kind"], d.get("key"))
                                                                      for d in report["decisions"]]


def _wrapper_on_s(monkeypatch):
    """The layers with an S step the table model ignores: prompt_sensitivity is 0 on every case, so an
    "up" prediction on it misses at once."""
    from evalrx.eval_agent.probing import LAYERS
    from evalrx.eval_agent.probing.actions import PlanStep
    from evalrx.eval_agent.probing.rules import Layer

    monkeypatch.setitem(CARDS, "wrapped", (Metric("prompt_sensitivity", "the answer moves with a wrapper", 0.1),))
    step = PlanStep("prompt_contrast", {"strategies": {"baseline": "{prompt}", "w": "Q: {prompt}"}}, "Z1",
                    "re-ask inside a wrapper", key="wrapped")
    at = next(i for i, x in enumerate(LAYERS) if x.rule_id == "Sb")
    return (*LAYERS[:at], Layer("Z1", "S", "does a wrapper move it", (step,)), *LAYERS[at:])


def test_a_decided_prediction_that_thins_a_route_asks_the_judge_again(tmp_path, monkeypatch):
    judge = _Judge(primary={"wrapped"})
    report = _explained(tmp_path / "a", judge, layers=_wrapper_on_s(monkeypatch))
    first, second = report["explanations"]["sets"]
    assert len(judge.prompts) == 2 and "settled" in first["reason"] and "fewer than two live" in second["reason"]
    assert first["final"]["S.h1"]["state"] == "contradicted" and first["final"]["S.h2"]["state"] == "live"
    previous = json.loads(judge.prompts[1][judge.prompts[1].index("Evidence (JSON):") + 16:])["previous"]
    assert {p["id"]: p["state"] for p in previous}["S.h1"] == "contradicted"
    # what ran is shown to the judge as measured (still offered while chunks of it are left)
    assert "wrapped" in second["seen"]["S"]
    # the re-ask adds to what is still live: S.h2 is kept, the judge's new "up" explanation takes a fresh id
    # (S.h1 was the contradicted one), and its repeat of S.h2 does not compete with it
    assert second["carried"] == ["N.h1", "N.h2", "S.h2"]
    assert [h["id"] for h in second["hypotheses"]] == ["N.h1", "N.h2", "S.h2", "S.h3"]
    assert "cannot be told apart from S.h2" in {d["what"]: d["reason"] for d in second["dropped"]}["S.h2"]


def test_a_route_left_with_only_steps_the_judge_is_not_asked_about_does_not_ask_again(tmp_path, monkeypatch):
    import evalrx.eval_agent.probing.controller as controller

    judge = _Judge(primary={"wrapped"})
    # every S step re-measures S's cut here, the wrapper too once the judge has been asked: after the
    # first ask S has tools left (the wrapper's other chunks among them), but none the judge could be
    # asked to predict, so the wrapper contradicting S.h1 asks nothing
    monkeypatch.setattr(controller, "restates_route",
                        lambda route, step: route == "S" and (step != "wrapped" or bool(judge.prompts)))
    report = _explained(tmp_path / "a", judge, layers=_wrapper_on_s(monkeypatch))
    (only,) = report["explanations"]["sets"]
    assert len(judge.prompts) == 1 and only["final"]["S.h1"]["state"] == "contradicted"
    assert set(only["offered"]["S"]) == {"wrapped"}


def test_a_failed_judge_call_leaves_D_at_zero_and_the_run_completes(tmp_path):
    report = _explained(tmp_path / "f", _Judge(fail=True))
    sets = report["explanations"]["sets"]
    assert sets and all(s["source"] == "failed" and "timed out" in s["error"] for s in sets)
    assert all(c["D"] == 0.0 for d in report["decisions"] for c in d.get("candidates") or [])
    assert report["cost"]["judge"]["n_failed"] == len(sets) and report["stop"]["reason"]


def test_the_rules_scheduler_does_not_take_an_explainer(tmp_path):
    with pytest.raises(ValueError, match="scored scheduler"):
        run_triage(_Table(_pope_like()[1]), _pope_like()[0],
                   **_kw(tmp_path / "r", explainer=Explainer(_Judge(), identity={})))


def test_compare_applies_the_explaining_runs_explanations_to_both_runs_evidence(tmp_path, capsys):
    from evalrx.eval_agent.probing.compare import compare, main, markdown

    out = tmp_path / "c"
    rules = run_triage(_Table(_pope_like()[1]), _pope_like()[0], **_kw(out, max_fail=8, min_route_fail=1))
    explained = _explained(out, _Judge())
    diff = compare(rules, explained)
    d = diff["discrimination"]
    assert d["baseline"]["revision"] == d["other"]["revision"] == explained["explanations"]["sets"][-1]["revision"]
    assert d["other"]["n_conflicting_pairs"] >= 1 and d["baseline"]["n_conflicting_pairs"] == d["other"][
        "n_conflicting_pairs"]
    assert diff["judge"]["other"]["n_calls"] == 1 and "tells apart" in markdown(diff)
    json.dumps(diff)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(rules, default=str)), b.write_text(json.dumps(explained, default=str))
    assert main([str(a), str(b)]) == 0 and "tells apart" in capsys.readouterr().out
    assert main([str(a), str(b), "--json"]) == 0 and json.loads(capsys.readouterr().out)["same_sample"] is True


@pytest.mark.parametrize("scheduler", ["rules", "scored"])
def test_a_budgeted_rerun_on_the_same_ledger_decides_as_the_run_that_executed(tmp_path, monkeypatch, scheduler):
    import evalrx.eval_agent.probing.controller as controller

    real = controller.execute

    def metered(action, *args, **kwargs):   # a second per case, so the budget binds the same way every run
        obs = real(action, *args, **kwargs)
        obs.cost["wall_sec"] = float(len(obs.planned_ids))
        return obs

    monkeypatch.setattr(controller, "execute", metered)
    cases, table = _pope_like()
    kw = dict(max_fail=8, min_route_fail=1, scheduler=scheduler)
    full = run_triage(_Table(table), cases, **_kw(tmp_path / "full", **kw))
    budget = round(full["cost"]["charged"]["wall_sec"] * 0.8)
    first = run_triage(_Table(table), cases, **_kw(tmp_path / "b", budget_sec=budget, **kw))
    assert any(e["status"] == "skipped" for e in first["executions"])   # the budget bound
    assert first["cost"]["charged"]["wall_sec"] < full["cost"]["charged"]["wall_sec"]
    # resumed on its own ledger, and run on the ledger of the unbudgeted run (which holds
    # evidence the budget excludes): the same steps, skips and charge, without a model call
    for out in ("b", "full"):
        model = _Table(table)
        again = run_triage(model, cases, **_kw(tmp_path / out, budget_sec=budget, **kw))
        assert model.n_generate == 0 and again["stop"] == first["stop"]
        assert [(e["key"], e["status"], e.get("n_cases")) for e in again["executions"]] == [
            (e["key"], e["status"], e.get("n_cases")) for e in first["executions"]]
        assert again["cost"]["charged"] == first["cost"]["charged"]


class _SlowExplainer(Explainer):
    """Every judge call is charged 100 seconds."""

    def propose(self, root, revision, context, offered):
        es = super().propose(root, revision, context, offered)
        es.cost = {**es.cost, "wall_sec": 100.0}
        return es


def test_a_judge_call_is_asked_only_while_its_estimate_fits_the_budget(tmp_path, monkeypatch):
    from evalrx.eval_agent.probing import LAYERS
    from evalrx.eval_agent.probing.actions import PlanStep
    from evalrx.eval_agent.probing.rules import Layer

    # the re-ask scenario: two asks when nothing binds; the evidence costs milliseconds
    monkeypatch.setitem(CARDS, "wrapped", (Metric("prompt_sensitivity", "the answer moves with a wrapper", 0.1),))
    step = PlanStep("prompt_contrast", {"strategies": {"baseline": "{prompt}", "w": "Q: {prompt}"}}, "Z1",
                    "re-ask inside a wrapper", key="wrapped")
    at = next(i for i, x in enumerate(LAYERS) if x.rule_id == "Sb")
    layers = (*LAYERS[:at], Layer("Z1", "S", "does a wrapper move it", (step,)), *LAYERS[at:])

    def run(out, budget):
        explainer = _SlowExplainer(_Judge(primary={"wrapped"}), identity={"provider": "stub", "model": "slow"})
        return run_triage(_Table(_pope_like()[1]), _pope_like()[0],
                          **_kw(tmp_path / out, max_fail=8, min_route_fail=1, scheduler="scored", layers=layers,
                                explainer=explainer, budget_sec=budget))

    assert len(run("full", 0.0)["explanations"]["sets"]) == 2
    for budget, asks in ((50.0, 1), (150.0, 1), (250.0, 2)):
        r = run(f"b{budget}", budget)
        assert len(r["explanations"]["sets"]) == asks, budget
        # the first call's cost is unknown, so it may overshoot by one call; a later one is not
        # made unless the largest call so far fits
        assert r["cost"]["charged"]["wall_sec"] - budget <= 100.0 if asks == 1 else r["cost"]["charged"]["wall_sec"] <= budget


class _Wavering(_Table):
    """A wrong answer flips inside a neutral wrapper: those FAIL cases are on route U."""

    def generate(self, inputs, **kwargs):
        ans = super().generate(inputs, **kwargs)
        wrapped = str(inputs.prompt).startswith(("Question:", "Answer the following question"))
        wrong = self.wrong.get((next((q for q, img in self.table if q in str(inputs.prompt)), None), inputs.image))
        return ("Yes" if ans == "No" else "No") if wrapped and wrong else ans


def _wavering(out):
    cases, table = _pope_like()
    model = _Wavering(table)
    model.wrong = {(c.inputs.prompt, c.inputs.image): c.label.value == "fail" for c in cases}
    judge = _Judge()
    report = run_triage(model, cases, **_kw(out, max_fail=8, min_route_fail=1, scheduler="scored",
                                            explainer=Explainer(judge, identity={"provider": "stub", "model": "stub"})))
    return report, judge


def _offered_on(out):
    _, judge = _wavering(out)
    asked: dict[str, set] = {}
    for p in judge.prompts:
        for route, r in json.loads(p[p.index("Evidence (JSON):") + 16:])["routes"].items():
            asked.setdefault(route, set()).update(c["step"] for c in r["to_run"] + r["already_run"])
    return asked


def test_the_judge_is_not_asked_to_predict_the_check_a_route_is_cut_on(tmp_path, monkeypatch):
    import evalrx.eval_agent.probing.controller as controller

    with monkeypatch.context() as m:
        m.setattr(controller, "restates_route", lambda route, step: False)
        assert "self_consistency_n10" in _offered_on(tmp_path / "unfiltered")["U"]   # U1's candidate here
    # filtered, U has nothing left to run here, and is explained on the tools already run on it
    assert "self_consistency_n10" not in _offered_on(tmp_path / "filtered").get("U", set())


def test_a_route_with_nothing_left_to_run_is_explained_on_the_tools_already_run(tmp_path):
    from evalrx.eval_agent.probing.explanations import restates_route

    report, judge = _wavering(tmp_path / "w")
    (prompt,) = judge.prompts
    u = json.loads(prompt[prompt.index("Evidence (JSON):") + 16:])["routes"]["U"]
    assert not u["to_run"] and u["already_run"]
    assert {c["step"] for c in u["already_run"]} == {k for k in u["measured"] if not restates_route("U", k)}
    (asked,) = report["explanations"]["sets"]
    explained = [h for h in asked["hypotheses"] if h["route"] == "U"]
    # kept, standing on readings the judge saw: a fit here, a test only on new cases
    assert len(explained) == 2 and all(step in asked["seen"]["U"] for h in explained for step, _ in h["primary"])
    # nothing on U is run again, so they give no D
    assert not any(c["route"] == "U" and c["D"] > 0 for d in report["decisions"] for c in d.get("candidates") or [])


def test_a_new_ask_keeps_the_live_explanations_and_adds_only_competing_ones():
    from evalrx.eval_agent.probing.explanations import carry_over

    on = (("gradcam", "gc_top5_share"),)
    kept = [Hypothesis("S.h2", "S", "flat", {"gradcam": {"gc_top5_share": "same"}}, primary=on)]
    new = [Hypothesis("S.h1", "S", "peaked", {"gradcam": {"gc_top5_share": "up"}}, primary=on),
           Hypothesis("S.h2", "S", "flat again", {"gradcam": {"gc_top5_share": "same"}}, primary=on),
           Hypothesis("N.h1", "N", "another route", {"gradcam": {"gc_top5_share": "same"}}, primary=on)]
    out, dropped = carry_over(kept, new, taken={"S.h1", "S.h2"})
    # S.h1 was an earlier explanation's id: the new one is renumbered; another route is not compared
    assert [(h.id, h.claim) for h in out] == [("S.h2", "flat"), ("S.h3", "peaked"), ("N.h1", "another route")]
    assert dropped == [{"what": "S.h2", "reason": "cannot be told apart from S.h2 (kept from the last ask) on a "
                                                  "primary prediction"}]
