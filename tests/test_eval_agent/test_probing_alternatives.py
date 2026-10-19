"""The scored scheduler's redundancy term and its exploration share.

* an answered layer offers its other candidates as alternatives: R = 0.5 when
  one gives the kind of evidence the tool that answered the layer gave, 0 when
  it gives another kind; an alternative closes no gap and covers no unit, so it
  only runs on a D above its R, and the judge is offered it;
* exploration spends at most its share of the evidence and draws only from its
  targets, the actions the value never takes (an inactive route's next
  candidate, another route's representative); what it started on a route is
  completed on value (T); ``explore_from="candidates"`` (ablation) draws from
  the candidates other than the best instead, alternatives included.
"""

from __future__ import annotations

import json

import pytest

import evalrx.eval_agent.probing.controller as controller
from evalrx.eval_agent.probing import LAYERS, run_triage
from evalrx.eval_agent.probing.actions import PlanStep
from evalrx.eval_agent.probing.explanations import CARDS, Explainer, Metric
from evalrx.eval_agent.probing.rules import Layer
from evalrx.eval_agent.probing.scoring import Weights
from tests.test_eval_agent.test_probing_mvp2 import _kw, _pope_like, _Table


def _wrapper(key, text, evidence):
    return PlanStep("prompt_contrast", {"strategies": {"baseline": "{prompt}", key: text + " {prompt}"}}, "Z1",
                    f"re-ask inside the wrapper {text!r}", key=key, evidence=evidence)


FIRST = _wrapper("wrap_a", "Q:", "wrapper")
SAME_KIND = _wrapper("wrap_b", "Ask:", "wrapper")
OTHER_KIND = _wrapper("wrap_c", "Please answer:", "another kind")


@pytest.fixture
def wrappers(monkeypatch):
    """An S layer with three candidates: the first answers it, the second gives the same kind
    of evidence, the third another kind. The table model ignores wrappers."""
    for step in (FIRST, SAME_KIND, OTHER_KIND):
        monkeypatch.setitem(CARDS, step.name, (Metric("prompt_sensitivity", "the answer moves with a wrapper", 0.1),))
    at = next(i for i, x in enumerate(LAYERS) if x.rule_id == "Sb")
    return (*LAYERS[:at], Layer("Z1", "S", "does a wrapper move it", (FIRST, SAME_KIND, OTHER_KIND)), *LAYERS[at:])


class _Judge:
    """Two explanations per route that agree on every step but the alternatives: on an
    alternative's first reading one predicts up, the other that it stays the same."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        ctx = json.loads(prompt[prompt.index("Evidence (JSON):") + len("Evidence (JSON):"):])
        routes = {}
        for route, r in ctx["routes"].items():
            routes[route] = [{"id": f"h{i}", "claim": f"claim {i}",
                              "predictions": {c["step"]: {c["readings"][0]["field"]: d if c.get("alternative_to")
                                                          else "same"} for c in r["to_run"]}}
                             for i, d in enumerate(("up", "same"), 1)]
        return json.dumps({"routes": routes})


class _Flipper(_Table):
    """The table model, except that inside the "Q:" wrapper it answers its wrong cases right: wrap_a's
    prompt_sensitivity is 1 on every FAIL case and 0 on every PASS case."""

    def __init__(self, table, cases):
        super().__init__(table)
        self.gold = {(c.inputs.prompt, c.inputs.image): c.expected for c in cases}

    def generate(self, inputs, **kwargs):
        if str(inputs.prompt).startswith("Q: "):
            key = (str(inputs.prompt)[3:], inputs.image)
            if key in self.gold:
                self.n_generate += 1
                return self.gold[key]
        return super().generate(inputs, **kwargs)


def _run(out, layers, judge=None, model=None, **over):
    explainer = Explainer(judge, identity={"provider": "stub", "model": "stub"}) if judge is not None else None
    return run_triage(model or _Table(_pope_like()[1]), _pope_like()[0],
                      **_kw(out, **{"max_fail": 8, "min_route_fail": 1, "scheduler": "scored", "layers": layers,
                                    "explainer": explainer, **over}))


def _ran(report):
    return {e["key"] for e in report["executions"] if e["status"] == "ok"}


def test_an_answered_layer_offers_alternatives_that_earn_nothing_without_D(tmp_path, wrappers):
    report = _run(tmp_path / "a", wrappers)
    rows = [c for d in [*report["decisions"], report["stop"]] for c in d.get("candidates") or []]
    alts = {c["key"]: c for c in rows if c.get("alternative_to")}
    assert set(alts) == {"wrap_b", "wrap_c"} and {c["alternative_to"] for c in alts.values()} == {"wrap_a"}
    assert alts["wrap_b"]["R"] > 0 and alts["wrap_c"]["R"] == 0.0     # same kind of evidence, another kind
    assert all(c["T"] == c["C"] == c["D"] == c["gain"] == 0.0 for c in alts.values())
    assert "wrap_a" in _ran(report) and not {"wrap_b", "wrap_c"} & _ran(report)


def test_an_alternative_runs_on_a_D_above_its_redundancy_and_the_judge_is_offered_it(tmp_path, wrappers):
    judge = _Judge()
    report = _run(tmp_path / "d", wrappers, judge)
    # asked while Z1 was open: its other candidates are offered as alternatives to its current one
    asked = [json.loads(p[p.index("Evidence (JSON):") + 16:])["routes"]["S"]["to_run"] for p in judge.prompts]
    offered = {c["step"]: c.get("alternative_to") for c in asked[0]}
    assert offered["wrap_a"] is None and offered["wrap_b"] == offered["wrap_c"] == "wrap_a"
    # an unmeasured up-vs-same conflict is at most half decidable: enough against R = 0, not against 0.5
    assert "wrap_c" in _ran(report) and "wrap_b" not in _ran(report)
    (taken,) = [d for d in report["decisions"] if d.get("key") == "wrap_c" and d["kind"] == "follow_up"]
    assert taken["alternative_to"] == "wrap_a" and "alternative to wrap_a" in taken["rationale"]
    assert ("S", "Z1", "wrap_c") in {(f["route"], f["rule_id"], f["key"]) for f in report["follow_ups"]}
    # without the redundancy term the same-kind alternative ties and runs first; its result
    # contradicts the "up" explanation, and then wrap_c has nothing left to tell apart
    no_r = _run(tmp_path / "r0", wrappers, _Judge(), weights=Weights(r=0.0))
    assert "wrap_b" in _ran(no_r) and "wrap_c" not in _ran(no_r)


@pytest.fixture
def metered(monkeypatch):
    """One second per case: the exploration share is checked on deterministic costs."""
    real = controller.execute

    def execute(action, *args, **kwargs):
        obs = real(action, *args, **kwargs)
        if obs.cost.get("calls"):
            obs.cost["wall_sec"] = float(len(obs.planned_ids))
        return obs

    monkeypatch.setattr(controller, "execute", execute)


def test_exploration_takes_its_targets_within_its_share_and_pays_for_completing_them(tmp_path, wrappers, metered):
    # one case a chunk, so that an explored step has chunks left on the route
    report = _run(tmp_path / "e", wrappers, explore_share=0.3, chunk=1)
    explored = [d for d in report["decisions"] if d.get("choice") == "explore"]
    assert explored and report["explore_share"] == 0.3 and report["explore_from"] == "targets"
    # within the share, but for the one chunk whose estimate fell short
    assert 0 < report["explored"]["wall_sec"] <= 0.3 * report["cost"]["evidence"]["wall_sec"] + 2.0
    for d in explored:   # a target, never a candidate the value could take, and never an alternative
        assert d["explore_target"] and "exploration of" in d["rationale"] and d["gain"] == 0.0
        assert (d["route"], d["key"]) not in {(c["route"], c["key"]) for c in d["candidates"]}
        assert d["key"] not in ("wrap_b", "wrap_c")
    # wrap_a is Z1's representative on S: on N it lies outside the route's method family. The
    # table model ignores wrappers, so it finds nothing, and completing it stays exploration
    on_n = [d for d in report["decisions"] if (d.get("route"), d.get("key")) == ("N", "wrap_a")]
    assert "method family" in on_n[0]["explore_target"]
    assert len(on_n) > 1 and all(d["choice"] == "explore" for d in on_n)
    assert all("no reading outside its band" in d["explore_target"] for d in on_n[1:])
    none = _run(tmp_path / "n", wrappers, explore_share=0.0, chunk=2)
    assert not [d for d in none["decisions"] if d.get("choice") == "explore"] and not none["explored"]


def test_an_exploration_that_finds_something_is_completed_on_value(tmp_path, monkeypatch, metered):
    monkeypatch.setitem(CARDS, FIRST.name, (Metric("prompt_sensitivity", "the answer moves with a wrapper", 0.1),))
    # wrap_a answers an N question; on S (six FAIL cases) it lies outside the route's method family
    at = next(i for i, x in enumerate(LAYERS) if x.rule_id == "N3")
    layers = (*LAYERS[:at], Layer("Z1", "N", "does a wrapper move it", (FIRST,)), *LAYERS[at:])
    cases, table = _pope_like()
    report = _run(tmp_path / "f", layers, model=_Flipper(table, cases), explore_share=0.3, chunk=1)
    on_s = [d for d in report["decisions"] if (d.get("route"), d.get("key")) == ("S", "wrap_a")]
    assert on_s[0]["choice"] == "explore" and "method family" in on_s[0]["explore_target"]
    # once two FAIL cases and two controls show the wrapper moving FAIL answers only, it graduates
    graduated = [d for d in on_s if d["choice"] == "best"]
    assert graduated and all(d["kind"] == "extend" for d in graduated)
    found = next(c for c in graduated[0]["candidates"] if (c["route"], c["key"]) == ("S", "wrap_a"))
    assert found["T"] > 0 and "found prompt_sensitivity up" in found["note"]
    assert report["outcomes"]["S"]["wrap_a"]["prompt_sensitivity"]["observed"] == "up"


def test_exploration_takes_a_route_too_small_to_be_active(tmp_path, wrappers, metered):
    report = _run(tmp_path / "i", wrappers, explore_share=0.3, chunk=2, min_route_fail=3)
    small = [r["route"] for r in report["routes"] if not r["active"]]
    assert small == ["N"]
    assert any(d.get("choice") == "explore" and d["route"] == "N" and "too small" in d["explore_target"]
               for d in report["decisions"])


def test_the_candidates_ablation_explores_the_other_candidates_alternatives_included(tmp_path, wrappers, metered):
    report = _run(tmp_path / "c", wrappers, explore_share=0.3, chunk=2, explore_from="candidates")
    explored = {d["key"] for d in report["decisions"] if d.get("choice") == "explore"}
    assert explored & {"wrap_b", "wrap_c"} and report["explore_from"] == "candidates"
    assert not any(d.get("explore_targets") for d in report["decisions"])
