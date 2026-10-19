"""MVP-2 triage: nested expansion with per-case reuse, and follow-ups chosen by fixed rules.

* expanding 4 -> 8 FAIL cases gives every case the same record as drawing 8 at
  once, for the same model calls: only the new cases run;
* the per-case rows of every ``reuse`` step do not depend on the batch;
* a rerun with a larger cap reads the earlier chunks back and extends the
  follow-ups only to the new cases;
* a failed or refused candidate falls back within its layer, a valid one
  advances to the next layer, and a refusal on any chunk refuses the step;
* the expansion rule fires on an uncertain routing signal or a small S
  cohort, and stops at the cap.
"""

from __future__ import annotations

import pytest

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent.probing import BATTERY, LAYERS, run_triage
from evalrx.eval_agent.probing.actions import FAILED, OK, UNFAITHFUL, Observation
from evalrx.eval_agent.probing.evidence import StepEvidence
from evalrx.eval_agent.probing.rules import (
    active_routes,
    candidate_steps,
    expansion_gaps,
    next_for_route,
    next_size,
)
from evalrx.eval_agent.probing.triage import difference_interval

OBJECTS = ("cat", "dog", "car", "cup")


def _pope_like(n_images: int = 24):
    """POPE-like: each image is asked about one present (Yes) and one absent (No) object, and
    every question recurs across images; the model answers wrong on 10 of the 48 cases."""
    cases, table = [], {}
    for i in range(n_images):
        img = f"img{i}"
        for j, (obj, gold) in enumerate(((OBJECTS[i % 4], "Yes"), (OBJECTS[(i + 1) % 4], "No"))):
            q = f"Is there a {obj} in the image?"
            wrong = (7 * i + j) % 5 == 0
            ans = ("No" if gold == "Yes" else "Yes") if wrong else gold
            table[(q, img)] = ans
            cases.append(FailureCase(id=f"c{i}-{j}", inputs=Inputs(prompt=q, image=img), expected=gold,
                                     observed=ans, label=Label.FAIL if wrong else Label.PASS,
                                     metadata={"gold": gold, "finish_reason": "stop"}))
    return CaseBatch(cases), table


class _Table(Model):
    """A black box that answers from a table keyed by (question, image), whatever wraps the
    question: deterministic, so its answers cannot depend on the batch."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self, table):
        self.table, self.n_generate = table, 0

    def generate(self, inputs, **kwargs):
        self.n_generate += 1
        for (q, img), ans in self.table.items():
            if q in str(inputs.prompt) and img == inputs.image:
                return ans
        return "No"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _yes_no(case, text):
    t = str(text).strip().lower()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else ""


def _kw(out, **over):
    kw = dict(out_dir=out, model_identity={"spec": "table"}, n_fail=4, seed=0, host_assumed_finish=True,
              n_needs_counterfactual=True, task_kind="yes_no", answer_fn=_yes_no, min_cohort=100,
              analyzer_kwargs={"self_consistency": {"gen_kwargs": {"temperature": 1.0}}}, progress=None,
              explore_share=0.0)   # the scored scheduler explores only in the tests about exploration
    return {**kw, **over}


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "action_id"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _generated(report, kinds=("battery", "expansion")):
    return sum((e.get("cost") or {}).get("calls", {}).get("generate", 0) for e in report["executions"]
               if e["kind"] in kinds and not e["reused"])


def test_expansion_runs_only_new_cases_and_matches_drawing_the_larger_sample_at_once(tmp_path):
    cases, table = _pope_like()
    grown_model, direct_model = _Table(table), _Table(table)
    grown = run_triage(grown_model, cases, **_kw(tmp_path / "grown", max_fail=8, max_actions=1))
    direct = run_triage(direct_model, cases, **_kw(tmp_path / "direct", n_fail=8, max_actions=0))

    (expand,) = [d for d in grown["decisions"] if d["kind"] == "expand"]
    assert (expand["from_fail"], expand["to_fail"]) == (4, 8)
    assert {g["gap"] for g in expand["gaps"]} == {"s_cohort_small"}
    assert grown["sample"]["case_ids"] == direct["sample"]["case_ids"]
    n_new = expand["n_new_cases"]
    for e in grown["executions"]:
        if e["kind"] == "expansion" and e["status"] != "not_applicable":
            assert next(s for s in BATTERY if s.name == e["key"]).reuse and e["n_cases"] == n_new
    # the batch-dependent steps the routes do not read run once, on the settled sample
    # the descriptive steps no route reads wait for the settled sample, then run on it once (in chunks)
    for key in ("embed_lens", "patchscope"):
        runs = [e for e in grown["executions"] if e["key"] == key]
        assert {e["revision"] for e in runs} == {1}
        assert sum(e["n_cases"] for e in runs) == len(direct["sample"]["case_ids"])
    # every case reads the same as when the 8-FAIL sample is drawn at once, for the same calls
    assert {r["case_id"]: _strip(r) for r in grown["cases"]} == {r["case_id"]: _strip(r) for r in direct["cases"]}
    assert _generated(grown) == _generated(direct) == grown_model.n_generate == direct_model.n_generate
    checks = grown["checks"]
    assert all(checks["reused_chunks_disjoint"].values()) and all(checks["battery_covers_sample"].values())
    # the cap is reached with the gap still open: recorded, not retried; the next spend is refused
    assert [d["kind"] for d in grown["decisions"]] == ["expand", "expand_not_possible"]
    assert grown["expansion"]["state"] == "not_possible" and "cap of 8" in grown["expansion"]["reason"]
    assert grown["stop"]["reason"] == "max_actions" and grown["stop"]["pending"]
    assert [r["revision"] for r in grown["revisions"]] == [0, 1]


def test_rerun_reads_everything_back_and_a_larger_cap_extends_only_to_new_cases(tmp_path):
    cases, table = _pope_like()
    out = tmp_path / "p"
    small_model = _Table(table)
    small = run_triage(small_model, cases, **_kw(out, max_fail=4, min_route_fail=1))
    assert small["stop"]["reason"] == "no_candidate" and small["follow_ups"]
    first = {(d["route"], d["key"]) for d in small["decisions"] if d["kind"] == "follow_up"}
    assert ("N", "describe_first") in first or ("S", "describe_first") in first

    # same command: every action read back, the same decisions, no model call
    again_model = _Table(table)
    again = run_triage(again_model, cases, **_kw(out, max_fail=4, min_route_fail=1))
    assert again_model.n_generate == 0 and all(e["reused"] for e in again["executions"])
    assert [(d["kind"], d.get("key")) for d in again["decisions"]] == [(d["kind"], d.get("key"))
                                                                      for d in small["decisions"]]

    # a larger cap: the battery and the follow-ups run on the cases the larger sample adds, only
    big_model = _Table(table)
    big = run_triage(big_model, cases, **_kw(out, max_fail=8, min_route_fail=1))
    assert big["revisions"][-1]["n_fail"] == 8
    fresh = [e for e in big["executions"] if e["kind"] == "follow_up" and not e["reused"]]
    reread = [e for e in big["executions"] if e["kind"] == "follow_up" and e["reused"]]
    assert reread, "the smaller run's follow-up chunks are read back"
    new_ids = set(big["sample"]["case_ids"]) - set(small["sample"]["case_ids"])
    for fu in big["follow_ups"]:
        assert fu["not_run_on"] in (0, None)
    calls = sum(e["cost"]["calls"].get("generate", 0) for e in fresh if e.get("cost"))
    assert calls == big_model.n_generate - _generated(big)
    assert all(e["n_cases"] <= len(new_ids) for e in fresh if e["key"] == "describe_first")
    assert all(big["checks"]["reused_chunks_disjoint"].values())


def test_describe_first_asks_for_its_own_output_room_where_the_backend_honours_it(tmp_path):
    cases, table = _pope_like()
    ident = {"spec": "table", "generation": {"max_new_tokens": 16}}

    class _Recording(_Table):
        def generate(self, inputs, **kwargs):
            self.kwargs = [*getattr(self, "kwargs", []), kwargs]
            return super().generate(inputs, **kwargs)

    # per-call settings honoured: the describe step asks for 256 tokens and runs
    model = _Recording(table)
    report = run_triage(model, cases, **_kw(tmp_path / "room", max_fail=4, min_route_fail=1, model_identity=ident))
    keys = [d["key"] for d in report["decisions"] if d["kind"] == "follow_up"]
    assert "describe_first" in keys and {"max_new_tokens": 256} in model.kwargs
    assert all(report["checks"]["configs_match_plan"].values())

    # dropped per call (an OpenAI-compatible endpoint): the step runs without its
    # gen_kwargs, so the host's 16-token cap applies and leaves no room
    from evalrx.eval_agent.probing.controller import _Run
    from evalrx.eval_agent.probing.ledger import Ledger

    step, ids = _step("describe_first"), [str(c.id) for c in cases][:4]
    for per_call, expected in ((True, None), (False, "not_applicable")):
        run = _Run(_Table(table), cases, Ledger(tmp_path / f"g{per_call}"), context={"model": ident}, extras={},
                   task_kind="yes_no", budget_sec=0.0, price_fn=None, run_logger=None, say=lambda _m: None,
                   per_call_generation=per_call)
        why = run.gate(step, ids)
        assert (why[0] if why else None) == expected
        assert ("gen_kwargs" in run.config(step)) is per_call
    assert "output cap of 16" in why[1]


def test_max_actions_zero_runs_only_the_battery(tmp_path):
    cases, table = _pope_like()
    report = run_triage(_Table(table), cases, **_kw(tmp_path / "z", max_fail=8, max_actions=0))
    assert report["decisions"] == [] and report["stop"] == {"reason": "max_actions", "max_actions": 0,
                                                            "pending": "expansion"}
    assert {e["kind"] for e in report["executions"]} == {"battery"}


# -- rules ---------------------------------------------------------------------------


def _obs(step_key, ids, status=OK, rows=True):
    return Observation(action_id=f"{step_key}-{status}-{ids[0]}", analyzer=step_key, attempt=1, status=status,
                       params={}, planned_ids=list(ids),
                       measured_ids=list(ids) if status == OK else [],
                       findings={"per_case": [{"sample_id": i, "v": 1.0} for i in ids]} if status == OK and rows
                       else None)


def _records(n_fail_route=None):
    """Three N FAILs, two S FAILs (stratum yes), one H FAIL, and PASS controls in both strata."""
    rows = []
    for i, route in enumerate(n_fail_route or ["N", "N", "N", "S", "S", "H"]):
        rows.append({"case_id": f"f{i}", "label": "fail", "stratum": "yes" if i % 2 else "no",
                     "routes": [route], "stable": route == "S", "N": {"state": "clear"}})
    for i in range(8):
        rows.append({"case_id": f"p{i}", "label": "pass", "stratum": "yes" if i % 2 else "no",
                     "routes": [], "stable": True, "N": {"state": "clear"}})
    return rows


def _step(key):
    by = {s.name: s for s in [*BATTERY, *candidate_steps(LAYERS)]}
    return by[key]


def _no_gate(step, ids):
    return None


def test_routes_go_h_first_then_by_failures_and_layers_fall_back_within_and_advance_between():
    records = _records()
    order = [r["route"] for r in active_routes(records, min_route_fail=1)]
    assert order == ["H", "N", "S"]
    assert [r["route"] for r in active_routes(records, min_route_fail=3)
            if r["active"]] == ["N"]

    n_ids = ["f0", "f1", "f2"]
    ev = {"attention_knockout": StepEvidence(_step("attention_knockout"), [_obs("attention_knockout", n_ids)])}
    # N1 is served by the battery's knockout; N2's representative is next
    p = next_for_route("N", records, ev, _no_gate, str)
    assert (p.layer.rule_id, p.step.name, p.kind) == ("N2", "attention_temperature", "open")
    assert sorted(p.fail_ids) == n_ids and len(p.control_ids) == 3  # matched per stratum

    # a refused representative moves to its fallback in the SAME layer
    target = [*p.fail_ids, *p.control_ids]
    ev["attention_temperature"] = StepEvidence(_step("attention_temperature"),
                                               [_obs("attention_temperature", target, status=UNFAITHFUL)])
    p = next_for_route("N", records, ev, _no_gate, str)
    assert (p.layer.rule_id, p.step.name) == ("N2", "attention_sensitivity")
    assert p.tried == [{"key": "attention_temperature", "status": "unfaithful", "rule_id": "N2"}]

    # a valid observation (whatever it shows) completes the layer: on to N3
    ev["attention_sensitivity"] = StepEvidence(_step("attention_sensitivity"), [_obs("attention_sensitivity", target)])
    p = next_for_route("N", records, ev, _no_gate, str)
    assert (p.layer.rule_id, p.step.name) == ("N3", "describe_first")

    # when the knockout failed, N1's own fallback comes first; gated-off candidates are passed over
    ev["attention_knockout"] = StepEvidence(_step("attention_knockout"),
                                            [_obs("attention_knockout", n_ids, status=FAILED)])

    def gate(step, ids):
        return ("unsupported", "model lacks capabilities ['hidden_states']") if step.analyzer == "causal_trace" else None

    p = next_for_route("N", records, ev, gate, str)
    assert (p.layer.rule_id, p.step.name) == ("N3", "describe_first")
    assert [(t["rule_id"], t["key"], t["status"]) for t in p.tried] == [
        ("N1", "attention_knockout", "failed"), ("N1", "causal_trace", "unsupported")]


def test_a_layer_done_on_fewer_cases_is_extended_to_the_routes_new_cases_first():
    records = _records()
    s_ids = ["f3", "f4"]
    ev = {"describe_first": StepEvidence(_step("describe_first"), [_obs("describe_first", ["f3", "p1"])])}
    p = next_for_route("S", records, ev, _no_gate, str)
    assert (p.layer.rule_id, p.kind) == ("Sb", "extend")
    assert sorted(p.todo) == sorted(set([*s_ids, *p.control_ids]) - {"f3", "p1"})
    assert sorted(p.fail_ids) == s_ids


def test_a_refusal_on_any_chunk_refuses_the_step_on_every_case():
    step = _step("attention_knockout")
    ev = StepEvidence(step, [_obs("ko", ["a", "b"]), _obs("ko", ["c"], status=UNFAITHFUL)])
    assert ev.for_case("a").status == UNFAITHFUL and not ev.valid_on(["a", "b"])
    plain = StepEvidence(_step("self_consistency"), [_obs("sc", ["a", "b"]), _obs("sc", ["c"], status=FAILED)])
    assert plain.for_case("a").status == OK and plain.for_case("c").status == FAILED
    assert plain.todo(["a", "b", "c", "d"]) == ["d"]
    from evalrx.eval_agent.probing.actions import PlanStep

    batchy = StepEvidence(PlanStep("x", {}, "S", "reads the batch", reuse=False, reuse_note="test"),
                          [_obs("el", ["a", "b"])])
    assert batchy.todo(["a", "b"]) == [] and batchy.todo(["a", "b", "c"]) == ["a", "b", "c"]


def test_expansion_gaps_and_sizes():
    lo, hi = difference_interval(3, 8, 0, 8)
    assert lo < 0 < hi                      # 3/8 vs 0/8: not sure the difference is not zero
    assert difference_interval(13, 32, 1, 32)[0] > 0
    assert difference_interval(1, 0, 0, 4) is None

    def summary(k_fail, k_pass, n):
        rate = {"fail": {"n_fired": k_fail, "n_tested": n}, "pass": {"n_fired": k_pass, "n_tested": n}}
        d = (k_fail - k_pass) / n
        return {"U": {**rate, "routing_signal": max(0.0, d), "signal_ci95": difference_interval(k_fail, n, k_pass, n)},
                "N": {**rate, "routing_signal": 0.0, "signal_ci95": None},
                "H_label_free": {**rate, "routing_signal": None, "signal_ci95": None}}

    records = _records()
    assert [g["route"] for g in expansion_gaps(summary(3, 0, 8), records, min_cohort=1)
            if g["gap"] == "signal_uncertain"] == ["U"]
    assert not expansion_gaps(summary(13, 1, 32), records, min_cohort=1)
    cohort = expansion_gaps(summary(13, 1, 32), records, min_cohort=8)
    assert cohort == [{"gap": "s_cohort_small", "stratum": "yes", "n_S": 1, "n_stable_pass": 4, "min_cohort": 8},
                      {"gap": "s_cohort_small", "stratum": "no", "n_S": 1, "n_stable_pass": 4, "min_cohort": 8}][::-1]
    assert next_size(8, 32, 100) == 16 and next_size(16, 32, 20) == 20 and next_size(32, 32, 100) is None


def test_budget_estimates_come_from_the_same_steps_earlier_seconds_and_charge_the_evidence(tmp_path):
    from evalrx.eval_agent.probing.controller import _Run
    from evalrx.eval_agent.probing.ledger import Ledger

    cases, table = _pope_like()
    ledger = Ledger(tmp_path / "b")
    run = _Run(_Table(table), cases, ledger, context={}, extras={}, task_kind="yes_no", budget_sec=10.0,
               price_fn=None, run_logger=None, say=lambda _m: None)
    step = _step("self_consistency")
    obs = _obs("sc", ["c0-0", "c1-0"])
    obs.cost = {"wall_sec": 4.0, "calls": {"generate": 10}}
    run.evidence[step.name] = StepEvidence(step, [obs])
    assert run.estimate_sec(step, 4) == 8.0
    assert "exceeds the 6.0s left" in run.over_budget(8.0)
    assert run.over_budget(5.0) is None
    # what another invocation spent on the same ledger is not this run's evidence
    ledger.cost({"kind": "action", "wall_sec": 70.0})
    assert run.over_budget(5.0) is None
    other = _obs("sc2", ["c2-0"])
    other.cost = {"wall_sec": 7.0, "calls": {"generate": 5}}
    run.evidence[step.name].add(other)
    assert run.charged("wall_sec") == 11.0 and "spent" in run.over_budget(None)
    run.evidence[step.name].add(other)   # one observation read twice is charged once
    assert run.charged("wall_sec") == 11.0
    run.judge_cost["wall_sec"] = 2.0
    assert run.charged("wall_sec") == 13.0


# -- the reuse declarations hold --------------------------------------------------------

#: Steps declared ``reuse`` whose rows these tests show do not depend on the batch.
#: ``per_case`` steps run once per case by construction.
_CHECKED = {"answer_extraction_audit", "termination_audit", "modality_ablation", "prompt_contrast",
            "attention_knockout", "attention_temperature", "attention_sensitivity", "gradcam",
            "generic_attention", "layer_contrast", "format_sensitivity",
            "embed_lens", "patchscope", "tuned_lens", "causal_trace"}


def test_every_reuse_step_is_covered_by_a_decomposability_check():
    steps = [*BATTERY, *candidate_steps(LAYERS)]
    assert {s.analyzer for s in steps if s.reuse and not s.per_case} <= _CHECKED
    for s in steps:
        if not s.reuse:
            assert s.reuse_note, s.name


def _rows(result):
    from evalrx.eval_agent.probing.actions import canonical

    findings = canonical(result.findings or {})
    rows = {str(r["sample_id"]): r for r in findings.get("per_case") or []}
    outputs = ((result.artifacts or {}).get("outputs") or {}) if hasattr(result, "artifacts") else {}
    return rows, canonical(outputs)


def _split_equals_whole(analyzer, model, batch, cut=3):
    whole_rows, whole_out = _rows(analyzer.run(model, batch))
    a_rows, a_out = _rows(analyzer.run(model, CaseBatch(list(batch)[:cut])))
    b_rows, b_out = _rows(analyzer.run(model, CaseBatch(list(batch)[cut:])))
    merged_out = {s: {**a_out.get(s, {}), **b_out.get(s, {})} for s in set(a_out) | set(b_out)}
    return whole_rows == {**a_rows, **b_rows} and whole_out == merged_out, whole_rows


@pytest.fixture()
def vlm_cases():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.tiny_whitebox import image

    layout = [("cat", "Yes"), ("dog", "No"), ("cat", "No"), ("dog", "Yes"),
              ("cat", "Yes"), ("dog", "No"), ("cat", "No"), ("dog", "Yes")]
    imgs = [image(i) for i in range(4)]
    out = []
    for i, (obj, gold) in enumerate(layout):
        fail = i in (1, 5)
        out.append(FailureCase(
            id=f"pope-{i}", inputs=Inputs(prompt=f"Is there a {obj} in the image?", image=imgs[i // 2]),
            expected=gold, observed="Yes" if fail or gold == "Yes" else "No",
            label=Label.FAIL if fail else Label.PASS, metadata={"gold": gold, "finish_reason": "stop"}))
    return CaseBatch(out)


REF = "<the cases themselves, as a fixed reference>"


@pytest.mark.parametrize("name, config", [
    ("attention_knockout", {"target": "gold"}),
    ("attention_temperature", {"target": "gold"}),
    ("attention_sensitivity", {"target": "gold"}),
    ("gradcam", {"target": "gold"}),
    ("generic_attention", {"target": "gold"}),
    ("layer_contrast", {}),
    # with a fixed reference / fit set and per-case noise seeds (the triage passes EXPLORE)
    ("embed_lens", {"layer": 0.9, "reference": REF}),
    ("patchscope", {"target": "gold", "media_layers": 2, "reference": REF}),
    ("tuned_lens", {"steps": 10, "fit_cases": REF}),
    ("causal_trace", {"target": "gold", "min_effect": 0.0, "n_points": 3}),
])
def test_white_box_reuse_steps_give_the_same_rows_whatever_shares_the_batch(vlm_cases, name, config):
    from evalrx.core.registry import registry
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    config = {k: (vlm_cases if v == REF else v) for k, v in config.items()}
    same, rows = _split_equals_whole(registry.analyzers.get(name)(**config), model, vlm_cases)
    assert rows and same


def test_behavioural_reuse_steps_give_the_same_rows_whatever_shares_the_batch():
    from evalrx.core.registry import registry

    cases, table = _pope_like(6)
    model = _Table(table)
    for name, config in [("answer_extraction_audit", {"reask": False}), ("answer_extraction_audit", {"reask": True}),
                         ("termination_audit", {"continue_non_clean": True}),
                         ("modality_ablation", {"mode": "swap", "donor_rule": "counterfactual", "donor_pool": cases}),
                         ("modality_ablation", {"mode": "drop"}),
                         ("prompt_contrast", {"strategies": {"baseline": "{prompt}", "x": "Q: {prompt}"}})]:
        same, rows = _split_equals_whole(registry.analyzers.get(name)(**config), model, cases, cut=5)
        assert rows and same, (name, config)

    mc = CaseBatch([FailureCase(id=f"m{i}", inputs=Inputs(prompt=f"Which {i}?\nA. cat\nB. dog\nC. cup"),
                                expected="A", observed="A", label=Label.PASS) for i in range(5)])

    class _Letters(_Table):
        def generate(self, inputs, **kwargs):
            return "B" if "B. cat" in str(inputs.prompt) else "A"

    same, rows = _split_equals_whole(registry.analyzers.get("format_sensitivity")(), _Letters({}), mc, cut=2)
    assert rows and same
