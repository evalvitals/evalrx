"""MVP-0 triage controller (``--m1-policy triage_active``): the acceptance behaviours.

* the planned actions run exactly as planned (no backfill, cap or codegen);
* unsupported / failed / unfaithful outcomes carry a reason and never a zero;
* fidelity passed, failed, unchecked and not-applicable stay distinct;
* the Stage-0 baseline cost is separate from the actions' new calls;
* swap donors and the actual mode are traceable per case;
* a resumed run reads finished actions back instead of re-running them;
* route labels never turn a missing measurement into a negative one.
"""

from __future__ import annotations

import json

import pytest

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent.probing import explore_confirm_split, matched_sample, run_triage
from evalrx.eval_agent.probing.actions import (
    FAILED,
    OK,
    UNFAITHFUL,
    UNSUPPORTED,
    Observation,
    ProbeAction,
)
from evalrx.eval_agent.probing.executor import execute
from evalrx.eval_agent.probing.triage import summarize, triage_cases


def _cases(n_fail=6, n_pass=8, gold_fail="No"):
    out = []
    for i in range(n_fail):
        out.append(FailureCase(id=f"f{i}", inputs=Inputs(prompt=f"is there a cat {i}?"),
                               expected=gold_fail, observed="Yes", label=Label.FAIL,
                               metadata={"gold": gold_fail}))
    for i in range(n_pass):
        gold = "No" if i % 2 else "Yes"
        out.append(FailureCase(id=f"p{i}", inputs=Inputs(prompt=f"is there a dog {i}?"),
                               expected=gold, observed=gold, label=Label.PASS,
                               metadata={"gold": gold}))
    return CaseBatch(out)


# -- sampler ---------------------------------------------------------------------


def test_matched_sample_is_nested_matched_and_order_free():
    cases = _cases()
    small, d_small = matched_sample(cases, 2, seed=3)
    big, d_big = matched_sample(cases, 4, seed=3)
    assert set(d_small.fail_ids) <= set(d_big.fail_ids)
    assert set(d_small.pass_ids) <= set(d_big.pass_ids)
    # every FAIL has gold No, so every control does too
    gold = {c.id: c.metadata["gold"] for c in cases}
    assert {gold[i] for i in d_big.pass_ids} == {"No"}
    assert d_big.strata == {"no": {"fail": 4, "pass": 4, "pass_available": 4}}
    assert list(d_big.case_ids) == [c.id for c in cases if c.id in set(d_big.case_ids)]
    # the draw depends on ids and seed, not on the input order
    _, d_rev = matched_sample(CaseBatch(list(cases)[::-1]), 4, seed=3)
    assert set(d_rev.case_ids) == set(d_big.case_ids)


def test_matched_sample_reports_shortfall_instead_of_borrowing():
    _, draw = matched_sample(_cases(n_fail=6), 6, seed=0)
    assert draw.strata["no"] == {"fail": 6, "pass": 4, "pass_available": 4}
    assert draw.shortfall == {"no": 2}
    assert len(draw.pass_ids) == 4  # no gold-Yes PASS filled in


def test_explore_split_is_the_loops_split():
    from evalrx.eval_agent.loop import VLDiagnoseLoop
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol

    cases = _cases(10, 14)
    loop = VLDiagnoseLoop(model=_BlackBox(), protocol=ExperimentProtocol(description="d"),
                          confirm_split=0.5, confirm_split_seed=20260818)
    want_e, want_c = loop._split_explore_confirm(cases)
    got_e, got_c = explore_confirm_split(cases, 0.5, 20260818)
    assert [c.id for c in got_e] == [c.id for c in want_e]
    assert [c.id for c in got_c] == [c.id for c in want_c]
    by_case = explore_confirm_split(cases, 0.5, 20260818, by="case")
    assert [c.id for c in by_case[0]] == [c.id for c in got_e] and [c.id for c in by_case[1]] == [c.id for c in got_c]
    with pytest.raises(ValueError):
        explore_confirm_split(cases, 0.5, 20260818, by="chart")


# -- executor --------------------------------------------------------------------


class _BlackBox(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self):
        self.calls = 0

    def generate(self, inputs, **kwargs):
        self.calls += 1
        return "Yes"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _analyzer(behaviour, requires=frozenset()):
    from evalrx.analyzers._internals.attn_tap import FidelityError
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.result import Result

    class _A(Analyzer):
        name = "fake"

        def _run(self, model, cases):
            if behaviour == "raise":
                raise ValueError("boom")
            if behaviour == "unfaithful":
                raise FidelityError("edit untrustworthy", {"tap_faithful": False, "tap_max_logit_diff": 3.0})
            model.generate(cases[0].inputs)
            rows = [{"sample_id": c.id, "v": 1} for c in list(cases)[1:]]
            return Result(analyzer="fake", model=repr(model), cases=cases,
                          findings={"per_case": rows, "skip_reasons": {"no_media": 1}})

    _A.requires = requires
    return _A()


def _action(cases, name="fake"):
    return ProbeAction(analyzer=name, config={}, case_ids=tuple(c.id for c in cases), route="N", purpose="t")


def test_executor_types_every_outcome_and_never_writes_a_zero():
    cases = _cases(2, 2)
    model = _BlackBox()
    ok = execute(_action(cases), "a1", 1, _analyzer("ok"), model, cases)
    assert ok.status == OK and ok.cost["calls"] == {"generate": 1}
    assert ok.measured_ids == [c.id for c in list(cases)[1:]]
    assert set(ok.missing) == {cases[0].id} and "no_media" in ok.missing[cases[0].id]
    assert ok.fidelity == {"status": "not_applicable"}

    failed = execute(_action(cases), "a2", 1, _analyzer("raise"), model, cases)
    assert failed.status == FAILED and failed.findings is None and failed.error_type == "ValueError"

    unfaithful = execute(_action(cases), "a3", 1, _analyzer("unfaithful"), model, cases, fidelity_check="edit")
    assert unfaithful.status == UNFAITHFUL and unfaithful.findings is None
    assert unfaithful.fidelity["status"] == "failed" and unfaithful.fidelity["verdict"]["tap_max_logit_diff"] == 3.0

    before = model.calls
    gated = execute(_action(cases), "a4", 1, _analyzer("ok", frozenset({Capability.ATTENTION})),
                    model, cases, fidelity_check="edit")
    assert gated.status == UNSUPPORTED and "attention" in gated.error and model.calls == before
    assert gated.fidelity["status"] == "unchecked"


def test_executor_refuses_a_case_set_other_than_the_plan():
    cases = _cases(2, 2)
    with pytest.raises(ValueError, match="planned"):
        execute(_action(list(cases)[:2]), "a", 1, _analyzer("ok"), _BlackBox(), cases)


def test_fidelity_failures_are_typed_runtime_errors(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("transformers")
    from evalrx.analyzers._internals import edits
    from evalrx.analyzers._internals.attn_tap import FidelityError
    from tests.tiny_whitebox import TinyVLM, image

    monkeypatch.setattr(edits, "check_fidelity", lambda *a, **k: {"tap_faithful": False})
    with pytest.raises(FidelityError) as info:
        edits.require_edit_fidelity(TinyVLM("qwen2_5_vl"), Inputs(prompt="hi", image=image(0)), [3])
    assert isinstance(info.value, RuntimeError) and info.value.verdict["faithful"] is False


# -- triage mapping ------------------------------------------------------------------


def _obs(analyzer, rows, status=OK, params=None, error=None):
    return Observation(action_id=f"id-{analyzer}", analyzer=analyzer, attempt=1, status=status,
                       params=params or {}, planned_ids=[r["sample_id"] for r in rows],
                       measured_ids=[r["sample_id"] for r in rows],
                       findings={"per_case": rows} if status == OK else None, error=error)


def test_triage_never_reads_missing_as_negative():
    cases = CaseBatch([
        FailureCase(id="f", inputs=Inputs(prompt="q", image="f.png"), expected="No", observed="Yes",
                    label=Label.FAIL),
        FailureCase(id="p", inputs=Inputs(prompt="q", image="p.png"), expected="No", observed="No",
                    label=Label.PASS),
    ])
    aea = _obs("answer_extraction_audit", [
        {"sample_id": "f", "extraction_suspect": 1, "extraction_point_miss": 0, "output_truncated": 0, "gave_up": 0},
        {"sample_id": "p", "extraction_suspect": 0, "extraction_point_miss": 0, "output_truncated": 0, "gave_up": 0},
    ])
    ta = _obs("termination_audit", [
        {"sample_id": "f", "termination_class": "no_answer", "looks_truncated": 0, "gave_up": 0},
        {"sample_id": "p", "termination_class": "no_answer", "looks_truncated": 0, "gave_up": 0},
    ])
    swap = _obs("modality_ablation", [
        {"sample_id": "f", "ablation_base_answer": "Yes", "grounded_in_image": False,
         "ablation_image_is_swap": True, "ablation_image_donor": "p", "ablation_image_answer": "Yes"},
        {"sample_id": "p", "ablation_base_answer": "No", "grounded_in_image": False,
         "ablation_image_is_swap": False, "ablation_image_donor": None, "ablation_image_answer": "No"},
    ], params={"mode": "swap"})
    ko = _obs("attention_knockout", [], status=FAILED, error="no case could be analysed")
    records = triage_cases(cases, {"answer_extraction_audit": aea, "termination_audit": ta,
                                   "modality_ablation": swap, "attention_knockout": ko},
                           host_assumed_finish=True, answer_contract=False)
    f, p = records
    # extraction_suspect fires, but only the label-free H state feeds the routing signal
    assert f["H"]["state"] == "fired" and f["H"]["state_label_free"] == "clear"
    fields = {x["field"]: x for x in f["H"]["fields"] if x["analyzer"] == "answer_extraction_audit"}
    assert fields["output_truncated"]["state"] == "untested"  # host-set finish_reason
    term = [x for x in f["H"]["fields"] if x["field"] == "termination_class"][0]
    assert term["state"] == "not_applicable"
    # real swap, unchanged answer -> N; drop fallback -> N untested, not clear
    assert f["N"]["state"] == "fired" and f["N"]["slots"][0]["donor"] == "p"
    assert p["N"]["state"] == "untested" and p["routes"] == ["X"]
    assert any("N untested" in r for r in p["x_reasons"])
    assert f["knockout"]["state"] == "untested" and "no case" in f["knockout"]["reason"]
    s = summarize(records)
    assert s["N"]["pass"]["n_tested"] == 0 and s["N"]["routing_signal"] is None
    assert s["H_any_audit_only"]["fail"]["n_fired"] == 1
    # only the real swap counts, split by the unablated answer
    assert s["N_by_base_answer"]["yes"] == {"fail": {"n_tested": 1, "n_fired": 1},
                                            "pass": {"n_tested": 0, "n_fired": 0}}


def test_few_answer_tasks_test_n_only_with_counterfactual_donors():
    cases = CaseBatch([
        FailureCase(id="cf", inputs=Inputs(prompt="q", image="cf.png"), expected="No", observed="Yes",
                    label=Label.FAIL),
        FailureCase(id="ct", inputs=Inputs(prompt="q", image="ct.png"), expected="No", observed="No",
                    label=Label.PASS),
    ])
    swap = _obs("modality_ablation", [
        {"sample_id": "cf", "ablation_base_answer": "Yes", "grounded_in_image": True,
         "ablation_image_is_swap": True, "ablation_image_donor": "d1", "ablation_image_answer": "No",
         "ablation_image_donor_kind": "counterfactual", "ablation_image_donor_expected": "no"},
        {"sample_id": "ct", "ablation_base_answer": "No", "grounded_in_image": False,
         "ablation_image_is_swap": True, "ablation_image_donor": "d2", "ablation_image_answer": "No",
         "ablation_image_donor_kind": "content", "ablation_image_donor_expected": None},
    ], params={"mode": "swap"})
    few = triage_cases(cases, {"modality_ablation": swap}, host_assumed_finish=True,
                       answer_contract=False, n_needs_counterfactual=True)
    assert few[0]["N"]["state"] == "clear" and few[0]["N"]["slots"][0]["follows_donor"] is True
    assert few[1]["N"]["state"] == "untested" and "counterfactual" in few[1]["N"]["reason"]
    open_answer = triage_cases(cases, {"modality_ablation": swap}, host_assumed_finish=True,
                               answer_contract=False, n_needs_counterfactual=False)
    assert open_answer[1]["N"]["state"] == "fired" and "assumption" in open_answer[1]["N"]["slots"][0]
    donors = summarize(few)["N_donors"]
    assert donors["fail"]["counterfactual_followed_donor"] == 1 and donors["pass"]["content"] == 1


def _no_answer(case, output):
    """The answer key of a task whose outputs commit to no categorical answer (open captions)."""
    return None


def test_u_is_not_applicable_where_outputs_commit_to_no_answer_and_s_needs_h_and_n():
    from evalrx.eval_agent.probing.triage import commits_to_answers

    cases = CaseBatch([
        _yn_case("s", Label.FAIL, "Yes", "A cat on a sofa."),   # H and N clear: S without U
        _yn_case("p", Label.PASS, "Yes", "A dog on a rug."),    # a stable control
    ])
    obs = _clean_h_and_n([c.id for c in cases], {"s": "No", "p": "No"})
    assert not commits_to_answers(cases, _no_answer) and commits_to_answers(cases, None)
    kw = dict(host_assumed_finish=True, answer_contract=False, n_needs_counterfactual=False)
    recs = {r["case_id"]: r for r in triage_cases(cases, obs, answer_fn=_no_answer, **kw)}
    assert recs["s"]["U"]["state"] == "not_applicable"
    assert [(u["part"], u["state"]) for u in recs["s"]["U"]["parts"]] == [("sampling", "not_applicable"),
                                                                          ("reword", "not_applicable")]
    assert recs["s"]["routes"] == ["S"] and recs["s"]["stable"] and recs["p"]["stable"]
    # with an answer key and no U observations, U is untested and the case is X
    keyed = {r["case_id"]: r for r in triage_cases(cases, obs, **kw)}
    assert keyed["s"]["U"]["state"] == "untested" and keyed["s"]["routes"] == ["X"]


def test_open_answer_records_match_controls_without_the_gold_answer():
    from evalrx.eval_agent.probing.rules import matched_controls

    cases = CaseBatch([
        FailureCase(id="f1", inputs=Inputs(prompt="q1"), expected="1.577", observed="1.5", label=Label.FAIL),
        FailureCase(id="f2", inputs=Inputs(prompt="q2"), expected="Taiwan", observed="China", label=Label.FAIL),
        FailureCase(id="p1", inputs=Inputs(prompt="q3"), expected="47", observed="47", label=Label.PASS),
        FailureCase(id="p2", inputs=Inputs(prompt="q4"), expected="2019", observed="2019", label=Label.PASS),
    ])
    gold = triage_cases(cases, {}, host_assumed_finish=True, answer_contract=False)
    assert matched_controls(gold, ["f1", "f2"], str) == ([], {"1.577": 1, "taiwan": 1})
    # the sampler's rule for open answers: every PASS case can be a FAIL case's control
    none = triage_cases(cases, {}, host_assumed_finish=True, answer_contract=False, stratum="none")
    assert {r["stratum"] for r in none} == {"all"}
    assert matched_controls(none, ["f1", "f2"], str) == (["p1", "p2"], {})


def test_triage_truncation_counts_when_the_backend_reported_it():
    cases = CaseBatch([FailureCase(id="f", inputs=Inputs(prompt="q"), expected="1", observed="1 2 3",
                                   label=Label.FAIL, metadata={"finish_reason": "length"})])
    aea = _obs("answer_extraction_audit", [{"sample_id": "f", "output_truncated": 1, "gave_up": 0}])
    rec = triage_cases(cases, {"answer_extraction_audit": aea}, host_assumed_finish=False,
                       answer_contract=False)[0]
    trunc = [x for x in rec["H"]["fields"] if x["field"] == "output_truncated"][0]
    assert trunc["state"] == "fired" and trunc["provenance"] == "backend"
    assert rec["routes"] == ["H"] and rec["H"]["state_label_free"] == "fired"


def test_executor_runs_a_one_case_analyzer_once_per_case_and_keeps_artifacts():
    from evalrx.analyzers._internals.attn_tap import FidelityError
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.result import Result

    class _OneCase(Analyzer):  # like self_consistency: measures cases[0] only
        name = "one"

        def __init__(self, fail_on=None):
            super().__init__(fail_on=fail_on)

        def _run(self, model, cases):
            if cases[0].id == self.fail_on:
                raise FidelityError("bad", {"faithful": False})
            out = model.generate(cases[0].inputs)
            return Result(analyzer="one", model=repr(model), cases=cases, artifacts={"samples": [out], "big": 1},
                          findings={"n_seen": len(cases), "_caveat": "note"})

    cases = _cases(2, 1)
    model = _BlackBox()
    obs = execute(_action(cases, "one"), "a", 1, _OneCase(), model, cases, per_case=True,
                  keep_artifacts=("samples",))
    assert obs.status == OK and obs.cost["calls"] == {"generate": 3} and obs.measured_ids == [c.id for c in cases]
    row = obs.per_case()["f0"]
    assert row["n_seen"] == 1 and row["artifacts"] == {"samples": ["Yes"]} and "_caveat" not in row
    assert obs.findings["_caveat"] == "note" and obs.findings["per_case_mode"] is True
    refused = execute(_action(cases, "one"), "b", 1, _OneCase(fail_on="f1"), model, cases, per_case=True,
                      fidelity_check="patch")
    assert refused.status == UNFAITHFUL and refused.findings is None
    batch = execute(_action(cases, "one"), "c", 1, _OneCase(), model, cases, keep_artifacts=("samples",))
    assert batch.findings["artifacts"] == {"samples": ["Yes"]} and len(batch.measured_ids) == 0


def _clean_h_and_n(ids, gold_after_swap):
    """H rows that fire nothing, and a counterfactual swap each model followed."""
    aea = _obs("answer_extraction_audit", [
        {"sample_id": i, "extraction_suspect": 0, "extraction_point_miss": 0, "output_truncated": 0, "gave_up": 0}
        for i in ids])
    ta = _obs("termination_audit", [
        {"sample_id": i, "termination_class": "clean", "looks_truncated": 0, "gave_up": 0} for i in ids])
    swap = _obs("modality_ablation", [
        {"sample_id": i, "ablation_base_answer": "x", "grounded_in_image": True, "ablation_image_is_swap": True,
         "ablation_image_donor": "d", "ablation_image_answer": gold_after_swap[i],
         "ablation_image_donor_kind": "counterfactual", "ablation_image_donor_expected": gold_after_swap[i].lower()}
        for i in ids], params={"mode": "swap"})
    return {"answer_extraction_audit": aea, "termination_audit": ta, "modality_ablation": swap}


def _u_obs(samples, reworded, gen_kwargs):
    sc = _obs("self_consistency", [{"sample_id": i, "artifacts": {"samples": v}} for i, v in samples.items()],
              params={"gen_kwargs": gen_kwargs})
    outputs = {"baseline": {i: v[0] for i, v in reworded.items()},
               "question_frame": {i: v[1] for i, v in reworded.items()},
               "instruction_frame": {i: v[2] for i, v in reworded.items()}}
    rw = _obs("prompt_contrast", [{"sample_id": i} for i in reworded])
    rw.findings["artifacts"] = {"outputs": outputs}
    return {"self_consistency": sc, "prompt_reword": rw}


def _yes_no(case, text):
    from evalrx.benchmark.scoring import answer_key

    return answer_key("yes_no", text)


def _yn_case(cid, label, gold, observed):
    return FailureCase(id=cid, inputs=Inputs(prompt=f"Is there a thing in {cid}?", image=f"{cid}.png"),
                       expected=gold, observed=observed, label=label, metadata={"gold": gold})


def test_u_needs_both_parts_and_s_is_a_fully_checked_stable_failure():
    cases = CaseBatch([
        _yn_case("s", Label.FAIL, "Yes", "No"),    # every check clear -> S
        _yn_case("u", Label.FAIL, "Yes", "No"),    # most samples say Yes -> U
        _yn_case("m", Label.FAIL, "Yes", "No"),    # one sample says Yes: noise, the Stage-0 answer is modal -> S
        _yn_case("t", Label.FAIL, "Yes", "No"),    # a tie with the Stage-0 answer stays clear -> S
        _yn_case("w", Label.FAIL, "Yes", "No"),    # a wrapper flips it -> U
        _yn_case("i", Label.FAIL, "Yes", "No"),    # wrappers unreadable -> invalid controls -> X
        _yn_case("p", Label.PASS, "Yes", "Yes"),   # every check clear -> stable control, no route
    ])
    ids = [c.id for c in cases]
    obs = _clean_h_and_n(ids, {i: "No" if i == "p" else "Yes" for i in ids})
    obs.update(_u_obs(
        samples={"s": ["No"] * 5, "u": ["Yes", "Yes, there is", "No", "yes", "No."],
                 "m": ["No", "Yes, there is", "No", "no", "No."], "t": ["No", "Yes", "no", "yes", "I cannot tell"],
                 "w": ["No"] * 5, "i": ["No"] * 5, "p": ["Yes"] * 4 + ["I cannot tell"]},
        reworded={"s": ["No", "No.", "no"], "u": ["No", "No", "No"], "m": ["No", "No", "No"], "t": ["No", "No", "No"],
                  "w": ["No", "Yes", "No"], "i": ["No", "Maybe", "Hard to say"], "p": ["Yes", "Yes", "Yes"]},
        gen_kwargs={"temperature": 1.0}))
    rec = {r["case_id"]: r for r in triage_cases(cases, obs, host_assumed_finish=True, answer_contract=True,
                                                n_needs_counterfactual=True, answer_fn=_yes_no)}
    assert rec["s"]["routes"] == ["S"] and rec["s"]["stable"]
    sampling, reword = rec["u"]["U"]["parts"]
    assert sampling["state"] == "fired" and sampling["agreement"] == 0.4 and sampling["answers"] == {"yes": 3, "no": 2}
    assert rec["u"]["routes"] == ["U"]
    minority, tie = rec["m"]["U"]["parts"][0], rec["t"]["U"]["parts"][0]
    assert minority["state"] == "clear" and minority["agreement"] == 0.8 and minority["base_is_modal"]
    assert tie["state"] == "clear" and tie["answers"] == {"no": 2, "yes": 2}
    assert rec["m"]["routes"] == rec["t"]["routes"] == ["S"]
    assert rec["w"]["routes"] == ["U"] and rec["w"]["U"]["parts"][1]["changed"] == ["question_frame"]
    assert rec["i"]["routes"] == ["X"] and rec["i"]["U"]["parts"][1]["invalid_controls"] == [
        "instruction_frame", "question_frame"]
    assert any(r.startswith("U untested: reword") for r in rec["i"]["x_reasons"])
    # an unreadable sample is set aside, not counted as a different answer
    assert rec["p"]["U"]["parts"][0]["n_unreadable"] == 1 and rec["p"]["U"]["state"] == "clear"
    assert rec["p"]["routes"] == [] and rec["p"]["stable"]
    s = summarize(list(rec.values()))
    assert s["S"] == {"fail": 3, "stable_pass": 1} and s["X"] == {"fail": 1, "pass": 0}
    assert s["U"]["fail"] == {"n": 6, "n_tested": 5, "n_fired": 2, "rate": 0.4}

    # greedy sampling settings: the sampling part is untested, so nothing becomes S
    obs.update(_u_obs(samples={i: ["No"] * 5 for i in ids}, reworded={i: ["No"] * 3 for i in ids}, gen_kwargs={}))
    greedy = {r["case_id"]: r for r in triage_cases(cases, obs, host_assumed_finish=True, answer_contract=True,
                                                   answer_fn=_yes_no)}
    assert greedy["s"]["routes"] == ["X"] and "greedy" in greedy["s"]["U"]["reason"]
    # a case without media: N does not apply, so it does not block S
    text = CaseBatch([FailureCase(id="t", inputs=Inputs(prompt="2+2?"), expected="4", observed="5",
                                  label=Label.FAIL)])
    t_obs = {k: v for k, v in _clean_h_and_n(["t"], {"t": "5"}).items() if k != "modality_ablation"}
    t_obs.update(_u_obs(samples={"t": ["5"] * 5}, reworded={"t": ["5", "5", "5"]}, gen_kwargs={"do_sample": True}))
    t = triage_cases(text, t_obs, host_assumed_finish=True, answer_contract=True)[0]
    assert t["N"]["state"] == "not_applicable" and t["routes"] == ["S"]


def _num_case(cid, observed):
    return FailureCase(id=cid, inputs=Inputs(prompt=f"How much in {cid}?", image=f"{cid}.png"), expected=["12.5"],
                       observed=observed, label=Label.FAIL,
                       metadata={"task": "exact_or_numeric", "numeric_tolerance": 0.05})


def test_u_and_s_compare_answers_under_the_tasks_tolerance():
    """ChartQA grades 12.6 against 12.5 as right (5 %), so the triage must not call them two answers:
    the host's agreement rule (``agree_fn``) decides, and the samples are clustered by it."""
    from evalrx.benchmark import tasks
    from evalrx.eval_agent.probing.triage import _contrast_values

    cases = CaseBatch([_num_case("r", "100"), _num_case("p", "12.5"), _num_case("w", "12.4")])
    ids = [c.id for c in cases]
    obs = _clean_h_and_n(ids, {i: "7" for i in ids})
    obs.update(_u_obs(
        samples={"r": ["101", "102", "99", "250", "250"],    # roundings of the base vs a twice-sampled rival
                 "p": ["12.6", "13.5", "13.6", "13.7", "12.5"],  # a rival the model gives in three roundings
                 "w": ["12.4"] * 5},
        reworded={"r": ["100", "100", "100"], "p": ["12.5", "12.5", "12.5"], "w": ["12.5", "12.6", "13.5"]},
        gen_kwargs={"temperature": 1.0}))
    th = _obs("prompt_contrast", [{"sample_id": i} for i in ids])
    th.findings.update(artifacts={"outputs": {"baseline": {"w": "12.5"}, "sensitive": {"w": "12.6"}}},
                       by_strategy={"baseline": {"w": 0.0}, "sensitive": {"w": 0.0}})
    obs["prompt_threshold"] = th
    kw = dict(host_assumed_finish=True, answer_contract=True, stratum="none", answer_fn=tasks.answer_key)
    exact = {r["case_id"]: r for r in triage_cases(cases, obs, **kw)}
    within = {r["case_id"]: r for r in triage_cases(cases, obs, agree_fn=tasks.same_answer, **kw)}

    # r: by key the base was never sampled (fires); within 5 % three samples are the base, two the rival
    assert exact["r"]["U"]["parts"][0]["state"] == "fired" and exact["r"]["U"]["parts"][0]["agreement"] == 0.0
    r = within["r"]["U"]["parts"][0]
    assert r["state"] == "clear" and r["agreement"] == 0.6 and r["rivals"] == {"250": 2} and r["base_is_modal"]
    assert r["answers"] == {"101": 1, "102": 1, "99": 1, "250": 2}      # the exact counts stay on record
    # p: by key five different answers tie (clear); within 5 % the rival 13.5-13.7 outnumbers the base 3 to 2
    assert exact["p"]["U"]["parts"][0]["state"] == "clear"
    p = within["p"]["U"]["parts"][0]
    assert p["state"] == "fired" and p["agreement"] == 0.4 and p["rivals"] == {"13.7": 3}
    # w: a wrapper that moved the answer by less than the tolerance did not move it; the Stage-0 answer
    # 12.4 is the re-asked 12.5; the threshold instruction's 12.6 is no change
    assert exact["w"]["U"]["parts"][1]["changed"] == ["instruction_frame", "question_frame"]
    w = within["w"]["U"]["parts"][1]
    assert w["changed"] == ["instruction_frame"] and w["baseline_matches_stage0"]
    assert not exact["w"]["U"]["parts"][1]["baseline_matches_stage0"]
    assert exact["w"]["S_readings"]["threshold"]["changed"] and not within["w"]["S_readings"]["threshold"]["changed"]
    assert within["r"]["routes"] == ["S"] and within["p"]["routes"] == within["w"]["routes"] == ["U"]
    # the follow-up reading of a prompt_contrast step uses the same rule
    case_w = next(c for c in cases if c.id == "w")
    assert _contrast_values(th, case_w, tasks.answer_key, tasks.same_answer)["changed_sensitive"] == 0.0
    assert _contrast_values(th, case_w, tasks.answer_key, lambda c, a, b: a == b)["changed_sensitive"] == 1.0


def test_s_readings_are_summarised_per_cohort_and_stratum():
    cases = CaseBatch([
        _yn_case("s1", Label.FAIL, "Yes", "No"), _yn_case("s2", Label.FAIL, "Yes", "No"),
        _yn_case("p1", Label.PASS, "Yes", "Yes"), _yn_case("p2", Label.PASS, "No", "No"),
    ])
    ids = [c.id for c in cases]
    obs = _clean_h_and_n(ids, {"s1": "Yes", "s2": "Yes", "p1": "No", "p2": "Yes"})
    base = {"s1": "No", "s2": "No", "p1": "Yes", "p2": "No"}
    obs.update(_u_obs(samples={i: [base[i]] * 5 for i in ids}, reworded={i: [base[i]] * 3 for i in ids},
                      gen_kwargs={"temperature": 1.0}))
    th = _obs("prompt_contrast", [{"sample_id": i} for i in ids])
    after = {"s1": "Yes", "s2": "No", "p1": "Yes", "p2": "Yes"}   # fixes s1, breaks p2
    th.findings.update(
        artifacts={"outputs": {"baseline": base, "sensitive": after}},
        by_strategy={"baseline": {"s1": 0.0, "s2": 0.0, "p1": 1.0, "p2": 1.0},
                     "sensitive": {"s1": 1.0, "s2": 0.0, "p1": 1.0, "p2": 0.0}})
    obs["prompt_threshold"] = th
    obs["embed_lens"] = _obs("embed_lens", [
        {"sample_id": "s1", "el_query_word": "thing", "el_query_sim": 0.2},
        {"sample_id": "s2", "el_query_word": "thing", "el_query_sim": 0.4},
        {"sample_id": "p1", "el_query_word": "thing", "el_query_sim": 0.5}])
    obs["patchscope"] = _obs("patchscope", [
        {"sample_id": "s1", "ps_first_depth_frac": 0.6, "ps_peak_prob": 0.7, "ps_final_prob": 0.2,
         "ps_late_drop": 0.5, "ps_media_query_hits": 3},
        {"sample_id": "s2", "ps_first_depth_frac": 1.0, "ps_peak_prob": 0.1, "ps_final_prob": 0.05,
         "ps_late_drop": 0.05, "ps_media_query_hits": 0},
        {"sample_id": "p1", "ps_first_depth_frac": 0.5, "ps_peak_prob": 0.9, "ps_final_prob": 0.9,
         "ps_late_drop": 0.0}], params={"target": "gold"})
    rec = triage_cases(cases, obs, host_assumed_finish=True, answer_contract=True, n_needs_counterfactual=True,
                       answer_fn=_yes_no)
    by = {r["case_id"]: r for r in rec}
    assert [by[i]["routes"] for i in ids] == [["S"], ["S"], [], []]
    assert by["s1"]["S_readings"]["threshold"] == {
        "action_id": "id-prompt_contrast", "strategy": "sensitive", "base_answer": "no", "answer": "yes",
        "changed": True, "fixed": True, "broken": False, "state": "measured"}
    assert by["p2"]["S_readings"]["embed_lens"]["state"] == "untested"
    s = summarize(rec)
    groups = {(g["label"], g["stratum"]): g for g in s["S_threshold"]["groups"]}
    assert groups[("fail", "yes")]["n_fixed"] == 1 and groups[("pass", "no")]["n_broken"] == 1
    cohort = s["S_readings"]["S"]["yes"]
    assert cohort["n"] == 2 and cohort["embed_lens"] == {"n": 2, "mean_query_sim": 0.3}
    assert cohort["patchscope"]["n_target_first_before_last"] == 1
    assert cohort["patchscope"]["n_media_query_hit"] == 1 and cohort["patchscope"]["n_media_query_read"] == 2
    assert s["S_readings"]["stable_pass"]["yes"]["patchscope"]["n_media_query_read"] == 0


# -- controller end to end ---------------------------------------------------------------


@pytest.fixture()
def vlm_cases():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.tiny_whitebox import image

    # like POPE: four images, two adjacent questions each (one Yes, one No), and
    # the same question asked on several images with opposite golds
    layout = [("cat", "Yes"), ("dog", "No"), ("cat", "No"), ("dog", "Yes"),
              ("cat", "Yes"), ("dog", "No"), ("cat", "No"), ("dog", "Yes")]
    imgs = [image(i) for i in range(4)]
    out = []
    for i, (obj, gold) in enumerate(layout):
        fail = i in (1, 5)  # "dog" absent, answered Yes
        out.append(FailureCase(
            id=f"pope-{i}", inputs=Inputs(prompt=f"Is there a {obj} in the image?", image=imgs[i // 2]),
            expected=gold, observed="Yes" if fail or gold == "Yes" else "No",
            label=Label.FAIL if fail else Label.PASS,
            metadata={"gold": gold, "finish_reason": "stop"},
        ))
    return CaseBatch(out)


def _steps(report):
    """The battery's executions on the first sample, by step key."""
    return {e["key"]: e for e in report["executions"] if e["kind"] == "battery"}


def test_mvp0_end_to_end_and_resume(tmp_path, vlm_cases):
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    ident = {"spec": model.spec.key, "backend": "tiny"}
    kw = dict(out_dir=tmp_path / "probing", model_identity=ident, n_fail=2, seed=0,
              host_assumed_finish=True, n_needs_counterfactual=True, task_kind="yes_no",
              analyzer_kwargs={"self_consistency": {"gen_kwargs": {"temperature": 1.0}}},
              stage0_cost={"wall_sec": 1.0, "calls": {"generate": 8}}, progress=None)
    report = run_triage(model, vlm_cases, **kw)

    steps = _steps(report)
    assert [s["status"] for s in report["executions"]] == [OK] * 9  # 2 FAIL: no route proposes
    assert steps["answer_extraction_audit"]["cost"]["calls"] == {}  # Stage-0 outputs re-read
    assert steps["modality_ablation"]["cost"]["calls"] == {"generate": 8}  # 4 cases x (base + swap)
    assert steps["attention_knockout"]["fidelity"]["status"] == "passed"
    assert steps["termination_audit"]["fidelity"]["status"] == "not_applicable"
    assert steps["self_consistency"]["cost"]["calls"] == {"generate": 20}  # one call of 5 per case
    assert steps["prompt_reword"]["cost"]["calls"] == {"generate": 12}
    assert steps["prompt_threshold"]["cost"]["calls"] == {"generate": 8}
    assert steps["patchscope"]["fidelity"]["status"] == "passed"
    checks = report["checks"]
    assert checks["only_rule_actions"] and checks["stage0_cost_separate"]
    assert all(checks["battery_covers_sample"].values()) and all(checks["configs_match_plan"].values())
    assert all(checks["batch_dependent_read_from_current_sample"].values())
    assert report["stop"]["reason"] == "no_candidate" and report["decisions"] == []

    sample = report["sample"]
    assert len(sample["fail_ids"]) == 2 and sample["strata"]["no"]["pass"] == 2
    expected_donor = {"pope-1": "pope-3", "pope-5": "pope-7", "pope-2": "pope-4", "pope-6": "pope-0"}
    for rec in report["cases"]:
        slot = rec["N"]["slots"][0]
        assert slot["donor_kind"] == "counterfactual" and slot["donor"] == expected_donor[rec["case_id"]]
        # TinyVLM answers "x" whatever it sees: an answer that had to change did not
        assert rec["N"]["state"] == "fired" and slot["follows_donor"] is False
        assert rec["knockout"]["state"] == "measured" and rec["knockout"]["fidelity"] == "passed"
        # ... and not the Stage-0 answer either: re-sampling moved it; the wrappers did not
        sampling, reword = rec["U"]["parts"]
        assert sampling["state"] == "fired" and sampling["agreement"] == 0.0 and sampling["answers"] == {"x": 5}
        assert reword["state"] == "clear" and reword["baseline_matches_stage0"] is False
        assert rec["routes"] == ["N", "U"] and not rec["stable"]
        assert rec["S_readings"]["threshold"]["state"] == "measured"
        assert rec["S_readings"]["patchscope"]["state"] == "measured"
        assert rec["S_readings"]["patchscope"]["target"] == "prediction"
    assert report["summary"]["N_donors"]["fail"]["counterfactual"] == 2
    assert report["summary"]["S"] == {"fail": 0, "stable_pass": 0}

    out = tmp_path / "probing"
    costs = [json.loads(line) for line in (out / "cost_ledger.jsonl").read_text().splitlines()]
    assert [c["kind"] for c in costs] == ["stage0"] + ["action"] * 9
    assert costs[1]["baseline_reused"] is True and costs[3]["baseline_reused"] is False
    for name in ("split_manifest.json", "triage_report.json", "triage_report.md"):
        assert (out / name).is_file()

    # resume: same identities -> read back, no model calls, no new action cost
    # (only this invocation's Stage 0, which the caller did spend)
    n_forwards = model.n_forwards
    again = run_triage(model, vlm_cases, **kw)
    assert all(s["reused"] for s in again["executions"]) and model.n_forwards == n_forwards
    costs2 = [json.loads(line) for line in (out / "cost_ledger.jsonl").read_text().splitlines()]
    assert [c["kind"] for c in costs2[len(costs):]] == ["stage0"] and costs2[-1]["invocation"] == 2
    events = [json.loads(line) for line in (out / "selection_trace.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events].count("reused") == 9

    # a different model identity is a different action: nothing is reused
    other = run_triage(model, vlm_cases, **{**kw, "model_identity": {**ident, "dtype": "fp32"}})
    assert not any(s["reused"] for s in other["executions"])


def _rerun(report):
    """Steps that executed again instead of being read back (not-applicable steps never run)."""
    return [s["key"] for s in report["executions"] if s["status"] != "not_applicable" and not s["reused"]]


def test_mvp0_changed_tool_code_is_a_new_action(tmp_path, vlm_cases, monkeypatch):
    from evalrx.eval_agent.probing import controller
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    kw = dict(out_dir=tmp_path / "v", model_identity={"spec": "tiny"}, n_fail=2, progress=None)
    first = run_triage(model, vlm_cases, **kw)
    real = controller.tool_version
    monkeypatch.setattr(controller, "tool_version",
                        lambda a: real(a) + ("-edited" if a.name == "attention_knockout" else ""))
    second = run_triage(model, vlm_cases, **kw)
    assert _rerun(second) == ["attention_knockout"]
    assert _steps(second)["attention_knockout"]["action_id"] != _steps(first)["attention_knockout"]["action_id"]


def test_mvp0_code_version_is_recorded_but_not_part_of_the_identity(tmp_path, vlm_cases, monkeypatch):
    from evalrx.eval_agent.probing import controller
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    kw = dict(out_dir=tmp_path / "c", model_identity={"spec": "tiny"}, n_fail=2,
              stage0_cost={"wall_sec": 1.0, "calls": {"generate": 8}}, progress=None)
    first = run_triage(model, vlm_cases, **kw)
    code = controller.code_version()
    assert first["code"] == code and "eval_agent/probing" in code["parts"]
    out = tmp_path / "c"
    planned = [json.loads(line) for line in (out / "selection_trace.jsonl").read_text().splitlines()
               if '"planned"' in line]
    assert planned and all(e["code"] == code["tree"] for e in planned)
    assert f"code tree `{code['tree']}`" in (out / "triage_report.md").read_text()
    # other code (a backend, the scheduler) changed: the ledger is still read back, the new tree recorded
    monkeypatch.setattr(controller, "code_version", lambda: {**code, "tree": "000000000000"})
    second = run_triage(model, vlm_cases, **kw)
    assert not _rerun(second) and second["code"]["tree"] == "000000000000"
    costs = [json.loads(line) for line in (out / "cost_ledger.jsonl").read_text().splitlines()]
    assert [c["code"] for c in costs if c["kind"] == "stage0"] == [code["tree"], "000000000000"]


def test_mvp0_steps_that_compare_answers_do_not_run_where_outputs_commit_to_none(tmp_path, vlm_cases):
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    report = run_triage(model, vlm_cases, out_dir=tmp_path / "cap", model_identity={"spec": "tiny"}, n_fail=2,
                        task_kind="chair_caption", answer_fn=_no_answer, progress=None,
                        analyzer_kwargs={"self_consistency": {"gen_kwargs": {"temperature": 1.0}}})
    steps = _steps(report)
    for key in ("self_consistency", "prompt_reword", "prompt_threshold"):
        assert steps[key]["status"] == "not_applicable" and "cost" not in steps[key]
    assert steps["self_consistency"]["reason"] == ("step is for tasks whose outputs commit to a categorical "
                                                   "answer; this task's do not")
    assert steps["modality_ablation"]["status"] == steps["attention_knockout"]["status"] == OK
    assert {r["U"]["state"] for r in report["cases"]} == {"not_applicable"}


def test_code_version_hashes_per_package_directory_and_reads_a_snapshot_ref(tmp_path):
    from evalrx.eval_agent.probing.controller import _code_version, code_label

    root = tmp_path / "evalrx"
    for rel, text in {"__init__.py": "", "models/hf.py": "a = 1\n", "eval_agent/probing/rules.py": "b = 2\n",
                      "reporting/web/node_modules/gyp/x.py": "", "models/__pycache__/hf.py": ""}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    first = _code_version(root)
    assert sorted(first["parts"]) == [".", "eval_agent/probing", "models"]
    assert first["git"] is None and code_label(first).endswith("commit unknown")
    (root / "models" / "hf.py").write_text("a = 3\n")
    (tmp_path / "CODE_REF.json").write_text('{"commit": "0123456789abcdef", "dirty": true}')
    second = _code_version(root)
    assert second["tree"] != first["tree"] and second["parts"]["models"] != first["parts"]["models"]
    assert second["parts"]["eval_agent/probing"] == first["parts"]["eval_agent/probing"]
    assert second["git"] == {"commit": "0123456789abcdef", "dirty": True}
    assert code_label(second) == f"code tree `{second['tree']}`, commit `0123456789` + uncommitted changes"
    assert code_label(None) == "code not recorded"


def test_mvp0_uncertain_attempt_gets_a_new_number(tmp_path, vlm_cases):
    from evalrx.eval_agent.probing.ledger import Ledger
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    kw = dict(out_dir=tmp_path / "p", model_identity={"spec": "tiny"}, n_fail=2, progress=None)
    first = run_triage(model, vlm_cases, **kw)
    ko_id = _steps(first)["attention_knockout"]["action_id"]
    ledger = Ledger(tmp_path / "p")
    (ledger.obs_dir / f"{ko_id}.json").unlink()  # as if the process died before settling
    second = run_triage(model, vlm_cases, **kw)
    planned = [e for e in ledger.trace_events() if e.get("event") == "planned" and e["action_id"] == ko_id]
    assert [e["attempt"] for e in planned] == [1, 2]
    assert planned[1]["previous_attempts_uncertain"] == 1
    assert _rerun(second) == ["attention_knockout"]


def test_mvp0_black_box_gates_off_the_white_box_step(tmp_path, vlm_cases):
    model = _BlackBox()
    report = run_triage(model, vlm_cases, out_dir=tmp_path / "bb", model_identity={"spec": "bb"},
                             n_fail=2, progress=None)
    steps = _steps(report)
    assert steps["attention_knockout"]["status"] == UNSUPPORTED
    assert steps["attention_knockout"]["cost"]["calls"] == {}
    assert steps["modality_ablation"]["status"] == OK
    assert steps["embed_lens"]["status"] == steps["patchscope"]["status"] == UNSUPPORTED
    assert steps["prompt_threshold"]["status"] == "not_applicable"  # no task kind given
    assert steps["self_consistency"]["status"] == OK
    assert report["checks"]["no_zero_for_failure"] is True
    assert all(r["knockout"]["state"] == "untested" for r in report["cases"])


class _FakeAPI(Model):
    """A black-box model that answers Yes and counts requests and tokens like an API runtime."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self):
        self.counters = {"requests": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0,
                         "thought_tokens": 0, "responses_without_usage": 0}

    def generate(self, inputs, **kwargs):
        self.counters["requests"] += 1
        self.counters["prompt_tokens"] += 300
        self.counters["completion_tokens"] += 2
        return "Yes"

    def usage_counters(self):
        return dict(self.counters)

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_api_run_meters_tokens_and_dollars_and_runs_its_behavioural_follow_up(tmp_path, vlm_cases):
    from evalrx.models.backends.api.prices import usd

    model = _FakeAPI()
    report = run_triage(
        model, vlm_cases, out_dir=tmp_path / "api", model_identity={"spec": "fake-api"}, n_fail=2,
        min_route_fail=1, task_kind="yes_no", n_needs_counterfactual=True, progress=None,
        analyzer_kwargs={"self_consistency": {"gen_kwargs": {"temperature": 1.0}}},
        price_fn=lambda u: usd("gemini-3.1-flash-lite", u),
        stage0_cost={"wall_sec": 1.0, "calls": {"generate": 8}, "api_usage": {"requests": 8}, "usd": 0.001})
    steps = _steps(report)
    for key in ("attention_knockout", "embed_lens", "patchscope"):   # no internals: gated, nothing spent
        assert steps[key]["status"] == UNSUPPORTED and steps[key]["cost"]["calls"] == {}
    swap = steps["modality_ablation"]["cost"]
    assert swap["api_usage"]["requests"] == swap["calls"]["generate"] == 8
    assert swap["usd"] == usd("gemini-3.1-flash-lite", swap["api_usage"])
    run = report["cost"]["actions_run"]
    assert run["api_usage"]["requests"] == model.counters["requests"] == sum(run["calls"].values())
    assert run["usd"] == round(sum(s["cost"].get("usd") or 0 for s in report["executions"] if s.get("cost")), 6)
    assert report["cost"]["stage0"]["usd"] == 0.001
    # sampling ran: the U sampling part is tested, not left untested as on greedy settings
    assert all(r["U"]["parts"][0]["state"] in ("fired", "clear") for r in report["cases"])
    # the FAIL cases keep "Yes" after a swap that should make it "no": N. Its internal
    # layers are gated off at no cost, and the behavioural one runs on the two FAIL
    # cases and their two matched PASS controls
    (decision,) = report["decisions"]
    assert (decision["route"], decision["rule_id"], decision["key"]) == ("N", "N3", "describe_first")
    passed = {t["key"]: t for t in decision["passed_over"]}
    assert passed["attention_knockout"]["battery"] and passed["attention_knockout"]["status"] == "unsupported"
    assert passed["causal_trace"]["status"] == "unsupported" and "hidden_states" in passed["causal_trace"]["reason"]
    assert passed["attention_temperature"]["status"] == passed["attention_sensitivity"]["status"] == "unsupported"
    follow = report["executions"][-1]
    assert follow["kind"] == "follow_up" and follow["n_cases"] == 4 and follow["cost"]["calls"] == {"generate": 8}
    (fu,) = report["follow_ups"]
    assert fu["reading"]["route_fail"]["n_measured"] == 2 and fu["reading"]["controls"]["n_measured"] == 2
    assert fu["reading"]["route_fail"]["means"]["changed_describe_first"] == 0.0  # "Yes" whatever the prompt
    assert fu["reading"]["route_fail"]["n_by_field"]["changed_describe_first"] == 2
    assert fu["reading"]["route_fail"]["means"]["complied_describe_first"] == 0.0  # it never described
    assert report["stop"]["reason"] == "no_candidate"
    layers = {st["rule_id"]: st["state"] for st in report["routes"][0]["layers"]}
    assert layers == {"N1": "exhausted", "N2": "exhausted", "N3": "done"}
    md = (tmp_path / "api" / "triage_report.md").read_text()
    assert "## Decisions" in md and "Follow-ups" in md and "USD" in md and "did not add the text" in md
    assert "Not measured: unsupported (model lacks capabilities" in md      # knockout, no table of Nones
    assert "Not measured: embed_lens unsupported" in md


def test_cli_default_policy_is_judge_and_triage_refuses_held_out():
    from evalrx.benchmark.run import build_parser, run_args

    assert build_parser().parse_args([]).m1_policy == "judge"
    args = build_parser().parse_args(["--modality", "vlm", "--model", "qwen3.5-0.8b-vl",
                                      "--m1-policy", "triage_active", "--held-out",
                                      "--diagnosis-mode", "fixed"])
    with pytest.raises(ValueError, match="triage_active"):
        run_args(args)


def test_per_call_sampling_is_requested_only_where_the_backend_honours_it():
    from types import SimpleNamespace

    from evalrx.benchmark.runner import _sampling_kwargs

    args = SimpleNamespace(top_p=0.95, top_k=20)
    assert _sampling_kwargs(SimpleNamespace(backend="hf_local"), args) == {"temperature": 1.0, "top_p": 0.95,
                                                                          "top_k": 20}
    assert _sampling_kwargs(SimpleNamespace(backend="gemini"), args) == {"temperature": 1.0}  # merged per call
    assert _sampling_kwargs(SimpleNamespace(backend="endpoint"), args) is None             # dropped per call
