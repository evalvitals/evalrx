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
from evalrx.eval_agent.probing import explore_confirm_split, matched_sample, run_triage_mvp0
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
        FailureCase(id="f", inputs=Inputs(prompt="q"), expected="No", observed="Yes", label=Label.FAIL),
        FailureCase(id="p", inputs=Inputs(prompt="q"), expected="No", observed="No", label=Label.PASS),
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


def test_triage_truncation_counts_when_the_backend_reported_it():
    cases = CaseBatch([FailureCase(id="f", inputs=Inputs(prompt="q"), expected="1", observed="1 2 3",
                                   label=Label.FAIL, metadata={"finish_reason": "length"})])
    aea = _obs("answer_extraction_audit", [{"sample_id": "f", "output_truncated": 1, "gave_up": 0}])
    rec = triage_cases(cases, {"answer_extraction_audit": aea}, host_assumed_finish=False,
                       answer_contract=False)[0]
    trunc = [x for x in rec["H"]["fields"] if x["field"] == "output_truncated"][0]
    assert trunc["state"] == "fired" and trunc["provenance"] == "backend"
    assert rec["routes"] == ["H"] and rec["H"]["state_label_free"] == "fired"


# -- controller end to end ---------------------------------------------------------------


@pytest.fixture()
def vlm_cases():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.tiny_whitebox import image

    imgs = [image(i) for i in range(4)]
    out = []
    for i in range(8):
        img = imgs[i // 2]  # two questions per image, adjacent, like POPE
        fail = i % 4 == 1
        gold = "No" if i % 2 else "Yes"
        out.append(FailureCase(
            id=f"pope-{i}", inputs=Inputs(prompt=f"Is there a cat in the image {i}?", image=img),
            expected=gold, observed="Yes" if fail or gold == "Yes" else "No",
            label=Label.FAIL if fail else Label.PASS,
            metadata={"gold": gold, "finish_reason": "stop"},
        ))
    return CaseBatch(out)


def test_mvp0_end_to_end_and_resume(tmp_path, vlm_cases):
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    ident = {"spec": model.spec.key, "backend": "tiny"}
    kw = dict(out_dir=tmp_path / "probing", model_identity=ident, n_fail=2, seed=0,
              host_assumed_finish=True, stage0_cost={"wall_sec": 1.0, "calls": {"generate": 8}},
              progress=None)
    report = run_triage_mvp0(model, vlm_cases, **kw)

    steps = {s["analyzer"]: s for s in report["steps"]}
    assert [s["status"] for s in report["steps"]] == [OK] * 4
    assert steps["answer_extraction_audit"]["cost"]["calls"] == {}  # Stage-0 outputs re-read
    assert steps["modality_ablation"]["cost"]["calls"] == {"generate": 8}  # 4 cases x (base + swap)
    assert steps["attention_knockout"]["fidelity"]["status"] == "passed"
    assert steps["termination_audit"]["fidelity"]["status"] == "not_applicable"
    checks = report["checks"]
    assert checks["only_planned_actions"] and checks["stage0_cost_separate"]
    assert all(checks["case_sets_match_plan"].values()) and all(checks["configs_match_plan"].values())

    sample = report["sample"]
    assert len(sample["fail_ids"]) == 2 and sample["strata"]["no"]["pass"] == 2
    for rec in report["cases"]:
        assert rec["N"]["state"] in ("fired", "clear")  # TinyVLM answers "x" whatever it sees
        donor = rec["N"]["slots"][0]["donor"]
        assert donor != rec["case_id"]
        assert rec["knockout"]["state"] == "measured" and rec["knockout"]["fidelity"] == "passed"

    out = tmp_path / "probing"
    costs = [json.loads(line) for line in (out / "cost_ledger.jsonl").read_text().splitlines()]
    assert [c["kind"] for c in costs] == ["stage0"] + ["action"] * 4
    assert costs[1]["baseline_reused"] is True and costs[3]["baseline_reused"] is False
    for name in ("split_manifest.json", "triage_report.json", "triage_report.md"):
        assert (out / name).is_file()

    # resume: same identities -> read back, no model calls, no new action cost
    # (only this invocation's Stage 0, which the caller did spend)
    n_forwards = model.n_forwards
    again = run_triage_mvp0(model, vlm_cases, **kw)
    assert all(s["reused"] for s in again["steps"]) and model.n_forwards == n_forwards
    costs2 = [json.loads(line) for line in (out / "cost_ledger.jsonl").read_text().splitlines()]
    assert [c["kind"] for c in costs2[len(costs):]] == ["stage0"] and costs2[-1]["invocation"] == 2
    events = [json.loads(line) for line in (out / "selection_trace.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events].count("reused") == 4

    # a different model identity is a different action: nothing is reused
    other = run_triage_mvp0(model, vlm_cases, **{**kw, "model_identity": {**ident, "dtype": "fp32"}})
    assert not any(s["reused"] for s in other["steps"])


def test_mvp0_uncertain_attempt_gets_a_new_number(tmp_path, vlm_cases):
    from evalrx.eval_agent.probing.ledger import Ledger
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    kw = dict(out_dir=tmp_path / "p", model_identity={"spec": "tiny"}, n_fail=2, progress=None)
    first = run_triage_mvp0(model, vlm_cases, **kw)
    ko_id = [s for s in first["steps"] if s["analyzer"] == "attention_knockout"][0]["action_id"]
    ledger = Ledger(tmp_path / "p")
    (ledger.obs_dir / f"{ko_id}.json").unlink()  # as if the process died before settling
    second = run_triage_mvp0(model, vlm_cases, **kw)
    planned = [e for e in ledger.trace_events() if e.get("event") == "planned" and e["action_id"] == ko_id]
    assert [e["attempt"] for e in planned] == [1, 2]
    assert planned[1]["previous_attempts_uncertain"] == 1
    assert [s["reused"] for s in second["steps"]] == [True, True, True, False]


def test_mvp0_black_box_gates_off_the_white_box_step(tmp_path, vlm_cases):
    model = _BlackBox()
    report = run_triage_mvp0(model, vlm_cases, out_dir=tmp_path / "bb", model_identity={"spec": "bb"},
                             n_fail=2, progress=None)
    steps = {s["analyzer"]: s for s in report["steps"]}
    assert steps["attention_knockout"]["status"] == UNSUPPORTED
    assert steps["attention_knockout"]["cost"]["calls"] == {}
    assert steps["modality_ablation"]["status"] == OK
    assert report["checks"]["no_zero_for_failure"] is True
    assert all(r["knockout"]["state"] == "untested" for r in report["cases"])


def test_cli_default_policy_is_judge_and_triage_refuses_held_out():
    from evalrx.benchmark.run import build_parser, run_args

    assert build_parser().parse_args([]).m1_policy == "judge"
    args = build_parser().parse_args(["--modality", "vlm", "--model", "qwen3.5-0.8b-vl",
                                      "--m1-policy", "triage_active", "--held-out",
                                      "--diagnosis-mode", "fixed"])
    with pytest.raises(ValueError, match="triage_active"):
        run_args(args)
