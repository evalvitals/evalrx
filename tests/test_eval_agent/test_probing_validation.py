"""MVP-4: issues frozen from a diagnosis and validated once on cases it never saw.

* the partition keeps every validation case off the media DISCOVERY saw;
* freezing keeps route signals and readings whose discovery interval clears the
  band, one issue per route and step (its reading furthest outside the band),
  and drops a step that re-measures the check its route is cut on;
* the family is written before the validation partition is read, every issue is
  tested at 1 - alpha/K, and each arm is credited with its own issues' verdicts:
  an effect that holds on new cases is supported, one that held only on the
  discovery cases is refuted;
* no carded reading reads the label (a step targeting the gold answer is
  refused), and the validator refuses a report from another rules version;
* readings whose per-case values go together within FAIL and within the
  controls are one phenomenon (a FAIL-vs-control shift alone does not join
  them), and an arm's phenomenon takes the verdicts of its own issues only.
"""

from __future__ import annotations

import json

import pytest

from evalrx.eval_agent.probing import LAYERS, explore_confirm_split, run_triage
from evalrx.eval_agent.probing.controller import BATTERY
from evalrx.eval_agent.probing.explanations import CARDS, Metric, label_bound
from evalrx.eval_agent.probing.rules import Layer, candidate_steps
from evalrx.eval_agent.probing.validation import (
    ALPHA,
    IssueRecord,
    credit_arms,
    freeze_issues,
    group_phenomena,
    partition_bundle,
    source_key,
    supported_curve,
    validate,
    within_group_rho,
)
from tests.test_eval_agent.test_probing_alternatives import FIRST, _Flipper
from tests.test_eval_agent.test_probing_mvp2 import _kw, _pope_like, _yes_no

SEED = 7


def _split(n_images=96):
    cases, table = _pope_like(n_images)
    explore, confirm = explore_confirm_split(cases, 0.5, SEED)
    return cases, table, explore, confirm


def test_the_partition_keeps_every_validation_case_off_the_media_discovery_saw():
    cases, _, explore, confirm = _split()
    bundle, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5)
    seen = {source_key(c) for c in explore}
    assert validation and not {source_key(c) for c in validation} & seen
    assert set(bundle.validation_ids) <= {str(c.id) for c in confirm}
    assert len(bundle.validation_ids) + bundle.excluded["shares_a_source_with_discovery"] == len(confirm)
    assert bundle.excluded["shares_a_source_with_discovery"] > 0   # a POPE image has a yes and a no question
    assert bundle.roles["repair_confirm"] is None and bundle.to_dict()["sha1"] == bundle.to_dict()["sha1"]
    assert bundle.diagnostic_val_source == "internal" and bundle.validation_from == {"confirm": len(validation)}


def test_a_media_split_keeps_every_image_on_one_side_so_the_validation_drops_nothing():
    from evalrx.core.case import Label

    cases, _, by_case_explore, by_case_confirm = _split()
    explore, confirm = explore_confirm_split(cases, 0.5, SEED, by="media")
    assert not {source_key(c) for c in explore} & {source_key(c) for c in confirm}
    assert len(explore) + len(confirm) == len(cases) and {c.id for c in explore} | {c.id for c in confirm} == {c.id for c in cases}
    assert abs(len(confirm) - len(cases) / 2) <= 4                         # 96 images, 2 questions each: 48 a side
    fail = lambda batch: sum(c.label == Label.FAIL for c in batch) / len(batch)  # noqa: E731
    assert abs(fail(confirm) - fail(cases)) <= 0.1                        # stratified by the group's labels
    # the per-case split of the same batch shares images across the sides; this one is deterministic and
    # moves whole images at a time
    assert {source_key(c) for c in by_case_explore} & {source_key(c) for c in by_case_confirm}
    again = explore_confirm_split(cases, 0.5, SEED, by="media")
    assert [c.id for c in again[1]] == [c.id for c in confirm]
    bundle, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5, split_by="media")
    assert bundle.excluded["shares_a_source_with_discovery"] == 0 and len(validation) == len(confirm)
    assert bundle.split_by == "media" and bundle.rule.startswith("DISCOVERY = EXPLORE, split from CONFIRM by media")
    assert bundle.to_dict()["split_by"] == "media"
    # too small to split, or everything on one image: no split, as the loop
    assert explore_confirm_split(cases[:3], 0.5, SEED, by="media")[1] is None
    one_image = [c for c in cases if c.inputs.image == "img0"]
    assert explore_confirm_split(one_image + one_image, 0.5, SEED, by="media")[1] is None


def test_a_validation_only_source_joins_the_validation_cases_under_the_same_media_rule(tmp_path):
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label

    cases, _, explore, confirm = _split()
    seen = sorted({c.inputs.image for c in explore})[:3]
    (taken,) = [str(c.id) for c in confirm][:1]
    source = CaseBatch([
        FailureCase(id=f"s{i}" if img != "repeat" else taken, expected="Yes", observed="No", label=Label.FAIL,
                    inputs=Inputs(prompt="Is there a dog in the image?", image="new9" if img == "repeat" else img),
                    metadata={"gold": "Yes", "finish_reason": "stop"})
        for i, img in enumerate([*seen, "new0", "new1", "new2", "new3", "repeat"])])
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    base, _ = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5)
    bundle, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5, source=source,
                                          source_manifest=manifest)
    # the confirm half is unchanged; the source adds its cases on media discovery never saw, once per id
    assert bundle.validation_ids[:len(base.validation_ids)] == base.validation_ids
    assert bundle.validation_ids[len(base.validation_ids):] == ["s3", "s4", "s5", "s6"]
    assert bundle.excluded == {"shares_a_source_with_discovery": base.excluded["shares_a_source_with_discovery"],
                               "source_shares_a_source_with_discovery": 3, "source_repeats_an_id": 1}
    assert bundle.validation_from == {"confirm": len(base.validation_ids), "source": 4}
    assert not {source_key(c) for c in validation} & {source_key(c) for c in explore}
    assert bundle.diagnostic_val_source == "internal+source" and "validation-only source" in bundle.rule
    assert bundle.source_manifest["path"] == str(manifest) and bundle.manifest == {}
    assert bundle.to_dict()["sha1"] != base.to_dict()["sha1"]


def _outcome(observed, ci, n=(8, 8), diff=None):
    return {"n": list(n), "observed": observed, "ci95": ci, "diff": diff if diff is not None else sum(ci) / 2,
            "fail_mean": 0.5, "control_mean": 0.2}


def test_freezing_keeps_what_clears_the_band_one_issue_per_route_and_step():
    report = {
        "summary": {"U": {"signal_ci95": [0.2, 0.5], "fail": {"n_fired": 5, "n_tested": 10, "rate": 0.5},
                          "pass": {"n_fired": 1, "n_tested": 10, "rate": 0.1}},
                    "N": {"signal_ci95": [0.05, 0.3], "fail": {"n_fired": 3, "n_tested": 10, "rate": 0.3},
                          "pass": {"n_fired": 1, "n_tested": 10, "rate": 0.1}}},
        "outcomes": {
            "S": {"gradcam": {"gc_entropy": _outcome("down", [-0.4, -0.2]),
                              "gc_top5_share": _outcome("up", [0.1, 0.4]),
                              "gc_media_grad_share": _outcome("undetermined", [-0.1, 0.2])},
                  "self_consistency_n10": {"consistency": _outcome("up", [0.2, 0.4])}},
            "U": {"logprob_entropy": {"mean_top_entropy": _outcome("up", [0.1, 0.25])},
                  "patchscope": {"ps_peak_prob": _outcome("down", [-0.3, -0.1])}},
            "X": {"media_drop": {"grounded_in_image": _outcome("up", [0.3, 0.9])}},
            "N": {"patchscope": {"ps_peak_prob": _outcome("undetermined", [-0.2, 0.1])}},
        },
    }
    issues = {(i.kind, i.route, i.check or i.step): i for i in freeze_issues(report)}
    # re-sampling on S and the output's close calls on U read the check the route is cut on
    assert set(issues) == {("route_signal", "U", "U"), ("reading", "S", "gradcam"),
                           ("reading", "U", "patchscope"), ("reading", "X", "media_drop")}
    gc = issues[("reading", "S", "gradcam")]
    # the reading furthest outside its band (0.15 past 0.05) is tested, the other is listed
    assert (gc.reading, gc.direction, gc.secondary) == ("gc_entropy", "down", ["gc_top5_share: up"])
    assert gc.spread is not None and gc.rule_id is None
    assert issues[("reading", "X", "media_drop")].rule_id == "X1"     # X1 narrows its cases
    assert len({i.issue_id for i in issues.values()}) == 4


@pytest.fixture
def wrapper_on_s(monkeypatch):
    monkeypatch.setitem(CARDS, FIRST.name, (Metric("prompt_sensitivity", "the answer moves with a wrapper", 0.1),))
    at = next(i for i, x in enumerate(LAYERS) if x.rule_id == "Sb")
    return (*LAYERS[:at], Layer("Z1", "S", "does a wrapper move it", (FIRST,)), *LAYERS[at:])


class _DiscoveryOnly(_Flipper):
    """Wrong cases come out right inside the wrapper only on the discovery cases: an effect that
    does not hold on new cases."""

    def __init__(self, table, cases, only):
        super().__init__(table, [c for c in cases if str(c.id) in only])


def _diagnose_and_validate(tmp_path, model, layers, explore, validation, bundle, name):
    report = run_triage(model, explore, **_kw(tmp_path / "probing", max_fail=8, min_route_fail=1, layers=layers))
    issues = freeze_issues(report, layers)
    out = validate(model, explore, validation, {name: issues}, {name: report}, out_dir=tmp_path / "validation",
                   model_identity={"spec": "table"}, bundle=bundle, seed=0, host_assumed_finish=True,
                   n_needs_counterfactual=True, task_kind="yes_no", answer_fn=_yes_no, layers=layers,
                   analyzer_kwargs={"self_consistency": {"gen_kwargs": {"temperature": 1.0}}}, progress=None)
    return report, issues, out


def test_an_effect_that_holds_on_new_cases_is_supported_and_frozen_before_validation(tmp_path, wrapper_on_s):
    cases, table, explore, confirm = _split()
    bundle, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5)
    report, issues, out = _diagnose_and_validate(tmp_path, _Flipper(table, cases), wrapper_on_s, explore,
                                                 validation, bundle, "rules")
    (wrap,) = [i for i in issues if i.step == "wrap_a"]
    assert (wrap.route, wrap.direction) == ("S", "up")
    row = next(r for r in out["issues"] if r["issue_id"] == wrap.issue_id)
    assert row["validation"]["verdict"] == "supported"
    # one-sided at alpha / K on the claimed side: the interval at 1 - 2 alpha / K
    assert out["level"] == pytest.approx(1 - 2 * ALPHA / out["k"]) and out["arms"]["rules"]["supported"] >= 1
    frozen = json.loads((tmp_path / "validation" / "frozen.json").read_text())
    assert frozen["sha1"] == out["frozen_sha1"] and wrap.issue_id in frozen["arms"]["rules"]
    # the diagnosis never touched a validation case; the validator wrote to its own directory
    val_ids = set(bundle.validation_ids)
    planned = [json.loads(line) for line in (tmp_path / "probing" / "selection_trace.jsonl").read_text().splitlines()
               if '"planned"' in line]
    ran_on = {cid for e in planned for cid in e["action"]["case_ids"]}
    assert planned and ran_on and not val_ids & ran_on
    assert not val_ids & set(report["sample"]["case_ids"])
    assert set(out["sample"]["case_ids"]) <= val_ids
    assert (tmp_path / "validation" / "selection_trace.jsonl").is_file()
    assert "## Arms" in (tmp_path / "validation" / "validation_report.md").read_text()
    # both reports and every planned action say which code ran
    from evalrx.eval_agent.probing.controller import code_version

    assert out["code"] == report["code"] == code_version()
    assert all(e["code"] == out["code"]["tree"] for e in planned)
    assert f"code tree `{out['code']['tree']}`" in (tmp_path / "validation" / "validation_report.md").read_text()


def test_an_effect_that_held_only_on_the_discovery_cases_is_refuted(tmp_path, wrapper_on_s):
    cases, table, explore, confirm = _split()
    bundle, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5)
    model = _DiscoveryOnly(table, cases, {str(c.id) for c in explore})
    _, issues, out = _diagnose_and_validate(tmp_path, model, wrapper_on_s, explore, validation, bundle, "rules")
    (wrap,) = [i for i in issues if i.step == "wrap_a"]
    row = next(r for r in out["issues"] if r["issue_id"] == wrap.issue_id)
    assert row["validation"]["verdict"] == "refuted" and row["validation"]["diff"] == 0.0


def test_a_live_explanation_is_checked_on_new_cases_even_where_no_issue_ran_its_step(tmp_path, wrapper_on_s):
    cases, table, explore, confirm = _split()
    bundle, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5)
    model = _Flipper(table, cases)
    report = run_triage(model, explore, **_kw(tmp_path / "probing", max_fail=8, min_route_fail=1,
                                               layers=wrapper_on_s))
    claim = {"id": "S.h1", "route": "S", "claim": "a neutral wrapper moves these errors",
             "predictions": {FIRST.name: {"prompt_sensitivity": "up"}}, "primary": [[FIRST.name, "prompt_sensitivity"]]}
    tie = {"id": "U.h1", "route": "U", "claim": "a near tie", "primary": [["self_consistency_n10", "consistency"]],
           "predictions": {"self_consistency_n10": {"consistency": "down"}}}
    report["explanations"] = {"sets": [{"hypotheses": [claim, {**claim, "id": "S.h2"}, tie],
                                        "final": {"S.h1": {"state": "live"}, "S.h2": {"state": "contradicted"},
                                                  "U.h1": {"state": "live"}}}]}
    out = validate(model, explore, validation, {"d": []}, {"d": report}, out_dir=tmp_path / "validation",
                   model_identity={"spec": "table"}, bundle=bundle, seed=0, host_assumed_finish=True,
                   n_needs_counterfactual=True, task_kind="yes_no", answer_fn=_yes_no, layers=wrapper_on_s,
                   analyzer_kwargs={"self_consistency": {"gen_kwargs": {"temperature": 1.0}}}, progress=None)
    frozen = json.loads((tmp_path / "validation" / "frozen.json").read_text())
    assert [h["id"] for h in frozen["explanations"]["d"]] == ["S.h1", "U.h1"]     # the live ones, frozen first
    row, tied = out["explanations"]["arms"]["d"]
    assert row["state"] == "supported" and row["predictions"][0]["verdict"] == "hit" and row["predictions"][0]["primary"]
    # one test in the primary family (the definitional one is not a test): one-sided at alpha
    assert out["explanations"]["k"] == 1 and out["explanations"]["level"] == pytest.approx(1 - 2 * ALPHA)
    # U is cut on re-sampling: re-sampling more cannot support or contradict an explanation of U
    assert tied["state"] == "untestable" and tied["predictions"][0]["verdict"] == "definitional"
    assert "self_consistency_n10" not in (tmp_path / "validation" / "selection_trace.jsonl").read_text()


def test_no_carded_reading_reads_the_label_and_a_gold_target_is_refused(tmp_path):
    assert not label_bound((*BATTERY, *candidate_steps(LAYERS)))
    assert not any(m.field.startswith("fixed_") for ms in CARDS.values() for m in ms)
    cases, table, explore, _ = _split(16)
    with pytest.raises(ValueError, match="read the gold answer"):
        run_triage(_Flipper(table, cases), explore,
                   **_kw(tmp_path, analyzer_kwargs={"attention_knockout": {"target": "gold"}}))


def test_the_validator_refuses_a_report_from_another_rules_version(tmp_path):
    _, _, explore, confirm = _split(16)
    _, validation = partition_bundle(explore, confirm, seed=SEED, confirm_split=0.5)
    with pytest.raises(ValueError, match="another rules version"):
        validate(None, explore, validation, {"old": []}, {"old": {"rules_version": "mvp2-rules-1"}},
                 out_dir=tmp_path, model_identity={}, progress=None)


def _reading(route, step):
    return IssueRecord(kind="reading", route=route, direction="up", band=0.1, metric_kind="mean", claim="-",
                       step=step, reading="r")


def test_readings_that_go_together_within_groups_are_one_phenomenon_and_a_shift_is_not():
    fail, ctrl = [f"f{i}" for i in range(6)], [f"c{i}" for i in range(6)]
    base = dict(zip(fail + ctrl, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]))
    same = {c: 2 * v + 10 for c, v in base.items()}              # the same order inside each group
    order = [3, 6, 1, 5, 2, 4]                                   # FAIL and controls far apart, order unrelated
    shifted = {**dict(zip(fail, order)), **{c: 100 + v for c, v in zip(ctrl, order)}}
    a, b, c, d = (_reading("S", "knockout"), _reading("S", "causal_trace"), _reading("S", "gradcam"),
                  _reading("U", "knockout"))
    signal = IssueRecord(kind="route_signal", route="U", direction="up", band=0.1, metric_kind="rate", claim="-",
                         check="U")
    per_case = {a.issue_id: (base, fail, ctrl), b.issue_id: (same, fail, ctrl), c.issue_id: (shifted, fail, ctrl),
                d.issue_id: (base, fail, ctrl)}
    assert within_group_rho(base, same, [fail, ctrl]) == pytest.approx(1.0)
    assert within_group_rho(base, shifted, [fail, ctrl]) == pytest.approx(-3 / 35)
    results = {signal.issue_id: {"verdict": "inconclusive"}, a.issue_id: {"verdict": "supported"},
               b.issue_id: {"verdict": "inconclusive"}, c.issue_id: {"verdict": "supported"},
               d.issue_id: {"verdict": "refuted"}}
    phenomena, links = group_phenomena([signal, a, b, c, d], results, per_case)
    assert len(links) == 3                                       # every pair on route S; d is alone on U
    assert {frozenset(ph["issues"]): ph["verdict"] for ph in phenomena} == {
        frozenset({signal.issue_id}): "inconclusive", frozenset({a.issue_id, b.issue_id}): "supported",
        frozenset({c.issue_id}): "supported", frozenset({d.issue_id}): "refuted"}   # d: same values, other route
    of = {i: ph["id"] for ph in phenomena for i in ph["issues"]}
    credit = credit_arms({"both": [a, b], "twin_only": [b]}, results, of, {})
    assert (credit["both"]["n_phenomena"], credit["both"]["phenomena_supported"], credit["both"]["n_issues"]) == (1, 1, 2)
    # the twin's phenomenon is supported through the knockout, which this arm did not freeze
    assert (credit["twin_only"]["phenomena_supported"], credit["twin_only"]["phenomena_inconclusive"]) == (0, 1)


def test_an_arm_found_more_only_where_it_holds_at_every_grouping_threshold():
    verdicts = {"a": "supported", "b": "supported", "c": "inconclusive"}
    links = [{"issues": ["a", "b"], "rho": 0.55}, {"issues": ["b", "c"], "rho": -0.35}]
    arms = {"two": ["a", "b"], "one": ["a"], "none": ["c"]}
    curve = supported_curve(arms, verdicts, links)
    # a and b are one phenomenon up to 0.55 and two above it
    assert curve["supported"]["two"] == [1, 1, 1, 1, 1, 1, 2, 2, 2]
    assert curve["supported"]["one"] == [1] * 9 and curve["supported"]["none"] == [0] * 9
    # "two" beats "one" only above 0.55, so that difference does not count; both beat "none" everywhere
    assert {(d["higher"], d["lower"]) for d in curve["differences"]} == {("two", "none"), ("one", "none")}
    above = supported_curve(arms, verdicts, links, grid=(0.6, 0.7))
    assert ("two", "one") in {(d["higher"], d["lower"]) for d in above["differences"]}
