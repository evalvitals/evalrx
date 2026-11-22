"""Held-out comparison designs, bounded predictions and the single correction family."""

from __future__ import annotations

import json

import pytest

from evalrx.analysis.holdout import holdout_confirm, outcome_values
from evalrx.analysis.holdout_designs import estimate_design, representative_rows
from evalrx.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    compile_recipe,
    evaluate_expr,
    parse_expectation,
)


def _cell(n: int, n_fail: int, **cols) -> list[dict]:
    """*n* rows sharing *cols*, the first *n_fail* of them failing."""
    return [{**cols, "label": "fail" if i < n_fail else "pass"} for i in range(n)]


def _signal(name: str, expr: str, **recipe) -> dict:
    return {"name": name, "rationale": "", "recipe": {"name": name, "kind": "expr",
                                                      "expr": expr, **recipe}}


def _confirm(signals: list[dict], rows: list[dict], **kwargs) -> dict:
    report = {"candidate_signals": signals,
              "hypotheses": kwargs.pop("hypotheses", [])}
    return holdout_confirm(report, rows, outcome_col="label", **kwargs)


def _by_name(confirm: dict) -> dict:
    return {v["name"]: v for v in confirm["signal_verdicts"]}


# ── recipe fields ────────────────────────────────────────────────────────────


def test_recipe_optional_fields_round_trip_and_tolerate_bad_shapes():
    recipe = SignalRecipe.from_dict({
        "name": "r", "expr": "x > 1", "where": "kind == 'i2v'", "design": "stratified",
        "strata": "generator", "bins": [0.2, "0.5"], "modifier": "late == 1",
        "expect": {"relation": "abs<", "value": 0.1},
    })
    assert recipe.where == "kind == 'i2v'" and recipe.strata == ["generator"]
    assert recipe.bins == [0.2, 0.5]
    assert parse_expectation(recipe.expect) == ("abs<", 0.1)

    sloppy = SignalRecipe.from_dict({"name": "r", "expr": "x", "bins": ["low", 1],
                                     "strata": 7, "expect": "small"})
    assert sloppy.bins == [] and sloppy.strata == [] and sloppy.expect is None
    assert parse_expectation({"relation": "~", "value": 1}) is None
    assert parse_expectation({"relation": "abs<", "value": 0}) is None
    assert parse_expectation({"relation": "nonzero", "value": 9}) == ("nonzero", 0.0)


def test_where_leaves_rows_out_instead_of_scoring_them_zero():
    rows = [{"case_id": "a", "kind": "i2v", "x": 5}, {"case_id": "b", "kind": "t2v", "x": 5},
            {"case_id": "c", "kind": "i2v", "x": 0}, {"case_id": "d", "x": 5}]
    recipe = SignalRecipe(name="r", expr="x > 1", where="kind == 'i2v'")
    assert compile_recipe(recipe, rows) == {"a": 1.0, "c": 0.0}


def test_no_expression_may_read_the_outcome_column_whatever_it_is_called():
    rows = [{"case_id": "a", "x": 1, "verdict": "bad"}]
    with pytest.raises(RecipeError, match="reserved outcome/id"):
        compile_recipe(SignalRecipe(name="r", expr="verdict == 'bad'"), rows,
                       reserved=("verdict",))
    with pytest.raises(RecipeError, match="recipe where may not reference"):
        compile_recipe(SignalRecipe(name="r", expr="x > 0", where="label == 'fail'"), rows)
    assert evaluate_expr("x + 1", rows) == [2]


def test_keyword_functions_in_the_recipe_language():
    rows = [{"case_id": "a", "t": "Please restart the ART module"},
            {"case_id": "b", "t": "start again"}]
    word = compile_recipe(SignalRecipe(name="w", expr='has_word(t, "art")'), rows)
    piece = compile_recipe(SignalRecipe(name="p", expr='contains(t, "art")'), rows)
    assert word == {"a": 1.0, "b": 0.0}
    assert piece == {"a": 1.0, "b": 1.0}


# ── designs ──────────────────────────────────────────────────────────────────


def test_subgroup_condition_tests_the_effect_inside_one_subgroup():
    rows = (_cell(40, 30, kind="i2v", late=1) + _cell(40, 6, kind="i2v", late=0)
            + _cell(40, 8, kind="t2v", late=1) + _cell(40, 6, kind="t2v", late=0))
    out = _by_name(_confirm([
        _signal("late_i2v", "late == 1", where="kind == 'i2v'"),
        _signal("late_t2v", "late == 1", where="kind == 't2v'"),
    ], rows))
    assert out["late_i2v"]["n_holdout"] == 80
    assert out["late_i2v"]["details"]["n_excluded_by_where"] == 80
    assert out["late_i2v"]["effect"] == pytest.approx(0.60)
    assert out["late_i2v"]["reject"] is True
    assert out["late_t2v"]["effect"] == pytest.approx(0.05)
    assert out["late_t2v"]["reject"] is False


def test_difference_in_differences_estimates_the_interaction():
    rows = (_cell(60, 45, kind="i2v", late=1) + _cell(60, 9, kind="i2v", late=0)
            + _cell(60, 12, kind="t2v", late=1) + _cell(60, 9, kind="t2v", late=0))
    out = _by_name(_confirm([
        _signal("late_more_in_i2v", "late == 1", design="did", modifier="kind == 'i2v'"),
    ], rows))["late_more_in_i2v"]
    assert out["design"] == "did"
    assert out["details"]["effect_where_modifier_1"] == pytest.approx(0.60)
    assert out["details"]["effect_where_modifier_0"] == pytest.approx(0.05)
    assert out["effect"] == pytest.approx(0.55)
    assert out["ci"][0] > 0.3 and out["reject"] is True
    assert out["details"]["cells"]["modifier_1_flagged"] == {"n": 60, "mean": 0.75}


def test_stratified_comparison_removes_a_gap_that_is_only_a_difference_in_mix():
    # The flag does nothing inside either generator; it is just far more
    # common under the generator that fails more.
    rows = (_cell(90, 54, gen="weak", flag=1) + _cell(30, 18, gen="weak", flag=0)
            + _cell(30, 3, gen="strong", flag=1) + _cell(90, 9, gen="strong", flag=0))
    out = _by_name(_confirm([
        _signal("raw", "flag == 1"),
        _signal("matched", "flag == 1", design="stratified", strata=["gen"],
                expect={"relation": "abs<", "value": 0.15}),
    ], rows))
    assert out["raw"]["effect"] == pytest.approx(0.475 - 0.225)
    assert out["raw"]["reject"] is True
    matched = out["matched"]
    assert matched["design"] == "stratified"
    assert matched["effect"] == pytest.approx(0.0, abs=1e-9)
    assert matched["details"]["unadjusted_difference"] == pytest.approx(0.25)
    assert matched["details"]["n_strata_used"] == 2
    assert matched["claim"]["status"] == "holds"       # predicted small, is small
    assert matched["reject"] is False                  # which is not a separation


def test_strata_with_only_one_group_are_dropped_and_counted():
    rows = (_cell(20, 10, gen="a", flag=1) + _cell(20, 4, gen="a", flag=0)
            + _cell(15, 5, gen="b", flag=1))           # nothing to compare in "b"
    out = _by_name(_confirm([
        _signal("m", "flag == 1", design="stratified", strata=["gen"]),
    ], rows))["m"]
    assert out["n_holdout"] == 40
    assert out["details"]["n_rows_in_one_sided_strata"] == 15
    assert out["details"]["n_strata"] == 2 and out["details"]["n_strata_used"] == 1


def test_dose_response_uses_the_frozen_cut_points():
    rows = (_cell(50, 5, score=0.1) + _cell(50, 15, score=0.4) + _cell(50, 30, score=0.8))
    out = _by_name(_confirm([
        _signal("trend", "score", bins=[0.25, 0.6]),
        _signal("no_bins", "score * 2"),
    ], rows))
    trend = out["trend"]
    assert trend["design"] == "dose_response"
    assert [b["n"] for b in trend["details"]["bins"]] == [50, 50, 50]
    assert [b["mean"] for b in trend["details"]["bins"]] == [0.1, 0.3, 0.6]
    assert trend["effect"] == pytest.approx(0.25)      # per step up the bins
    assert trend["reject"] is True
    # A continuous recipe with no cut points still cannot be re-tested, and says how to fix it.
    assert out["no_bins"]["status"] == "skipped"
    assert "binary flag" in out["no_bins"]["reason"] and "bins" in out["no_bins"]["reason"]


def test_designs_that_cannot_be_estimated_are_skipped_with_a_reason():
    rows = _cell(20, 10, flag=1, score=0.5, kind="a") + _cell(20, 4, flag=0, score=0.2, kind="a")
    out = _by_name(_confirm([
        _signal("bad_design", "flag == 1", design="regression"),
        _signal("no_strata", "flag == 1", design="stratified"),
        _signal("no_modifier", "flag == 1", design="did"),
        _signal("one_cell", "flag == 1", design="did", modifier="kind == 'b'"),
        _signal("continuous_did", "score", design="did", modifier="flag == 1"),
        _signal("one_sided", "score > 0"),
        _signal("bad_syntax", "flag ==="),
    ], rows))
    assert all(v["status"] == "skipped" for v in out.values())
    assert "unknown design" in out["bad_design"]["reason"]
    assert "strata" in out["no_strata"]["reason"]
    assert "modifier" in out["no_modifier"]["reason"]
    assert "no held-out row" in out["one_cell"]["reason"]
    assert "needs a 0/1 recipe" in out["continuous_did"]["reason"]
    assert "one side" in out["one_sided"]["reason"]
    assert "recipe error" in out["bad_syntax"]["reason"]


# ── claims, bounds and the correction family ─────────────────────────────────


def test_a_control_expected_to_be_small_is_not_recorded_as_a_finding():
    # A real difference of nine points on a comparison that was predicted to
    # be under ten: it neither "holds" as a finding nor clearly stays in bound.
    rows = _cell(300, 87, flag=1) + _cell(300, 60, flag=0)
    out = _by_name(_confirm([
        _signal("as_finding", "flag == 1"),
        _signal("as_control", "flag == 1", expect={"relation": "abs<", "value": 0.10}),
        _signal("loose_control", "flag == 1", expect={"relation": "abs<", "value": 0.30}),
        _signal("tight_control", "flag == 1", expect={"relation": "abs<", "value": 0.01}),
    ], rows))
    assert out["as_finding"]["effect"] == pytest.approx(0.09)
    assert out["as_finding"]["claim"]["text"] == "effect is not zero"
    assert out["as_control"]["claim"]["text"] == "|effect| < 0.1"
    assert out["as_control"]["claim"]["status"] == "inconclusive"
    assert out["loose_control"]["claim"]["status"] == "holds"
    assert out["tight_control"]["claim"]["status"] == "contradicted"
    assert all(out[k]["reject"] is False
               for k in ("as_control", "loose_control", "tight_control"))


def test_directional_claims_can_hold_or_be_contradicted():
    rows = _cell(100, 60, flag=1) + _cell(100, 20, flag=0)
    out = _by_name(_confirm([
        _signal("up", "flag == 1", expect={"relation": ">", "value": 0}),
        _signal("down", "flag == 1", expect={"relation": "<", "value": 0}),
        _signal("big", "flag == 1", expect={"relation": ">", "value": 0.9}),
    ], rows))
    assert out["up"]["claim"]["status"] == "holds" and out["up"]["reject"] is True
    assert out["up"]["details"]["p_value_method"] == "exact (Fisher)"
    assert out["down"]["claim"]["status"] == "contradicted"
    assert out["big"]["claim"]["status"] == "contradicted"
    assert out["big"]["details"]["p_value_method"] == "percentile bootstrap"


def test_header_lists_the_tests_the_p_values_came_from():
    rows = _cell(100, 60, flag=1) + _cell(100, 20, flag=0)
    exact_only = _confirm([_signal("up", "flag == 1")], rows)["adjudication"]
    assert exact_only["p_values"] == ["exact (Fisher)"]
    both = _confirm([
        _signal("up", "flag == 1"),
        _signal("bounded", "flag == 1", expect={"relation": "abs<", "value": 0.9}),
    ], rows)["adjudication"]
    assert both["p_values"] == ["exact (Fisher)", "percentile bootstrap"]
    grouped = _confirm([_signal("up", "flag == 1")],
                       [{**r, "g": i // 4} for i, r in enumerate(rows)],
                       group_col="g")["adjudication"]
    assert grouped["p_values"] == ["percentile bootstrap"]
    empty = _confirm([{"name": "prose"}], rows)["adjudication"]
    assert empty["p_values"] is None and empty["method"] == "none"


def test_all_claims_share_one_correction_family():
    # One strong flag, one borderline flag (p about 0.03 on its own) and
    # twenty flags that are exactly balanced against the outcome.
    rows = []
    for i in range(200):
        fail = i < 80
        row = {"label": "fail" if fail else "pass",
               "strong": int(i < 60 or 160 <= i),          # 60/80 fails, 40/120 passes
               "borderline": int(i < 46 or 80 <= i < 130)}  # 46/80 fails, 50/120 passes
        for k in range(20):
            row[f"null{k}"] = int((i + k) % 2 == 0)         # half of each side
        rows.append(row)
    nulls = [_signal(f"null{k}", f"null{k} == 1") for k in range(20)]
    alone = _by_name(_confirm([_signal("borderline", "borderline == 1")], rows))
    assert 0.01 < alone["borderline"]["p_value"] < 0.05
    assert alone["borderline"]["reject"] is True

    confirm = _confirm(
        [_signal("strong", "strong == 1"), _signal("borderline", "borderline == 1")] + nulls,
        rows,
    )
    meta = confirm["adjudication"]
    assert meta["method"] == "BH" and meta["n_in_family"] == 22
    assert meta["rejected"] == ["strong"]
    verdicts = _by_name(confirm)
    # The borderline flag clears 0.05 on its own and does not survive the family.
    assert verdicts["borderline"]["raw_reject"] is True
    assert verdicts["borderline"]["reject"] is False
    assert verdicts["borderline"]["fdr_corrected"] is False
    assert verdicts["strong"]["fdr_corrected"] is True


def test_header_does_not_name_a_correction_when_nothing_was_tested():
    confirm = _confirm([{"name": "prose_only", "rationale": "no recipe"}],
                       _cell(10, 5, x=1))
    meta = confirm["adjudication"]
    assert meta["method"] == "none" and meta["n_in_family"] == 0
    assert "nothing was corrected" in meta["note"]


def test_small_samples_get_the_exact_test_and_a_warning():
    rows = _cell(4, 4, flag=1) + _cell(4, 0, flag=0)
    out = _by_name(_confirm([_signal("tiny", "flag == 1")], rows))["tiny"]
    assert out["p_value"] == pytest.approx(2 / 70)     # not 1/2001 from a degenerate bootstrap
    assert out["details"]["min_units_per_side"] == 4
    assert "unreliable" in out["details"]["small_sample_warning"]


# ── groups ───────────────────────────────────────────────────────────────────


def test_group_column_makes_intervals_count_groups_not_rows():
    # Ten source items, eight rows each; the flag and the outcome are both
    # properties of the item.
    rows = []
    for g, (flag, fail) in enumerate([(1, 1), (1, 1), (1, 1), (1, 0), (1, 0),
                                      (0, 0), (0, 0), (0, 0), (0, 0), (0, 1)]):
        rows += [{"item": f"i{g}", "flag": flag, "label": "fail" if fail else "pass"}] * 8
    signals = [_signal("flag", "flag == 1")]
    by_row = _by_name(_confirm(signals, rows))["flag"]
    by_item = _by_name(_confirm(signals, rows, group_col="item"))["flag"]
    assert by_row["effect"] == pytest.approx(by_item["effect"]) == pytest.approx(0.4)
    assert by_row["reject"] is True and by_item["reject"] is False
    assert by_item["n_clusters"] == 10 and "n_clusters" not in by_row
    assert by_item["details"]["min_units_per_side"] == 5
    width = lambda v: v["ci"][1] - v["ci"][0]  # noqa: E731
    assert width(by_item) > 1.5 * width(by_row)


# ── numeric outcome, representatives, hypotheses ─────────────────────────────


def test_numeric_outcome_is_compared_by_mean():
    rows = ([{"flag": 1, "score": 70.0 + i % 5} for i in range(40)]
            + [{"flag": 0, "score": 50.0 + i % 5} for i in range(40)])
    values, kind, positive, note = outcome_values(rows, "score")
    assert kind == "continuous" and positive is None and "differences of its mean" in note
    report = {"candidate_signals": [_signal("flag", "flag == 1")], "hypotheses": []}
    confirm = holdout_confirm(report, rows, outcome_col="score")
    verdict = confirm["signal_verdicts"][0]
    assert confirm["outcome"]["kind"] == "continuous" and confirm["n_validate_fail"] == 0
    assert verdict["effect"] == pytest.approx(20.0)
    assert verdict["details"]["mean_flagged"] == pytest.approx(72.0)
    assert "fail_rate_flagged" not in verdict
    assert verdict["reject"] is True


def test_representative_rows_sit_at_the_median_not_the_extreme():
    rows = ([{"id": f"f{i}", "size": float(i), "label": "fail"} for i in range(1, 22)]
            + [{"id": f"p{i}", "size": float(-i), "label": "pass"} for i in range(1, 22)])
    confirm = _confirm([_signal("big", "size > 0")], rows, id_col="id")
    cases = confirm["signal_verdicts"][0]["representative_cases"]
    assert [c["id"] for c in cases["flagged"]] == ["f11", "f10", "f12"]
    assert [c["id"] for c in cases["unflagged"]] == ["p11", "p10", "p12"]
    assert cases["flagged"][0]["row"] == 10            # position in the held-out rows
    # Not offered for a signal whose claim did not hold.
    flat = _confirm([_signal("coin", "size > 100 or size < -100 or size == 1")], rows)
    assert "representative_cases" not in flat["signal_verdicts"][0]

    outcome = [1.0] * 21 + [0.0] * 21
    estimate = estimate_design(SignalRecipe(name="b", expr="size > 0"), rows, outcome)
    assert representative_rows(estimate, rows, outcome, ["size"], k=1) == {
        "unflagged": [31], "flagged": [10]}


class _Judge:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({
            "verdict": "partial", "reasoning": "numbers hold; the cause is open.",
            "needs_surgery": True,
            "experiment": {"data_needed": "200 held-out prompts", "operation": "swap the encoder",
                           "expected_result": "the gap closes", "falsified_by": "the gap stays"},
        })


def test_numeric_predictions_are_checked_by_the_host_not_the_judge():
    # The flag raises the failure rate from 20% to 50%; the control does nothing.
    rows = (_cell(100, 50, flag=1, ctrl=1) + _cell(100, 50, flag=1, ctrl=0)
            + _cell(100, 20, flag=0, ctrl=1) + _cell(100, 20, flag=0, ctrl=0))
    hypotheses = [
        {"statement": "Flagged rows fail more and the control is flat.",
         "predictions": [{"signal": "flag", "relation": ">", "value": 0.0},
                         {"signal": "ctrl", "relation": "abs<", "value": 0.2}]},
        {"statement": "The flag effect is huge.",
         "predictions": [{"signal": "flag", "relation": ">", "value": 0.8}]},
        {"statement": "Something about a signal nobody froze.",
         "predictions": [{"signal": "missing", "relation": ">", "value": 0.0},
                         {"signal": "flag", "relation": "roughly", "value": 1}]},
        {"statement": "No numbers at all."},
    ]
    signals = [_signal("flag", "flag == 1"), _signal("ctrl", "ctrl == 1")]
    confirm = _confirm(signals, rows, hypotheses=hypotheses)
    first, second, third, fourth = confirm["hypothesis_verdicts"]
    assert first["host_verdict"] == "supported"
    assert [r["status"] for r in first["prediction_results"]] == ["holds", "holds"]
    assert second["host_verdict"] == "refuted"
    assert second["prediction_results"][0]["status"] == "contradicted"
    assert third["host_verdict"] == "none"
    assert [r["status"] for r in third["prediction_results"]] == ["not_testable"] * 2
    assert fourth["host_verdict"] == "none" and "prediction_results" not in fourth
    # No judge ran: the host's verdicts stand on their own.
    assert all(h["verdict"] == "not_judged" for h in confirm["hypothesis_verdicts"])
    # Signal claims and hypothesis predictions were corrected together.
    meta = confirm["adjudication"]
    assert (meta["n_signal_claims"], meta["n_hypothesis_claims"], meta["n_in_family"]) == (2, 3, 5)
    assert not any("_p_contrary" in r for h in confirm["hypothesis_verdicts"]
                   for r in h.get("prediction_results", []))


def test_judge_sees_host_results_and_can_fill_in_the_experiment_plan():
    rows = _cell(100, 60, flag=1) + _cell(100, 20, flag=0)
    judge = _Judge()
    kept = {"data_needed": "d", "operation": "o", "expected_result": "e", "falsified_by": "f"}
    confirm = _confirm(
        [_signal("flag", "flag == 1")], rows, judge=judge,
        hypotheses=[
            {"statement": "A cause.", "predictions": [
                {"signal": "flag", "relation": ">", "value": 0.0}]},
            {"statement": "Another cause.", "experiment": kept},
        ],
    )
    first, second = confirm["hypothesis_verdicts"]
    assert "PREDICTIONS THE HOST ALREADY CHECKED" in judge.prompts[0]
    assert "flag > 0.0: holds" in judge.prompts[0]
    assert "PREDICTIONS THE HOST ALREADY CHECKED" not in judge.prompts[1]
    assert first["verdict"] == "partial" and first["host_verdict"] == "supported"
    assert first["experiment"]["falsified_by"] == "the gap stays"
    assert second["experiment"] == kept                # the proposer's plan is not overwritten
