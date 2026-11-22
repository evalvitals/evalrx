"""Group-aware split, group-column suggestion, explicit roles and the leakage check."""

from __future__ import annotations

import random

from evalrx.analysis.fused_pipeline import _split_records, group_overlap
from evalrx.analysis.holdout import split_records
from evalrx.analysis.profile import (
    intraclass_correlation,
    leakage_suspects,
    profile_records,
    suggest_group_columns,
)


def _grouped_rows(seed: int = 0, n_groups: int = 200) -> list[dict]:
    """Several rows per source item; rows of an item tend to share the outcome."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for g in range(n_groups):
        tendency = rng.random()
        for _ in range(rng.randint(2, 8)):
            score = tendency + rng.gauss(0, 0.15)
            rows.append({
                "case_id": f"c{len(rows)}",
                "clip": f"clip{g}",
                "model": rng.choice(["a", "b", "c"]),
                "noise": f"n{rng.randint(0, 40)}",
                "x": round(rng.random(), 2),
                "label": "fail" if score < 0.4 else "pass",
            })
    return rows


def test_group_split_keeps_every_group_on_one_side():
    rows = _grouped_rows()
    explore, holdout = split_records(rows, 0.3, seed=0, outcome_col="label", group_col="clip")
    assert holdout is not None
    assert not ({r["clip"] for r in explore} & {r["clip"] for r in holdout})
    assert group_overlap(explore, holdout, "clip")["share"] == 0.0
    # Complete, and close to the requested share and to the overall fail rate.
    assert len(explore) + len(holdout) == len(rows)
    assert abs(len(holdout) / len(rows) - 0.3) < 0.03

    def fail_rate(rs: list[dict]) -> float:
        return sum(r["label"] == "fail" for r in rs) / len(rs)

    assert abs(fail_rate(holdout) - fail_rate(rows)) < 0.05
    # Deterministic for a seed, different for another.
    again = split_records(rows, 0.3, seed=0, outcome_col="label", group_col="clip")[1]
    other = split_records(rows, 0.3, seed=1, outcome_col="label", group_col="clip")[1]
    assert [r["case_id"] for r in again] == [r["case_id"] for r in holdout]
    assert [r["case_id"] for r in other] != [r["case_id"] for r in holdout]


def test_row_level_split_leaves_most_held_out_rows_in_explored_groups():
    rows = _grouped_rows()
    explore, holdout = split_records(rows, 0.3, seed=0, outcome_col="label")
    assert group_overlap(explore, holdout, "clip")["share"] > 0.8


def test_rows_without_a_group_are_their_own_group():
    rows = [{"g": None, "label": "fail" if i % 3 == 0 else "pass"} for i in range(60)]
    explore, holdout = _split_records(rows, frac=0.3, seed=0, label_col="label", group_col="g")
    assert holdout is not None and len(explore) + len(holdout) == 60
    assert 12 <= len(holdout) <= 24


def test_group_split_handles_a_many_valued_outcome():
    rng = random.Random(3)
    rows = []
    for g in range(150):
        base = rng.random()
        rows.extend({"g": f"g{g}", "score": base + rng.gauss(0, 0.1)}
                    for _ in range(rng.randint(1, 9)))
    explore, holdout = _split_records(rows, frac=0.3, seed=0, label_col="score", group_col="g")
    assert holdout is not None
    assert abs(len(holdout) / len(rows) - 0.3) < 0.04
    assert group_overlap(explore, holdout, "g")["share"] == 0.0

    def mean(rs: list[dict]) -> float:
        return sum(r["score"] for r in rs) / len(rs)

    assert abs(mean(holdout) - mean(explore)) < 0.08


def test_intraclass_correlation_extremes():
    assert intraclass_correlation(["a", "a", "b", "b"], [0.0, 0.0, 1.0, 1.0]) == 1.0
    icc = intraclass_correlation(["a", "a", "b", "b"], [0.0, 1.0, 0.0, 1.0])
    assert icc is not None and icc < 0
    assert intraclass_correlation(["a", "b"], [0.0, 1.0]) is None   # no repeats
    assert intraclass_correlation(["a", "a"], [0.0, 1.0]) is None   # one group


def test_profile_suggests_the_grouping_column_from_the_data():
    rows = _grouped_rows()
    profile = profile_records(rows, outcome_col="label")
    (suggestion,) = profile.suggested_group_columns
    assert suggestion["column"] == "clip"
    assert suggestion["n_groups"] == 200 and suggestion["icc"] > 0.3
    assert any("'clip'" in w and "group column" in w for w in profile.warnings)
    # A factor with three levels, an id that never repeats and a float are not units.
    assert suggest_group_columns(rows, outcome_col="label", exclude={"clip"}) == []


def test_explicit_roles_override_the_name_heuristic():
    rows = _grouped_rows()
    for i, r in enumerate(rows):
        r["batch"] = f"2024-{1 + i % 12:02d}"
        r["wave"] = i % 4
    inferred = profile_records(rows, outcome_col="label")
    assert "clip" not in inferred.group_columns
    assert "batch" in inferred.time_columns        # ISO year-month values
    assert "wave" not in inferred.time_columns

    named = profile_records(rows, outcome_col="label", group_cols=["clip"],
                            time_cols=["wave", "absent"])
    assert "clip" in named.group_columns and named.columns["clip"].role == "group"
    assert "wave" in named.time_columns
    assert named.suggested_group_columns == []     # already named
    assert any("'absent'" in w and "not in the data" in w for w in named.warnings)


def test_time_like_names_are_recognised_as_whole_words_only():
    rows = [{"month_reported": "x", "created_at": "y", "candidate": "z",
             "content_group": "g", "label": "pass"} for _ in range(4)]
    profile = profile_records(rows, outcome_col="label")
    assert set(profile.time_columns) == {"month_reported", "created_at"}
    assert "content_group" in profile.group_columns
    assert profile.columns["candidate"].role == "predictor"


def test_leakage_check_flags_columns_that_determine_the_outcome():
    rng = random.Random(1)
    rows = []
    for i in range(300):
        score = rng.random()
        fail = score < 0.3
        rows.append({
            "case_id": f"c{i}",
            "label": "fail" if fail else "pass",
            "grader_note": "wrong" if fail else "ok",      # written after grading
            "raw_score": score,                            # the outcome before the cut
            "length": rng.randint(5, 50),
            "mild": (score + rng.gauss(0, 0.5)),
        })
    found = {s["column"]: s for s in leakage_suspects(rows, outcome_col="label",
                                                      exclude={"case_id"})}
    assert set(found) == {"grader_note", "raw_score"}
    assert found["grader_note"]["measure"] == "accuracy_from_value"
    assert found["raw_score"]["measure"] == "auc"

    profile = profile_records(rows, outcome_col="label")
    assert profile.columns["raw_score"].leakage_suspect is True
    assert profile.columns["length"].leakage_suspect is False
    assert sum("almost fully determines" in w for w in profile.warnings) == 2


def test_leakage_check_on_a_numeric_outcome_uses_correlation():
    rng = random.Random(2)
    rows = [{"y": v, "copy": 2 * v + 1, "other": rng.random()}
            for v in (rng.random() for _ in range(100))]
    (found,) = leakage_suspects(rows, outcome_col="y")
    assert found["column"] == "copy" and found["measure"] == "correlation"
