"""The Case Studio shows the frozen batch the way the run split it.

The loop divides the batch before anything runs -- explore for M1-M3, a
withheld pool for M4 / M5 -- and the report has to show that division, not a
flat list. New runs tag each case record; runs from before the tag are read
from ``run_start``'s explore count and the order the cases were logged in.
"""

from __future__ import annotations

from evalrx.core import CaseBatch, FailureCase, Label
from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
from evalrx.reporting.dynamic import build_report_data


def _case(i: int) -> FailureCase:
    return FailureCase.from_prompt(
        f"question {i}", id=f"c{i}", expected="yes",
        observed="yes" if i % 2 else "no", label=Label.PASS if i % 2 else Label.FAIL,
    )


def _write_run(root, *, n_explore: int, batches) -> None:
    logger = RunLoggerV2(root)
    logger.log_run_start({"model": "m", "benchmark_name": "b", "n_cases": n_explore})
    for cases, split in batches:
        logger.log_cases(CaseBatch(cases), split=split)
    logger.close()


def _by_split(data) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for case in data["cases"]:
        out.setdefault(case.get("split") or "", []).append(case["id"])
    return out


def test_tagged_records_are_read_as_recorded(tmp_path):
    _write_run(tmp_path, n_explore=3, batches=[
        ([_case(0), _case(1), _case(2)], "explore"),
        ([_case(3), _case(4)], "confirm"),
        ([_case(5)], "test"),
    ])
    data = build_report_data(tmp_path)
    assert _by_split(data) == {"explore": ["c0", "c1", "c2"], "confirm": ["c3", "c4"], "test": ["c5"]}
    rows = data["setting"]["partitions"]
    assert [(r["code"], r["label"], r["n"]) for r in rows] == [
        ("E", "Explore", 3), ("H", "Held-out", 2), ("C", "Confirm", 1),
    ]
    assert not any(r["inferred"] for r in rows)
    # The sheet's headline counts come from the same table, not from the
    # length of the case-record list (which holds every partition).
    headline = data["case_study"]["headline"] if data.get("case_study") else None
    if headline:
        assert (headline["n_explore"], headline["n_heldout"]) == (3, 3)


def test_an_untagged_two_way_run_is_inferred_from_the_record_order(tmp_path):
    # Logged the way run() logs: explore first, then the withheld pool, with
    # run_start counting only the explore partition.
    _write_run(tmp_path, n_explore=2, batches=[
        ([_case(0), _case(1)], None),
        ([_case(2), _case(3), _case(4)], None),
    ])
    data = build_report_data(tmp_path)
    assert _by_split(data) == {"explore": ["c0", "c1"], "confirm": ["c2", "c3", "c4"]}
    rows = data["setting"]["partitions"]
    # One withheld pool served as both the held-out and the confirm set.
    assert [(r["code"], r["n"], r["inferred"]) for r in rows] == [("E", 2, True), ("H/C", 3, True)]


def test_a_run_without_a_split_shows_no_partitions(tmp_path):
    _write_run(tmp_path, n_explore=3, batches=[([_case(0), _case(1), _case(2)], None)])
    data = build_report_data(tmp_path)
    assert data["setting"]["partitions"] == []
    assert all(case.get("split") is None for case in data["cases"])
