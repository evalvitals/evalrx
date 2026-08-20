

def test_outcome_regrades_go_to_the_sanity_lane():
    """'Is the answer correct' columns (an analyzer's own re-grade of the
    outcome) agree with the label at ~84% — under the leak threshold, but they
    were every BH survivor on qwen3.5-2b/minervamath and then M5's tautological
    'evidence'. They are isolated by NAME; derived mechanism flags stay."""
    from evalvitals.analysis.stats_tools import (
        OUTCOME_REGRADE_METRICS, StatsInput, isolate_label_leaks,
    )

    labels = {f"c{i}": i < 10 for i in range(20)}
    # 84% agreement with the label — a re-grade with a different matcher
    regrade = {f"c{i}": float((i < 10) != (i % 6 == 0)) for i in range(20)}
    inp = StatsInput(
        labels=labels,
        per_case={
            "answer_extraction_audit.gold_in_answer_region": dict(regrade),
            "self_repair.baseline_correct": dict(regrade),
            "coverage_verification_gap.majority_correct": dict(regrade),
            "answer_extraction_audit.label_disagrees": {f"c{i}": float(i % 6 == 0) for i in range(20)},
            "self_repair.changed_answer": {f"c{i}": float(i % 3 == 0) for i in range(20)},
        },
    )
    moved = isolate_label_leaks(inp)
    assert set(moved) == {
        "answer_extraction_audit.gold_in_answer_region",
        "self_repair.baseline_correct",
        "coverage_verification_gap.majority_correct",
    }
    assert all("outcome re-grade" in r for r in moved.values())
    assert set(inp.per_case) == {"answer_extraction_audit.label_disagrees",
                                 "self_repair.changed_answer"}
    assert "gold_in_answer_region" in OUTCOME_REGRADE_METRICS
