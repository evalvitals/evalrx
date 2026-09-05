

def test_outcome_regrades_go_to_the_sanity_lane():
    """'Is the answer correct' columns (an analyzer's own re-grade of the
    outcome) agree with the label at ~84% — under the leak threshold, but they
    were every BH survivor on qwen3.5-2b/minervamath and then M4's tautological
    'evidence'. They are isolated by NAME; derived mechanism flags stay."""
    from evalrx.analysis.stats_tools import (
        OUTCOME_REGRADE_METRICS,
        StatsInput,
        isolate_label_leaks,
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


def test_count_of_correct_samples_is_an_outcome_regrade_too():
    """coverage_verification_gap.n_correct is 'how many of k samples were
    correct' -- the outcome graded k times. As a 0..k count it dodges the
    binary-only leak score, and with degenerate sampling it is {0, k}, i.e.
    the label; spatial457/qwen2.5-vl (2026-08-20) had it as the sole BH
    survivor. n_unique (sample diversity) is a mechanism signal and stays."""
    from evalrx.analysis.stats_tools import StatsInput, isolate_label_leaks

    labels = {f"c{i}": i < 10 for i in range(20)}
    inp = StatsInput(
        labels=labels,
        per_case={
            "coverage_verification_gap.n_correct": {f"c{i}": 0.0 if i < 10 else 5.0 for i in range(20)},
            "coverage_verification_gap.n_unique": {f"c{i}": float(1 + i % 3) for i in range(20)},
        },
    )
    moved = isolate_label_leaks(inp)
    assert "coverage_verification_gap.n_correct" in moved
    assert "coverage_verification_gap.n_unique" not in moved
    assert "coverage_verification_gap.n_unique" in inp.per_case
