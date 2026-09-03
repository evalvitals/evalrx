"""WS1 — label-leak detection + sanity-lane isolation (the deferred 'leak-1' check).

A leak is the FAIL label *in disguise* (a binary flag that recomputes the
outcome), NOT a strong predictor. A continuous feature that perfectly separates
the classes is legitimate discovery and must never be misfiled.
"""

from __future__ import annotations

from evalrx.analysis.stats_tools import (
    StatsInput,
    build_stats_input,
    describe_data,
    isolate_label_leaks,
    label_leak_score,
)
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result


def _labels(n_fail: int, n_pass: int) -> dict[str, bool]:
    labels = {f"f{i}": True for i in range(n_fail)}
    labels.update({f"p{i}": False for i in range(n_pass)})
    return labels


class TestLabelLeakScore:
    def test_binary_signal_equal_to_label_is_flagged(self):
        labels = _labels(15, 15)
        # flag == 1 exactly on the FAIL cases → the label in disguise
        sig = {cid: (1.0 if is_fail else 0.0) for cid, is_fail in labels.items()}
        sc = label_leak_score(sig, labels)
        assert sc["leak"] is True
        assert sc["kind"] == "binary"
        assert sc["score"] >= 0.95

    def test_continuous_perfect_separator_is_not_flagged(self):
        labels = _labels(15, 15)
        # object size perfectly separates (20 vs 80) — a finding, NOT a leak
        sig = {cid: (20.0 if is_fail else 80.0) for cid, is_fail in labels.items()}
        sc = label_leak_score(sig, labels)
        assert sc["leak"] is False
        assert sc["kind"] == "continuous"
        # the rank-separation score is still reported for transparency
        assert sc["score"] == 1.0

    def test_strong_but_imperfect_binary_is_not_flagged(self):
        # ~80% accuracy binary flag → a real signal, below the near-equality bar
        labels = _labels(20, 20)
        sig = {}
        for i, (cid, is_fail) in enumerate(labels.items()):
            # 4 of every 20 disagree → ~0.8 accuracy
            sig[cid] = float(is_fail) if i % 5 != 0 else float(not is_fail)
        sc = label_leak_score(sig, labels)
        assert sc["leak"] is False
        assert sc["score"] < 0.95

    def test_small_n_is_not_flagged(self):
        labels = _labels(2, 2)  # below _LEAK_MIN_N
        sig = {cid: (1.0 if is_fail else 0.0) for cid, is_fail in labels.items()}
        assert label_leak_score(sig, labels)["leak"] is False


class TestIsolateLabelLeaks:
    def test_moves_leak_to_sanity_keeps_real_signals(self):
        labels = _labels(15, 15)
        leak = {cid: (1.0 if is_fail else 0.0) for cid, is_fail in labels.items()}
        real = {cid: (20.0 if is_fail else 80.0) for cid, is_fail in labels.items()}
        inp = StatsInput(labels=labels, per_case={"probe_flag": leak, "obj_size": real})
        moved = isolate_label_leaks(inp)
        assert "probe_flag" in moved
        assert "probe_flag" in inp.sanity
        assert "obj_size" in inp.per_case        # real signal untouched
        assert "probe_flag" not in inp.per_case
        assert describe_data(inp)["sanity_signals"] == ["probe_flag"]

    def test_denylist_isolates_by_name(self):
        labels = _labels(15, 15)
        weak = {cid: float(i % 2) for i, cid in enumerate(labels)}
        inp = StatsInput(labels=labels, per_case={"some.audit_col": weak})
        moved = isolate_label_leaks(inp, denylist=("audit_col",))
        assert "some.audit_col" in moved and "some.audit_col" in inp.sanity

    def test_idempotent(self):
        labels = _labels(15, 15)
        leak = {cid: (1.0 if is_fail else 0.0) for cid, is_fail in labels.items()}
        inp = StatsInput(labels=labels, per_case={"probe_flag": dict(leak)})
        isolate_label_leaks(inp)
        again = isolate_label_leaks(inp)  # nothing left to move
        assert again == {}
        assert list(inp.sanity) == ["probe_flag"]


class TestBuildStatsInputIsolation:
    def test_label_equal_per_case_column_routed_to_sanity(self):
        cases = []
        per_case = []
        for i in range(15):
            cases.append(FailureCase(id=f"f{i}", inputs=Inputs(prompt="q"), label=Label.FAIL))
            per_case.append({"sample_id": f"f{i}", "is_fail_flag": 1, "attn": 0.1 + i * 0.001})
        for i in range(15):
            cases.append(FailureCase(id=f"p{i}", inputs=Inputs(prompt="q"), label=Label.PASS))
            per_case.append({"sample_id": f"p{i}", "is_fail_flag": 0, "attn": 0.6 + i * 0.001})
        results = {
            "probe": Result(analyzer="probe", model="m", cases=CaseBatch(cases),
                            findings={"per_case": per_case})
        }
        inp = build_stats_input(results, CaseBatch(cases))
        assert "probe.is_fail_flag" in inp.sanity         # the label-equal flag
        assert "probe.is_fail_flag" not in inp.per_case
        assert "probe.attn" in inp.per_case               # the real continuous signal
