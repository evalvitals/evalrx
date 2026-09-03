"""Failure-mode clustering (analysis.failure_modes): the numpy fallback path
runs unconditionally (no sklearn/hdbscan dependency); sklearn-backed paths are
skipped when not installed.
"""

from __future__ import annotations

import sys

import pytest

from evalrx.analysis.failure_modes import (
    FailureMode,
    FailureModeReport,
    _boundary_contrast_pairs,
    _cluster_cosine_greedy,
    _compute_error_signals,
    _hash_vectorize,
    _rank_by_centroid_distance,
    cluster_failures,
)


def _two_mode_records(n_per_mode: int = 8) -> list[dict]:
    records = []
    for i in range(n_per_mode):
        records.append({
            "case_id": f"small_{i}", "label": "FAIL",
            "prompt": "small object missed detection too tiny to see",
        })
    for i in range(n_per_mode):
        records.append({
            "case_id": f"occl_{i}", "label": "FAIL",
            "prompt": "occluded object hidden behind another blocking item",
        })
    for i in range(5):
        records.append({
            "case_id": f"pass_{i}", "label": "PASS",
            "prompt": "correct detection clean and clear visible",
        })
    return records


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_fail_cases_returns_empty_report():
    report = cluster_failures([{"case_id": "x", "label": "PASS", "prompt": "ok"}])
    assert report.n_fail_cases == 0
    assert report.clusters == []
    assert report.method == "none"


def test_too_few_fail_cases_returns_a_single_cluster():
    records = [
        {"case_id": "a", "label": "FAIL", "prompt": "a weird one-off bug"},
        {"case_id": "b", "label": "PASS", "prompt": "fine"},
    ]
    report = cluster_failures(records, min_cluster_size=3)
    assert report.method == "single_cluster"
    assert len(report.clusters) == 1
    assert report.clusters[0].case_ids == ["a"]
    assert report.n_fail_cases == 1


# ---------------------------------------------------------------------------
# Numpy fallback (always runs, no optional deps)
# ---------------------------------------------------------------------------

def test_hash_vectorize_is_deterministic_and_normalized():
    import numpy as np

    X = _hash_vectorize(["hello world", "hello world", "goodbye"], n_features=64)
    assert X.shape == (3, 64)
    assert (X[0] == X[1]).all()  # identical text -> identical vector
    assert not (X[0] == X[2]).all()
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)


def test_cosine_greedy_recovers_two_obviously_distinct_groups():
    X = _hash_vectorize([
        "small object missed detection too tiny to see",
        "small object missed detection too tiny to see",
        "occluded object hidden behind another blocking item",
        "occluded object hidden behind another blocking item",
    ], n_features=128)
    labels = _cluster_cosine_greedy(X, max_clusters=4, threshold=0.5)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_failures_forced_cosine_greedy_recovers_two_modes():
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=4,
    )
    assert report.method == "cosine_greedy"
    assert report.n_fail_cases == 16
    sizes = sorted(c.size for c in report.clusters)
    assert sizes[-1] >= 6  # at least one large, coherent cluster recovered
    # every FAIL case lands in exactly one cluster or unclustered
    all_ids = {cid for c in report.clusters for cid in c.case_ids} | set(report.unclustered_ids)
    assert len(all_ids) == 16


def test_cluster_failures_falls_back_to_hashing_vectorizer_when_sklearn_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=4,
    )
    assert report.params["vectorizer"] == "hashing"
    assert report.n_fail_cases == 16


def test_cluster_failures_forced_hdbscan_raises_when_not_installed():
    with pytest.raises(ImportError):
        cluster_failures(_two_mode_records(), method="hdbscan")


def test_auto_method_falls_back_to_cosine_greedy_when_no_optional_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "hdbscan", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    report = cluster_failures(_two_mode_records(), min_cluster_size=3, max_clusters=4)
    assert report.method == "cosine_greedy"
    assert report.params["vectorizer"] == "hashing"


# ---------------------------------------------------------------------------
# sklearn-backed paths (skipped when not installed)
# ---------------------------------------------------------------------------

def test_cluster_failures_with_sklearn_agglomerative_recovers_two_modes():
    pytest.importorskip("sklearn")
    report = cluster_failures(
        _two_mode_records(), method="agglomerative", min_cluster_size=3, max_clusters=2,
    )
    assert report.method == "agglomerative"
    assert len(report.clusters) == 2
    sizes = sorted(c.size for c in report.clusters)
    assert sizes == [8, 8]


def test_cluster_failures_with_hdbscan_or_skip():
    pytest.importorskip("hdbscan")
    report = cluster_failures(_two_mode_records(), method="hdbscan", min_cluster_size=3)
    assert report.method == "hdbscan"


# ---------------------------------------------------------------------------
# LLM naming tier
# ---------------------------------------------------------------------------

class _FakeJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def generate(self, prompt, **kwargs) -> str:
        self.calls += 1
        return next(self._responses)


def test_llm_naming_renames_clusters_on_valid_response():
    judge = _FakeJudge([
        '[{"cluster_id": 0, "name": "small_object_miss", "description": "objects too small to detect"},'
        ' {"cluster_id": 1, "name": "occlusion_miss", "description": "objects hidden behind others"}]'
    ])
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3,
        max_clusters=2, judge=judge,
    )
    names = {c.name for c in report.clusters}
    assert "small_object_miss" in names or "occlusion_miss" in names
    assert report.named_by.startswith("llm:")
    assert judge.calls == 1


def test_llm_naming_repairs_once_then_falls_back_to_top_terms(monkeypatch):
    judge = _FakeJudge(["not json at all", "still not json"])
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3,
        max_clusters=2, judge=judge,
    )
    # Falls back to top_terms naming — never crashes, never leaves clusters unnamed.
    assert report.named_by == "top_terms"
    assert all(c.name for c in report.clusters)
    assert judge.calls == 2  # one repair attempt, then give up


# ---------------------------------------------------------------------------
# to_dict() contract (pyod-style stable-key check)
# ---------------------------------------------------------------------------

def test_failure_mode_to_dict_has_stable_keys():
    fm = FailureMode(name="n", description="d", case_ids=["a"], exemplars=[{"x": 1}], size=1, top_terms=["t"])
    assert set(fm.to_dict()) == {
        "name", "description", "case_ids", "exemplars", "size", "top_terms", "boundary_pairs",
    }


def test_failure_mode_report_to_dict_has_stable_keys():
    report = FailureModeReport(clusters=[], n_fail_cases=0, unclustered_ids=[], method="none")
    assert set(report.to_dict()) == {
        "clusters", "n_fail_cases", "unclustered_ids", "method", "named_by", "params",
    }


def test_as_hypothesis_context_empty_when_no_clusters():
    assert FailureModeReport().as_hypothesis_context() == ""


def test_as_hypothesis_context_summarizes_clusters():
    report = FailureModeReport(
        clusters=[FailureMode(name="small_obj", description="too small", size=5)],
        method="cosine_greedy",
    )
    text = report.as_hypothesis_context()
    assert "small_obj" in text and "too small" in text and "n=5" in text


# ---------------------------------------------------------------------------
# Failure-aware embedding (expected_col / error_fn) — opt-in, default off
# ---------------------------------------------------------------------------

def test_compute_error_signals_prefers_injected_error_fn():
    fail_rows = [(0, {"expected": "a", "output": "b"})]
    signals = _compute_error_signals(
        fail_rows, expected_col="expected", error_fn=lambda r: "CUSTOM", judge=None,
    )
    assert signals == ["CUSTOM"]


def test_compute_error_signals_empty_when_unconfigured():
    """Default behavior (no expected_col, no error_fn): today's topic-only path."""
    fail_rows = [(0, {"expected": "a", "output": "b"})]
    signals = _compute_error_signals(fail_rows, expected_col=None, error_fn=None, judge=None)
    assert signals == [""]


def test_compute_error_signals_deterministic_fallback_without_judge():
    fail_rows = [(0, {"expected": "a", "output": "b"})]
    signals = _compute_error_signals(fail_rows, expected_col="expected", error_fn=None, judge=None)
    assert signals == ["expected=a got=b"]


def test_compute_error_signals_llm_batched_success():
    judge = _FakeJudge([
        '[{"index": 0, "error": "off by one"}, {"index": 1, "error": "wrong entity"}]'
    ])
    fail_rows = [(0, {"expected": "10", "output": "9"}), (1, {"expected": "dog", "output": "cat"})]
    signals = _compute_error_signals(fail_rows, expected_col="expected", error_fn=None, judge=judge)
    assert signals == ["off by one", "wrong entity"]
    assert judge.calls == 1


def test_compute_error_signals_llm_falls_back_to_deterministic_on_malformed():
    judge = _FakeJudge(["not json at all", "still not json"])
    fail_rows = [(0, {"expected": "10", "output": "9"})]
    signals = _compute_error_signals(fail_rows, expected_col="expected", error_fn=None, judge=judge)
    assert signals == ["expected=10 got=9"]
    assert judge.calls == 2


def _same_topic_two_mechanisms_records() -> list[dict]:
    records = []
    for i in range(4):
        records.append({"case_id": f"a{i}", "label": "FAIL", "prompt": "q", "output": "5", "expected": "10"})
    for i in range(4):
        records.append({"case_id": f"b{i}", "label": "FAIL", "prompt": "q", "output": "cat", "expected": "dog"})
    for i in range(3):
        records.append({"case_id": f"p{i}", "label": "PASS", "prompt": "q", "output": "ok", "expected": "ok"})
    return records


def test_error_fn_lets_clustering_separate_same_topic_different_mechanisms():
    records = _same_topic_two_mechanisms_records()
    # Restrict to the (identical, "q") prompt text only, so "output"/"expected"
    # can't leak in via _row_text's auto-detected default text columns — the
    # only way the mismatch can reach clustering is through error_fn itself.
    plain = cluster_failures(
        records, text_cols=["prompt"], method="cosine_greedy", min_cluster_size=3, max_clusters=4,
    )
    assert len(plain.clusters) == 1

    # With error_fn folding in the mismatch, the two mechanisms separate.
    aware = cluster_failures(
        records, text_cols=["prompt"], method="cosine_greedy", min_cluster_size=3, max_clusters=4,
        error_fn=lambda r: f"expected {r['expected']} got {r['output']}",
    )
    assert len(aware.clusters) == 2
    sizes = sorted(c.size for c in aware.clusters)
    assert sizes == [4, 4]


# ---------------------------------------------------------------------------
# Boundary-aware induction (boundary_aware=True) — opt-in, default off
# ---------------------------------------------------------------------------

def test_rank_by_centroid_distance_separates_outlier_into_boundary():
    import numpy as np

    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [0.0, 0.1]])
    core, boundary = _rank_by_centroid_distance([0, 1, 2, 3], X, n_core=2, n_boundary=1)
    assert 2 not in core
    assert boundary == [2]


def test_boundary_contrast_pairs_picks_nearest_pass_by_cosine():
    import numpy as np

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    fail_rows = [(0, {"id": "f0"}), (1, {"id": "f1"})]
    pass_rows = [{"id": "p0"}, {"id": "p1"}]
    X_pass = np.array([[0.9, 0.1], [0.1, 0.9]])
    pairs = _boundary_contrast_pairs([0, 1], X, fail_rows, pass_rows, X_pass)
    assert pairs[0]["fail"]["id"] == "f0" and pairs[0]["nearest_pass"]["id"] == "p0"
    assert pairs[1]["fail"]["id"] == "f1" and pairs[1]["nearest_pass"]["id"] == "p1"


def test_boundary_contrast_pairs_empty_without_pass_rows():
    import numpy as np

    X = np.array([[1.0, 0.0]])
    assert _boundary_contrast_pairs([0], X, [(0, {"id": "f0"})], [], X[:0]) == []


def test_boundary_pairs_empty_by_default():
    report = cluster_failures(_two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=4)
    assert all(c.boundary_pairs == [] for c in report.clusters)


def test_boundary_aware_produces_nearest_pass_contrast_pairs():
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=4,
        boundary_aware=True,
    )
    assert report.n_fail_cases == 16
    for cluster in report.clusters:
        assert cluster.boundary_pairs
        for bp in cluster.boundary_pairs:
            assert bp["nearest_pass"]["label"] == "PASS"
            assert "similarity" in bp


def test_boundary_aware_with_no_pass_rows_does_not_crash():
    records = [r for r in _two_mode_records() if r["label"] == "FAIL"]
    report = cluster_failures(
        records, method="cosine_greedy", min_cluster_size=3, max_clusters=4, boundary_aware=True,
    )
    assert all(c.boundary_pairs == [] for c in report.clusters)


class _CapturingJudge:
    """Like _FakeJudge, but records the prompt it was last called with."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt, **kwargs) -> str:
        self.last_prompt = prompt
        return self.response


def test_boundary_pairs_included_in_naming_prompt():
    judge = _CapturingJudge(
        '[{"cluster_id": 0, "name": "n0", "description": "d0"},'
        ' {"cluster_id": 1, "name": "n1", "description": "d1"}]'
    )
    cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=2,
        boundary_aware=True, judge=judge,
    )
    assert "Boundary contrast" in judge.last_prompt
