"""Standalone library entry point: evalvitals.analysis.api.explore().

Same underlying pipeline as explore_run.run_explore (CLI-facing), but library
style: takes a path OR in-memory records, returns a structured result instead
of printing + an exit code, and only persists artifacts when `out` is given.
"""

from __future__ import annotations

import json

from evalvitals.analysis import api as explore_api
from evalvitals.analysis.api import ExploreRunResult, explore
from evalvitals.analysis.explorer import ExploratoryAnalysisReport
from evalvitals.analysis.hypothesis_agent import Hypothesis


class _FakeExploreAgent:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def explore_records(self, records, **_kwargs):
        return ExploratoryAnalysisReport(
            question="What predicts yield?",
            ok=True,
            observations=[f"{len(records)} rows, no missing values."],
            takeaways=[],
            data_profile={"loaded_rows": len(records)},
        )

    def explore_path(self, path, **_kwargs):
        return ExploratoryAnalysisReport(
            question="What predicts yield?",
            ok=True,
            observations=["30 rows loaded from disk."],
            takeaways=[],
            data_profile={"loaded_rows": 30},
        )


class _FakeHypothesisAgent:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def propose(self, report, **_kwargs):
        return [Hypothesis(statement="Temperature drives yield.", basis="b", test_design="t")]


def test_explore_accepts_in_memory_records_and_skips_persistence_by_default(monkeypatch):
    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeExploreAgent)
    monkeypatch.setattr(explore_api, "HypothesisAgent", _FakeHypothesisAgent)

    records = [{"case_id": "c0", "label": "pass"}, {"case_id": "c1", "label": "fail"}]
    result = explore(records, provider="llm")

    assert isinstance(result, ExploreRunResult)
    assert result.ok is True
    assert result.out_dir is None
    assert result.hypotheses == [
        {"statement": "Temperature drives yield.", "plain_statement": "", "basis": "b", "test_design": "t"}
    ]
    assert "2 rows" in result.report.observations[0]


def test_explore_accepts_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeExploreAgent)
    monkeypatch.setattr(explore_api, "HypothesisAgent", _FakeHypothesisAgent)

    result = explore(tmp_path / "records.json", provider="llm")

    assert result.ok is True
    assert result.report.data_profile["loaded_rows"] == 30


def test_explore_persists_artifacts_only_when_out_is_given(tmp_path, monkeypatch):
    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeExploreAgent)
    monkeypatch.setattr(explore_api, "HypothesisAgent", _FakeHypothesisAgent)

    out_dir = tmp_path / "out"
    result = explore([{"case_id": "c0", "label": "pass"}], provider="llm", out=out_dir)

    assert result.out_dir == out_dir.resolve()
    saved = json.loads((out_dir / "exploratory_report.json").read_text())
    assert saved["ok"] is True
    assert saved["hypotheses"] == [
        {"statement": "Temperature drives yield.", "plain_statement": "", "basis": "b", "test_design": "t"}
    ]


def test_explore_persists_m2_partial_and_emits_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeExploreAgent)
    monkeypatch.setattr(explore_api, "HypothesisAgent", _FakeHypothesisAgent)

    class _Sink:
        def __init__(self):
            self.events = []

        def emit(self, stage, status, message, **kwargs):
            self.events.append((stage, status, message, kwargs))

    sink = _Sink()
    out_dir = tmp_path / "out"
    explore([{"case_id": "c0", "label": "pass"}], provider="llm", out=out_dir, progress_sink=sink)

    assert json.loads((out_dir / "partial_report.json").read_text())["hypotheses"] == []
    assert (out_dir / "exploratory_report.json").exists()
    assert [(stage, status) for stage, status, *_ in sink.events] == [
        ("m2", "started"), ("m2", "completed"), ("m3", "started"),
        ("m3", "completed"), ("persist", "completed"),
    ]


def test_explore_skips_m3_when_disabled(monkeypatch):
    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeExploreAgent)

    def _boom(*_args, **_kwargs):
        raise AssertionError("HypothesisAgent should not be constructed when disabled")

    monkeypatch.setattr(explore_api, "HypothesisAgent", _boom)

    result = explore([{"case_id": "c0", "label": "pass"}], provider="llm", propose_hypotheses=False)

    assert result.hypotheses == []


def test_top_level_evalvitals_explore_is_the_same_function():
    import evalvitals

    assert evalvitals.explore is explore
