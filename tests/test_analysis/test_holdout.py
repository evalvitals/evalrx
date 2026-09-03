"""Generic held-out verification (`analysis.holdout`) and its explore-CLI wiring.

The deco_hallu pipeline's phase 2 generalized to any dataset: split before
exploration, frozen-recipe re-evaluation on the held-out rows, e-BH
adjudication with split_label="held_out", optional LLM-judge hypothesis
grading — the upload workbench's "Explore + held-out verification" mode.
"""

from __future__ import annotations

import json

from evalrx.analysis.holdout import (
    failure_indicator,
    holdout_confirm,
    split_records,
)


def _rows(n_fail: int = 20, n_pass: int = 80) -> list[dict]:
    rows = []
    for i in range(n_fail):
        rows.append({"case": f"f{i}", "x": 0.9, "label": "fail"})
    for i in range(n_pass):
        rows.append({"case": f"p{i}", "x": 0.1, "label": "pass"})
    return rows


# ── split_records ────────────────────────────────────────────────────────────


def test_split_is_deterministic_and_stratified():
    rows = _rows()
    explore1, hold1 = split_records(rows, 0.4, seed=0, outcome_col="label")
    explore2, hold2 = split_records(rows, 0.4, seed=0, outcome_col="label")
    assert hold1 is not None
    assert len(hold1) == 40 and len(explore1) == 60
    assert [r["case"] for r in hold1] == [r["case"] for r in hold2]
    # stratified: the 20% fail share survives in both halves
    fail_share = sum(r["label"] == "fail" for r in hold1) / len(hold1)
    assert abs(fail_share - 0.2) < 0.05
    # disjoint and complete
    assert {r["case"] for r in explore1} | {r["case"] for r in hold1} == {
        r["case"] for r in rows
    }
    assert not ({r["case"] for r in explore1} & {r["case"] for r in hold1})


def test_split_noop_when_frac_zero_or_tiny_batch():
    rows = _rows()
    kept, hold = split_records(rows, 0.0, outcome_col="label")
    assert hold is None and len(kept) == len(rows)
    kept, hold = split_records(rows[:3], 0.4, outcome_col="label")
    assert hold is None


# ── failure_indicator ────────────────────────────────────────────────────────


def test_failure_indicator_fail_like_and_binary_and_minority():
    ind, pos, note = failure_indicator(
        [{"label": "fail"}, {"label": "pass"}, {"label": None}], "label"
    )
    assert ind == [1, 0, None] and pos == "fail" and "fail-like" in note

    ind, pos, _ = failure_indicator([{"y": 1}, {"y": 0}], "y")
    assert ind == [1, 0] and pos == "truthy"

    ind, pos, note = failure_indicator(
        [{"grade": "ok"}] * 9 + [{"grade": "bad-case"}], "grade"
    )
    assert pos == "bad-case" and "minority" in note
    assert sum(ind) == 1

    ind, pos, note = failure_indicator([{"g": "a"}, {"g": "b"}, {"g": "c"}], "g")
    assert pos is None and ind == [None] * 3 and "not binary" in note


# ── holdout_confirm ──────────────────────────────────────────────────────────


class _FakeJudge:
    def __init__(self, verdict="supported"):
        self.verdict = verdict
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {"verdict": self.verdict, "reasoning": "held-out fail rates differ.",
             "needs_surgery": False}
        )


def _report() -> dict:
    return {
        "candidate_signals": [
            {"name": "high_x", "rationale": "x separates",
             "recipe": {"name": "high_x", "kind": "expr", "expr": "x >= 0.5"},
             "effect": 0.5, "reject": True},
            {"name": "no_recipe", "rationale": "descriptive only"},
            {"name": "continuous", "rationale": "no frozen threshold",
             "recipe": {"name": "continuous", "kind": "expr", "expr": "x * 2"}},
        ],
        "hypotheses": [
            {"statement": "High x marks failures.", "basis": "explore half",
             "test_design": "re-test frozen recipe on holdout"},
        ],
    }


def test_holdout_confirm_adjudicates_and_judges():
    judge = _FakeJudge()
    confirm = holdout_confirm(
        _report(), _rows(n_fail=12, n_pass=28), outcome_col="label",
        judge=judge, judge_meta={"model": "fake"},
    )
    assert confirm["phase"] == "holdout_confirm"
    assert confirm["adjudication"]["split"] == "held_out"
    assert confirm["n_validate_rows"] == 40 and confirm["n_validate_fail"] == 12

    by_name = {v["name"]: v for v in confirm["signal_verdicts"]}
    # perfect separation on the frozen threshold -> a real held-out REJECT
    assert by_name["high_x"]["status"] == "adjudicated"
    assert by_name["high_x"]["reject"] is True
    assert by_name["high_x"]["fail_rate_flagged"] == 1.0
    assert by_name["high_x"]["fail_rate_unflagged"] == 0.0
    assert by_name["no_recipe"]["status"] == "skipped"
    assert "binary flag" in by_name["continuous"]["reason"]

    (hv,) = confirm["hypothesis_verdicts"]
    assert hv["verdict"] == "supported" and hv["needs_surgery"] is False
    # the judge saw the held-out evidence table, not explore-half numbers
    assert "flagged-group fail rate" in judge.prompts[0]
    assert confirm["judge"] == {"model": "fake"}


def test_holdout_confirm_without_judge_marks_not_judged():
    confirm = holdout_confirm(_report(), _rows(), outcome_col="label", judge=None)
    (hv,) = confirm["hypothesis_verdicts"]
    assert hv["verdict"] == "not_judged" and hv["needs_surgery"] is True
    assert confirm["judge"] is None


def test_holdout_confirm_non_binary_outcome_degrades_honestly():
    rows = [{"x": 0.9, "label": f"grade{i % 3}"} for i in range(12)]
    confirm = holdout_confirm(_report(), rows, outcome_col="label", judge=None)
    assert confirm["n_validate_fail"] == 0
    assert all(v["status"] == "skipped" for v in confirm["signal_verdicts"])
    assert "not binary" in confirm["outcome"]["note"]


# ── run_explore wiring (split before explore, confirm after) ─────────────────


def test_run_explore_holdout_splits_and_writes_confirm(monkeypatch, tmp_path):
    import evalrx.analysis.api as explore_api
    import evalrx.analysis.explore_run as er
    from evalrx.analysis.explorer import CandidateSignal, ExploratoryAnalysisReport

    captured: dict = {}

    class _FakeAgent:
        def __init__(self, **kw):
            pass

        def explore_records(self, records, *, question, outcome_col=None):
            captured["n_records"] = len(records)
            captured["question"] = question
            report = ExploratoryAnalysisReport(
                question=question, ok=True, workdir=str(tmp_path)
            )
            report.candidate_signals = [
                CandidateSignal(
                    name="high_x", rationale="x separates",
                    recipe={"name": "high_x", "kind": "expr", "expr": "x >= 0.5"},
                )
            ]
            report.hypotheses = [
                {"statement": "High x marks failures.", "basis": "b",
                 "test_design": "t"}
            ]
            return report

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)
    monkeypatch.setattr(
        er, "load_records_from_path",
        lambda path, **kw: _rows(n_fail=20, n_pass=80),
    )
    monkeypatch.setattr(er, "_build_judge", lambda model: _FakeJudge("partial"))

    out = tmp_path / "out"
    rc = er.run_explore(
        tmp_path / "data", out=out, coder_provider="claude_code",
        outcome_col="label", propose_hypotheses=False,
        holdout_frac=0.4, holdout_confirm=True,
    )
    assert rc == 0
    # the explorer saw ONLY the explore share, with the frozen-recipe demand
    assert captured["n_records"] == 60
    assert "FROZEN" in captured["question"]

    confirm = json.loads((out / "confirm_report.json").read_text())
    assert confirm["adjudication"]["split"] == "held_out"
    assert confirm["n_validate_rows"] == 40
    assert confirm["split_meta"]["n_explore"] == 60
    assert confirm["hypothesis_verdicts"][0]["verdict"] == "partial"
    # audit trail: the exact held-out rows
    assert len(json.loads((out / "holdout_records.json").read_text())) == 40


def test_run_explore_holdout_without_confirm_reserves_rows(monkeypatch, tmp_path):
    import evalrx.analysis.api as explore_api
    import evalrx.analysis.explore_run as er
    from evalrx.analysis.explorer import ExploratoryAnalysisReport

    class _FakeAgent:
        def __init__(self, **kw):
            pass

        def explore_records(self, records, *, question, outcome_col=None):
            assert "FROZEN" not in question  # no confirm -> no recipe demand
            return ExploratoryAnalysisReport(question=question, ok=True,
                                             workdir=str(tmp_path))

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)
    monkeypatch.setattr(er, "load_records_from_path", lambda path, **kw: _rows())

    out = tmp_path / "out"
    er.run_explore(
        tmp_path / "data", out=out, coder_provider="claude_code",
        outcome_col="label", propose_hypotheses=False, holdout_frac=0.25,
    )
    assert not (out / "confirm_report.json").exists()
    assert len(json.loads((out / "holdout_records.json").read_text())) == 25
    assert json.loads((out / "holdout_split.json").read_text())["n_explore"] == 75
