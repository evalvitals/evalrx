"""Explore-run wiring: column roles, data card, keyword recipes, judge backend,
row-limit warning and sandbox cleanup."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

import evalrx.analysis.api as explore_api
import evalrx.analysis.explore_run as er
from evalrx.analysis.explorer import (
    CandidateSignal,
    ExploratoryAnalysisAgent,
    ExploratoryAnalysisReport,
    load_records_from_path,
    truncation_warning,
)
from evalrx.analysis.hypothesis_agent import HypothesisAgent
from evalrx.analysis.keywords import keyword_candidates, tokenize
from evalrx.cli import main


def _grouped_rows(seed: int = 0, n_groups: int = 200) -> list[dict]:
    """Rows cut from source clips; a clip's rows share the flag and tend to share the outcome."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for g in range(n_groups):
        flagged = g % 2 == 0
        clip_fails = rng.random() < (0.85 if flagged else 0.15)
        for _ in range(rng.randint(3, 7)):
            fail = clip_fails if rng.random() < 0.9 else not clip_fails
            rows.append({
                "case": f"c{len(rows)}", "clip": f"clip{g}", "flag": int(flagged),
                "note": ("timeout " if fail and rng.random() < 0.6 else "")
                        + rng.choice(["alpha beta", "gamma delta", "epsilon zeta"]),
                "label": "fail" if fail else "pass",
            })
    return rows


class _FakeAgent:
    """Stands in for the coding agent: proposes one frozen recipe and one hypothesis."""

    seen: dict = {}

    def __init__(self, **kwargs):
        pass

    def explore_records(self, records, *, question, outcome_col=None, **roles):
        type(self).seen = {"n": len(records), "question": question, "roles": roles,
                           "clips": {r["clip"] for r in records}}
        report = ExploratoryAnalysisReport(question=question, ok=True,
                                           workdir=str(Path(tempfile.gettempdir())))
        report.candidate_signals = [CandidateSignal(
            name="flag", rationale="flagged clips fail more",
            recipe={"name": "flag", "kind": "expr", "expr": "flag == 1"},
        )]
        report.hypotheses = [{"statement": "Flagged clips fail more.", "basis": "b",
                              "test_design": "t",
                              "predictions": [{"signal": "flag", "relation": ">", "value": 0}]}]
        return report


class _Judge:
    def generate(self, prompt: str) -> str:
        return json.dumps({"verdict": "supported", "reasoning": "held-out rates differ.",
                           "needs_surgery": False})


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    rows = _grouped_rows()
    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)

    def _load(path, info=None, **kwargs):
        if info is not None:
            info.update(n_loaded=len(rows), n_available=len(rows), n_files=1, truncated=False)
        return rows

    monkeypatch.setattr(er, "load_records_from_path", _load)
    monkeypatch.setattr(er, "_build_judge", lambda model: _Judge())

    def _run(**kwargs):
        out = tmp_path / "out"
        rc = er.run_explore(tmp_path / "data", out=out, coder_provider="claude_code",
                            outcome_col="label", propose_hypotheses=False,
                            holdout_frac=0.4, holdout_confirm=True, **kwargs)
        assert rc == 0
        return (json.loads((out / "confirm_report.json").read_text()),
                json.loads((out / "exploratory_report.json").read_text()),
                json.loads((out / "holdout_records.json").read_text()))

    return _run


def test_group_col_separates_the_split_and_clusters_the_intervals(fake_run, capsys):
    confirm, report, holdout = fake_run(group_col="clip")
    assert not (_FakeAgent.seen["clips"] & {r["clip"] for r in holdout})
    assert _FakeAgent.seen["roles"] == {"group_col": "clip"}
    meta = confirm["split_meta"]
    assert meta["group_col"] == "clip" and meta["group_overlap"]["share"] == 0.0
    assert confirm["adjudication"]["clustered_by"] == "clip"
    (verdict,) = confirm["signal_verdicts"]
    assert verdict["n_clusters"] == len({r["clip"] for r in holdout})
    assert not any("looks like the unit" in c for c in confirm["caveats"])
    assert "groups: clip" in capsys.readouterr().out


def test_without_group_col_the_run_reports_the_overlap_and_names_the_column(fake_run, capsys):
    confirm, report, holdout = fake_run()
    (caveat,) = [c for c in confirm["caveats"] if "looks like the unit" in c]
    assert "'clip'" in caveat and "--group-col clip" in caveat
    (suggested,) = confirm["split_meta"]["suggested_group_columns"]
    assert suggested["column"] == "clip"
    assert suggested["holdout_share_in_explore_groups"] > 0.8
    # The same warning reaches the terminal and the exploratory report.
    assert "WARNING: column 'clip' looks like the unit" in capsys.readouterr().out
    assert caveat in report["caveats"]
    # Row-level intervals: narrower than the grouped ones on the same data.
    (by_row,) = confirm["signal_verdicts"]
    grouped = fake_run(group_col="clip")[0]["signal_verdicts"][0]
    assert (by_row["ci"][1] - by_row["ci"][0]) < (grouped["ci"][1] - grouped["ci"][0])


def test_a_named_group_column_that_is_missing_is_reported(fake_run):
    confirm, _, _ = fake_run(group_col="no_such_column")
    assert any("'no_such_column' is absent or empty" in c for c in confirm["caveats"])
    assert "group_overlap" not in confirm["split_meta"]


def test_data_card_answers_and_gaps_are_printed_in_the_caveats(fake_run):
    confirm, report, _ = fake_run(
        data_card={"unit": "one generated clip", "threshold": "score below 60"})
    caveats = report["caveats"]
    assert "data card — what one row is: one generated clip" in caveats
    assert "data card — how the outcome or its threshold was set: score below 60" in caveats
    (gap,) = [c for c in caveats if c.startswith("data card — not stated")]
    assert "how a row was cut from the raw object" in gap
    assert "how results are combined" in gap and "what one row is" not in gap
    assert set(caveats) <= set(confirm["caveats"]) | set(report["caveats"])
    assert "DATA CARD (from the user)" in _FakeAgent.seen["question"]
    assert "one generated clip" in _FakeAgent.seen["question"]


def test_text_column_keywords_are_frozen_on_explore_and_tested_on_holdout(fake_run):
    confirm, report, _ = fake_run(text_col="note", group_col="clip")
    names = [c["name"] for c in report["candidate_signals"]]
    assert names[0] == "flag" and "kw_timeout" in names
    verdicts = {v["name"]: v for v in confirm["signal_verdicts"]}
    timeout = verdicts["kw_timeout"]
    assert timeout["recipe"]["expr"] == "has_word(note, 'timeout')"
    assert timeout["claim"]["text"] == "effect > 0" and timeout["claim"]["stated_by_recipe"]
    assert timeout["claim"]["status"] == "holds"
    # Filler words were picked too (something always leans one way). Each enters
    # the family with the direction it showed on the exploration rows, so it
    # has to repeat that direction on rows it was not chosen on. Whether a
    # given filler survives is a matter of chance at the family's error rate
    # and is measured by simulation (docs/validation), not asserted here.
    fillers = [v for n, v in verdicts.items() if n.startswith("kw_") and n != "kw_timeout"]
    assert fillers
    assert all(v["claim"]["stated_by_recipe"] and v["claim"]["relation"] in (">", "<")
               for v in fillers)
    assert all(v["explore_reject_in_sample"] is False for v in fillers)
    assert confirm["adjudication"]["n_in_family"] == len(verdicts) + 1   # + the prediction


def test_hypothesis_prediction_gets_a_host_verdict_in_the_run(fake_run):
    confirm, _, _ = fake_run(group_col="clip")
    (hv,) = confirm["hypothesis_verdicts"]
    assert hv["verdict"] == "supported" and hv["host_verdict"] == "supported"


# ── judge backend ────────────────────────────────────────────────────────────


def test_judge_follows_the_explorer_backend(monkeypatch):
    monkeypatch.setattr(er, "_build_judge", lambda model: ("claude-judge", model))
    judge, meta = er.resolve_judge("same", "", coder_provider="codex", coder_model="gpt-x",
                                   coder_binary="/opt/codex")
    assert judge.cli_config.provider == "codex" and judge.cli_config.model == "gpt-x"
    assert judge.cli_config.binary_path == "/opt/codex"
    assert meta == {"backend": "codex", "model": "gpt-x", "same_as_proposer": "yes"}

    judge, meta = er.resolve_judge("same", "", coder_provider="claude_code")
    assert judge == ("claude-judge", "claude-opus-4-8")
    assert meta["backend"] == "claude" and meta["same_as_proposer"] == "same_backend"

    judge, meta = er.resolve_judge("claude", "", coder_provider="codex", coder_model="gpt-x")
    assert judge == ("claude-judge", "claude-opus-4-8") and meta["same_as_proposer"] == "no"

    judge, meta = er.resolve_judge("gemini_cli", "", coder_provider="codex",
                                   coder_model="gpt-x", coder_binary="/opt/codex")
    # Another backend gets neither the explorer's model nor its binary.
    assert judge.cli_config.model == "" and judge.cli_config.binary_path == ""
    assert meta == {"backend": "gemini_cli", "model": "(backend default)",
                    "same_as_proposer": "no"}

    _, meta = er.resolve_judge("same", "gpt-y", coder_provider="codex", coder_model="gpt-x")
    assert meta["same_as_proposer"] == "no"


def test_same_reviewer_is_recorded_and_warned_about(fake_run, capsys):
    confirm, _, _ = fake_run()
    assert confirm["judge"]["same_as_proposer"] == "same_backend"
    (caveat,) = [c for c in confirm["caveats"] if "same backend" in c]
    assert "--judge-backend" in caveat
    assert "WARNING: the hypotheses were proposed and graded on the same backend" in (
        capsys.readouterr().out)
    assert er.same_reviewer_caveat({"same_as_proposer": "no"}, "codex") == ""
    assert "same model (m on codex)" in er.same_reviewer_caveat(
        {"same_as_proposer": "yes", "model": "m"}, "codex")


def test_cli_passes_the_new_flags(monkeypatch, tmp_path):
    import evalrx.cli as cli_mod

    captured: dict = {}
    monkeypatch.setattr(cli_mod, "run_explore",
                        lambda path, **kwargs: captured.update(kwargs) or 0)
    card = tmp_path / "card.yaml"
    card.write_text("unit: one clip\nextraction: first 4 seconds\nignored: x\n")
    assert main(["explore", "data", "--group-col", "clip", "--time-col", "wave",
                 "--text-col", "note", "--data-card", str(card),
                 "--outcome-threshold", "score < 60", "--judge-backend", "codex",
                 "--judge-model", "m", "--keep-sandbox"]) == 0
    assert captured["group_col"] == "clip" and captured["time_col"] == "wave"
    assert captured["text_col"] == "note" and captured["keep_sandbox"] is True
    assert captured["judge_backend"] == "codex" and captured["judge_model"] == "m"
    assert captured["data_card"] == {"unit": "one clip", "extraction": "first 4 seconds",
                                     "threshold": "score < 60"}

    captured.clear()
    assert main(["explore", "data"]) == 0
    assert captured["data_card"] is None and captured["group_col"] is None
    assert captured["judge_backend"] == "same" and captured["keep_sandbox"] is False


def test_data_card_file_must_be_a_mapping(tmp_path):
    card = tmp_path / "card.json"
    card.write_text('["not", "a", "mapping"]')
    with pytest.raises(ValueError, match="must be a mapping"):
        er.load_data_card(card)
    card.write_text('{"unit": " a row ", "aggregation": ""}')
    assert er.load_data_card(card) == {"unit": "a row"}


# ── row limit ────────────────────────────────────────────────────────────────


def test_row_limit_reports_how_many_rows_were_dropped(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    for k in range(3):
        (data / f"part{k}.json").write_text(json.dumps(
            [{"i": i, "part": k, "label": "fail" if i % 4 == 0 else "pass"}
             for i in range(50)]))
    info: dict = {}
    rows = load_records_from_path(data, max_rows=60, info=info)
    assert len(rows) == 60 and [sum(r["part"] == k for r in rows) for k in range(3)] == [20] * 3
    assert info == {"n_loaded": 60, "n_available": 150, "n_files": 3, "truncated": True}
    assert "only 60 of 150 rows" in truncation_warning(info, 60)
    # Without `info` the rows returned are the same.
    assert load_records_from_path(data, max_rows=60) == rows

    full: dict = {}
    assert len(load_records_from_path(data, max_rows=500, info=full)) == 150
    assert full["truncated"] is False and truncation_warning(full, 500) == ""

    class _Agent(_FakeAgent):
        def explore_records(self, records, *, question, outcome_col=None, **roles):
            return ExploratoryAnalysisReport(question=question, ok=True,
                                             workdir=str(tmp_path / "nowhere"))

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _Agent)
    out = tmp_path / "out"
    assert er.run_explore(data, out=out, outcome_col="label", propose_hypotheses=False,
                          holdout_frac=0.3, max_rows=60) == 0
    assert "WARNING: only 60 of 150 rows were loaded" in capsys.readouterr().out
    report = json.loads((out / "exploratory_report.json").read_text())
    assert any("only 60 of 150 rows" in c for c in report["caveats"])


# ── sandbox cleanup ──────────────────────────────────────────────────────────


def test_agent_removes_only_a_sandbox_it_created(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    own = ExploratoryAnalysisAgent()
    workdir = Path(own._sandbox.workdir)
    assert workdir.name.startswith("evalrx_sandbox_") and workdir.exists()
    (workdir / "records.json").write_text("[]")
    assert own.discard_sandbox() is True and not workdir.exists()

    given = ExploratoryAnalysisAgent(sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False))
    assert given.discard_sandbox() is False and tmp_path.exists()


def test_explore_deletes_the_temp_sandbox_once_artifacts_are_written(monkeypatch, tmp_path):
    made: list[Path] = []

    class _Agent:
        def __init__(self, **kwargs):
            self.workdir = Path(tempfile.mkdtemp(prefix="evalrx_sandbox_"))
            made.append(self.workdir)

        def explore_records(self, records, *, question, outcome_col=None, **roles):
            (self.workdir / "tables").mkdir()
            (self.workdir / "tables" / "t.csv").write_text("a,b\n1,2\n")
            (self.workdir / "records.json").write_text(json.dumps(records))
            return ExploratoryAnalysisReport(question=question, ok=True,
                                             workdir=str(self.workdir))

        def discard_sandbox(self) -> bool:
            import shutil

            shutil.rmtree(self.workdir)
            return True

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _Agent)
    rows = [{"x": 1, "label": "pass"}]
    out = tmp_path / "out"
    explore_api.explore(rows, out=out, propose_hypotheses=False)
    assert not made[0].exists()
    assert (out / "tables" / "t.csv").exists() and (out / "records.json").exists()
    # The report on disk points at the output directory, not the deleted one.
    assert json.loads((out / "exploratory_report.json").read_text())["workdir"] == str(out)

    kept = tmp_path / "kept"
    explore_api.explore(rows, out=kept, propose_hypotheses=False, keep_sandbox=True)
    assert made[1].exists()
    # With no output directory the sandbox is the only copy and is never removed.
    result = explore_api.explore(rows, propose_hypotheses=False)
    assert made[2].exists() and result.report.workdir == str(made[2])


# ── keywords and hypothesis structure ────────────────────────────────────────


def test_keyword_candidates_rank_by_evidence_and_freeze_direction():
    rng = random.Random(0)
    rows, side = [], []
    for _ in range(600):
        fail = rng.random() < 0.3
        words = rng.sample([f"w{i:03d}" for i in range(300)], 12)
        if rng.random() < (0.5 if fail else 0.1):
            words.append("Timeout")
        if rng.random() < (0.05 if fail else 0.4):
            words.append("cached")
        rows.append({"text": " ".join(words)})
        side.append(1.0 if fail else 0.0)
    picked = keyword_candidates(rows, "text", side, top_k=1)
    assert [c.name for c in picked] == ["kw_timeout", "kw_cached"]
    assert picked[0].recipe["expect"] == {"relation": ">", "value": 0.0}
    assert picked[1].recipe["expect"] == {"relation": "<", "value": 0.0}
    assert "Chosen on those rows, so untested" in picked[0].rationale
    assert picked[0].sufficient is None and picked[0].reject is None
    assert keyword_candidates(rows, "text", [1.0] * 600) == []     # one side only
    assert keyword_candidates(rows, "absent", side) == []
    assert tokenize("It's re-run_42, OK?") == {"it's", "run", "42"} - {"42"} | {"run"}


class _ProposalJudge:
    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps({"hypotheses": [{
            "hypothesis": "The encoder drops late frames, which causes the failures.",
            "plain_statement": "Late frames get lost, and that is when it fails.",
            "basis": "flag", "test": "re-run with frames kept",
            "predictions": [{"signal": "flag", "relation": ">", "value": 0.1},
                            {"signal": "prose_only", "relation": ">", "value": 0}],
            "experiment": {"data_needed": "200 clips", "operation": "keep late frames",
                           "expected_result": "failures drop",
                           "falsified_by": "failures stay the same"},
        }, {
            "hypothesis": "Something else entirely explains the pattern seen.",
            "plain_statement": "Something else explains it.",
            "experiment": {"data_needed": "", "operation": "", "expected_result": "",
                           "falsified_by": ""},
        }]})


def test_hypothesis_agent_carries_predictions_and_a_structured_plan():
    judge = _ProposalJudge()
    report = {"question": "q", "takeaways": [{"title": "t", "analysis": "a"}],
              "candidate_signals": [
                  {"name": "flag", "rationale": "r",
                   "recipe": {"name": "flag", "kind": "expr", "expr": "flag == 1"}},
                  {"name": "prose_only", "rationale": "no recipe"}]}
    first, second = HypothesisAgent(judge=judge).propose(report)
    # The prompt names the signal and its recipe, so a prediction can refer to it.
    assert "[signal name: flag; recipe: flag == 1]" in judge.prompt
    # A prediction about a signal with no recipe cannot be checked and is dropped.
    assert first.predictions == [{"signal": "flag", "relation": ">", "value": 0.1}]
    assert first.experiment["falsified_by"] == "failures stay the same"
    assert first.to_dict()["experiment"]["operation"] == "keep late frames"
    # An all-empty plan is no plan; the dict then has the shape it always had.
    assert second.experiment is None and second.predictions == []
    assert set(second.to_dict()) == {"statement", "plain_statement", "basis", "test_design"}
