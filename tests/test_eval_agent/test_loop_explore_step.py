"""The loop's optional in-cycle explore step (VLDiagnoseLoop(explorer=...)).

Contract under test:

  * it runs between M1 and M2 over the SAME per-case table M2 sees (M1 analyzer
    signals + labels), and only when an explorer is configured;
  * its output reaches M3 as an ``ExploreContext`` (observations / rendered
    charts / caveats) and the dashboard as files under ``<run>/explore/`` —
    never M2's confirmatory family, M4, or the fix gate;
  * it is best-effort: an explorer failure costs the notes, not the run;
  * ``run_confirm`` (M4 → fix) never explores — there is no M3 to inform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalrx.analysis.explorer import ExploratoryAnalysisReport
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.loop import VLDiagnoseLoop
from evalrx.eval_agent.run_logger import RunLogger
from evalrx.eval_agent.stages.diagnosis import DiagnosisResult, ExploreContext
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel

# ── stubs ─────────────────────────────────────────────────────────────────────


def _batch(n_fail: int = 3, n_pass: int = 3) -> CaseBatch:
    cases = []
    for i in range(n_fail):
        cases.append(FailureCase(id=f"f{i}", inputs=Inputs(prompt=f"q{i}"), label=Label.FAIL))
    for i in range(n_pass):
        cases.append(FailureCase(id=f"p{i}", inputs=Inputs(prompt=f"q{i}"), label=Label.PASS))
    return CaseBatch(cases)


class _Probe:
    """M1 stand-in: one analyzer with a per-case numeric signal (fail cases high)."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_schema = None
        self.last_selection_prompt = ""
        self.last_selection_raw = ""
        self._failed_analyzers: dict = {}
        self._generated_probes: list = []
        self.run_logger = None

    def probe(self, model, data, **kw):
        self.calls.append("m1")
        return {"chain": Result(
            analyzer="chain", model="fake", cases=CaseBatch([]),
            findings={"per_case": [
                {"sample_id": c.id, "n_steps": (9.0 if c.label == Label.FAIL else 2.0)}
                for c in data
            ]},
        )}


class _Stats:
    """M2 stand-in: records the call and returns a minimal StatsAnalysisReport."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.run_logger = None

    def analyze(self, results, model_name="", protocol=None, data=None, **kw):
        from evalrx.analysis.stats_agent import StatsAnalysisReport

        self.calls.append("m2")
        return StatsAnalysisReport(model_name=model_name, findings=[], severity="low",
                                   narrative="n", raw_results=dict(results), stats_plan=[])


class _Diag:
    """M3 stand-in: records the explore_context it was handed."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.seen_context: list[Any] = []

    def diagnose(self, analysis, prior_cycles=None, explore_context=None,
                 failure_modes=None):
        self.calls.append("m3")
        self.seen_context.append(explore_context)
        return DiagnosisResult(
            model_name="fake",
            hypotheses=[Hypothesis(statement="chains are long", target_model="fake",
                                   predicted_failure_mode="long_chain")],
            raw_judge_output="raw",
        )


class _Explorer:
    """ExploratoryAnalysisAgent stand-in: records its inputs, returns a canned report."""

    def __init__(self, calls: list[str], report: ExploratoryAnalysisReport | None = None,
                 raise_exc: Exception | None = None) -> None:
        self.calls = calls
        self.report = report
        self.raise_exc = raise_exc
        self.records: list[dict] | None = None
        self.question = ""
        self.outcome_col = None

    def explore_records(self, records, *, question="", outcome_col=None):
        self.calls.append("explore")
        if self.raise_exc is not None:
            raise self.raise_exc
        self.records = list(records)
        self.question = question
        self.outcome_col = outcome_col
        return self.report


def _report(workdir: Path | None = None, **over) -> ExploratoryAnalysisReport:
    kw: dict[str, Any] = dict(
        question="q", ok=True,
        observations=["FAIL cases have ~4x longer chains than PASS cases"],
        charts=[{"name": "n_steps_by_label", "kind": "bar", "data": "tables/n_steps_by_label.csv",
                 "x": "label", "y": "mean_n_steps", "title": "Mean n_steps by label"}],
        caveats=["in-sample; n=6"],
        workdir=str(workdir) if workdir else "",
    )
    kw.update(over)
    return ExploratoryAnalysisReport(**kw)


def _seed_workdir(workdir: Path) -> None:
    """A tables/ CSV the chart spec above can be rendered from."""
    (workdir / "tables").mkdir(parents=True)
    (workdir / "tables" / "n_steps_by_label.csv").write_text(
        "label,mean_n_steps\nfail,9.0\npass,2.0\n")


def _loop(calls, explorer, run_logger=None, **kw):
    diag = _Diag(calls)
    loop = VLDiagnoseLoop(
        model=FakeModel(), protocol=ExperimentProtocol(description="chains"),
        probe_agent=_Probe(calls), stats_agent=_Stats(calls), diagnosis_agent=diag,
        explorer=explorer, run_logger=run_logger, max_cycles=1, **kw,
    )
    return loop, diag


# ── ordering + data contract ─────────────────────────────────────────────────


def test_explore_runs_between_m1_and_m2_on_the_m1_per_case_table():
    calls: list[str] = []
    explorer = _Explorer(calls, report=_report())
    loop, diag = _loop(calls, explorer)

    loop.run(_batch())

    assert calls[:4] == ["m1", "explore", "m2", "m3"]
    # the explorer sees exactly what M2 sees: one row per labelled case with the
    # analyzer's per-case signal (sanitized `analyzer_metric`) + a label column
    assert explorer.outcome_col == "label"
    assert explorer.records is not None and len(explorer.records) == 6
    row = explorer.records[0]
    assert "chain_n_steps" in row and row["label"] in {"pass", "fail"}
    # the default question names the outcome, the protocol and the recipe rule
    assert "label=fail" in explorer.question and "chains" in explorer.question
    assert "FROZEN" in explorer.question


def test_explore_notes_reach_m3_as_a_descriptive_context_only():
    calls: list[str] = []
    rep = _report(candidate_signals=[])
    loop, diag = _loop(calls, _Explorer(calls, report=rep))

    loop.run(_batch())

    ctx = diag.seen_context[0]
    assert isinstance(ctx, ExploreContext)
    assert ctx.source == "loop_explorer"
    assert ctx.observations == rep.observations and ctx.caveats == rep.caveats
    # ExploreContext carries observations/charts/caveats — never a verdict
    assert not hasattr(ctx, "candidate_signals") and not hasattr(ctx, "reject")


def test_no_explorer_is_a_noop():
    calls: list[str] = []
    loop, diag = _loop(calls, explorer=None)
    loop.run(_batch())
    assert "explore" not in calls
    assert diag.seen_context == [None]


def test_explorer_failure_costs_the_notes_not_the_run(tmp_path):
    calls: list[str] = []
    logger = RunLogger(run_dir=tmp_path / "logs")
    loop, diag = _loop(calls, _Explorer(calls, raise_exc=RuntimeError("coder down")),
                       run_logger=logger)

    report = loop.run(_batch())
    logger.close()

    assert calls[:4] == ["m1", "explore", "m2", "m3"]
    assert report.final_hypotheses            # M3 still ran, on M2 alone
    assert diag.seen_context == [None]
    events = [json.loads(line) for line in (tmp_path / "logs" / "run_log.jsonl").read_text().splitlines()]
    ex = [e for e in events if e["event"] == "explore"]
    assert len(ex) == 1 and ex[0]["ok"] is False and "no report" in ex[0]["error"]
    # the failed step is still accounted for
    assert "explore" in (events[-1].get("timings_sec") or {})


def test_explore_report_without_notes_keeps_the_prior_context():
    calls: list[str] = []
    empty = _report(observations=[], charts=[], caveats=[])
    loop, diag = _loop(calls, _Explorer(calls, report=empty),
                       explore_report={"observations": ["from Step-1 fused report"]})
    loop.run(_batch())
    ctx = diag.seen_context[0]
    assert ctx is not None and ctx.observations == ["from Step-1 fused report"]


# ── persistence + logging ────────────────────────────────────────────────────


def test_explore_persists_beside_logs_and_m3_sees_the_rendered_chart(tmp_path):
    pytest.importorskip("matplotlib")
    calls: list[str] = []
    workdir = tmp_path / "sandbox"
    _seed_workdir(workdir)
    logger = RunLogger(run_dir=tmp_path / "logs")
    loop, diag = _loop(calls, _Explorer(calls, report=_report(workdir)), run_logger=logger)

    loop.run(_batch())
    logger.close()

    # default explore_dir: sibling of logs/ (where the dashboard's
    # _find_explore_report looks: <root>/*/exploratory_report.json)
    out = tmp_path / "explore"
    report_json = json.loads((out / "exploratory_report.json").read_text())
    assert (out / "tables" / "n_steps_by_label.csv").exists()
    figs = list((out / "figures").glob("*.png"))
    assert figs, "chart spec + csv must render to a PNG the judge can be shown"
    assert report_json["charts"][0]["figure_path"] == str(figs[0])
    # ...and that PNG is exactly what M3 was handed as an image
    ctx = diag.seen_context[0]
    assert ctx.figure_paths == [str(figs[0])]

    events = [json.loads(line) for line in (tmp_path / "logs" / "run_log.jsonl").read_text().splitlines()]
    ex = next(e for e in events if e["event"] == "explore")
    assert ex["ok"] is True and ex["n_charts"] == 1 and ex["n_charts_rendered"] == 1
    assert ex["report_path"] == str(out / "exploratory_report.json")
    assert ex["figures"] == [str(figs[0])]
    assert ex["observations"] == _report().observations
    # M3's own event records the explore figures it was shown
    m3 = next(e for e in events if e["event"] == "diagnosis")
    assert m3.get("explore_figures") == [str(figs[0])]


def test_explicit_explore_dir_wins(tmp_path):
    calls: list[str] = []
    workdir = tmp_path / "sandbox"
    _seed_workdir(workdir)
    logger = RunLogger(run_dir=tmp_path / "logs")
    loop, _ = _loop(calls, _Explorer(calls, report=_report(workdir)), run_logger=logger,
                    explore_dir=tmp_path / "elsewhere")
    loop.run(_batch())
    logger.close()
    assert (tmp_path / "elsewhere" / "exploratory_report.json").exists()
    assert not (tmp_path / "explore").exists()


def test_run_analysis_explores_and_run_confirm_does_not(tmp_path):
    calls: list[str] = []
    explorer = _Explorer(calls, report=_report())
    loop, diag = _loop(calls, explorer)

    report = loop.run_analysis(_batch())
    assert calls[:4] == ["m1", "explore", "m2", "m3"]
    assert diag.seen_context[0] is not None

    calls.clear()
    loop.run_confirm(_batch(), list(report.final_hypotheses))
    assert "explore" not in calls   # M4 only — no M3 to inform


def test_default_question_is_built_from_the_protocol():
    from evalrx.eval_agent.prompts.explore_step import default_explore_question

    q = default_explore_question(ExperimentProtocol(
        description="A text LLM sorts words.", task_domain="sorting",
        failure_patterns="drops words"))
    assert "label=fail" in q and "label=pass" in q
    assert "A text LLM sorts words." in q and "sorting" in q and "drops words" in q
    generic = default_explore_question(None)
    assert "Context:" not in generic and "FROZEN" in generic


# ── end-to-end with the real M3: prompt + image ──────────────────────────────


class _ImageJudge(FakeModel):
    """M3 judge that declares images= and records what it was shown."""

    def __init__(self) -> None:
        from evalrx.core.capability import Capability

        super().__init__(capabilities={Capability.GENERATE})
        self.calls: list[dict] = []

    def generate(self, inputs, images=None, **kw):
        self.calls.append({"prompt": str(inputs), "images": [str(p) for p in (images or [])]})
        return ("HYPOTHESIS: long chains lose words, consistent with the Mean n_steps "
                "by label chart\nFAILURE_MODE: long_chain\n"
                "KEEP: long chains lose words, consistent with the Mean n_steps by label chart")


def test_real_m3_gets_the_explore_notes_and_the_rendered_png(tmp_path):
    pytest.importorskip("matplotlib")
    from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent

    calls: list[str] = []
    workdir = tmp_path / "sandbox"
    _seed_workdir(workdir)
    judge = _ImageJudge()
    logger = RunLogger(run_dir=tmp_path / "logs")
    loop = VLDiagnoseLoop(
        model=FakeModel(), protocol=ExperimentProtocol(description="chains"),
        probe_agent=_Probe(calls), stats_agent=_Stats(calls),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        explorer=_Explorer(calls, report=_report(workdir)),
        run_logger=logger, max_cycles=1,
    )
    loop.run(_batch())
    logger.close()

    prompt = judge.calls[0]["prompt"]
    assert "EXPLORATORY MECHANISM NOTES" in prompt and "UNCONFIRMED" in prompt
    assert "FAIL cases have ~4x longer chains" in prompt
    png = next((tmp_path / "explore" / "figures").glob("*.png"))
    assert str(png) in judge.calls[0]["images"]
    events = [json.loads(line) for line in (tmp_path / "logs" / "run_log.jsonl").read_text().splitlines()]
    m3 = next(e for e in events if e["event"] == "diagnosis")
    assert m3.get("explore_context_used") is True
    assert "Mean n_steps by label" in (m3.get("referenced_charts") or [])


# ── default explore_dir: always inside the run, where the dashboard looks ────


def test_default_explore_dir_with_a_run_context_is_under_its_root(tmp_path):
    """V1 layout: explore/ is a real, persisted directory under the run root,
    named in manifest.json. Pinned to logger_version="v1" — the V2 contract
    (ephemeral, under ctx.runtime_root, captured into M2/log.json rather than
    left on disk) is asserted separately below."""
    from evalrx.eval_agent.run_context import RunContext

    calls: list[str] = []
    workdir = tmp_path / "sandbox"
    _seed_workdir(workdir)
    with RunContext(tmp_path / "run", logger_version="v1") as ctx:
        loop, _ = _loop(calls, _Explorer(calls, report=_report(workdir)), run_logger=ctx.logger)
        assert loop._explore_out_dir() == ctx.root / "explore" == ctx.explore_dir
        loop.run(_batch())
    assert (tmp_path / "run" / "explore" / "exploratory_report.json").exists()
    assert not (tmp_path / "explore").exists()          # never outside the run
    # the manifest/README knows the directory
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert any("explore/" in str(f) for f in json.dumps(manifest).split('"'))


def test_default_explore_dir_with_a_v2_run_context_stays_off_the_run_root(tmp_path):
    """Regression test for the bug where _explore_out_dir() hand-derived
    ctx.root/"explore" instead of calling ctx.explore_dir, ignoring is_v2 —
    which would have left real files outside V2's JSON-only run artifact,
    never cleaned up by finalize()'s runtime_root rmtree."""
    from evalrx.eval_agent.run_context import RunContext

    calls: list[str] = []
    workdir = tmp_path / "sandbox"
    _seed_workdir(workdir)
    with RunContext(tmp_path / "run", logger_version="v2") as ctx:
        loop, _ = _loop(calls, _Explorer(calls, report=_report(workdir)), run_logger=ctx.logger)
        out_dir = loop._explore_out_dir()
        assert out_dir == ctx.explore_dir
        assert ctx.runtime_root in out_dir.parents
        assert out_dir != ctx.root / "explore"
        loop.run(_batch())
    # V2 never leaves real files at <root>/explore; the runtime tree that held
    # them was captured into M2/log.json and then deleted by finalize().
    assert not (tmp_path / "run" / "explore").exists()
    assert (tmp_path / "run" / "M2" / "log.json").is_file()


def test_default_explore_dir_beside_a_standalone_logs_dir_and_inside_other_dirs(tmp_path):
    calls: list[str] = []
    for name, expected in (("logs", tmp_path / "explore"),
                           ("logs_confirm", tmp_path / "explore"),
                           ("run_2026", tmp_path / "run_2026" / "explore")):
        logger = RunLogger(run_dir=tmp_path / name)
        loop, _ = _loop(calls, _Explorer(calls, report=_report()), run_logger=logger)
        assert loop._explore_out_dir() == expected, name
        logger.close()
    # no logger and no explicit dir: nothing is persisted, notes stay in memory
    loop, diag = _loop(calls, _Explorer(calls, report=_report()))
    assert loop._explore_out_dir() is None
    loop.run(_batch())
    assert diag.seen_context[0] is not None
