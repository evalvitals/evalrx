"""evalrx.analysis.run_codebase: run a user's codebase, harvest per-case
records, and feed them to explore(). No real CLI provider/GPU is exercised —
CodegenRunner and explore() are monkeypatched with fakes.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from evalrx.analysis.api import ExploreRunResult
from evalrx.analysis.explorer import ExploratoryAnalysisReport
from evalrx.analysis.run_codebase import CodebaseRunResult, run_codebase

# `evalrx.analysis`'s __init__ does `from .run_codebase import run_codebase`,
# which shadows the `run_codebase` *submodule* attribute with the function of
# the same name — `importlib.import_module` (sys.modules lookup) sidesteps that.
run_codebase_mod = importlib.import_module("evalrx.analysis.run_codebase")

_RECORDS = [
    {"case_id": "c0", "label": "PASS", "input": "x", "prediction": "x"},
    {"case_id": "c1", "label": "FAIL", "input": "y", "prediction": "z"},
]


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "run.py").write_text("print('eval')\n", encoding="utf-8")
    return repo


class _FakeCliResult:
    def __init__(self, error: str | None = None, audit: dict | None = None) -> None:
        self.error = error
        self.audit = audit


class _WritesRecordsRunner:
    """Fake CodegenRunner whose .run() writes valid records on the first call."""

    def __init__(self, cli_config) -> None:
        self.cli_config = cli_config

    def run(self, prompt, *, workdir, timeout_sec=None):
        Path(workdir, "records.json").write_text(json.dumps(_RECORDS), encoding="utf-8")
        return _FakeCliResult(audit={"provider": self.cli_config.provider})


class _NeverWritesRunner:
    """Fake CodegenRunner whose .run() never produces records — exercises the
    repair-turn + eventual failure path."""

    call_count = 0

    def __init__(self, cli_config) -> None:
        self.cli_config = cli_config

    def run(self, prompt, *, workdir, timeout_sec=None):
        type(self).call_count += 1
        return _FakeCliResult(error="boom")


def _fake_explore(records, **kwargs):
    return ExploreRunResult(
        report=ExploratoryAnalysisReport(question="q", ok=True, observations=[f"{len(records)} rows"]),
        out_dir=kwargs.get("out"),
        ok=True,
        hypotheses=[],
    )


def _boom_explore(*_args, **_kwargs):
    raise AssertionError("explore() should not be called")


def test_run_codebase_happy_path_harvests_records_and_calls_explore(tmp_path, monkeypatch):
    monkeypatch.setattr(run_codebase_mod, "CodegenRunner", _WritesRecordsRunner)
    monkeypatch.setattr(run_codebase_mod, "explore", _fake_explore)

    repo = _make_repo(tmp_path)
    result = run_codebase(repo, provider="llm")

    assert isinstance(result, CodebaseRunResult)
    assert result.ran_ok is True
    assert result.error is None
    assert len(result.records) == 2
    assert {r["label"] for r in result.records} == {"PASS", "FAIL"}
    assert result.explore is not None
    assert result.explore.ok is True
    # no `out` given -> the temp workspace is cleaned up before returning
    assert result.workspace is not None
    assert not result.workspace.exists()


def test_run_codebase_analyze_false_skips_explore(tmp_path, monkeypatch):
    monkeypatch.setattr(run_codebase_mod, "CodegenRunner", _WritesRecordsRunner)
    monkeypatch.setattr(run_codebase_mod, "explore", _boom_explore)

    repo = _make_repo(tmp_path)
    result = run_codebase(repo, provider="llm", analyze=False)

    assert result.ran_ok is True
    assert len(result.records) == 2
    assert result.explore is None


def test_run_codebase_gives_up_after_max_attempts_when_no_records_produced(tmp_path, monkeypatch):
    _NeverWritesRunner.call_count = 0
    monkeypatch.setattr(run_codebase_mod, "CodegenRunner", _NeverWritesRunner)
    monkeypatch.setattr(run_codebase_mod, "explore", _boom_explore)

    repo = _make_repo(tmp_path)
    result = run_codebase(repo, provider="llm", max_attempts=2)

    assert result.ran_ok is False
    assert result.records == []
    assert result.explore is None
    assert "boom" in result.error
    assert _NeverWritesRunner.call_count == 2


def test_run_codebase_persists_records_and_workspace_under_out(tmp_path, monkeypatch):
    monkeypatch.setattr(run_codebase_mod, "CodegenRunner", _WritesRecordsRunner)
    monkeypatch.setattr(run_codebase_mod, "explore", _fake_explore)

    repo = _make_repo(tmp_path)
    out_dir = tmp_path / "out"
    result = run_codebase(repo, out=out_dir, provider="llm")

    assert result.workspace == out_dir / "workspace"
    assert result.workspace.exists()
    saved = json.loads((out_dir / "records.json").read_text())
    assert len(saved) == 2


def test_run_codebase_does_not_mutate_the_original_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(run_codebase_mod, "CodegenRunner", _WritesRecordsRunner)
    monkeypatch.setattr(run_codebase_mod, "explore", _fake_explore)

    repo = _make_repo(tmp_path)
    run_codebase(repo, provider="llm")

    assert not (repo / "records.json").exists()


def test_run_codebase_missing_path_returns_error_without_side_effects(tmp_path):
    result = run_codebase(tmp_path / "does-not-exist", provider="llm")

    assert result.ran_ok is False
    assert "does not exist" in result.error


def test_top_level_evalrx_run_codebase_is_the_same_function():
    import evalrx

    assert evalrx.run_codebase is run_codebase


def test_analysis_package_exports_run_codebase():
    from evalrx import analysis

    assert analysis.run_codebase is run_codebase
    assert analysis.CodebaseRunResult is CodebaseRunResult
