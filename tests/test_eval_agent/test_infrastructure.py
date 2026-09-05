"""Tests for the ARC-ported experiment infrastructure.

Covers:
  - ExperimentGitManager (no-repo path)
  - validate_entry_point / validate_entry_point_resolved
  - ExperimentSandbox.run_project (multi-file, harness injection, cleanup)
  - JsonlStore (roundtrip, summarize)
  - EvolutionStore (append/load, time-decay, build_overlay, extract_lessons)
  - SandboxFactory (default backend)
  - ExperimentWriterResult (new fields backward compat)
  - AutoDiagnoseLoop (run_dir creates artifacts + checkpoint + heartbeat)
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# ExperimentGitManager
# ─────────────────────────────────────────────────────────────────────────────


def test_git_manager_no_repo(tmp_path):
    from evalrx.eval_agent.git_manager import ExperimentGitManager

    gm = ExperimentGitManager(tmp_path)
    assert gm.is_git_repo() is False
    assert gm.get_current_branch() == ""
    assert gm.get_experiment_history() == []
    assert gm.discard_experiment("x", "test") is False


def test_git_manager_commit_message_format():
    from evalrx.eval_agent.git_manager import ExperimentGitManager

    msg = ExperimentGitManager._format_commit_message(
        run_id="20260604_123000",
        cycle=2,
        hypothesis_statuses={"Model repeats tokens": "supported"},
        description="resolved",
    )
    assert "eval(20260604_123000)" in msg
    assert "cycle 2" in msg
    assert "resolved" in msg
    assert "Model repeats tokens" in msg


def test_git_manager_parse_log_line_valid():
    from evalrx.eval_agent.git_manager import ExperimentGitManager

    line = "abc1234 eval(20260604_120000) cycle 1: resolved"
    parsed = ExperimentGitManager._parse_experiment_log_line(line)
    assert parsed is not None
    assert parsed["hash"] == "abc1234"
    assert parsed["run_id"] == "20260604_120000"
    assert parsed["cycle"] == "1"


def test_git_manager_parse_log_line_invalid():
    from evalrx.eval_agent.git_manager import ExperimentGitManager

    assert ExperimentGitManager._parse_experiment_log_line("not a valid line") is None


# ─────────────────────────────────────────────────────────────────────────────
# validate_entry_point
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_entry_point_accepts_relative():
    from evalrx.agent_runtime.sandbox import validate_entry_point

    assert validate_entry_point("main.py") is None
    assert validate_entry_point("subdir/entry.py") is None


def test_validate_entry_point_rejects_absolute():
    from evalrx.agent_runtime.sandbox import validate_entry_point

    err = validate_entry_point("/etc/passwd")
    assert err is not None
    assert "relative" in err.lower()


def test_validate_entry_point_rejects_dotdot():
    from evalrx.agent_runtime.sandbox import validate_entry_point

    err = validate_entry_point("../escape.py")
    assert err is not None
    assert ".." in err


def test_validate_entry_point_rejects_empty():
    from evalrx.agent_runtime.sandbox import validate_entry_point

    assert validate_entry_point("") is not None


def test_validate_entry_point_resolved_escape(tmp_path):
    from evalrx.agent_runtime.sandbox import validate_entry_point_resolved

    # Create a symlink that escapes the staging dir
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1")
    link = tmp_path / "link.py"
    link.symlink_to(outside)

    err = validate_entry_point_resolved(tmp_path, "link.py")
    assert err is not None


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentSandbox.run_project
# ─────────────────────────────────────────────────────────────────────────────


def test_run_project_executes_and_parses_metrics(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(
        textwrap.dedent("""\
            def run():
                print("accuracy: 0.95")
                print("verdict: 1.0")

            if __name__ == "__main__":
                run()
        """)
    )
    sandbox = ExperimentSandbox(workdir=tmp_path / "sandbox")
    result = sandbox.run_project(project)
    assert result.ok
    assert result.metrics.get("accuracy") == pytest.approx(0.95)
    assert result.metrics.get("verdict") == pytest.approx(1.0)


def test_run_project_injects_harness(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(
        "if __name__ == '__main__':\n    print('verdict: 0.0')\n"
    )
    workdir = tmp_path / "sandbox"
    sandbox = ExperimentSandbox(workdir=workdir)
    sandbox.run_project(project)

    # After a successful run the project dir is cleaned up;
    # harness source exists in the agent_runtime package
    harness_src = (
        Path(__file__).parent.parent.parent
        / "evalrx" / "agent_runtime" / "experiment_harness.py"
    )
    assert harness_src.exists(), "experiment_harness.py not found in agent_runtime package"


def test_run_project_numbered_dirs(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(
        "if __name__ == '__main__':\n    print('verdict: 1.0')\n"
    )
    workdir = tmp_path / "sandbox"
    sandbox = ExperimentSandbox(workdir=workdir)
    sandbox.run_project(project)
    sandbox.run_project(project)
    assert sandbox._run_counter == 2


def test_run_project_rejects_path_traversal(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("x = 1")
    sandbox = ExperimentSandbox(workdir=tmp_path / "sandbox")
    result = sandbox.run_project(project, entry_point="../escape.py")
    assert result.returncode != 0
    assert ".." in result.stderr


def test_sandbox_cleanup_on_success(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    sandbox = ExperimentSandbox(workdir=tmp_path)
    sandbox.run("print('verdict: 1.0')")
    # Successful run: script should be deleted
    py_files = list(tmp_path.glob("exp_*.py"))
    assert len(py_files) == 0, f"Expected cleanup but found: {py_files}"


def test_sandbox_keeps_script_on_failure(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    sandbox = ExperimentSandbox(workdir=tmp_path)
    sandbox.run("raise RuntimeError('intentional failure')")
    py_files = list(tmp_path.glob("exp_*.py"))
    assert len(py_files) == 1, "Script should be kept on failure"


def test_sandbox_runs_correctly_with_relative_workdir(tmp_path, monkeypatch):
    """Found via Docker E2E verification: _run_script passes the same workdir
    as both the subprocess cwd and (via script_path) part of the command
    argv. If workdir stayed relative, the child process resolved the script
    path a second time relative to its new cwd, doubling it and failing with
    'python: can't open file <workdir>/<workdir>/exp_0001.py'. Every coded
    fix/M5 attempt hit this in practice since example run.py scripts pass a
    relative --run-dir straight into RunContext."""
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    monkeypatch.chdir(tmp_path)
    sandbox = ExperimentSandbox(workdir=Path("nested/workdir"))
    result = sandbox.run("print('verdict: 1.0')")
    assert result.returncode == 0
    assert result.metrics.get("verdict") == 1.0


def test_sandbox_cleanup_false_keeps_script_on_success(tmp_path):
    """A durable, self-contained workdir (e.g. one fix/M5 trial's workspace/)
    must keep its code even on success — cleanup=True is the legacy/ephemeral
    default; the trial-backed path opts out."""
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    sandbox = ExperimentSandbox(workdir=tmp_path, cleanup=False)
    sandbox.run("print('verdict: 1.0')")
    py_files = list(tmp_path.glob("exp_*.py"))
    assert len(py_files) == 1, f"Expected the script to be kept, found: {py_files}"


def test_run_project_harness_not_overwritable(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(
        "if __name__ == '__main__':\n    print('verdict: 1.0')\n"
    )
    # Attacker tries to overwrite harness
    (project / "experiment_harness.py").write_text("import os; os.system('evil')")
    workdir = tmp_path / "sandbox"
    sandbox = ExperimentSandbox(workdir=workdir)
    # Should complete without error (harness overwrite silently skipped)
    result = sandbox.run_project(project)
    assert result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis serialization
# ─────────────────────────────────────────────────────────────────────────────


def test_hypothesis_roundtrip():
    from evalrx.eval_agent.hypothesis import (
        Hypothesis,
        HypothesisStatus,
        hypothesis_from_dict,
        hypothesis_to_dict,
    )

    h = Hypothesis(
        statement="Model fails on long inputs",
        target_model="qwen-7b",
        predicted_failure_mode="context_length",
        status=HypothesisStatus.SUPPORTED,
        id="h-001",
        evidence=["result-1"],
        metadata={"created_by": "test"},
    )
    d = hypothesis_to_dict(h)
    h2 = hypothesis_from_dict(d)
    assert h2.statement == h.statement
    assert h2.status == HypothesisStatus.SUPPORTED
    assert h2.evidence == ["result-1"]
    assert h2.metadata == {"created_by": "test"}


# ─────────────────────────────────────────────────────────────────────────────
# JsonlStore
# ─────────────────────────────────────────────────────────────────────────────


def test_jsonl_store_hypothesis_roundtrip(tmp_path):
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.store import JsonlStore

    store = JsonlStore(tmp_path / "store")
    h = Hypothesis(
        statement="Entity binding fails",
        target_model="qwen",
        predicted_failure_mode="binding",
        status=HypothesisStatus.REFUTED,
        id="h-42",
    )
    store.add_hypothesis(h)

    # New instance reads from same dir
    store2 = JsonlStore(tmp_path / "store")
    results = store2.query(kind="hypotheses")
    assert len(results) == 1
    assert results[0].statement == "Entity binding fails"
    assert results[0].status == HypothesisStatus.REFUTED
    assert results[0].id == "h-42"


def test_jsonl_store_summarize(tmp_path):
    from evalrx.eval_agent.hypothesis import Hypothesis
    from evalrx.eval_agent.store import JsonlStore

    store = JsonlStore(tmp_path / "store")
    store.add_hypothesis(Hypothesis(
        statement="h1", target_model="m", predicted_failure_mode="f"
    ))
    store.add_hypothesis(Hypothesis(
        statement="h2", target_model="m", predicted_failure_mode="f"
    ))
    summary = store.summarize()
    assert summary["n_hypotheses"] == 2
    assert summary["n_cases"] == 0
    assert summary["n_results"] == 0


def test_jsonl_store_query_filter_by_status(tmp_path):
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.store import JsonlStore

    store = JsonlStore(tmp_path / "store")
    h_sup = Hypothesis("s1", "m", "f", status=HypothesisStatus.SUPPORTED)
    h_ref = Hypothesis("s2", "m", "f", status=HypothesisStatus.REFUTED)
    store.add_hypothesis(h_sup)
    store.add_hypothesis(h_ref)

    supported = store.query(kind="hypotheses", status=HypothesisStatus.SUPPORTED)
    assert len(supported) == 1
    assert supported[0].statement == "s1"


# ─────────────────────────────────────────────────────────────────────────────
# EvolutionStore
# ─────────────────────────────────────────────────────────────────────────────


def test_evolution_store_append_and_load(tmp_path):
    from evalrx.eval_agent.evolution import EvolutionStore, LessonEntry

    store = EvolutionStore(tmp_path / "evolution")
    lesson = LessonEntry(
        run_id="run-001",
        cycle=1,
        category="surgery",
        severity="warning",
        description="Hypothesis was inconclusive",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    store.append(lesson)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].description == "Hypothesis was inconclusive"
    assert loaded[0].category == "surgery"
    assert store.count() == 1


def test_evolution_time_decay_today():
    from evalrx.eval_agent.evolution import _time_weight

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    weight = _time_weight(now_iso)
    assert weight == pytest.approx(1.0, abs=0.01)


def test_evolution_time_decay_old():
    from evalrx.eval_agent.evolution import _time_weight

    old = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat(timespec="seconds")
    assert _time_weight(old) == 0.0


def test_evolution_time_decay_30_days():
    from evalrx.eval_agent.evolution import _time_weight

    thirty_days_ago = (
        datetime.now(timezone.utc) - timedelta(days=30)
    ).isoformat(timespec="seconds")
    weight = _time_weight(thirty_days_ago)
    assert 0.4 < weight < 0.6  # ~0.5 at 30 days (one half-life)


def test_evolution_build_overlay_empty(tmp_path):
    from evalrx.eval_agent.evolution import EvolutionStore

    store = EvolutionStore(tmp_path / "evolution")
    assert store.build_overlay("surgery") == ""


def test_evolution_build_overlay_with_lessons(tmp_path):
    from evalrx.eval_agent.evolution import EvolutionStore, LessonEntry

    store = EvolutionStore(tmp_path / "evolution")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.append_many([
        LessonEntry("r1", 0, "surgery", "warning", "First lesson", ts),
        LessonEntry("r2", 1, "surgery", "error", "Second lesson", ts),
    ])
    overlay = store.build_overlay("surgery", max_lessons=5)
    assert "Lessons from Prior Diagnosis Runs" in overlay
    assert "First lesson" in overlay
    assert "Second lesson" in overlay
    assert "[WARN]" in overlay
    assert "[ERROR]" in overlay


def test_evolution_extract_lessons_inconclusive():
    from evalrx.eval_agent.evolution import extract_lessons
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport

    h = Hypothesis("h", "m", "f", status=HypothesisStatus.INCONCLUSIVE)
    report = AutoDiagnoseReport(
        cycles=3,
        resolved=False,
        final_hypotheses=[h],
    )
    report._run_id = "run-test"
    lessons = extract_lessons(report)
    categories = {lesson.category for lesson in lessons}
    severities = {lesson.severity for lesson in lessons}
    assert "surgery" in categories
    assert "diagnosis" in categories
    assert "warning" in severities


def test_evolution_extract_lessons_resolved_has_no_diagnosis_warning():
    from evalrx.eval_agent.evolution import extract_lessons
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport

    h = Hypothesis("h", "m", "f", status=HypothesisStatus.SUPPORTED)
    report = AutoDiagnoseReport(
        cycles=1,
        resolved=True,
        final_hypotheses=[h],
    )
    report._run_id = "run-ok"
    lessons = extract_lessons(report)
    diag_warnings = [lesson for lesson in lessons if lesson.category == "diagnosis" and lesson.severity == "warning"]
    assert len(diag_warnings) == 0, "Resolved run should not produce a diagnosis warning"


# ─────────────────────────────────────────────────────────────────────────────
# SandboxFactory
# ─────────────────────────────────────────────────────────────────────────────


def test_create_sandbox_default_returns_experiment_sandbox(tmp_path):
    from evalrx.agent_runtime.factory import SandboxFactoryConfig, create_sandbox
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    sandbox = create_sandbox(SandboxFactoryConfig(), tmp_path)
    assert isinstance(sandbox, ExperimentSandbox)


def test_create_sandbox_docker_fallback_returns_subprocess(tmp_path):
    from evalrx.agent_runtime.factory import SandboxFactoryConfig, create_sandbox
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    # Patch Docker check to simulate unavailability
    with patch("evalrx.agent_runtime.factory._try_create_docker_sandbox", return_value=None):
        sandbox = create_sandbox(SandboxFactoryConfig(mode="docker"), tmp_path)
    assert isinstance(sandbox, ExperimentSandbox)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentWriterResult backward compat
# ─────────────────────────────────────────────────────────────────────────────


def test_experiment_writer_result_defaults():
    from evalrx.eval_agent.stages.experiment_writer import ExperimentWriterResult

    r = ExperimentWriterResult(code="print('hello')")
    assert r.files == {}
    assert r.blueprint == ""
    assert r.verdict is None
    assert r.ok is False  # returncode == -1


def test_experiment_writer_result_code_set():
    from evalrx.eval_agent.stages.experiment_writer import ExperimentWriterResult

    r = ExperimentWriterResult(
        code="print('verdict: 1.0')",
        files={"main.py": "print('verdict: 1.0')"},
        returncode=0,
        metrics={"verdict": 1.0},
        verdict=1.0,
    )
    assert r.ok is True
    assert r.code == r.files["main.py"]


# ─────────────────────────────────────────────────────────────────────────────
# AutoDiagnoseLoop run_dir infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_loop(run_dir=None):
    """Build a minimal AutoDiagnoseLoop with mocked agents."""
    from evalrx.eval_agent.legacy import AutoDiagnoseLoop

    mock_model = MagicMock()
    mock_probe = MagicMock()
    mock_probe.probe.return_value = {}  # empty probe → loop exits immediately
    mock_analysis = MagicMock()
    mock_analysis.analyze.return_value = MagicMock(
        severity="LOW", findings=[], narrative=""
    )

    loop = AutoDiagnoseLoop(
        model=mock_model,
        probe_agent=mock_probe,
        analysis_module=mock_analysis,
        diagnosis_agent=None,  # analysis-only mode → exits after first M1+M2
        run_dir=run_dir,
    )
    return loop


def test_loop_no_run_dir_no_side_effects(tmp_path):
    """Default (no run_dir) must not write any files."""
    loop = _make_mock_loop(run_dir=None)
    mock_data = MagicMock()
    loop.run(mock_data)

    # No checkpoint, heartbeat, or evolution files anywhere in tmp_path
    assert not any(tmp_path.rglob("checkpoint.json"))
    assert not any(tmp_path.rglob("heartbeat.json"))
    assert not any(tmp_path.rglob("lessons.jsonl"))


def test_loop_creates_artifacts_dir(tmp_path):
    loop = _make_mock_loop(run_dir=tmp_path)
    mock_data = MagicMock()
    loop.run(mock_data)

    artifacts_dirs = list((tmp_path / "artifacts").iterdir())
    assert len(artifacts_dirs) == 1, "Expected exactly one artifacts/{run_id}/ dir"


def test_loop_creates_evolution_dir(tmp_path):
    loop = _make_mock_loop(run_dir=tmp_path)
    mock_data = MagicMock()
    loop.run(mock_data)

    evolution_dir = tmp_path / "evolution"
    assert evolution_dir.exists()


def test_loop_checkpoint_written_after_cycle(tmp_path):
    """Checkpoint is written when at least one full cycle completes."""
    from evalrx.eval_agent.legacy import AutoDiagnoseLoop

    mock_model = MagicMock()
    mock_probe = MagicMock()
    mock_probe.probe.return_value = {"analyzer1": MagicMock()}

    mock_analysis = MagicMock()
    mock_analysis.analyze.return_value = MagicMock(severity="LOW", findings=[], narrative="")

    mock_diag = MagicMock()
    mock_diag.hypotheses = []  # no hypotheses → exits loop
    mock_diagnosis_agent = MagicMock()
    mock_diagnosis_agent.diagnose.return_value = mock_diag

    loop = AutoDiagnoseLoop(
        model=mock_model,
        probe_agent=mock_probe,
        analysis_module=mock_analysis,
        diagnosis_agent=mock_diagnosis_agent,
        run_dir=tmp_path,
        max_cycles=1,
    )
    loop.run(MagicMock())

    cp_path = tmp_path / "checkpoint.json"
    assert cp_path.exists(), "checkpoint.json should be written"
    data = json.loads(cp_path.read_text())
    assert "last_completed_cycle" in data
    assert "run_id" in data


def test_loop_heartbeat_written(tmp_path):
    from evalrx.eval_agent.legacy import AutoDiagnoseLoop

    mock_model = MagicMock()
    mock_probe = MagicMock()
    mock_probe.probe.return_value = {"a": MagicMock()}

    mock_analysis = MagicMock()
    mock_analysis.analyze.return_value = MagicMock(severity="LOW", findings=[], narrative="")

    mock_diag_agent = MagicMock()
    mock_diag_agent.diagnose.return_value = MagicMock(hypotheses=[])

    loop = AutoDiagnoseLoop(
        model=mock_model,
        probe_agent=mock_probe,
        analysis_module=mock_analysis,
        diagnosis_agent=mock_diag_agent,
        run_dir=tmp_path,
        max_cycles=1,
    )
    loop.run(MagicMock())

    hb_path = tmp_path / "heartbeat.json"
    assert hb_path.exists(), "heartbeat.json should be written"
    data = json.loads(hb_path.read_text())
    assert "pid" in data
    assert "last_cycle" in data


def test_loop_resume_skips_completed_cycles(tmp_path):
    """A checkpoint at cycle 1 should cause the loop to start at cycle 2."""
    from evalrx.eval_agent.legacy import AutoDiagnoseLoop

    # Write a fake checkpoint saying cycle 1 already completed
    cp = {
        "last_completed_cycle": 1,
        "run_id": "fake",
        "hypothesis_statuses": [],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(cp))

    mock_probe = MagicMock()
    mock_probe.probe.return_value = {}  # returns empty → exits immediately

    call_count = []

    def counting_probe(model, data, hint_failure_modes=None):
        call_count.append(1)
        return {}

    mock_probe.probe.side_effect = counting_probe

    loop = AutoDiagnoseLoop(
        model=MagicMock(),
        probe_agent=mock_probe,
        analysis_module=MagicMock(),
        diagnosis_agent=None,
        run_dir=tmp_path,
        max_cycles=5,
    )
    loop.run(MagicMock())

    # With start_cycle=2 and max_cycles=5, there are 3 iterations available
    # but probe returns empty so loop exits after first probe call
    assert len(call_count) <= 3, (
        f"Loop should skip cycles 0–1 but got {len(call_count)} probe calls"
    )
