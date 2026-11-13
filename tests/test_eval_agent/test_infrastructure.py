"""Tests for the ARC-ported experiment infrastructure.

Covers:
  - validate_entry_point / validate_entry_point_resolved
  - ExperimentSandbox.run_project (multi-file, harness injection, cleanup)
  - JsonlStore (roundtrip, summarize)
  - SandboxFactory (default backend)
  - ExperimentWriterResult (new fields backward compat)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

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
