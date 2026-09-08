"""Tests for RunContext's V1 (``logger_version="v1"``) layout contract.

Every ``RunContext(...)`` call here pins ``logger_version="v1"`` explicitly —
this file asserts V1-specific paths (``figures_dir``, ``fixes_dir``,
``run_log.jsonl``, ``manifest.json``, ``README.txt``) that V2 either places
elsewhere or never writes. It stays the permanent regression suite for V1's
layout regardless of what RunContext's own default is; V2's RunContext-level
behavior (ephemeral trial/explore trees, ``run.json``, no manifest/README) is
covered separately in ``test_run_logger_v2.py``.

Covers:
  - directory layout creation (lazy mkdir per subdirectory)
  - new_workdir / figure_path path allocation
  - write_report_file / write_diagnose_report (VLDiagnoseReport-style and
    duck-typed AutoDiagnoseReport-style inputs)
  - manifest.json + README.txt generation at finalize()
  - context-manager protocol

No GPU/model required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _make_hypothesis(statement="model confuses left/right", status=None):
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus

    return Hypothesis(
        statement=statement,
        target_model="qwen3-vl-4b-instruct",
        predicted_failure_mode="spatial_attention_failure",
        status=status or HypothesisStatus.SUPPORTED,
    )


def _make_test_result(hypothesis=None):
    from evalrx.eval_agent.hypothesis import HypothesisStatus
    from evalrx.eval_agent.stages.hypothesis_tester import HypothesisTestResult

    return HypothesisTestResult(
        hypothesis=hypothesis or _make_hypothesis(),
        status=HypothesisStatus.SUPPORTED,
        test_name="fail_rate_comparison",
        effect_size=0.32,
        is_consistent_with_protocol=True,
        confidence=0.81,
        verdict="Spatial queries fail significantly more than controls.",
        evidence={"n_fail": 6, "n_pass": 6},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────


def test_root_created_on_init(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    root = tmp_path / "run1"
    assert not root.exists()
    ctx = RunContext(root, logger_version="v1")
    assert ctx.root == root
    assert root.is_dir()


def test_root_resolved_to_absolute_for_relative_input(tmp_path, monkeypatch):
    """A relative root (e.g. ``Path("outputs/run1")``, the common case in
    examples) must end up absolute. ExperimentSandbox/run_coded_pipeline run
    subprocesses with cwd=<workdir under root> *and* a script path built from
    that same (relative) workdir — a relative root makes the child process
    resolve the script path a second time relative to its new cwd, doubling
    it and raising FileNotFoundError on every coded fix/M5 attempt."""
    from evalrx.eval_agent.run_context import RunContext

    monkeypatch.chdir(tmp_path)
    ctx = RunContext(Path("outputs/run1"), logger_version="v1")
    assert ctx.root.is_absolute()
    assert ctx.root == (tmp_path / "outputs" / "run1").resolve()


def test_subdirectories_lazily_created(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    # Nothing but the root should exist before first access.
    for name in ("report", "figures", "artifacts", "prompts",
                 "experiments", "tools", "workspace", "fixes"):
        assert not (ctx.root / name).exists()

    assert ctx.report_dir == ctx.root / "report"
    assert ctx.figures_dir == ctx.root / "figures"
    assert ctx.artifacts_dir == ctx.root / "artifacts"
    assert ctx.prompts_dir == ctx.root / "prompts"
    assert ctx.experiments_dir == ctx.root / "experiments"
    assert ctx.tools_dir == ctx.root / "tools"
    assert ctx.workspace_dir == ctx.root / "workspace"
    assert ctx.fixes_dir == ctx.root / "fixes"

    for name in ("report", "figures", "artifacts", "prompts",
                 "experiments", "tools", "workspace", "fixes"):
        assert (ctx.root / name).is_dir()


def test_log_path_and_manifest_path(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    assert ctx.log_path == ctx.root / "run_log.jsonl"
    assert ctx.manifest_path == ctx.root / "manifest.json"


def test_run_id_defaults_to_root_name(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "spatial", logger_version="v1")
    assert ctx.run_id == "spatial"


def test_run_id_override(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "spatial", run_id="custom_id", logger_version="v1")
    assert ctx.run_id == "custom_id"


# ─────────────────────────────────────────────────────────────────────────────
# Logger binding
# ─────────────────────────────────────────────────────────────────────────────


def test_logger_property_builds_run_logger_bound_to_context(tmp_path):
    from evalrx.eval_agent.run_context import RunContext
    from evalrx.eval_agent.run_logger import RunLogger

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    logger = ctx.logger
    assert isinstance(logger, RunLogger)
    assert logger.run_dir == ctx.root
    assert logger.artifact_dir == ctx.artifacts_dir
    assert logger.log_path == ctx.log_path
    ctx.finalize()


def test_logger_property_is_cached(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    first = ctx.logger
    second = ctx.logger
    assert first is second
    ctx.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Producer-facing path allocation
# ─────────────────────────────────────────────────────────────────────────────


def test_new_workdir_unique_under_workspace(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    d1 = ctx.new_workdir("m5 surgery")
    d2 = ctx.new_workdir("m5 surgery")
    assert d1 != d2
    assert d1.parent == ctx.workspace_dir
    assert d2.parent == ctx.workspace_dir
    assert d1.is_dir()
    assert d2.is_dir()
    # label is slugified
    assert "m5_surgery" in d1.name


def test_figure_path_appends_png_by_default(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    assert ctx.figure_path("m2_effects") == ctx.figures_dir / "m2_effects.png"


def test_figure_path_preserves_known_extension(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    assert ctx.figure_path("heatmap.svg") == ctx.figures_dir / "heatmap.svg"


# ─────────────────────────────────────────────────────────────────────────────
# Trial allocation (per-attempt self-contained folders: fix candidates, M5)
# ─────────────────────────────────────────────────────────────────────────────


def test_new_trial_root_not_created_until_first_write(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    trial = ctx.new_trial("fixes", "L1 attend carefully")
    # Lazy: a deduped/discarded candidate must leave nothing on disk.
    assert not trial.root.exists()
    assert trial.root.parent == ctx.fixes_dir
    assert "L1_attend_carefully" in trial.root.name
    assert trial.root.name.startswith("01_")


def test_new_trial_numbering_is_monotonic_per_category(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    f1 = ctx.new_trial("fixes", "a")
    f2 = ctx.new_trial("fixes", "b")
    e1 = ctx.new_trial("experiments", "x")
    assert f1.root.name.startswith("01_")
    assert f2.root.name.startswith("02_")
    # Separate counter per category — experiments/ starts back at 01.
    assert e1.root.name.startswith("01_")
    assert e1.root.parent == ctx.experiments_dir


def test_new_trial_rejects_unknown_category(tmp_path):
    import pytest

    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    with pytest.raises(ValueError, match="fixes.*experiments"):
        ctx.new_trial("bogus", "x")


def test_trial_write_creates_root_lazily(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    trial = ctx.new_trial("fixes", "coded_pipeline")
    path = trial.write("prompt.txt", "do the thing")
    assert trial.root.exists()
    assert path == trial.root / "prompt.txt"
    assert path.read_text(encoding="utf-8") == "do the thing"


def test_trial_workspace_created_lazily(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    trial = ctx.new_trial("fixes", "coded_pipeline")
    assert not trial.root.exists()
    ws = trial.workspace
    assert ws == trial.root / "workspace"
    assert ws.is_dir()
    # Idempotent — same path on repeated access.
    assert trial.workspace == ws


def test_trial_write_record_and_result(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    trial = ctx.new_trial("fixes", "attend_carefully")
    record_path = trial.write_record("# Fix attempt 01\n")
    result_path = trial.write_result({"fixed": True, "effect": 0.3})
    assert record_path == trial.root / "record.md"
    assert result_path == trial.root / "result.json"
    assert json.loads(result_path.read_text()) == {"fixed": True, "effect": 0.3}


def test_two_trials_in_same_category_have_independent_workspaces(tmp_path):
    """The bug this whole feature exists to fix: two coded fix attempts must
    not share (and overwrite) one sandbox."""
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    t1 = ctx.new_trial("fixes", "coded_pipeline")
    t2 = ctx.new_trial("fixes", "coded_pipeline")
    (t1.workspace / "fix_pipeline_exec.py").write_text("v1")
    (t2.workspace / "fix_pipeline_exec.py").write_text("v2")
    assert t1.workspace != t2.workspace
    assert (t1.workspace / "fix_pipeline_exec.py").read_text() == "v1"
    assert (t2.workspace / "fix_pipeline_exec.py").read_text() == "v2"


# ─────────────────────────────────────────────────────────────────────────────
# Report API
# ─────────────────────────────────────────────────────────────────────────────


def test_write_report_file_text(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    path = ctx.write_report_file("notes.txt", "hello world")
    assert path == ctx.report_dir / "notes.txt"
    assert path.read_text(encoding="utf-8") == "hello world"


def test_write_diagnose_report_vl_style(tmp_path):
    """VLDiagnoseReport shape: all_hypotheses / all_test_results / stopped_by."""
    from evalrx.eval_agent.loop_reports import VLDiagnoseReport
    from evalrx.eval_agent.run_context import RunContext

    hyp = _make_hypothesis()
    test_result = _make_test_result(hyp)
    report = VLDiagnoseReport(
        cycles=2,
        stopped_by="criteria_met",
        verified_hypotheses=[test_result],
        final_hypotheses=[hyp],
        all_test_results=[test_result],
    )
    discovery_rows = [{"id": "c0", "label": "fail"}, {"id": "c1", "label": "pass"}]

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    written = ctx.write_diagnose_report(report, cases=[1, 2, 3], discovery=discovery_rows)

    assert set(written) == {"hypotheses", "m4_results", "summary_json", "summary_md", "discovery"}
    for path in written.values():
        assert path.exists()
        assert path.parent == ctx.report_dir

    hypotheses = json.loads((ctx.report_dir / "hypotheses.json").read_text())
    assert hypotheses == [{
        "statement": hyp.statement,
        "failure_mode": hyp.predicted_failure_mode,
        "status": hyp.status.value,
    }]

    m4 = json.loads((ctx.report_dir / "m4_results.json").read_text())
    assert m4[0]["hypothesis"] == hyp.statement
    assert m4[0]["effect_size"] == 0.32

    summary = json.loads((ctx.report_dir / "summary.json").read_text())
    assert summary["cycles"] == 2
    assert summary["stopped_by"] == "criteria_met"
    assert summary["n_cases"] == 3
    assert summary["n_verified"] == 1

    summary_md = (ctx.report_dir / "summary.md").read_text()
    assert "stopped_by: criteria_met" in summary_md
    assert hyp.predicted_failure_mode in summary_md

    discovery_out = json.loads((ctx.report_dir / "discovery_cases.json").read_text())
    assert discovery_out == discovery_rows


def test_write_diagnose_report_without_discovery_omits_file(tmp_path):
    from evalrx.eval_agent.loop_reports import VLDiagnoseReport
    from evalrx.eval_agent.run_context import RunContext

    report = VLDiagnoseReport(cycles=1, stopped_by="max_cycles")
    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    written = ctx.write_diagnose_report(report, cases=[])
    assert "discovery" not in written
    assert not (ctx.report_dir / "discovery_cases.json").exists()


def test_write_diagnose_report_duck_typed_auto_diagnose_style(tmp_path):
    """AutoDiagnoseReport shape: final_hypotheses / resolved, no all_test_results."""
    from evalrx.eval_agent.run_context import RunContext

    hyp = _make_hypothesis()
    report = SimpleNamespace(
        cycles=1,
        resolved=True,
        final_hypotheses=[hyp],
        # deliberately no all_hypotheses / all_test_results / stopped_by /
        # verified_hypotheses — write_diagnose_report must fall back cleanly.
    )

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    written = ctx.write_diagnose_report(report, cases=["c0"])

    hypotheses = json.loads((ctx.report_dir / "hypotheses.json").read_text())
    assert hypotheses == [{
        "statement": hyp.statement,
        "failure_mode": hyp.predicted_failure_mode,
        "status": hyp.status.value,
    }]
    m4 = json.loads((ctx.report_dir / "m4_results.json").read_text())
    assert m4 == []

    summary = json.loads((ctx.report_dir / "summary.json").read_text())
    assert summary["resolved"] is True
    assert summary["stopped_by"] is None
    assert summary["n_verified"] == 0
    assert "discovery" not in written


# ─────────────────────────────────────────────────────────────────────────────
# Manifest + README + finalize
# ─────────────────────────────────────────────────────────────────────────────


def test_finalize_writes_manifest_matching_disk(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", config={"model": "qwen3-vl-4b-instruct"}, logger_version="v1")
    ctx.write_report_file("summary.md", "# hi\n")
    (ctx.figures_dir / "plot.png").write_bytes(b"\x89PNG")
    (ctx.artifacts_dir / "attn.npy").write_bytes(b"\x00")
    ctx.new_workdir("c1_m5")

    ctx.finalize()

    assert ctx.manifest_path.exists()
    manifest = json.loads(ctx.manifest_path.read_text())
    assert manifest["run_id"] == ctx.run_id
    assert manifest["config"] == {"model": "qwen3-vl-4b-instruct"}

    on_disk = {
        str(f.relative_to(ctx.root))
        for f in ctx.root.rglob("*")
        if f.is_file() and f.name not in ("manifest.json", "README.txt")
    }
    in_manifest = {f for files in manifest["files"].values() for f in files}
    assert in_manifest == on_disk
    assert "report/summary.md" in manifest["files"]["report"]
    assert "figures/plot.png" in manifest["files"]["figures"]
    assert "artifacts/attn.npy" in manifest["files"]["artifacts"]


def test_finalize_writes_readme_without_stale_logs_prefix(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    ctx.write_report_file("summary.md", "# hi\n")
    ctx.logger  # touch so run_log.jsonl exists, like a real run
    ctx.finalize()

    readme = (ctx.root / "README.txt").read_text()
    assert "logs/" not in readme
    assert "report/" in readme
    assert "run_log.jsonl" in readme


def test_finalize_closes_logger(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    logger = ctx.logger
    assert logger._file_handler is not None
    ctx.finalize()
    # close() removes/closes handlers; a second call must stay idempotent.
    ctx.finalize()


def test_finalize_idempotent_without_logger_access(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1", logger_version="v1")
    # Never touched ctx.logger — finalize() must not require one.
    ctx.finalize()
    assert ctx.manifest_path.exists()
    assert (ctx.root / "README.txt").exists()


def test_context_manager_calls_finalize_on_exit(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    root = tmp_path / "run1"
    with RunContext(root, logger_version="v1") as ctx:
        ctx.write_report_file("summary.md", "# hi\n")
        assert not ctx.manifest_path.exists()

    assert (root / "manifest.json").exists()
    assert (root / "README.txt").exists()
