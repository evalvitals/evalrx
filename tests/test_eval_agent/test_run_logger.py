"""RunLogger core event contract: every run_log.jsonl line carries schema_version.

Downstream parsers of run_log.jsonl need a way to detect breaking changes to
event shapes without guessing from evalvitals_version (which tracks the
package, not the log format). See RUN_LOG_SCHEMA_VERSION in run_logger.py.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace


def test_log_run_start_carries_schema_version(tmp_path):
    from evalvitals.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION, RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    logger.log_run_start({"model": "fake-model"})
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["schema_version"] == RUN_LOG_SCHEMA_VERSION
    assert isinstance(entry["schema_version"], int)


def test_every_log_method_stamps_schema_version(tmp_path):
    """Spot-check a few distinct log_* methods, not just log_run_start."""
    from evalvitals.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION, RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    logger.log_run_start()
    logger.log_tool_codegen(
        module="m1_probe", name="fake_tool", need="testing", source="llm",
        ok=True, code="print(1)",
    )
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(entries) == 2
    assert all(e["schema_version"] == RUN_LOG_SCHEMA_VERSION for e in entries)


def test_codegen_persists_complete_raw_cli_stream(tmp_path):
    """The report audit trail must retain more than the summarized CLI output."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    logger.log_tool_codegen(
        module="m1_probe", name="fake_tool", need="testing", source="llm",
        ok=True, code="print(1)", raw_output="summary", raw_stream="complete stream",
    )
    logger.close()

    raw_stream = next((tmp_path / "run1" / "tools").glob("*_agent_raw_stream.txt"))
    assert raw_stream.read_text() == "complete stream"


def test_log_fix_accepts_serialized_best_candidate_name(tmp_path):
    """FixOutcome.to_dict() stores ``best`` as a candidate name, not a mapping."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    outcome = SimpleNamespace(
        to_dict=lambda: {
            "attempted": [{"name": "winning_patch", "tier": "prompt"}],
            "best": "winning_patch",
            "confirmed": True,
        }
    )
    logger.log_fix(outcome)
    logger.close()

    event = json.loads((tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()[0])
    assert event["best"]["name"] == "winning_patch"
    assert event["best"]["tier"] == "prompt"


def test_close_ends_live_langfuse_root_observation(tmp_path):
    """A live root span must be finalized when the diagnostic run closes."""
    from evalvitals.eval_agent.run_logger import RunLogger

    class Root:
        def __init__(self):
            self.output = None
            self.ended = False

        def update(self, *, output):
            self.output = output

        def end(self):
            self.ended = True

    logger = RunLogger(run_dir=tmp_path / "run1")
    root = Root()
    logger.tracer._live_root = root
    logger.close()

    assert root.ended
    assert root.output == {"spans": 0, "generations": 0, "scores": 0}


def _stats_report(stats_plan):
    from evalvitals.analysis.stats_agent import StatsAnalysisReport

    return StatsAnalysisReport(
        model_name="vlm",
        findings=[],
        severity="none",
        narrative="No anomalies detected.",
        raw_results={},
        stats_plan=stats_plan,
    )


def test_small_stats_payload_stays_inline(tmp_path):
    """A typical small analysis cycle keeps stats_plan inline, not externalized."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    small_plan = [{"tool": "signal_label_assoc", "rationale": "correlates with FAIL"}]
    logger.log_analysis(0, _stats_report(small_plan))
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["stats_plan"] == small_plan
    assert not (tmp_path / "run1" / "artifacts").exists() or not list(
        (tmp_path / "run1" / "artifacts").glob("*m2_stats_plan*")
    )


def test_large_stats_payload_externalized(tmp_path):
    """A stats_plan over the inline threshold is written to artifacts/ instead."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    big_plan = [
        {"tool": f"tool_{i}", "rationale": "x" * 200, "config": {"k": "v" * 50}}
        for i in range(30)
    ]
    logger.log_analysis(2, _stats_report(big_plan))
    logger.close()

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    summary = entry["stats_plan"]
    assert isinstance(summary, dict)
    assert summary["n_items"] == len(big_plan)
    assert summary["bytes"] > 4096
    artifact_path = tmp_path / "run1" / summary["path"]
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text()) == big_plan


def test_codegen_seq_increments_are_thread_safe(tmp_path):
    """Concurrent log_tool_codegen() calls must never collide on filename."""
    from evalvitals.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    n_threads = 20
    barrier = threading.Barrier(n_threads)

    def _call(i: int) -> None:
        barrier.wait()
        logger.log_tool_codegen(
            module="m1_probe", name=f"tool_{i}", need="testing", source="llm",
            ok=True, code="print(1)",
        )

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.close()

    code_files = list((tmp_path / "run1" / "tools").glob("*_code.py"))
    assert len(code_files) == n_threads, "filename collision dropped a codegen artifact"


def test_git_commit_falls_back_to_env(tmp_path, monkeypatch):
    """In Docker the image has no git; the commit must still come from the env.

    Without this fallback the code-version provenance promised in run_start is
    silently absent in exactly the containerised mode the examples run in.
    """
    import subprocess

    from evalvitals.eval_agent import run_logger as rl

    # Simulate "git unavailable" (raises like FileNotFoundError would). The
    # method imports subprocess locally, so patch the real module function.
    def _boom(*_a, **_k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setenv("EVALVITALS_GIT_COMMIT", "deadbee")

    logger = rl.RunLogger(run_dir=tmp_path / "run1")
    logger.log_run_start({"model": "fake-model"})
    logger.close()

    entry = json.loads((tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()[0])
    assert entry["git_commit"] == "deadbee"


def test_run_config_records_data_fingerprint_and_labels():
    """run_start must capture *which* cases ran + their label balance, not just count."""
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
    from evalvitals.eval_agent.run_metadata import _data_provenance

    cases = [
        FailureCase(id="a", inputs=Inputs(prompt="p1"), label=Label.FAIL),
        FailureCase(id="b", inputs=Inputs(prompt="p2"), label=Label.FAIL),
        FailureCase(id="c", inputs=Inputs(prompt="p3"), label=Label.PASS),
    ]
    prov = _data_provenance(CaseBatch(cases))
    assert prov["label_distribution"] == {"FAIL": 2, "PASS": 1}
    fp = prov["data_fingerprint"]
    assert isinstance(fp, str) and len(fp) == 12

    # Order-independent: same cases shuffled → same fingerprint.
    assert _data_provenance(CaseBatch(list(reversed(cases))))["data_fingerprint"] == fp
    # Different batch (one id changed) → different fingerprint.
    cases2 = cases[:-1] + [FailureCase(id="d", inputs=Inputs(prompt="p4"), label=Label.PASS)]
    assert _data_provenance(CaseBatch(cases2))["data_fingerprint"] != fp


def test_run_logger_creates_langfuse_trace_and_spans(tmp_path):
    """RunLogger must automatically record Langfuse spans, generations, and export bundle."""
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.core.result import Result

    run_dir = tmp_path / "langfuse_run"
    logger = RunLogger(run_dir=run_dir)
    logger.log_run_start({"model": "qwen-audio", "benchmark_name": "MMAU", "n_cases": 10})

    # Log M1 probe
    res = Result(analyzer="format_sensitivity", model="qwen-audio", findings={"accuracy": 0.85, "per_case": [{"sample_id": "c1", "score": 1.0}]})
    logger.log_probe(0, {"format_sensitivity": res})

    # Log M2 analysis
    report = _stats_report([{"tool": "signal_label_assoc", "effect": 0.42, "p_value": 0.001}])
    logger.log_analysis(0, report)

    # Close and check bundle export
    logger.close()

    bundle_file = run_dir / "langfuse_trace.json"
    assert bundle_file.exists()
    bundle = json.loads(bundle_file.read_text())
    assert bundle["trace"]["metadata"]["model"] == "qwen-audio"
    assert bundle["trace"]["metadata"]["benchmark"] == "MMAU"
    assert len(bundle["spans"]) >= 2
    assert any(s["stage"] == "M1" for s in bundle["spans"])
    assert any(s["stage"] == "M2" for s in bundle["spans"])
