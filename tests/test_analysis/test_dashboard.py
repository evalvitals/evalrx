from __future__ import annotations

import json

from evalrx.analysis.dashboard import load_loop_story, load_run


def _write_v2_run(root, events):
    """Inject raw event dicts into a real RunLoggerV2 bundle, the same way
    these tests used to hand-write raw run_log.jsonl lines — this tests
    load_loop_story's parsing/aggregation, not the logger's own public API,
    so routing goes through the private per-stage/run-level appenders
    directly rather than through log_probe/log_diagnosis/etc.'s domain-object
    signatures."""
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2, _resolve_stage

    logger = RunLoggerV2(root, observability_mode="offline")
    stage_tag = {"probe": "M1", "analysis": "M2", "diagnosis": "M3", "surgery": "M4", "fix": "M5"}
    run_key = {"loop_end": "loop_end", "agent_decision": "agent_decisions",
               "agent_tool": "agent_tool_calls", "case_record": "cases"}
    for ev in events:
        name = ev["event"]
        body = {k: v for k, v in ev.items() if k != "event"}
        if name == "run_start":
            logger.log_run_start(body)
        elif name in run_key:
            logger._append_run(run_key[name], dict(body))
        elif name in stage_tag:
            tag = (_resolve_stage(str(body.get("module", ""))) or stage_tag[name])
            logger._append_stage(tag, name, dict(body))
        else:
            raise ValueError(f"unhandled event type in test helper: {name}")
    logger.close()
    return logger


def test_load_run_reads_single_explore_report(tmp_path):
    (tmp_path / "exploratory_report.json").write_text(
        json.dumps({
            "ok": True,
            "question": "compare models",
            "observations": ["a"],
            "candidate_signals": [{"name": "trace_steps"}],
            "charts": [{"title": "C", "figure_path": "figures/00_c.png"}],
        }),
        encoding="utf-8",
    )

    run = load_run(tmp_path)

    assert run["root"] == str(tmp_path.resolve())
    assert run["kind"] == "explore"
    assert run["story"] is None
    assert len(run["runs"]) == 1
    assert run["runs"][0]["report"]["ok"] is True


def test_load_run_reads_fused_report(tmp_path):
    (tmp_path / "fused_report.json").write_text(
        json.dumps({"observations": ["x"], "charts": []}), encoding="utf-8"
    )
    run = load_run(tmp_path)
    assert run["kind"] == "explore"
    assert any(r["name"] == "fused_report" for r in run["runs"])


def test_load_run_detects_loop_run_and_parses_story(tmp_path):
    logs = tmp_path / "logs"
    events = [
        {"event": "analysis", "cycle": 1},
        {"event": "diagnosis", "cycle": 1, "n_hypotheses": 2,
         "referenced_charts": ["ObjSize by label"], "explore_context_used": True,
         "hypotheses": [{"statement": "h1", "failure_mode": "fm"}]},
        {"event": "surgery", "cycle": 1, "module": "m4", "status": "supported"},
        {"event": "fix", "cycle": 1},
    ]
    _write_v2_run(logs, events)

    run = load_run(tmp_path)
    assert run["kind"] == "loop"
    story = run["story"]
    assert story is not None
    assert len(story["diagnoses"]) == 1
    assert story["diagnoses"][0]["referenced_charts"] == ["ObjSize by label"]
    assert len(story["surgeries"]) == 1 and len(story["fixes"]) == 1


def test_load_loop_story_parses_events(tmp_path):
    """analysis/diagnosis/surgery/fix events all land, wherever the stage
    routes them — the multi-directory logs_m1/ + logs_m2_5/ merge this
    predecessor test guarded against was a llm_benchmark-specific artifact of
    the old flat-JSONL layout; RunLoggerV2 always writes one run.json, so
    there is nothing to merge."""
    events = [
        {"event": "probe", "cycle": 0},
        {"event": "analysis", "cycle": 1},
        {"event": "diagnosis", "cycle": 1, "n_hypotheses": 1,
         "hypotheses": [{"statement": "h", "failure_mode": "fm"}]},
        {"event": "surgery", "cycle": 1, "module": "m4", "status": "supported", "hypothesis": "h"},
    ]
    _write_v2_run(tmp_path, events)

    story = load_loop_story(tmp_path)
    assert story is not None
    assert len(story["diagnoses"]) == 1
    assert len(story["surgeries"]) == 1


def test_load_run_empty_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        run = load_run(d)
        assert run["kind"] == "empty"
        assert run["runs"] == []


def test_load_run_resolves_conventional_outputs_child(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _write_v2_run(outputs, [{"event": "analysis", "cycle": 1}])

    run = load_run(tmp_path)

    assert run["kind"] == "loop"
    assert run["root"] == str(outputs.resolve())
    assert len(run["story"]["analyses"]) == 1


def test_load_loop_story_returns_none_for_explore_output(tmp_path):
    (tmp_path / "exploratory_report.json").write_text("{}", encoding="utf-8")
    assert load_loop_story(tmp_path) is None


def test_load_loop_story_parses_run_lifecycle_and_agent_steps(tmp_path):
    events = [
        {"event": "run_start", "model": "FakeModel(...)", "decision_judge": "ClaudeModel(...)",
         "max_actions": 12, "n_cases": 10},
        {"event": "probe", "cycle": 0, "analyzers": ["attention"], "findings": {}, "artifact_paths": {}},
        {"event": "agent_decision", "step": 0, "action": "run_probe", "params": {},
         "rationale": "start with M1", "valid": True},
        {"event": "agent_tool", "step": 0, "tool": "run_probe", "ok": True,
         "summary": "ran 1 analyzer(s)"},
        {"event": "agent_decision", "step": 1, "action": "stop",
         "params": {"resolved": True, "reason": "too early"}, "rationale": "done",
         "valid": True},
        {"event": "agent_tool", "step": 1, "tool": "stop", "ok": False,
         "error": "no_supported_hypothesis", "summary": "cannot declare success yet"},
        {"event": "loop_end", "cycles": 2, "stopped_by": "max_actions", "n_verified": 0},
    ]
    _write_v2_run(tmp_path, events)

    story = load_loop_story(tmp_path)

    assert story is not None
    assert story["mode"] == "agentic"
    assert story["run_start"]["decision_judge"] == "ClaudeModel(...)"
    assert story["loop_end"]["stopped_by"] == "max_actions"
    assert len(story["probes"]) == 1

    steps = story["agent_steps"]
    assert [s["step"] for s in steps] == [0, 1]
    assert steps[0]["action"] == "run_probe"
    assert steps[0]["outcome"] == {
        "tool": "run_probe", "ok": True, "summary": "ran 1 analyzer(s)",
        "error": None, "duration_sec": None,
    }
    # The rejected stop dispatch must be visible, not silently dropped.
    assert steps[1]["outcome"]["ok"] is False
    assert steps[1]["outcome"]["error"] == "no_supported_hypothesis"


def test_load_loop_story_without_agent_events_has_loop_mode_and_empty_steps(tmp_path):
    _write_v2_run(tmp_path, [{"event": "analysis", "cycle": 0}])
    story = load_loop_story(tmp_path)
    assert story is not None
    assert story["mode"] == "loop"
    assert story["agent_steps"] == []
    assert story["run_start"] is None
    assert story["loop_end"] is None


def test_load_loop_story_reads_m4_results_and_failure_modes_files(tmp_path):
    _write_v2_run(tmp_path, [{"event": "analysis", "cycle": 0}])
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "m4_results.json").write_text(
        json.dumps([{"hypothesis": "h", "status": "supported", "effect_size": 1.0}]),
        encoding="utf-8",
    )
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "failure_modes.json").write_text(
        json.dumps({"clusters": [{"name": "small_object", "size": 5}], "method": "cosine_greedy"}),
        encoding="utf-8",
    )

    story = load_loop_story(tmp_path)

    assert story is not None
    assert story["m4_results"] == [{"hypothesis": "h", "status": "supported", "effect_size": 1.0}]
    assert story["failure_modes"]["clusters"][0]["name"] == "small_object"


def test_load_loop_story_degrades_gracefully_without_m4_or_failure_mode_files(tmp_path):
    _write_v2_run(tmp_path, [{"event": "analysis", "cycle": 0}])
    story = load_loop_story(tmp_path)
    assert story is not None
    assert story["m4_results"] == []
    assert story["failure_modes"] is None
