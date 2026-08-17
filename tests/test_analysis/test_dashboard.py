from __future__ import annotations

import json

from evalvitals.analysis.dashboard import load_loop_story, load_run


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
    logs = tmp_path / "logs_m2_5"
    logs.mkdir()
    events = [
        {"event": "analysis", "cycle": 1},
        {"event": "diagnosis", "cycle": 1, "n_hypotheses": 2,
         "referenced_charts": ["ObjSize by label"], "explore_context_used": True,
         "hypotheses": [{"statement": "h1", "failure_mode": "fm"}]},
        {"event": "surgery", "cycle": 1, "module": "m5", "status": "supported"},
        {"event": "fix", "cycle": 1},
    ]
    (logs / "run_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )

    run = load_run(tmp_path)
    assert run["kind"] == "loop"
    story = run["story"]
    assert story is not None
    assert len(story["diagnoses"]) == 1
    assert story["diagnoses"][0]["referenced_charts"] == ["ObjSize by label"]
    assert len(story["surgeries"]) == 1 and len(story["fixes"]) == 1


def test_load_loop_story_merges_multiple_logs(tmp_path):
    # A run split across logs_m1/ (M1) and logs_m2_5/ (M2-M5): the story must
    # merge both, not pick whichever sorts first (regression — logs_m1 has no
    # diagnoses, so picking it alone made the dashboard look empty).
    (tmp_path / "logs_m1").mkdir()
    (tmp_path / "logs_m1" / "run_log.jsonl").write_text(
        json.dumps({"event": "probe", "cycle": 0}) + "\n", encoding="utf-8"
    )
    (tmp_path / "logs_m2_5").mkdir()
    (tmp_path / "logs_m2_5" / "run_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [
            {"event": "analysis", "cycle": 1},
            {"event": "diagnosis", "cycle": 1, "n_hypotheses": 1,
             "hypotheses": [{"statement": "h", "failure_mode": "fm"}]},
            {"event": "surgery", "cycle": 1, "module": "m5", "status": "supported", "hypothesis": "h"},
        ]),
        encoding="utf-8",
    )

    story = load_loop_story(tmp_path)
    assert story is not None
    assert len(story["diagnoses"]) == 1    # came from logs_m2_5, not lost to logs_m1
    assert len(story["surgeries"]) == 1


def test_load_loop_story_keeps_only_newest_m2_arc(tmp_path):
    # A directory can hold a STALE confirmatory arc (logs_m2_5/, with surgeries)
    # AND a newer descriptive analysis-phase arc (logs_analysis/). Merging both
    # would resurrect the stale surgeries/verdicts on top of the descriptive run.
    # The loader must keep M1 + only the most-recent M2+ arc.
    import os

    (tmp_path / "logs_m1").mkdir()
    (tmp_path / "logs_m1" / "run_log.jsonl").write_text(
        json.dumps({"event": "probe", "cycle": 0}) + "\n", encoding="utf-8"
    )
    stale = tmp_path / "logs_m2_5" / "run_log.jsonl"
    stale.parent.mkdir()
    stale.write_text(
        "\n".join(json.dumps(e) for e in [
            {"event": "analysis", "cycle": 1, "descriptive_only": None},
            {"event": "surgery", "cycle": 1, "module": "m5", "status": "supported", "hypothesis": "h"},
        ]),
        encoding="utf-8",
    )
    fresh = tmp_path / "logs_analysis" / "run_log.jsonl"
    fresh.parent.mkdir()
    fresh.write_text(
        "\n".join(json.dumps(e) for e in [
            {"event": "analysis", "cycle": 0, "descriptive_only": True},
            {"event": "diagnosis", "cycle": 0, "n_hypotheses": 1,
             "hypotheses": [{"statement": "h", "failure_mode": "fm"}]},
        ]),
        encoding="utf-8",
    )
    # Make the descriptive arc unambiguously newer than the stale one.
    os.utime(stale, (1_000_000_000, 1_000_000_000))
    os.utime(fresh, (2_000_000_000, 2_000_000_000))

    story = load_loop_story(tmp_path)
    assert story is not None
    # Stale surgeries must NOT leak in; the descriptive analysis is the only M2+ arc.
    assert story["surgeries"] == []
    assert len(story["analyses"]) == 1
    assert story["analyses"][0]["descriptive_only"] is True
    assert len(story["diagnoses"]) == 1


def test_load_run_empty_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        run = load_run(d)
        assert run["kind"] == "empty"
        assert run["runs"] == []


def test_load_run_resolves_conventional_outputs_child(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "run_log.jsonl").write_text(
        json.dumps({"event": "analysis", "cycle": 1}) + "\n",
        encoding="utf-8",
    )

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
    (tmp_path / "run_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )

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
    (tmp_path / "run_log.jsonl").write_text(
        json.dumps({"event": "analysis", "cycle": 0}) + "\n", encoding="utf-8"
    )
    story = load_loop_story(tmp_path)
    assert story is not None
    assert story["mode"] == "loop"
    assert story["agent_steps"] == []
    assert story["run_start"] is None
    assert story["loop_end"] is None


def test_load_loop_story_reads_m5_results_and_failure_modes_files(tmp_path):
    (tmp_path / "run_log.jsonl").write_text(
        json.dumps({"event": "analysis", "cycle": 0}) + "\n", encoding="utf-8"
    )
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "m5_results.json").write_text(
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
    assert story["m5_results"] == [{"hypothesis": "h", "status": "supported", "effect_size": 1.0}]
    assert story["failure_modes"]["clusters"][0]["name"] == "small_object"


def test_load_loop_story_degrades_gracefully_without_m5_or_failure_mode_files(tmp_path):
    (tmp_path / "run_log.jsonl").write_text(
        json.dumps({"event": "analysis", "cycle": 0}) + "\n", encoding="utf-8"
    )
    story = load_loop_story(tmp_path)
    assert story is not None
    assert story["m5_results"] == []
    assert story["failure_modes"] is None
