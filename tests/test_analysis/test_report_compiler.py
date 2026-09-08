from __future__ import annotations

import json

from evalrx.analysis.dashboard import load_run
from evalrx.reporting.compiler import compile_diagnostic_report


def test_reader_report_is_generic_and_keeps_observation_distinct_from_proof():
    from evalrx.reporting.compiler import compile_reader_report

    report = compile_reader_report({
        "run": {"model": "demo", "benchmark_name": "audio questions", "n_cases": 10, "protocol": "Why errors?"},
        "m1": {"analyzers": ["format_sensitivity"]},
        "m2": {"explore": {"candidate_signals": [{
            "name": "arbitrary_task_signal", "display_name": "Arbitrary task signal",
            "rationale": "It differed between the two outcome groups.",
        }]}},
        "m3": {"hypotheses": []}, "m4": {"ran": False}, "m5_fix": {"ran": False},
    })

    payload = report.to_dict()
    assert "not a proven cause" in payload["headline"]
    assert payload["key_findings"][0]["title"] == "Arbitrary task signal"
    assert payload["key_findings"][0]["summary"] == "It differed between the two outcome groups."
    assert "does not prove" in payload["key_findings"][0]["limitation"]


def _explore_report():
    return {
        "question": "q",
        "conclusion": "Supported signal found.",
        "observations": ["obs"],
        "visual_plan": [{"name": "v", "plot_kind": "bar"}],
        "chart_readings": [{"chart": "C", "reading": "read"}],
        "dashboard_storyboard": [{
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": "Agent-owned panel text",
            "items": ["Method: held-out check"],
            "artifact_refs": ["candidate_signals"],
        }],
        "charts": [{"name": "c", "title": "C", "data": "tables/c.csv"}],
        "candidate_signals": [
            {"name": "low_focus", "effect": 0.4, "ci": [0.1, 0.7], "reject": True},
            {"name": "probe1_false_detection", "effect": 1.0, "ci": [1.0, 1.0],
             "reject": True},
        ],
        "critique": ["watch leakage"],
    }


def test_compile_diagnostic_report_is_claim_first():
    story = {
        "analyses": [{"cycle": 1, "conclusion": "M2 supported low_focus"}],
        "diagnoses": [{"cycle": 1, "hypotheses": [{"statement": "low_focus cause"}]}],
    }
    report = compile_diagnostic_report(story, _explore_report())

    assert report.answer.startswith("High attention focus is associated with FAIL cases")
    assert "sanity check" in report.answer
    by_text = {c.text: c for c in report.claims}
    assert any("High attention focus is associated" in text for text in by_text)
    leaky = next(c for c in report.claims if "Sanity check" in c.text)
    assert leaky.status == "descriptive"
    assert report.chart_readings == [{"chart": "C", "reading": "read"}]
    assert report.dashboard_storyboard[0]["summary"] == "Agent-owned panel text"
    assert report.critique == ["watch leakage"]


def test_question_falls_back_to_run_start_protocol_description_not_generic_text():
    """A loop/agentic run with no explore_report (so no explore_report["question"])
    must not show the generic "What distinguishes failures from passes?" filler —
    it should surface what run_start's own protocol actually says this run
    investigates, both in the compiled question and the problem_setting panel."""
    story = {
        "run_start": {"protocol": {"description": "Do FAIL cases share an attention signature?"}},
    }
    report = compile_diagnostic_report(story, explore_report=None)

    assert report.question == "Do FAIL cases share an attention signature?"
    problem_setting = next(p for p in report.dashboard_storyboard if p["id"] == "problem_setting")
    assert problem_setting["summary"] == "Do FAIL cases share an attention signature?"


def test_question_uses_generic_text_when_neither_source_is_available():
    report = compile_diagnostic_report(story=None, explore_report=None)
    assert report.question == "What distinguishes failures from passes?"


def test_load_run_attaches_diagnostic_report(tmp_path):
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    fused = tmp_path / "fused"
    logs = tmp_path / "logs"
    fused.mkdir()
    (fused / "fused_report.json").write_text(json.dumps(_explore_report()), encoding="utf-8")
    logger = RunLoggerV2(logs, observability_mode="offline")
    logger._append_stage("M2", "analysis", {"cycle": 1, "conclusion": "ok"})
    logger.close()

    loaded = load_run(tmp_path)

    assert loaded["kind"] == "loop"
    diag = loaded["story"]["diagnostic_report"]
    assert diag["answer"].startswith("High attention focus is associated with FAIL cases")
    assert diag["claims"]
