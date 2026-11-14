"""extract_run_data reads a run from its log alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalrx.reporting.run_data import extract_run_data
from evalrx.reporting.run_events import write_v2_bundle

TRACE = "00000000-0000-0000-0000-00000000000a"
LONG_NEED = "count how often the answer changes when the same question is asked " * 4


def _event(seq: int, name: str, stage: str = "RUN", **payload: Any) -> dict[str, Any]:
    return {"event": name, "stage": stage, "trace_id": TRACE, "event_seq": seq,
            "ts": f"2026-08-20T00:00:{seq:02d}+00:00", **payload}


def _run(root: Path, events: list[dict[str, Any]]) -> Path:
    write_v2_bundle(root / "logs", events)
    return root


def test_every_stage_is_read_from_its_events(tmp_path):
    root = _run(tmp_path / "outputs", [
        _event(1, "run_start", model="org/some-model", n_cases=4,
               label_distribution={"fail": 3, "pass": 1}, protocol={"description": "Answer the question."}),
        _event(2, "probe_search", "M1", n_synthesized=7),
        _event(3, "tool_codegen", "M1", module="m1", tool_name="probe1", need=LONG_NEED),
        _event(4, "probe", "M1", cycle=1, duration_sec=3.0,
               analyzers=["format_sensitivity", "generated:probe1", "mystery_probe", "calibration"],
               findings={
                   "format_sensitivity": {"n_cases": 4, "mean_flip_rate": 0.5,
                                          "per_case": [{"sample_id": "c1", "format_flip_rate": 0.5}]},
                   "generated:probe1": {"n_cases": 2, "flip_rate": 0.25, "mean_gap": 1.5, "ok": True},
                   "calibration": {"n_scored": 3, "logprob_channel": {"ece": 0.125}},
               }),
        _event(5, "analysis", "M2", conclusion="Order matters.", narrative="n", severity="high", duration_sec=1.0,
               stats_results=[
                   {"tool": "t_small", "summary": "small", "effect": 0.1, "reject": True, "p_value": 0.04},
                   {"tool": "t_null", "effect": 0.9, "reject": False, "underpowered": True},
                   {"tool": "t_big", "effect": -0.6, "ci": [-0.8, -0.4], "reject": True, "config": {"k": 1}},
               ],
               findings=["[HIGH] format_sensitivity.mean_flip_rate=0.5 > 0.2: answers flip with option order"]),
        _event(6, "explore", "M2", observations=["Short answers fail more; long ones pass."],
               workspace_snapshot={"files": {"analysis.py": "print(1)", "stdout.txt": "1\n",
                                             "agent_audit.json": json.dumps({"files_read": 2})}}),
        _event(7, "diagnosis", "M3", duration_sec=2.0,
               hypotheses=[{"id": "h1", "statement": "Option order drives the answer."}]),
        _event(8, "surgery", "M4", module="m4", hypothesis_id="h1", status="supported", confidence_score=0.9,
               evidence={"m4_effect_size": 0.4, "m4_verdict": "supported"}),
        _event(9, "surgery", "M5", module="m5", hypothesis_id="h1", status="supported"),
        _event(10, "fix", "M5", module="fix", fixed=True, attempted=[{"name": "a"}],
               best={"name": "think_first", "fixed": True, "payload": {"prompt_template": "Think first. {q}"}}),
        _event(11, "loop_end", cycles=2, stopped_by="budget", total_duration_sec=9.0),
    ])
    (root / "summary.json").write_text(json.dumps({"model": "some-model-as-typed", "dataset": "my_dataset"}))

    data = extract_run_data(root)

    assert data["run"]["kind"] == "loop"
    assert (data["run"]["model"], data["run"]["raw_model"]) == ("some-model-as-typed", "org/some-model")
    assert data["run"]["benchmark_name"] == "my_dataset"
    assert (data["run"]["n_cases"], data["run"]["cycles"], data["run"]["stopped_by"]) == (4, 2, "budget")
    assert data["pre_m1"] == {"ran": True, "n_cases": 7}

    assert [r["name"] for r in data["m1"]["results"]] == [
        "format_sensitivity", "calibration", "generated:probe1", "mystery_probe"]
    results = {r["name"]: r for r in data["m1"]["results"]}
    assert results["format_sensitivity"]["headline"] == [
        {"label": "Answer Flip Rate after Shuffling", "value": "50.0%"}]
    assert results["format_sensitivity"]["per_case"] == [{"sample_id": "c1", "format_flip_rate": 0.5}]
    assert [h["value"] for h in results["calibration"]["headline"]] == ["0.125", None]
    generated = results["generated:probe1"]
    assert generated["display_name"] == "Agent-written probe · probe1"
    assert generated["question"].endswith("…") and len(generated["question"]) == 178
    assert generated["description"] == LONG_NEED
    assert generated["headline"] == [{"label": "flip rate", "value": "25.0%"},
                                     {"label": "mean gap", "value": "1.5"}]
    assert (results["mystery_probe"]["display_name"], results["mystery_probe"]["findings"]) == ("Mystery Probe", {})

    assert [s["tool"] for s in data["m2"]["stats"]] == ["t_big", "t_small", "t_null"]
    assert data["m2"]["stats"][0] == {
        "tool": "t_big", "config": {"k": 1}, "summary": "", "effect": -0.6, "ci": [-0.8, -0.4],
        "reject": True, "p_value": None, "underpowered": False}
    # No explore report was inlined, so the events' own sentences stand in.
    assert [t["title"] for t in data["m2"]["explore"]["takeaways"]] == [
        "answers flip with option order", "Short answers fail more"]
    assert data["agents"]["explore"] == {"code": "print(1)", "stdout": "1\n", "audit": {"files_read": 2}}

    assert data["m3"]["hypotheses"] == [{"id": "h1", "statement": "Option order drives the answer."}]
    assert data["m4"]["results"] == [{
        "hypothesis_id": "h1", "hypothesis": None, "status": "supported", "effect_size": 0.4,
        "confidence": 0.9, "verdict": "supported", "evidence_grade": None, "protocol_consistent": None}]
    assert data["m4"]["event"]["module"] == "m4"
    assert [s["module"] for s in data["m5_surgery"]["surgeries"]] == ["m5"]
    fix = data["m5_fix"]
    assert (fix["ran"], fix["fixed"], fix["best"]["name"]) == (True, True, "think_first")
    assert fix["selection"] == [{"name": "a"}]
    assert fix["prompt_template"] == "Think first. {q}"


def test_a_run_with_no_summary_is_named_after_its_folder(tmp_path):
    root = _run(tmp_path / "my_bench", [
        _event(1, "run_start", model="EndpointModel(qwen3.5-2b)", protocol="Answer the question."),
        _event(2, "probe", "M1", analyzers=[]),
    ])
    (root / "summary.json").write_text("{not json")

    data = extract_run_data(root)

    assert (data["run"]["model"], data["run"]["benchmark_name"]) == ("qwen3.5-2b", "my_bench")
    assert data["run"]["protocol"] == "Answer the question."
    assert data["m2"]["stats"] == [] and data["m3"]["hypotheses"] == []
    assert data["m4"] == {"ran": False, "results": [], "event": {}}


def test_only_the_latest_run_in_a_reused_directory_is_read(tmp_path):
    earlier = {**_event(1, "run_start", model="old"), "trace_id": "earlier"}
    stale = {**_event(2, "diagnosis", "M3", hypotheses=[{"id": "stale"}]), "trace_id": "earlier"}
    root = _run(tmp_path / "outputs", [
        earlier, stale,
        _event(3, "run_start", model="new"),
        _event(4, "probe", "M1", analyzers=[]),
    ])

    data = extract_run_data(root)

    assert data["run"]["raw_model"] == "new"
    assert data["m3"]["hypotheses"] == []
