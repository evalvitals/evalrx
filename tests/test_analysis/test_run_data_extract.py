"""extract_run_data on the on-disk shapes older runs left behind.

Current runs keep everything inline on their events. These layouts still
load: per-analyzer result files, a fixes/ directory with the candidate's own
outputs, report/*.json fallbacks, and a manifest scored against a baseline.
"""

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


def _write(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")


MANIFEST = [
    {"id": "c1", "question": "Q1?", "options": ["A", "B"], "answer": "A", "audio": "a/c1.wav",
     "image": ["img/c1.png", "img/extra.png"], "duration_sec": 2.5, "metadata": {"mmau_task": "sound"}},
    {"id": "c2", "prompt": "Q2?", "candidates": ["X"], "gt": "X", "wav_path": "a/c2.wav",
     "image_path": "img/c2.png", "metadata": {"category": "speech"}},
    {"id": "c3", "query": "Q3?", "label": "yes", "image": []},
    {"id": "c4", "instruction": "Q4?", "expected": "no", "audio_file": "a/c4.wav",
     "image_file": "img/c4.png", "duration_sec": "not a number"},
]


def build_on_disk_run(root: Path) -> Path:
    """A run whose results sit in files beside the log, fixes/ included."""
    logs = root / "logs"
    write_v2_bundle(logs, [
        _event(1, "run_start", model="org/some-model", n_cases=4,
               label_distribution={"fail": 3, "pass": 1}, protocol={"description": "Answer the question."}),
        _event(2, "probe_search", "M1", n_synthesized=7),
        _event(3, "tool_codegen", "M1", module="m1", tool_name="probe1", need=LONG_NEED),
        _event(4, "probe", "M1", cycle=1, duration_sec=3.0,
               analyzers=["format_sensitivity", "generated:probe1", "mystery_probe", "calibration"],
               result_paths={"format_sensitivity": "artifacts/custom_fs.result.json"},
               findings={
                   "generated:probe1": {"n_cases": 2, "flip_rate": 0.25, "mean_gap": 1.5, "ok": True},
                   "calibration": {"n_scored": 3, "logprob_channel": {"ece": 0.125},
                                   "per_case": [{"sample_id": "c1", "conf_logprob": 0.95, "correct": 0}]},
               }),
        _event(5, "analysis", "M2", conclusion="Order matters.", narrative="n", severity="high",
               duration_sec=1.0, stats_results={"path": "artifacts/stats.json"},
               findings=["[HIGH] format_sensitivity.mean_flip_rate=0.5 > 0.2: answers flip with option order"]),
        _event(6, "diagnosis", "M3", hypotheses=[], duration_sec=2.0),
        _event(7, "surgery", "M5", module="m5", hypothesis_id="h1", status="supported"),
        _event(8, "fix", "M5", fixed=False, best={"name": "event-best"}, attempted=[{"name": "a"}]),
        _event(9, "loop_end", cycles=2, stopped_by="budget", total_duration_sec=9.0),
    ])
    _write(root / "data" / "manifest.jsonl", "\n".join(json.dumps(row) for row in MANIFEST))
    _write(root / "summary.json", {"model": "some-model-as-typed", "dataset": "my_dataset"})
    _write(logs / "artifacts" / "custom_fs.result.json", {"findings": {
        "n_cases": 4, "mean_flip_rate": 0.5,
        "per_case": [
            {"sample_id": "c1", "format_flip_rate": 0.5, "conf_logprob": 0.9, "correct": 0, "pass_at_k": 1.0},
            {"sample_id": "c2", "pass_at_k": 0.0},
            {"sample_id": ""},
            {"sample_id": "c3"},
        ],
    }})
    _write(logs / "artifacts" / "c1_mystery_probe.result.json", "{not json")
    _write(logs / "artifacts" / "stats.json", [
        {"tool": "t_small", "summary": "small", "effect": 0.1, "reject": True, "p_value": 0.04},
        {"tool": "t_null", "effect": 0.9, "reject": False, "underpowered": True},
        {"tool": "t_big", "effect": -0.6, "ci": [-0.8, -0.4], "reject": True, "config": {"k": 1}},
    ])
    _write(logs / "report" / "hypotheses.json", [{"id": "h1", "statement": "Option order drives the answer."}])
    _write(logs / "report" / "m4_results.json", [{"hypothesis_id": "h1", "status": "supported"}])
    _write(root / "fixes" / "0_unreadable" / "result.json", "{not json")
    _write(root / "fixes" / "1_failed" / "result.json", {"fixed": False})
    _write(root / "fixes" / "2_worked" / "result.json", {
        "fixed": True, "fixed_cases": ["c1"], "payload": {"prompt_template": "Think first. {q}"}})
    _write(root / "fixes" / "3_later" / "result.json", {"fixed": False})
    _write(root / "fixes" / "2_worked" / "outputs.jsonl", "\n".join([
        json.dumps({"case_id": "c2", "output": "X"}),
        "",
        json.dumps({"id": "c1", "status": "fixed", "output": "A"}),
        json.dumps({"output": "no id"}),
        "{not json",
        json.dumps({"case_id": "c4", "status": "broken", "output": "yes"}),
        json.dumps({"case_id": "not-in-manifest", "status": "untested", "output": 7}),
    ]))
    return root


def build_baseline_run(root: Path) -> Path:
    """A run with no candidate outputs: the manifest scored against a baseline."""
    logs = root / "logs"
    write_v2_bundle(logs, [
        _event(1, "run_start", model="org/some-model", protocol="Answer the question."),
        _event(2, "probe", "M1", analyzers=[]),
        _event(3, "analysis", "M2", stats_results={"path": "artifacts/stats.json"}),
        _event(4, "explore", "M2", observations=["Short answers fail more; long ones pass."],
               caveats=["small sample"], figures=["M2/artifacts/a.png"]),
        _event(5, "fix", "M5", fixed=True, best={"fixed": True, "fixed_cases": ["c1"], "broken_cases": ["c2"]}),
    ])
    _write(root / "data" / "manifest.jsonl", "\n".join(json.dumps(row) for row in MANIFEST[:3]))
    _write(root / "summary.json", "{not json")
    _write(logs / "artifacts" / "stats.json", "{not json")
    _write(logs / "explore" / "exploratory_report.json", "{not json")
    _write(logs / "report" / "hypotheses.json", "{not json")
    _write(logs / "report" / "m4_results.json", "{not json")
    _write(logs / "artifacts" / "baseline.json", {"cases": [
        {"id": "c1", "output": "B"}, {"id": "c3", "output": "no"}, {"output": "no id"}, "not a row"]})
    _write(logs / "M1" / "baseline.json", "{not json")
    return root


def test_results_kept_in_files_beside_the_log_are_read_back(tmp_path):
    data = extract_run_data(build_on_disk_run(tmp_path / "outputs"))

    assert data["run"]["model"] == "some-model-as-typed"
    assert data["run"]["raw_model"] == "org/some-model"
    assert data["run"]["benchmark_name"] == "my_dataset"
    assert (data["run"]["cycles"], data["run"]["stopped_by"]) == (2, "budget")
    assert data["pre_m1"] == {"ran": True, "n_cases": 7}

    results = {r["name"]: r for r in data["m1"]["results"]}
    assert [r["name"] for r in data["m1"]["results"]] == [
        "format_sensitivity", "calibration", "generated:probe1", "mystery_probe"]
    assert results["format_sensitivity"]["n"] == 4
    assert results["format_sensitivity"]["headline"] == [
        {"label": "Answer Flip Rate after Shuffling", "value": "50.0%"}]
    assert results["calibration"]["n"] == 3
    assert results["calibration"]["headline"][0]["value"] == "0.125"
    assert results["calibration"]["headline"][1]["value"] is None
    generated = results["generated:probe1"]
    assert generated["display_name"] == "Agent-written probe · probe1"
    assert generated["question"].endswith("…") and len(generated["question"]) == 178
    assert generated["description"] == LONG_NEED
    assert generated["headline"] == [{"label": "flip rate", "value": "25.0%"},
                                     {"label": "mean gap", "value": "1.5"}]
    assert results["mystery_probe"]["display_name"] == "Mystery Probe"
    assert results["mystery_probe"]["findings"] == {}

    assert [s["tool"] for s in data["m2"]["stats"]] == ["t_big", "t_small", "t_null"]
    assert data["m2"]["stats"][0] == {
        "tool": "t_big", "config": {"k": 1}, "summary": "", "effect": -0.6, "ci": [-0.8, -0.4],
        "reject": True, "p_value": None, "underpowered": False}
    assert data["m2"]["explore"]["takeaways"][0]["title"] == "answers flip with option order"

    assert data["m3"]["hypotheses"] == [{"id": "h1", "statement": "Option order drives the answer."}]
    assert data["m4"] == {"ran": True, "results": [{"hypothesis_id": "h1", "status": "supported"}], "event": {}}
    assert data["m5_surgery"]["ran"] is True
    fix = data["m5_fix"]
    assert (fix["ran"], fix["fixed"], fix["best"]) == (True, True, {"name": "event-best"})
    assert fix["selection"] == [{"name": "a"}]
    assert fix["confirm"]["fixed_cases"] == ["c1"]
    assert fix["prompt_template"] == "Think first. {q}"

    # The candidate's own outputs name the cases; rows that cannot be read
    # (no id, malformed, a manifest duration that is not a number) are skipped.
    assert data["cases"] == [
        {"id": "c1", "status": "fixed", "output": "A", "instruction": "Q1?", "choices": ["A", "B"],
         "expected": "A", "audio_path": "a/c1.wav", "image_path": "img/c1.png", "duration": 2.5,
         "task": "sound",
         "probe_flags": ["Option Order Flip (50%)", "Overconfident on Error (90%)", "Pass@5 Capable"]},
        {"id": "c2", "status": "unchanged", "output": "X", "instruction": "Q2?", "choices": ["X"],
         "expected": "X", "audio_path": "a/c2.wav", "image_path": "img/c2.png", "duration": 0.0,
         "task": "speech", "probe_flags": ["Pass@5 Zero-Coverage"]},
        {"id": "not-in-manifest", "status": "untested", "output": "7", "instruction": "", "choices": [],
         "expected": "", "audio_path": "", "image_path": "", "duration": 0.0, "task": "",
         "probe_flags": []},
    ]


def test_a_manifest_is_scored_against_the_baseline_when_no_outputs_exist(tmp_path):
    data = extract_run_data(build_baseline_run(tmp_path / "my_bench"))

    assert data["run"]["model"] == "org/some-model"
    assert data["run"]["benchmark_name"] == "my_bench"
    assert data["run"]["protocol"] == "Answer the question."
    assert data["run"]["n_cases"] == 3
    assert data["m2"]["stats"] == []
    # The explore report on disk is unreadable, so the event's own sentences stand in.
    assert data["m2"]["explore"]["takeaways"] == [{
        "title": "Short answers fail more", "plain_title": "Short answers fail more",
        "analysis": "Short answers fail more; long ones pass.", "chart_names": [], "table_names": []}]
    assert data["m2"]["explore"]["caveats"] == ["small sample"]
    assert data["m3"]["hypotheses"] == []
    assert data["m4"] == {"ran": False, "results": [], "event": {}}
    assert [(c["id"], c["status"], c["output"], c["image_path"]) for c in data["cases"]] == [
        ("c1", "fixed", "B", "img/c1.png"),
        ("c2", "broken", "", "img/c2.png"),
        ("c3", "unchanged", "no", ""),
    ]
