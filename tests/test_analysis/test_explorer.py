from __future__ import annotations

import json

from evalrx.agent_runtime.sandbox import ExperimentSandbox
from evalrx.analysis import ExploratoryAnalysisAgent, load_records_from_path, scan_folder

_GOOD_CODE = """
import json
from pathlib import Path

rows = json.loads(Path("records.json").read_text())
fails = [r for r in rows if r.get("label") == "fail"]
passes = [r for r in rows if r.get("label") == "pass"]
payload = {
    "observations": [f"{len(fails)} fail rows and {len(passes)} pass rows"],
    "visual_plan": [
        {
            "name": "flag_distribution",
            "question": "Does flag separate fail from pass rows?",
            "data_shape": "binary-signal-vs-binary-outcome",
            "plot_kind": "bar",
            "fallback_kind": "bar",
            "required_columns": ["flag", "label"],
            "rationale": "A binary signal is best summarized as grouped fail/pass counts.",
        }
    ],
    "chart_readings": [
        {
            "chart": "flag_distribution",
            "reading": "All fail rows have flag=1.",
            "do_not_infer": "This does not prove flag causes failure.",
        }
    ],
    "dashboard_storyboard": [
        {
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": "Agent-authored storyboard summary.",
            "items": ["Method: grouped comparison", "Takeaway: flag separates failures"],
            "artifact_refs": ["charts"],
        }
    ],
    "claims": [
        {
            "id": "C1",
            "text": "flag is a descriptive correlate of failure",
            "status": "descriptive",
            "evidence_ids": ["chart:flag_distribution"],
            "interpretation": "Confirm this on held-out data.",
            "do_not_infer": "No causality.",
        }
    ],
    "candidate_signals": [
        {
            "name": "flag",
            "rationale": "flag is concentrated in fail rows",
            "suggested_test": "signal_label_assoc",
        }
    ],
    "plots": [],
    "tables": {"counts": {"fail": len(fails), "pass": len(passes)}},
    "caveats": ["exploratory only"],
    "critique": ["small scripted fixture"],
    "recommended_confirmatory_tests": ["Run StatsAnalysisAgent.analyze_records on flag"],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""


class ScriptedJudge:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return _GOOD_CODE


def _rows() -> list[dict]:
    return [
        {"case_id": f"c{i}", "label": "fail" if i < 3 else "pass", "flag": int(i < 3)}
        for i in range(6)
    ]


def test_explorer_runs_generated_local_analysis(tmp_path):
    agent = ExploratoryAnalysisAgent(
        judge=ScriptedJudge(f"```python\n{_GOOD_CODE}\n```"),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )

    report = agent.explore_records(_rows(), question="Find failure patterns.")

    assert report.ok
    assert report.attempts == 1
    assert report.candidate_signal_names == ["flag"]
    assert report.visual_plan[0]["plot_kind"] == "bar"
    assert "rationale" in report.visual_plan[0]
    assert report.chart_readings[0]["chart"] == "flag_distribution"
    assert report.dashboard_storyboard[0]["summary"] == "Agent-authored storyboard summary."
    assert report.claims[0]["id"] == "C1"
    assert report.critique == ["small scripted fixture"]
    assert report.tables["counts"] == {"fail": 3, "pass": 3}
    assert "exploratory only" in report.caveats
    assert (tmp_path / "records.json").exists()


def test_explorer_prompt_requires_visual_plan(tmp_path):
    judge = ScriptedJudge(f"```python\n{_GOOD_CODE}\n```")
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )

    report = agent.explore_records(_rows(), question="Find failure patterns.")

    assert report.ok
    prompt = judge.prompts[0]
    assert "visualization plan" in prompt
    assert '"visual_plan"' in prompt
    assert "plot_kind" in prompt
    assert "chart_readings" in prompt
    assert "dashboard_storyboard" in prompt
    assert "critique" in prompt


def test_explorer_uses_inspector_for_repair(tmp_path):
    bad_code = "raise RuntimeError('broken')"
    judge = ScriptedJudge(f"```python\n{bad_code}\n```")
    inspector = ScriptedJudge(f"```python\n{_GOOD_CODE}\n```")
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        inspector=inspector,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=2,
    )

    report = agent.explore_records(_rows())

    assert report.ok
    assert report.attempts == 2
    assert len(judge.prompts) == 1
    assert len(inspector.prompts) == 1
    assert "Previous code" in inspector.prompts[0]


_CODE_JARGON_TAKEAWAY = """
import json
payload = {
    "observations": ["6 rows"],
    "visual_plan": [],
    "takeaways": [{
        "plain_title": "This signal is collinear with a high AUC.",
        "title": "flag has AUC=0.9 separating fail/pass",
        "chart_names": [], "table_names": [], "analysis": "x", "caveat": "",
    }],
    "chart_readings": [], "dashboard_storyboard": [], "claims": [],
    "candidate_signals": [], "plots": [], "tables": {}, "charts": [],
    "caveats": [], "critique": [], "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""

_CODE_CLEAN_TAKEAWAY = """
import json
payload = {
    "observations": ["6 rows"],
    "visual_plan": [],
    "takeaways": [{
        "plain_title": "Cases with the flag set almost always fail.",
        "title": "flag has AUC=0.9 separating fail/pass",
        "chart_names": [], "table_names": [], "analysis": "x", "caveat": "",
    }],
    "chart_readings": [], "dashboard_storyboard": [], "claims": [],
    "candidate_signals": [], "plots": [], "tables": {}, "charts": [],
    "caveats": [], "critique": [], "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""


def test_explorer_parses_plain_title(tmp_path):
    agent = ExploratoryAnalysisAgent(
        judge=ScriptedJudge(f"```python\n{_CODE_CLEAN_TAKEAWAY}\n```"),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )

    report = agent.explore_records(_rows())

    assert report.ok
    assert report.takeaways[0].plain_title == "Cases with the flag set almost always fail."
    assert report.takeaways[0].title == "flag has AUC=0.9 separating fail/pass"


def test_explorer_retries_a_jargon_plain_title(tmp_path):
    judge = ScriptedJudge(
        f"```python\n{_CODE_JARGON_TAKEAWAY}\n```",
        f"```python\n{_CODE_CLEAN_TAKEAWAY}\n```",
    )
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=2,
    )

    report = agent.explore_records(_rows())

    assert report.ok
    assert report.attempts == 2
    assert report.takeaways[0].plain_title == "Cases with the flag set almost always fail."
    assert report.critique == []
    # the retry prompt names the specific plain_title violation
    repair_prompt = judge.prompts[1]
    assert "plain_title" in repair_prompt
    assert "jargon" in repair_prompt


def test_explorer_gives_up_on_jargon_after_max_attempts_but_stays_ok(tmp_path):
    judge = ScriptedJudge(f"```python\n{_CODE_JARGON_TAKEAWAY}\n```")
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=1,
    )

    report = agent.explore_records(_rows())

    assert report.ok  # a language nitpick never fails the whole analysis
    assert report.attempts == 1
    assert report.takeaways[0].plain_title == "This signal is collinear with a high AUC."
    assert any("Plain-language check failed" in c for c in report.critique)


_JARGON_QUESTION = (
    "What predicts hallucination failures? Signals are collinear; compare "
    "ECDF distributions between FAIL and PASS."
)

_CODE_JARGON_QUESTION = """
import json
payload = {
    "plain_question": "This is still collinear jargon about ECDF.",
    "observations": [], "visual_plan": [], "takeaways": [],
    "chart_readings": [], "dashboard_storyboard": [], "claims": [],
    "candidate_signals": [], "plots": [], "tables": {}, "charts": [],
    "caveats": [], "critique": [], "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""

_CODE_CLEAN_QUESTION = """
import json
payload = {
    "plain_question": "Why does the model imagine objects that aren't there?",
    "observations": [], "visual_plan": [], "takeaways": [],
    "chart_readings": [], "dashboard_storyboard": [], "claims": [],
    "candidate_signals": [], "plots": [], "tables": {}, "charts": [],
    "caveats": [], "critique": [], "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""

_CODE_NO_PLAIN_QUESTION = """
import json
payload = {
    "observations": [], "visual_plan": [], "takeaways": [],
    "chart_readings": [], "dashboard_storyboard": [], "claims": [],
    "candidate_signals": [], "plots": [], "tables": {}, "charts": [],
    "caveats": [], "critique": [], "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""


def test_explorer_parses_plain_question(tmp_path):
    agent = ExploratoryAnalysisAgent(
        judge=ScriptedJudge(f"```python\n{_CODE_CLEAN_QUESTION}\n```"),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )

    report = agent.explore_records(_rows(), question=_JARGON_QUESTION)

    assert report.ok
    assert report.plain_question == "Why does the model imagine objects that aren't there?"
    assert report.question == _JARGON_QUESTION


def test_explorer_retries_a_jargon_plain_question(tmp_path):
    judge = ScriptedJudge(
        f"```python\n{_CODE_JARGON_QUESTION}\n```",
        f"```python\n{_CODE_CLEAN_QUESTION}\n```",
    )
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=2,
    )

    report = agent.explore_records(_rows(), question=_JARGON_QUESTION)

    assert report.ok
    assert report.attempts == 2
    assert report.plain_question == "Why does the model imagine objects that aren't there?"
    repair_prompt = judge.prompts[1]
    assert "plain_question" in repair_prompt


def test_explorer_does_not_force_a_rewrite_when_question_is_already_plain(tmp_path):
    """The caller's own question is often already fine (e.g. the default) —
    a missing plain_question must not trigger a pointless retry loop."""
    judge = ScriptedJudge(f"```python\n{_CODE_NO_PLAIN_QUESTION}\n```")
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=3,
    )

    report = agent.explore_records(_rows(), question="Does tool usage correlate with failures?")

    assert report.ok
    assert report.attempts == 1  # no retry consumed
    assert report.plain_question == ""
    assert report.critique == []


def test_explorer_keeps_working_report_when_plain_language_retry_crashes(tmp_path):
    """A jargon-y plain_title must never cost the whole analysis: if the
    retry aimed at fixing it crashes instead, fall back to the original
    working report rather than returning ok=False with nothing."""
    judge = ScriptedJudge(
        f"```python\n{_CODE_JARGON_TAKEAWAY}\n```",
        "```python\nraise RuntimeError('broken')\n```",
    )
    agent = ExploratoryAnalysisAgent(
        judge=judge,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=2,
    )

    report = agent.explore_records(_rows())

    assert report.ok
    assert report.takeaways[0].plain_title == "This signal is collinear with a high AUC."
    assert any("plain-language rewrite attempt failed" in c for c in report.critique)


def test_explorer_keeps_takeaway_with_only_plain_title(tmp_path):
    """A takeaway missing the technical 'title' but carrying 'plain_title'
    must not be silently dropped — plain_title alone is enough evidence a
    reader-facing finding was intended."""
    code = """
import json
payload = {
    "observations": [], "visual_plan": [],
    "takeaways": [{
        "plain_title": "Cases with the flag set almost always fail.",
        "chart_names": [], "table_names": [], "analysis": "x", "caveat": "",
    }],
    "chart_readings": [], "dashboard_storyboard": [], "claims": [],
    "candidate_signals": [], "plots": [], "tables": {}, "charts": [],
    "caveats": [], "critique": [], "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""
    agent = ExploratoryAnalysisAgent(
        judge=ScriptedJudge(f"```python\n{code}\n```"),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )

    report = agent.explore_records(_rows())

    assert report.ok
    assert len(report.takeaways) == 1
    assert report.takeaways[0].plain_title == "Cases with the flag set almost always fail."


def test_explorer_reports_missing_backend(tmp_path):
    agent = ExploratoryAnalysisAgent(sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False))

    report = agent.explore_records(_rows())

    assert not report.ok
    assert "no code-writing backend" in report.error
    assert report.data_profile["n_rows"] == 6


def test_load_records_from_path_reads_jsonl_and_skips_tool_calls(tmp_path):
    run_dir = tmp_path / "agent-a"
    run_dir.mkdir()
    main = run_dir / "agent-a_20260701.json"
    rows = [
        {"question_id": "q0", "is_correct": True, "input": {"question": "A?"}},
        {"question_id": "q1", "is_correct": False, "input": {"question": "B?"}},
    ]
    main.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (run_dir / "tool_calls_1.json").write_text(
        json.dumps([{"name": "tool", "content": "x"}]), encoding="utf-8"
    )

    loaded = load_records_from_path(tmp_path)

    assert len(loaded) == 2
    assert loaded[0]["label"] == "pass"
    assert loaded[1]["label"] == "fail"
    assert loaded[0]["input.question"] == "A?"
    assert loaded[0]["_source_dir"] == "agent-a"


def test_load_records_from_path_samples_across_files(tmp_path):
    for agent in ("agent-a", "agent-b"):
        run_dir = tmp_path / agent
        run_dir.mkdir()
        rows = [
            {"question_id": f"{agent}-{i}", "is_correct": i % 2 == 0}
            for i in range(10)
        ]
        (run_dir / f"{agent}_results.json").write_text(
            "\n".join(json.dumps(r) for r in rows),
            encoding="utf-8",
        )

    loaded = load_records_from_path(tmp_path, max_rows=4, max_files=2)

    assert len(loaded) == 4
    assert {r["_source_dir"] for r in loaded} == {"agent-a", "agent-b"}


def test_load_records_from_path_unpacks_dict_wrapped_case_list(tmp_path):
    """A common M1-output shape: one file per model/run, scalar run metadata
    plus a list of per-case dicts under a conventional key (e.g. "cases").
    This must load as one flat row per case — carrying the run metadata —
    with no bespoke pre-processing script, so `evalrx explore` can point
    directly at a raw M1 case directory."""
    run_dir = tmp_path / "cases"
    run_dir.mkdir()
    (run_dir / "model_a.json").write_text(json.dumps({
        "model": "model-a",
        "seed": 42,
        "cases": [
            {"case_id": "c0", "label": "pass"},
            {"case_id": "c1", "label": "fail"},
        ],
    }), encoding="utf-8")
    (run_dir / "model_b.json").write_text(json.dumps({
        "model": "model-b",
        "seed": 7,
        "cases": [{"case_id": "c2", "label": "pass"}],
    }), encoding="utf-8")

    loaded = load_records_from_path(tmp_path)

    assert len(loaded) == 3
    by_case = {r["case_id"]: r for r in loaded}
    assert by_case["c0"]["model"] == "model-a"
    assert by_case["c0"]["seed"] == 42
    assert by_case["c2"]["model"] == "model-b"
    # per-case fields win over same-named run metadata on collision
    assert by_case["c0"]["label"] == "pass"


def test_load_records_from_path_leaves_plain_dict_records_alone(tmp_path):
    """A dict with no recognizable list-of-dicts field is a single record
    (existing behavior), not something to unpack."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "single.json").write_text(
        json.dumps({"question_id": "q0", "is_correct": True}), encoding="utf-8"
    )

    loaded = load_records_from_path(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["question_id"] == "q0"


def test_scan_folder_reports_filesystem_structure_not_just_row_schema(tmp_path):
    for agent in ("agent-a", "agent-b"):
        run_dir = tmp_path / agent
        run_dir.mkdir()
        (run_dir / f"{agent}_results.json").write_text(
            json.dumps([{"question_id": f"{agent}-0", "is_correct": True}]),
            encoding="utf-8",
        )
    (tmp_path / "agent-a" / "tool_calls_1.json").write_text(
        json.dumps([{"name": "tool"}]), encoding="utf-8"
    )
    (tmp_path / "agent-a" / "notes.txt").write_text("scratch", encoding="utf-8")

    scan = scan_folder(tmp_path)

    assert scan["is_file"] is False
    assert scan["n_dirs"] == 2
    assert scan["n_files_total"] == 4  # 2 results + tool_calls + notes.txt
    assert scan["extensions"][".json"] == 3
    assert scan["extensions"][".txt"] == 1
    # tool_calls_1.json is discovered but excluded from the default sample
    assert scan["json_files_found"] == 3
    assert scan["json_files_used"] == 2
    assert any("notes.txt" in e for e in scan["entries"])


def test_scan_folder_handles_a_single_file(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps([{"a": 1}]), encoding="utf-8")

    scan = scan_folder(path)

    assert scan["is_file"] is True
    assert scan["n_files_total"] == 1
    assert scan["json_files_found"] == 1
    assert scan["json_files_used"] == 1


def test_explore_path_attaches_folder_scan_to_report(tmp_path):
    run_dir = tmp_path / "agent-a"
    run_dir.mkdir()
    (run_dir / "agent-a_results.json").write_text(
        json.dumps([{"question_id": "q0", "is_correct": True}]),
        encoding="utf-8",
    )
    agent = ExploratoryAnalysisAgent()
    report = agent.explore_path(tmp_path)

    scan = report.data_profile.get("folder_scan")
    assert scan is not None
    assert scan["n_files_total"] == 1
    assert scan["root"] == str(tmp_path)


def test_explorer_explore_path(tmp_path):
    data_dir = tmp_path / "logs" / "agent-a"
    data_dir.mkdir(parents=True)
    (data_dir / "agent-a_20260701.json").write_text(
        "\n".join(json.dumps(r) for r in _rows()),
        encoding="utf-8",
    )
    agent = ExploratoryAnalysisAgent(
        judge=ScriptedJudge(f"```python\n{_GOOD_CODE}\n```"),
        sandbox=ExperimentSandbox(workdir=tmp_path / "sandbox", cleanup=False),
    )

    report = agent.explore_path(data_dir.parent)

    assert report.ok
    assert report.data_profile["loaded_rows"] == 6
    assert report.data_profile["source_path"] == str(data_dir.parent)


def test_explore_path_hands_cli_agent_the_raw_folder_not_host_parsed_rows(tmp_path, monkeypatch):
    """With a CLI coding-agent backend, `explore_path` must not pre-parse the
    data itself: the host copies the raw files into the sandbox verbatim and
    the agent's own generated code is responsible for figuring out the shape
    (here: a dict-wrapped per-model case list, an arbitrary M1 output layout
    the host loader was never taught about) and organizing it into rows."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "model_a.json").write_text(json.dumps({
        "model": "model-a",
        "cases": [
            {"case_id": "c0", "label": "fail"},
            {"case_id": "c1", "label": "pass"},
        ],
    }), encoding="utf-8")
    (src / "model_b.json").write_text(json.dumps({
        "model": "model-b",
        "cases": [{"case_id": "c2", "label": "pass"}],
    }), encoding="utf-8")

    agent_written_code = """
import json
from pathlib import Path

rows = []
for f in sorted(Path("raw_input").glob("*.json")):
    payload = json.loads(f.read_text())
    for case in payload["cases"]:
        row = dict(case)
        row["model"] = payload["model"]
        rows.append(row)

Path("records.json").write_text(json.dumps(rows))
fails = [r for r in rows if r["label"] == "fail"]
result = {
    "observations": [f"{len(rows)} rows loaded from raw folder, {len(fails)} fail"],
    "visual_plan": [], "takeaways": [], "candidate_signals": [],
    "charts": [], "caveats": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(result))
"""

    class _FakeCliAgent:
        def run(self, prompt, *, workdir, timeout_sec):
            from evalrx.eval_agent.cli_agent import CliAgentResult
            return CliAgentResult(
                files={"analysis.py": agent_written_code},
                provider_name="fake", elapsed_sec=0.1, raw_output="fake trajectory",
            )

    monkeypatch.setattr(
        "evalrx.agent_runtime.providers.registry.create_cli_agent", lambda config: _FakeCliAgent()
    )

    from evalrx.eval_agent.cli_agent import CliAgentConfig

    sandbox = ExperimentSandbox(workdir=tmp_path / "wd", cleanup=False)
    agent = ExploratoryAnalysisAgent(cli_config=CliAgentConfig(provider="claude_code"), sandbox=sandbox)

    report = agent.explore_path(src, question="What predicts failure?")

    assert report.ok, report.error
    assert report.observations == ["3 rows loaded from raw folder, 1 fail"]
    # the host copied the raw files byte-for-byte -- no host-side unpacking
    copied = sorted(p.name for p in (sandbox.workdir / "raw_input").glob("*.json"))
    assert copied == ["model_a.json", "model_b.json"]
    raw_copy = json.loads((sandbox.workdir / "raw_input" / "model_a.json").read_text())
    assert raw_copy["cases"][0]["case_id"] == "c0"
    # data_profile carries the filesystem scan, not a host-computed row/outcome profile
    assert "outcome" not in report.data_profile
    assert report.data_profile["folder_scan"]["json_files_found"] == 2
