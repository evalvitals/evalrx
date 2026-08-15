"""The upload workbench (`evalvitals web`): zip in, M2+M3 run out.

Covers the streamlit-free helpers (zip staging with zip-slip protection, job
launch record, status classification) and — behind the dashboard extras — an
AppTest pass over the page itself, including rendering a finished run with the
same explore tabs as dashboard_app.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile

import pytest

from evalvitals.analysis.upload_app import (
    _route_followup,
    _timeline_steps,
    archive_run,
    build_explore_argv,
    job_status,
    launch_explore_job,
    list_runs,
    stage_zip,
)


def _zip_bytes(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return buf.getvalue()


# ── stage_zip ────────────────────────────────────────────────────────────────


def test_stage_zip_unwraps_single_root_and_skips_junk(tmp_path):
    payload = _zip_bytes({
        "mydata/cases.json": "[]",
        "mydata/sub/more.csv": "a,b\n1,2\n",
        "mydata/.DS_Store": "junk",
        "__MACOSX/mydata/._cases.json": "resource fork",
    })
    data_dir = stage_zip(payload, tmp_path)
    # single top-level folder is unwrapped: the explorer reads mydata/ directly
    assert data_dir == tmp_path / "data" / "mydata"
    assert (data_dir / "cases.json").read_text() == "[]"
    assert (data_dir / "sub" / "more.csv").exists()
    assert not (data_dir / ".DS_Store").exists()
    assert not (tmp_path / "data" / "__MACOSX").exists()


def test_stage_zip_flat_archive_uses_data_dir(tmp_path):
    payload = _zip_bytes({"a.json": "{}", "b.json": "{}"})
    assert stage_zip(payload, tmp_path) == tmp_path / "data"


def test_stage_zip_rejects_zip_slip(tmp_path):
    for evil in ("../evil.txt", "ok/../../evil.txt", "/abs.txt"):
        with pytest.raises(ValueError, match="unsafe path"):
            stage_zip(_zip_bytes({evil: "boom"}), tmp_path)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_stage_zip_rejects_empty_archive(tmp_path):
    with pytest.raises(ValueError, match="no files"):
        stage_zip(_zip_bytes({"__MACOSX/only-junk": "x"}), tmp_path)


# ── job launch + status ──────────────────────────────────────────────────────


def test_build_explore_argv_carries_form_choices(tmp_path):
    argv = build_explore_argv(
        tmp_path / "data", tmp_path / "output", question="why fail?",
        outcome_col="label", backend="claude_code", model="claude-opus-4-8",
        timeout_sec=900,
    )
    assert argv[:4] == [sys.executable, "-m", "evalvitals.cli", "explore"]
    assert "--outcome-col" in argv and argv[argv.index("--outcome-col") + 1] == "label"
    assert argv[argv.index("--backend") + 1] == "claude_code"
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
    assert argv[argv.index("-q") + 1] == "why fail?"

    bare = build_explore_argv(tmp_path, tmp_path, question="q", outcome_col="",
                              backend="codex", model="", timeout_sec=60)
    assert "--outcome-col" not in bare and "--model" not in bare
    assert "--holdout-frac" not in bare and "--holdout-confirm" not in bare

    verified = build_explore_argv(tmp_path, tmp_path, question="q",
                                  outcome_col="label", backend="claude_code",
                                  model="", timeout_sec=60,
                                  holdout_frac=0.4, holdout_confirm=True)
    assert verified[verified.index("--holdout-frac") + 1] == "0.4"
    assert "--holdout-confirm" in verified

    antigravity = build_explore_argv(
        tmp_path, tmp_path, question="q", outcome_col="label",
        backend="antigravity", model="", timeout_sec=60,
    )
    assert antigravity[antigravity.index("--backend") + 1] == "antigravity"


def test_launch_explore_job_verify_mode_records_split(tmp_path, monkeypatch):
    import evalvitals.analysis.upload_app as ua

    class _FakeProc:
        pid = 4242

    monkeypatch.setattr(ua.subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    job = launch_explore_job(
        run_dir, run_dir / "data", question="q", outcome_col="label",
        backend="claude_code", model="", timeout_sec=600,
        mode="verify", explore_share=0.6,
    )
    assert job["mode"] == "verify"
    assert job["explore_share"] == 0.6 and job["holdout_frac"] == 0.4
    wrapper = (run_dir / "job.sh").read_text()
    assert "--holdout-frac 0.4" in wrapper and "--holdout-confirm" in wrapper

    # explore-only mode with the default 1:0 split adds no holdout flags
    run2 = tmp_path / "run2"
    run2.mkdir()
    job2 = launch_explore_job(
        run2, run2 / "data", question="q", outcome_col="label",
        backend="claude_code", model="", timeout_sec=600,
        mode="explore", explore_share=1.0,
    )
    assert job2["holdout_frac"] == 0.0
    assert "--holdout" not in (run2 / "job.sh").read_text()


def test_launch_explore_job_writes_record_and_wrapper(tmp_path, monkeypatch):
    import evalvitals.analysis.upload_app as ua

    class _FakeProc:
        pid = 4242

    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(ua.subprocess, "Popen", _fake_popen)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    job = launch_explore_job(
        run_dir, run_dir / "data", question='what "drives" FAIL?',
        outcome_col="label", backend="claude_code", model="", timeout_sec=1200,
    )

    assert captured["cmd"][0] == "bash"
    assert captured["kwargs"]["start_new_session"] is True
    assert job["pid"] == 4242
    saved = json.loads((run_dir / "job.json").read_text())
    assert saved["out_dir"].endswith("output")
    wrapper = (run_dir / "job.sh").read_text()
    # free-text question survives shell quoting, and the exit code is persisted
    assert 'what "drives" FAIL?' in wrapper.replace('"\'"\'"', '"')
    assert "echo $? > exit_code" in wrapper


def test_job_status_transitions(tmp_path):
    run = tmp_path / "r"
    (run / "output").mkdir(parents=True)

    # no report, no exit code, dead pid -> stale
    (run / "job.json").write_text(json.dumps({"pid": 999999999}))
    assert job_status(run)["state"] == "stale"

    # our own pid is alive -> running
    import os
    (run / "job.json").write_text(json.dumps({"pid": os.getpid()}))
    assert job_status(run)["state"] == "running"

    # exited without a report -> failed, exit code surfaced
    (run / "exit_code").write_text("1\n")
    st = job_status(run)
    assert st["state"] == "failed" and st["exit_code"] == 1

    # report present wins regardless of exit code -> done
    (run / "output" / "exploratory_report.json").write_text("{}")
    assert job_status(run)["state"] == "done"


def test_list_runs_only_dirs_with_job_json(tmp_path):
    (tmp_path / "not_a_run").mkdir()
    a = tmp_path / "a"
    a.mkdir()
    (a / "job.json").write_text("{}")
    assert list_runs(tmp_path) == [a]


def test_archive_run_moves_only_direct_workspace_runs(tmp_path):
    run = tmp_path / "finished_run"
    run.mkdir()
    (run / "job.json").write_text("{}")
    (run / "output.txt").write_text("keep me")

    archived = archive_run(tmp_path, run)
    assert archived == tmp_path / ".trash" / "finished_run"
    assert not run.exists()
    assert archived.joinpath("output.txt").read_text() == "keep me"
    assert list_runs(tmp_path) == []

    attached = tmp_path.parent / "attached_results"
    attached.mkdir()
    (attached / "job.json").write_text("{}")
    with pytest.raises(ValueError, match="local workbench run"):
        archive_run(tmp_path, attached)
    assert attached.exists()


def test_timeline_collapses_event_history_and_closes_dangling_steps():
    events = [
        {"turn_id": "initial", "stage": "ingest", "status": "started", "message": "Extracting"},
        {"turn_id": "initial", "stage": "discover", "status": "completed", "message": "Found files"},
        {"turn_id": "initial", "stage": "job", "status": "started", "message": "Worker started"},
        {"turn_id": "initial", "stage": "m2", "status": "started", "message": "Starting M2"},
        {"turn_id": "initial", "stage": "m2_codegen", "status": "started", "message": "Generating code"},
        {"turn_id": "followup", "stage": "route", "status": "started", "message": "Other turn"},
    ]

    steps = _timeline_steps(events, turn_id="initial", run_state="done")
    by_stage = {str(step["stage"]): step for step in steps}
    assert set(by_stage) == {"ingest", "discover", "job", "m2", "m2_codegen"}
    assert all(step["status"] == "completed" for step in by_stage.values())
    assert "final report is available" in by_stage["job"]["message"]


def test_followup_router_and_canceled_turn_status(tmp_path):
    assert _route_followup("Explain the first takeaway") == "answer"
    assert _route_followup("Compare the two groups with a new chart") == "analyze"
    assert _route_followup("What hypothesis should we test?") == "hypothesize"

    turn = tmp_path / "turn"
    turn.mkdir()
    events = tmp_path / "events.jsonl"
    (turn / "job.json").write_text(json.dumps({
        "pid": 999999999, "turn_id": "turn_001", "events_path": str(events),
    }))
    events.write_text(json.dumps({"turn_id": "turn_001", "status": "canceled"}) + "\n")
    assert job_status(turn)["state"] == "canceled"


# ── the page itself (dashboard extras) ───────────────────────────────────────

pytest.importorskip("streamlit")
pytest.importorskip("pandas")


def _run_app(workspace):
    from streamlit.testing.v1 import AppTest

    sys.argv = ["upload_app.py", str(workspace)]
    at = AppTest.from_file("evalvitals/analysis/upload_app.py", default_timeout=30)
    at.run()
    return at


def _build_finished_run(workspace, name="demo_20260711_120000"):
    run = workspace / name
    out = run / "output"
    out.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({
        "pid": 1, "argv": ["python", "-m", "evalvitals.cli", "explore", "data"],
        "question": "what drives FAIL?", "outcome_col": "label",
        "backend": "claude_code", "model": "", "timeout_sec": 1200,
        "data_dir": str(run / "data"), "out_dir": str(out),
        "started_at": "2026-07-11T12:00:00",
    }))
    (run / "exit_code").write_text("0\n")
    (out / "exploratory_report.json").write_text(json.dumps({
        "ok": True,
        "question": "what drives FAIL?",
        "observations": ["all failures adversarial"],
        "takeaways": [{"title": "Peaked attention rides with FAIL.",
                       "chart_names": [], "table_names": [],
                       "analysis": "focus share separates.", "caveat": ""}],
        "hypotheses": [{"statement": "Peaked attention marks hallucinations.",
                        "basis": "d=1.3", "test_design": "re-test on holdout"}],
        "candidate_signals": [], "charts": [], "plots": [], "tables": {},
    }))
    return run


def test_upload_page_renders_form(tmp_path):
    at = _run_app(tmp_path)
    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Upload" in blob
    assert at.button[0].label == "Start analysis"
    # nothing uploaded yet -> the launch button is disabled
    assert at.button[0].disabled
    # the sidebar-reopen chevron must be exempted from the chrome-hiding CSS,
    # or a collapsed sidebar (narrow window) can never be reopened
    assert "stExpandSidebarButton" in blob


def test_mode_selection_drives_split_slider(tmp_path):
    at = _run_app(tmp_path)
    mode = next(r for r in at.radio if r.label == "Analysis mode")
    # Verification is the default: a run that confirms nothing is the weaker
    # product, so the user opts OUT of holding data back rather than opting in.
    assert mode.value == "Explore + held-out verification"
    slider = at.slider[0]
    assert slider.value == 0.6  # 0.6 : 0.4 default in verification mode
    blob = " ".join(str(m.value) for m in at.caption)
    assert "60%" in blob and "40%" in blob

    mode.set_value("Explore only (M2 + M3)")
    at.run()
    assert not at.exception
    slider = at.slider[0]
    assert slider.value == 1.0  # explore-only analyses everything in-sample


def test_finished_run_renders_explore_tabs(tmp_path):
    run = _build_finished_run(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    # the sidebar lists the run: AppTest's .options carry the format_func'd
    # display labels (a human-readable name, not the storage directory), while
    # set_value takes the raw option (the run name)
    from evalvitals.analysis.upload_app import _pretty_name

    radio = at.sidebar.radio[0]
    label = next(o for o in radio.options if str(o).endswith(_pretty_name(run.name)))
    assert "🟢" in label  # exit 0 + report present -> shown as done
    radio.set_value(run.name)
    at.run()
    assert not at.exception
    # uploads share the dashboard's FIXED five-tab layout; the stages this
    # M3-only run never reached grey out instead of disappearing
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting", "2 Exploratory Analysis", "3 Hypotheses",
        "4 Validation results", "5 Fix",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Peaked attention marks hallucinations." in blob
    assert blob.count("not available for this run") == 2


def test_finished_upload_run_uses_non_speculative_hypothesis_empty_state(tmp_path):
    run = _build_finished_run(tmp_path)
    out = run / "output"
    source = "\n".join(f"line_{i} = {i}" for i in range(80)) + "\nFULL_SOURCE_SENTINEL = True\n"
    (out / "analysis.py").write_text(source, encoding="utf-8")
    (out / "exploratory_report.json").write_text(json.dumps({
        "ok": True,
        "question": "what drives FAIL?",
        "observations": ["all failures adversarial"],
        "hypotheses": [],
        "candidate_signals": [{"name": "focus_share_high"}],
        "recommended_confirmatory_tests": ["Re-test this signal on held-out rows."],
        "charts": [], "plots": [], "tables": {},
    }), encoding="utf-8")

    at = _run_app(tmp_path)
    radio = at.sidebar.radio[0]
    radio.set_value(run.name)
    at.run()

    assert not at.exception
    info = " ".join(str(i.value) for i in at.info)
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "No formal hypotheses were recorded for this report." in info
    assert "M3 step enabled" not in info
    assert "too thin" not in info
    assert "Candidate signals — possible follow-ups, not validated hypotheses" in blob
    assert "Candidate signals — possible follow-ups, not validated hypotheses" in [
        e.label for e in at.expander
    ]
    assert "Run artifacts and developer details" in [e.label for e in at.expander]
    assert "Generated analysis.py preview" in blob
    assert any("line_0 = 0" in str(c.value) for c in at.code)
    assert not any("FULL_SOURCE_SENTINEL" in str(c.value) for c in at.code)


def test_attached_local_dir_renders_in_sidebar_and_body(tmp_path):
    """--attach lists an existing explore output next to uploads (read-only)
    and renders it with the same unified five-tab layout."""
    from streamlit.testing.v1 import AppTest

    local = tmp_path / "outputs_attn_full"
    local.mkdir()
    (local / "exploratory_report.json").write_text(json.dumps({
        "ok": True, "question": "q",
        "observations": ["obs"], "takeaways": [], "hypotheses": [],
        "candidate_signals": [], "charts": [], "plots": [], "tables": {},
    }))
    ws = tmp_path / "ws"
    ws.mkdir()

    sys.argv = ["upload_app.py", str(ws), "--attach", str(local)]
    at = AppTest.from_file("evalvitals/analysis/upload_app.py", default_timeout=30)
    at.run()
    assert not at.exception

    from evalvitals.analysis.upload_app import _pretty_name

    radio = at.sidebar.radio[0]
    label = next(o for o in radio.options if str(o).endswith(_pretty_name(local)))
    assert label.startswith("📁")
    radio.set_value(f"@{local}")
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting", "2 Exploratory Analysis", "3 Hypotheses",
        "4 Validation results", "5 Fix",
    ]
    # The on-disk location is provenance, not chrome: it must not be printed
    # beside the title where a screenshot or screen-share would carry it. It
    # survives only inside the collapsed provenance disclosure.
    assert any("read-only" in str(c.value) for c in at.caption)
    leaked = [str(c.value) for c in at.caption if str(local) in str(c.value)]
    assert leaked and all(t.startswith("Bundle:") for t in leaked), leaked
    blob = " ".join(str(m.value) for m in at.markdown)
    assert blob.count("not available for this run") == 2


def test_failed_run_shows_log_and_hint(tmp_path):
    run = tmp_path / "boom_20260711_120000"
    run.mkdir()
    (run / "job.json").write_text(json.dumps({"pid": 1, "argv": [], "backend": "claude_code"}))
    (run / "exit_code").write_text("1\n")
    (run / "explore.log").write_text("claude: command not found\n")
    at = _run_app(tmp_path)
    at.sidebar.radio[0].set_value(run.name)
    at.run()
    assert not at.exception
    assert any("exited with code 1" in str(e.value) for e in at.error)
    assert any("command not found" in str(c.value) for c in at.code)
