from __future__ import annotations

import logging

import pytest

import evalrx
from evalrx.analysis.cli import main as explore_main
from evalrx.cli import main
from evalrx.logging_utils import _MARKER_ATTR, TOP_LEVEL_LOGGER_NAME


def test_top_level_cli_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "EvalRX command-line interface" in out
    # chat REPL is retired; the single-shot explore entry replaces it.
    assert "explore" in out
    assert "chat" not in out


def test_verbose_flag_documented_in_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "-v, --verbose" in out


def test_verbose_flag_enables_console_logging():
    try:
        evalrx.disable_console_logging()
        top = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
        assert not any(getattr(h, _MARKER_ATTR, False) for h in top.handlers)
        assert main(["-v"]) == 0
        assert any(getattr(h, _MARKER_ATTR, False) for h in top.handlers)
    finally:
        evalrx.disable_console_logging()


def test_without_verbose_flag_logging_untouched():
    evalrx.disable_console_logging()
    top = logging.getLogger(TOP_LEVEL_LOGGER_NAME)
    assert main([]) == 0
    assert not any(getattr(h, _MARKER_ATTR, False) for h in top.handlers)


def test_top_level_explore_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["explore", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "single-shot exploratory analysis" in out.lower() or "no interactive repl" in out.lower()


def test_the_dashboard_command_is_retired(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "out"])
    assert exc.value.code == 2
    assert "invalid choice: 'dashboard'" in capsys.readouterr().err


def test_top_level_serve_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Generate (if needed)" in out
    assert "--no-browser" in out


def test_top_level_serve_dispatch(monkeypatch):
    import evalrx.cli as cli_mod

    captured = {}

    def _fake_serve(run_dir, *, port, no_audio, open_browser, block=True, runs_root=None):
        captured.update(run_dir=run_dir, port=port, no_audio=no_audio,
                        open_browser=open_browser, block=block, runs_root=runs_root)
        return 0

    monkeypatch.setattr(cli_mod, "serve_report", _fake_serve)
    assert main(["serve", "my_run", "--port", "8500", "--no-audio", "--no-browser"]) == 0
    assert captured == {"run_dir": "my_run", "port": 8500, "no_audio": True,
                        "open_browser": False, "block": True, "runs_root": None}


def test_langfuse_report_source_materializes_a_cache(monkeypatch, tmp_path):
    import evalrx.cli as cli_mod
    import evalrx.reporting.langfuse_source as source_mod

    captured = {}

    class Source:
        def materialize(self, trace_id, destination):
            captured["trace_id"] = trace_id
            captured["destination"] = destination
            return tmp_path

    def _build(**kwargs):
        captured.update(kwargs)
        return tmp_path / "report.html"

    monkeypatch.setattr(source_mod, "LangfuseRunSource", Source)
    monkeypatch.setattr("evalrx.reporting.static_export.export_static_report", _build)
    monkeypatch.setattr(cli_mod, "_langfuse_cache", lambda _trace: tmp_path / "cache")

    assert main(["report", "--source", "langfuse", "--trace-id", "trace-1"]) == 0
    assert captured["trace_id"] == "trace-1"
    assert captured["run_dir"] == str(tmp_path)


def test_top_level_explore_holdout_dispatch(monkeypatch):
    import evalrx.cli as cli_mod

    captured = {}

    def _fake_run_explore(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_mod, "run_explore", _fake_run_explore)
    assert main(["explore", "data", "--holdout-frac", "0.4",
                 "--holdout-confirm", "--holdout-seed", "7"]) == 0
    assert captured["holdout_frac"] == 0.4
    assert captured["holdout_confirm"] is True
    assert captured["holdout_seed"] == 7
    assert captured["judge_model"] == "claude-opus-4-8"
    assert captured["open_report"] is False


def test_serve_report_opens_the_explore_report(monkeypatch):
    import evalrx.cli as cli_mod

    captured = {}
    monkeypatch.setattr(cli_mod, "run_explore", lambda path, **kwargs: captured.update(kwargs) or 0)
    assert main(["explore", "data", "--serve-report"]) == 0
    assert captured["open_report"] is True


def test_explore_entry_help(capsys):
    with pytest.raises(SystemExit) as exc:
        explore_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "evalrx-explore" in out
    assert "--serve-report" in out and "--dashboard" not in out


def test_chat_repl_is_retired():
    # The interactive chat shell and its CLI entry no longer exist.
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("evalrx.analysis.chat")
    from evalrx.analysis import cli

    assert not hasattr(cli, "chat_main")


# --- every command reaches the function behind it ---------------------------

def test_run_dispatches_to_the_benchmark_runner(monkeypatch):
    import evalrx.benchmark.run as run_mod

    seen = []
    monkeypatch.setattr(run_mod, "run_args", lambda args: seen.append(args.command) or 3)
    monkeypatch.setattr(run_mod, "build_parser", lambda parser=None: parser)

    assert main(["run"]) == 3
    assert seen == ["run"]


def test_explore_without_a_path_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["explore"])
    assert exc.value.code == 2
    assert "evalrx explore requires a results path" in capsys.readouterr().err


@pytest.mark.parametrize("code, status", [(0, "completed"), (1, "failed")])
def test_explore_progress_path_brackets_the_run_with_job_events(monkeypatch, tmp_path, code, status):
    import json

    import evalrx.cli as cli_mod

    captured = {}
    monkeypatch.setattr(cli_mod, "run_explore", lambda path, **kwargs: captured.update(kwargs) or code)
    progress = tmp_path / "progress.jsonl"

    assert main(["explore", "data", "--progress-path", str(progress), "--thread-id", "t1"]) == code
    events = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [(e["stage"], e["status"], e["thread_id"], e["turn_id"]) for e in events] == [
        ("job", "started", "t1", "explore"), ("job", status, "t1", "explore")]
    assert captured["progress_sink"] is not None


def test_explore_no_narrate_without_progress_path_passes_no_sink(monkeypatch):
    import evalrx.cli as cli_mod

    captured = {}
    monkeypatch.setattr(cli_mod, "run_explore", lambda path, **kwargs: captured.update(kwargs) or 0)

    assert main(["explore", "data", "--no-narrate"]) == 0
    assert captured["progress_sink"] is None


def test_run_codebase_dispatch(monkeypatch):
    import evalrx.cli as cli_mod
    from evalrx.analysis.explorer import RECORDS_FILENAME
    from evalrx.analysis.narration import TerminalNarrator

    calls = []
    monkeypatch.setattr(cli_mod, "run_codebase_cli", lambda path, **kwargs: calls.append((path, kwargs)) or 0)

    assert main(["run-codebase", "repo"]) == 0
    assert main(["run-codebase", "repo", "--no-explore", "--no-narrate", "--records-name", "r.jsonl",
                 "--serve-report", "--model", "m", "-q", "why"]) == 0
    (path, default), (_, custom) = calls
    sink = default.pop("progress_sink")
    assert path == "repo" and isinstance(sink, TerminalNarrator)
    assert default == {
        "out": "evalrx_run_codebase_output", "coder_provider": "claude_code", "coder_model": "",
        "coder_binary": "", "outcome_col": None, "records_name": RECORDS_FILENAME, "timeout_sec": 1200,
        "max_attempts": 2, "question": "Explore this dataset and surface the patterns that matter.",
        "analyze": True, "open_report": False}
    assert (custom["analyze"], custom["progress_sink"], custom["records_name"], custom["open_report"],
            custom["coder_model"], custom["question"]) == (False, None, "r.jsonl", True, "m", "why")


def test_serve_reports_a_missing_ui_extra_as_a_usage_error(monkeypatch, capsys):
    import evalrx.cli as cli_mod

    def _no_ui(*args, **kwargs):
        raise ImportError("install evalrx[ui]")

    monkeypatch.setattr(cli_mod, "serve_report", _no_ui)
    with pytest.raises(SystemExit) as exc:
        main(["serve", "out"])
    assert exc.value.code == 2
    assert "install evalrx[ui]" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["serve", "report", "publish-report"])
def test_langfuse_source_without_a_trace_id_is_a_usage_error(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "out", "--source", "langfuse"])
    assert exc.value.code == 2
    assert "--trace-id is required with --source langfuse" in capsys.readouterr().err


def test_publish_report_prints_what_it_wrote(monkeypatch, capsys):
    from types import SimpleNamespace

    import evalrx.reporting.dynamic as dynamic_mod

    seen = {}

    def _publish(source_dir, example_dir=None):
        seen.update(source_dir=source_dir, example_dir=example_dir)
        return SimpleNamespace(data_path="d.json", spec_path="s.json", generated_by="deterministic")

    monkeypatch.setattr(dynamic_mod, "publish_report", _publish)

    assert main(["publish-report", "out", "--example-dir", "ex"]) == 0
    assert seen == {"source_dir": "out", "example_dir": "ex"}
    assert capsys.readouterr().out.splitlines() == [
        "Published report data: d.json", "Published report layout: s.json (deterministic)"]


def test_publish_report_failure_is_a_usage_error(monkeypatch, capsys):
    import evalrx.reporting.dynamic as dynamic_mod

    def _missing(source_dir, example_dir=None):
        raise LookupError("no run found")

    monkeypatch.setattr(dynamic_mod, "publish_report", _missing)
    with pytest.raises(SystemExit) as exc:
        main(["publish-report", "out"])
    assert exc.value.code == 2
    assert "no run found" in capsys.readouterr().err


def test_export_langfuse_writes_beside_the_run_or_syncs(monkeypatch):
    from pathlib import Path

    import evalrx.reporting.langfuse_exporter as exporter_mod

    written = []
    synced = []
    monkeypatch.setattr(exporter_mod, "export_to_langfuse_bundle", lambda run_dir, out: written.append((run_dir, out)))
    monkeypatch.setattr(exporter_mod, "sync_to_langfuse_live", lambda run_dir: synced.append(run_dir) or run_dir == "ok")

    assert main(["export-langfuse", "out"]) == 0
    assert main(["export-langfuse", "out", "-o", "t.json"]) == 0
    assert written == [("out", Path("out") / "langfuse_trace.json"), ("out", "t.json")]
    assert main(["export-langfuse", "ok", "--sync"]) == 0
    assert main(["export-langfuse", "down", "--sync"]) == 1
    assert synced == ["ok", "down"] and len(written) == 2


def test_backfill_langfuse_prints_a_summary(monkeypatch, capsys):
    import evalrx.observability as observability_mod

    seen = {}

    def _backfill(run_dir, dry_run=False):
        seen.update(run_dir=run_dir, dry_run=dry_run)
        return {"trace_id": "t", "events": 3, "published": 1, "pending": 2}

    monkeypatch.setattr(observability_mod, "backfill_run_to_langfuse", _backfill)

    assert main(["backfill-langfuse", "out", "--dry-run"]) == 0
    assert seen == {"run_dir": "out", "dry_run": True}
    assert capsys.readouterr().out.strip() == "Langfuse backfill: trace=t events=3 published=1 pending=2"
