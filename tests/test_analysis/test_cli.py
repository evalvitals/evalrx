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


def test_top_level_dashboard_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "--help"])
    assert exc.value.code == 0
    assert "Deprecated alias" in capsys.readouterr().out


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


def test_explore_entry_help(capsys):
    with pytest.raises(SystemExit) as exc:
        explore_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "evalrx-explore" in out
    assert "--dashboard" in out


def test_chat_repl_is_retired():
    # The interactive chat shell and its CLI entry no longer exist.
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("evalrx.analysis.chat")
    from evalrx.analysis import cli

    assert not hasattr(cli, "chat_main")
