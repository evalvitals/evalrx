"""Regression coverage for the OSC 8 terminal hyperlink helper."""

from __future__ import annotations

import io

from evalrx.term_links import hyperlink


class _FakeTty(io.StringIO):
    def isatty(self) -> bool:  # noqa: D102
        return True


def test_plain_text_on_a_non_tty_stream_has_no_escape_codes():
    out = hyperlink("http://127.0.0.1:8501", stream=io.StringIO())
    assert out == "http://127.0.0.1:8501"
    assert "\033" not in out


def test_wraps_a_url_in_osc8_on_a_tty():
    out = hyperlink("http://127.0.0.1:8501", stream=_FakeTty())
    assert out.startswith("\033]8;;http://127.0.0.1:8501\033\\")
    assert out.endswith("\033]8;;\033\\")
    assert "http://127.0.0.1:8501" in out  # visible label preserved


def test_local_path_becomes_a_file_uri():
    out = hyperlink("/tmp/report.html", stream=_FakeTty())
    assert "file:///tmp/report.html" in out


def test_custom_label_is_shown_but_target_is_still_the_link():
    out = hyperlink("http://127.0.0.1:8501", "click here", stream=_FakeTty())
    assert "click here" in out
    assert "http://127.0.0.1:8501" in out


def test_env_override_disables_hyperlinks_even_on_a_tty(monkeypatch):
    monkeypatch.setenv("EVALRX_NO_HYPERLINKS", "1")
    out = hyperlink("http://127.0.0.1:8501", stream=_FakeTty())
    assert out == "http://127.0.0.1:8501"
