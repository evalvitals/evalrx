"""The report viewer is a static server, not a Streamlit application."""

from __future__ import annotations


def test_serve_existing_report_without_recompiling(tmp_path, monkeypatch, capsys):
    from evalvitals.analysis import dashboard

    (tmp_path / "report.html").write_text("<!doctype html>", encoding="utf-8")
    observed = {}

    class FakeServer:
        server_port = 4321

        def __init__(self, address, handler):
            observed["address"] = address
            observed["handler"] = handler

        def serve_forever(self):
            observed["served"] = True

        def server_close(self):
            observed["closed"] = True

    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", FakeServer)
    assert dashboard.serve_report(
        tmp_path, port=0, open_browser=False, block=False,
    ) == 0

    assert observed["address"] == ("127.0.0.1", 0)
    assert "http://127.0.0.1:4321/report.html" in capsys.readouterr().out
