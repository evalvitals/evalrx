"""Tests for surfacing agy backend errors (quota/auth) from its log."""

from __future__ import annotations

from evalrx.eval_agent.cli_agent import _scan_agy_log


def test_scan_extracts_resource_exhausted(tmp_path):
    log = tmp_path / "agy.log"
    log.write_text(
        "I0608 01:09:00.583 printmode.go:147] Print mode: sending message\n"
        "E0608 01:09:01.101 log.go:398] agent executor error: RESOURCE_EXHAUSTED "
        "(code 429): You have exhausted your capacity on this model. "
        "Your quota will reset after 125h8m26s.\n"
        "I0608 01:09:01.235 manager.go:513] CLI store manager shutting down\n",
        encoding="utf-8",
    )
    reason = _scan_agy_log(str(log))
    assert "RESOURCE_EXHAUSTED" in reason
    assert "quota will reset" in reason
    # The glog "E0608 ...]" prefix is stripped.
    assert not reason.startswith("E0608")


def test_scan_returns_empty_without_error(tmp_path):
    log = tmp_path / "agy.log"
    log.write_text(
        "I0608 01:09:00.583 printmode.go:147] sending message\n"
        "I0608 01:09:01.235 manager.go:513] done\n",
        encoding="utf-8",
    )
    assert _scan_agy_log(str(log)) == ""


def test_scan_missing_file_is_empty():
    assert _scan_agy_log("/nonexistent/agy.log") == ""


def test_scan_picks_last_error(tmp_path):
    log = tmp_path / "agy.log"
    log.write_text(
        "E0608 00:00:00.000 a.go:1] transient quota blip\n"
        "E0608 01:00:00.000 b.go:2] UNAUTHENTICATED: token expired\n",
        encoding="utf-8",
    )
    assert "UNAUTHENTICATED" in _scan_agy_log(str(log))
