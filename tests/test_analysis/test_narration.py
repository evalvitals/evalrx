"""Regression coverage for the live terminal narration (`evalrx explore` /
`run-codebase`'s per-stage console output)."""

from __future__ import annotations

import io

from evalrx.analysis.narration import MultiSink, TerminalNarrator, _supports_color


def _narrator() -> tuple[TerminalNarrator, io.StringIO]:
    buf = io.StringIO()
    return TerminalNarrator(stream=buf, color=False), buf


def test_started_then_completed_prints_one_line_each_with_the_stage_label():
    n, buf = _narrator()
    n.emit("m2", "started", "Starting exploratory analysis (M2)")
    n.emit("m2", "completed", "Exploratory analysis completed",
           metrics={"n_figures": 23, "n_candidate_signals": 4})

    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("M2") and "explore" in lines[0] and "running" in lines[0]
    assert lines[1].startswith("M2")
    assert "23 figures" in lines[1]
    assert "4 candidate signals" in lines[1]
    assert "✓" in lines[1]


def test_failed_stage_shows_the_real_message_not_a_fabricated_one():
    n, buf = _narrator()
    n.emit("m2", "started", "Starting exploratory analysis (M2)")
    n.emit("m2", "failed", "backend produced no code")

    line = buf.getvalue().splitlines()[-1]
    assert "✗" in line
    assert "backend produced no code" in line


def test_m3_pluralizes_one_hypothesis_correctly():
    n, buf = _narrator()
    n.emit("m3", "completed", "Proposed 1 hypotheses", metrics={"n_hypotheses": 1})
    line = buf.getvalue().strip()
    assert "1 falsifiable hypothesis" in line
    assert "hypotheses" not in line  # singular, not the plural noun


def test_m4_summary_uses_real_holdout_counts():
    n, buf = _narrator()
    n.emit("m4", "completed", "Held-out verification complete",
           metrics={"n_rows": 242, "n_rejected": 1, "n_adjudicated": 3})
    line = buf.getvalue().strip()
    assert "held-out n=242" in line
    assert "1/3 reject" in line


def test_substage_is_silent_on_a_clean_first_attempt():
    n, buf = _narrator()
    n.emit("m2", "started", "Starting exploratory analysis (M2)")
    n.emit("m2_codegen", "started", "Generating analysis code", attempt=1)
    n.emit("m2_codegen", "completed", "Analysis code is ready", attempt=1)
    n.emit("m2_execute", "started", "Running generated analysis", attempt=1)
    n.emit("m2_execute", "completed", "Generated analysis finished", attempt=1)

    # Only the parent M2 "running…" line — no per-substage noise on a clean pass.
    assert len(buf.getvalue().splitlines()) == 1


def test_substage_retry_and_failure_are_narrated():
    n, buf = _narrator()
    n.emit("m2", "started", "Starting exploratory analysis (M2)")
    n.emit("m2_execute", "started", "Running generated analysis", attempt=1)
    n.emit("m2_execute", "failed", "Generated analysis needs repair", attempt=1)
    n.emit("m2_codegen", "started", "Generating analysis code", attempt=2)

    lines = buf.getvalue().splitlines()
    assert len(lines) == 3
    assert "✗" in lines[1] and "Generated analysis needs repair" in lines[1]
    assert "retry 2" in lines[2]


def test_run_stage_uses_the_codebase_label_not_a_redundant_run_run():
    n, buf = _narrator()
    n.emit("run", "started", "Running codebase at /tmp/x")
    line = buf.getvalue().strip()
    assert line.startswith("RUN")
    assert "run run" not in line.lower()


def test_multisink_fans_out_to_every_child_and_skips_none():
    n1, buf1 = _narrator()
    n2, buf2 = _narrator()
    sink = MultiSink(n1, None, n2)
    assert bool(sink)
    sink.emit("m2", "started", "Starting exploratory analysis (M2)")
    assert buf1.getvalue() == buf2.getvalue()
    assert "M2" in buf1.getvalue()


def test_multisink_with_only_none_entries_is_falsy():
    assert not MultiSink(None, None)


def test_color_is_off_for_a_non_tty_stream_even_without_no_color_set(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("EVALRX_FORCE_COLOR", raising=False)
    assert _supports_color(io.StringIO()) is False


def test_no_color_env_wins_even_when_the_stream_claims_to_be_a_tty(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert _supports_color(FakeTty()) is False
