"""Regression coverage for the M1-M5 loop's live terminal narration
(RunLoggerV2's `narrate=True` -> LoopNarrator)."""

from __future__ import annotations

import io

from evalrx.eval_agent.narration import LoopNarrator, _supports_color


def _narrator() -> tuple[LoopNarrator, io.StringIO]:
    buf = io.StringIO()
    return LoopNarrator(stream=buf, color=False), buf


def test_probe_event_reports_real_analyzer_count_and_duration():
    n, buf = _narrator()
    n.on_event("M1", "probe", {"cycle": 0, "analyzers": ["attention", "entropy"], "duration_sec": 4.2})
    line = buf.getvalue().strip()
    assert line.startswith("M1")
    assert "cycle 0" in line
    assert "2 analyzers run" in line
    assert "4.2s" in line


def test_analysis_event_reports_findings_severity_and_figures():
    n, buf = _narrator()
    n.on_event("M2", "analysis", {
        "cycle": 1, "n_findings": 3, "severity": "high", "figures": ["a.png", "b.png"],
    })
    line = buf.getvalue().strip()
    assert line.startswith("M2")
    assert "3 finding(s)" in line
    assert "severity=high" in line
    assert "2 figure(s)" in line


def test_diagnosis_event_pluralizes_hypotheses():
    n, buf = _narrator()
    n.on_event("M3", "diagnosis", {"cycle": 0, "n_hypotheses": 1})
    assert "1 falsifiable hypothesis" in buf.getvalue()
    assert "hypotheses" not in buf.getvalue()

    n2, buf2 = _narrator()
    n2.on_event("M3", "diagnosis", {"cycle": 0, "n_hypotheses": 3})
    assert "3 falsifiable hypotheses" in buf2.getvalue()


def test_surgery_event_shows_real_verdict_not_a_fabricated_one():
    n, buf = _narrator()
    n.on_event("M4", "surgery", {
        "cycle": 0, "status": "SUPPORTED", "fixed": True,
        "hypothesis": "peaked_attention correlates with hallucination",
    })
    line = buf.getvalue().strip()
    assert line.startswith("M4")
    assert "✓" in line
    assert "SUPPORTED" in line
    assert "peaked_attention correlates with hallucination" in line


def test_surgery_hypothesis_text_is_truncated_when_long():
    n, buf = _narrator()
    long_statement = "x" * 200
    n.on_event("M5", "surgery", {"status": "REFUTED", "fixed": False, "hypothesis": long_statement})
    line = buf.getvalue()
    assert "..." in line
    assert long_statement not in line


def test_fix_prints_one_line_per_tier_then_the_best_pick():
    n, buf = _narrator()
    n.on_event("M5", "fix", {
        "attempted": [
            {"tier": "L1_PROMPT", "name": "prompt_rewrite", "fixed": False, "effect": 0.004,
             "n_fixed": 1, "n_broken": 0},
            {"tier": "L2_SCAFFOLD", "name": "attention_crop", "fixed": True, "effect": 0.142,
             "n_fixed": 18, "n_broken": 2},
        ],
        "best": {"name": "attention_crop", "effect": 0.142, "n_fixed": 18, "n_broken": 2},
    })
    lines = buf.getvalue().splitlines()
    assert len(lines) == 3  # L1 + L2 + best
    assert "L1_PROMPT" in lines[0] and "✗" in lines[0]
    assert "L2_SCAFFOLD" in lines[1] and "✓" in lines[1]
    assert "best: attention_crop" in lines[2]
    assert "+0.142" in lines[2]


def test_fix_with_no_winner_says_so_honestly():
    n, buf = _narrator()
    n.on_event("M5", "fix", {"attempted": [{"tier": "L1_PROMPT", "name": "x", "fixed": False}], "best": {}})
    line = buf.getvalue().splitlines()[-1]
    assert "none selected" in line


def test_run_start_and_loop_end_bookend_the_run():
    n, buf = _narrator()
    n.on_run_start({"model": "qwen3-vl-8b-instruct", "n_cases": 606})
    n.on_run_event("loop_end", {"cycles": 2, "resolved": True, "n_hypotheses": 2})
    lines = buf.getvalue().splitlines()
    assert "RUN" in lines[0] and "qwen3-vl-8b-instruct" in lines[0] and "606" in lines[0]
    assert "RUN" in lines[1] and "2 cycle(s)" in lines[1] and "resolved=True" in lines[1]


def test_non_stage_and_non_narrated_events_are_silent():
    n, buf = _narrator()
    n.on_event("M1", "model_call", {"cycle": 0})  # real event, just not narration-worthy
    n.on_event("RUN", "probe", {"cycle": 0, "analyzers": []})  # not a real M-stage key
    n.on_run_event("agent_decisions", {})
    assert buf.getvalue() == ""


def test_color_is_off_for_a_non_tty_stream(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("EVALRX_FORCE_COLOR", raising=False)
    assert _supports_color(io.StringIO()) is False
