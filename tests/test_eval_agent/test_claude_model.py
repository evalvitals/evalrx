"""ClaudeModel — the Claude Code CLI wrapped as an M1–M5 judge.

Tested against a fake ``claude`` executable (a tiny shell script) so no real
CLI, auth, or network is involved.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from evalrx.core.capability import Capability
from evalrx.eval_agent import ClaudeModel


def _fake_claude(tmp_path: Path, body: str) -> str:
    """Write an executable fake claude binary and return its path."""
    p = tmp_path / "claude"
    p.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_missing_binary_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing named claude on PATH
    with pytest.raises(RuntimeError, match="CLAUDE_PATH"):
        ClaudeModel()


def test_generate_returns_stdout(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "HELLO FROM JUDGE"')
    judge = ClaudeModel(binary_path=binary)
    assert judge.generate("hi") == "HELLO FROM JUDGE"
    assert Capability.GENERATE in judge.capabilities


def test_model_flag_forwarded(tmp_path):
    # Echo back the argv so the test can assert the --model flag.
    binary = _fake_claude(tmp_path, 'echo "$@"')
    judge = ClaudeModel(binary_path=binary, model="claude-fable-5")
    out = judge.generate("question")
    assert "--model claude-fable-5" in out
    # A text judge asks for no tools, not for every permission. The blanket flag
    # is refused outright under root unless something sets IS_SANDBOX=1, which
    # made the judge unrunnable on a bare root box (Colab, a plain container).
    assert "--tools" in out
    assert "--dangerously-skip-permissions" not in out


def test_nonzero_exit_without_output_raises(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "auth expired" >&2; exit 3')
    judge = ClaudeModel(binary_path=binary)
    with pytest.raises(RuntimeError, match="exited 3"):
        judge.generate("hi")


def test_empty_response_warns_and_returns_empty(tmp_path):
    binary = _fake_claude(tmp_path, "exit 0")
    judge = ClaudeModel(binary_path=binary)
    with pytest.warns(UserWarning, match="empty response"):
        assert judge.generate("hi") == ""


def test_images_listed_in_prompt_and_workspace_added(tmp_path):
    # The prompt arrives on stdin now, so the fake must read both.
    binary = _fake_claude(tmp_path, 'echo "$@"; cat')
    img = tmp_path / "m2_effects.png"
    img.write_bytes(b"\x89PNG fake")
    judge = ClaudeModel(binary_path=binary)
    out = judge.generate("look at the figures", images=[img])
    assert "Images available in workspace: m2_effects.png" in out
    assert "--add-dir" in out
    # Reading the staged figures is the one thing a judge call does need a tool
    # for, so that call allows exactly Read -- still not the blanket bypass.
    assert "--allowed-tools Read" in out
    assert "--dangerously-skip-permissions" not in out


def test_timeout_raises(tmp_path):
    binary = _fake_claude(tmp_path, "sleep 5")
    judge = ClaudeModel(binary_path=binary, timeout_sec=1)
    with pytest.raises(RuntimeError, match="timed out"):
        judge.generate("hi")


def test_missing_image_paths_are_skipped(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "$@"')
    judge = ClaudeModel(binary_path=binary)
    out = judge.generate("q", images=[Path("/nonexistent/x.png")])
    assert "Images available" not in out


def test_binary_path_must_be_executable(tmp_path):
    p = tmp_path / "claude"
    p.write_text("not executable", encoding="utf-8")
    os.chmod(p, 0o644)
    with pytest.raises(RuntimeError):
        ClaudeModel(binary_path=str(p))


def test_utf8_output_decodes_under_any_locale(tmp_path, monkeypatch):
    """Regression: Fable's answers contain UTF-8 punctuation (em dashes etc.);
    subprocess text decoding must not depend on the container locale
    ('ascii' codec can't decode byte 0xc3 killed M3 in the first run)."""
    binary = _fake_claude(tmp_path, "printf 'HYPOTHESIS: caf\\303\\251 \\342\\200\\224 fine\\n'")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    judge = ClaudeModel(binary_path=binary)
    out = judge.generate("hi")
    assert "café — fine" in out


# ── the prompt must not be an argv entry ─────────────────────────────────────
#: Linux caps a SINGLE argv entry at 32 pages. Not ARG_MAX, not ulimit -s —
#: MAX_ARG_STRLEN, which nothing configures.
MAX_ARG_STRLEN = 32 * 4096


@pytest.mark.skipif(sys.platform != "linux", reason="MAX_ARG_STRLEN is a Linux limit")
def test_the_limit_this_guards_against_is_real(tmp_path):
    """The premise, so the guard below is never mistaken for superstition."""
    binary = _fake_claude(tmp_path, "wc -c")
    with pytest.raises(OSError, match="Argument list too long"):
        subprocess.run([binary, "x" * MAX_ARG_STRLEN], capture_output=True)


def test_prompt_larger_than_one_argv_entry_round_trips(tmp_path):
    """M2's prompt is the analyzers' findings JSON, so its size tracks how many
    analyzers M1 happened to select.

    Two runs of qwen3.5-2b over the SAME frozen bbh_word_sorting batch: 9
    analyzers -> 127,856 bytes, judge answered; 12 analyzers -> ~137,600,
    execve refused it with OSError(E2BIG). M2 caught the OSError, fell back to
    the threshold narrative, M3 had nothing to work from, and the run finished
    rc=0 reporting stopped_by=no_hypotheses — indistinguishable from a healthy
    run that found nothing.
    """
    binary = _fake_claude(tmp_path, "wc -c")
    judge = ClaudeModel(binary_path=binary)
    prompt = "x" * (MAX_ARG_STRLEN + 8000)

    assert int(judge.generate(prompt).strip()) == len(prompt)


def test_prompt_never_appears_in_argv(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "ARGV[$*]"')
    judge = ClaudeModel(binary_path=binary)

    assert "the prompt text" not in judge.generate("the prompt text")


def test_effort_flag_forwarded(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "$@"')
    judge = ClaudeModel(binary_path=binary, model="claude-fable-5", effort="high")
    out = judge.generate("q")
    assert "--effort high" in out
    # empty effort → flag omitted
    judge2 = ClaudeModel(binary_path=binary)
    assert "--effort" not in judge2.generate("q")
