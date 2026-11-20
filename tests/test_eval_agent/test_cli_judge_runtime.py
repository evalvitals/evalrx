"""The CLI judges share one runner: a timeout kills the whole process group,
and a limit notice is an error for every provider, not only Claude."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from evalrx.agent_runtime.judges import AgyModel, ClaudeModel, CodexModel
from evalrx.agent_runtime.providers.base import CliAgentBase


def _binary(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_timeout_kills_the_judges_child_processes_too(tmp_path):
    # A CLI's worker processes outlived a plain subprocess.run timeout, which
    # kills only the direct child; the runner kills the whole process group.
    pidfile = tmp_path / "grandchild.pid"
    body = f'sleep 30 & echo $! > "{pidfile}"; sleep 30'
    judge = ClaudeModel(binary_path=_binary(tmp_path, "claude", body), timeout_sec=1)
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        judge.generate("hi")
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(grandchild)


@pytest.mark.parametrize("make", [
    lambda b: ClaudeModel(binary_path=b),
    lambda b: AgyModel(binary_path=b),
])
def test_a_limit_notice_on_stdout_is_an_error_for_each_provider(tmp_path, make):
    binary = _binary(tmp_path, "cli", "cat >/dev/null 2>&1; echo \"You've hit your usage limit. Limit resets at 5pm.\"")
    with pytest.raises(RuntimeError, match="account limit reached"):
        make(binary).generate("hi")


def test_codex_limit_notice_in_its_answer_file_is_an_error(tmp_path):
    body = (
        'while [ "$1" != "--output-last-message" ]; do shift; done\n'
        "echo \"You've hit your usage limit.\" > \"$2\""
    )
    with pytest.raises(RuntimeError, match="account limit reached"):
        CodexModel(binary_path=_binary(tmp_path, "codex", body)).generate("hi")


def test_the_prompt_reaches_claude_on_stdin_and_the_cwd_is_not_the_callers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = ClaudeModel(binary_path=_binary(tmp_path, "claude", 'printf "%s|" "$PWD"; cat')).generate("PROMPT")
    cwd, prompt = out.split("|", 1)
    assert prompt == "PROMPT"
    assert Path(cwd).resolve() != tmp_path.resolve()


class _ScriptAgent(CliAgentBase):
    _provider_name = "script"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        return [self._binary]


def test_a_coding_agent_that_prints_a_limit_notice_reports_it(tmp_path):
    notice = "echo '[assistant] starting'; echo \"You've hit your usage limit. Limit resets at 5pm.\""
    result = _ScriptAgent(_binary(tmp_path, "cli", notice)).run("write it", workdir=tmp_path / "w")
    assert result.error and "account limit reached" in result.error


def test_a_coding_agent_that_wrote_files_is_not_read_as_limited(tmp_path):
    body = "echo 'x = 1' > main.py; echo 'Note: your usage limit resets on Monday.'"
    result = _ScriptAgent(_binary(tmp_path, "cli", body)).run("write it", workdir=tmp_path / "w")
    assert result.error is None and "main.py" in result.files
