"""The coding provider asks for its tools, not for every permission.

Live failure (VideoLLaMA2.1-7B-AV / Music-AVQA, 2026-08-23): M4's experiment
writer invoked claude_code, which exited in 0.7 seconds with

    --dangerously-skip-permissions cannot be used with root/sudo privileges

and ExperimentWriter aborted for want of a generated .py. The run log said only
"CLI agent produced no .py files", so a whole repair stage came back empty with
no indication that the coder had never started. Colab runs as root; so does
almost every plain container.

The flag was redundant on top of the explicit --allowed-tools grant that was
already there, which is why removing it costs nothing: the agent is still told
exactly what it may do.
"""

from __future__ import annotations

from pathlib import Path

from evalrx.agent_runtime.providers.claude_code import ClaudeCodeAgent


def _cmd(**kw) -> list[str]:
    agent = ClaudeCodeAgent(binary_path="/usr/bin/true", timeout_sec=10, **kw)
    return agent._build_cmd("do the thing", Path("/tmp/work"))


def test_no_blanket_permission_bypass():
    assert "--dangerously-skip-permissions" not in _cmd()


def test_the_tools_it_needs_are_still_granted():
    """Removing the bypass must not remove the grant -- the coder writes and runs
    code, and a coder that cannot use Write or Bash is as useless as one that
    cannot start."""
    cmd = _cmd()
    i = cmd.index("--allowed-tools")
    granted = set(cmd[i + 1].split())
    assert {"Bash", "Edit", "Write", "Read"} <= granted


def test_skills_are_granted_only_when_asked_for():
    assert "Skill" not in _cmd()
    assert "Skill" in _cmd(allow_skills=True)[_cmd(allow_skills=True).index("--allowed-tools") + 1]


def test_the_workspace_is_still_added():
    cmd = _cmd()
    assert cmd[cmd.index("--add-dir") + 1] == "/tmp/work"
