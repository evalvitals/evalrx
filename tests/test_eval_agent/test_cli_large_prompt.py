"""Large prompts must not blow past argv's OS size limit.

Every CLI judge/coder provider in ``agent_runtime`` passes its prompt as a
literal argv element (``-p <prompt>`` / ``exec <prompt>`` / ``--message
<prompt>``). This framework's prompts (a full M1 evidence dump, an L1/L2
fix-candidate brief with per-case tables, ...) routinely exceed the kernel's
argv+envp size limit -- observed in practice as ``OSError: [Errno 7]
Argument list too long`` on exec against a real ``agy`` binary, which the
caller then silently treats as "judge returned nothing." Both AgyModel
(the judge) and CliAgentBase (the shared coder-provider runner, covering
claude_code/antigravity/codex/gemini_cli/kimi_cli/opencode) spill oversized
prompts to a file in the sandboxed workspace instead of passing them inline.
"""

from __future__ import annotations

import shutil

import os
import stat
import textwrap
from pathlib import Path

import pytest

from evalvitals.agent_runtime.judges.agy import AgyModel
from evalvitals.agent_runtime.providers.base import CliAgentBase


def _echo_argv_script(tmp_path: Path) -> str:
    """A fake CLI binary that echoes back exactly what it was called with."""
    script = tmp_path / "fake_cli.sh"
    script.write_text('#!/bin/sh\necho GOT_ARGS:"$@"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)



#: A no-op executable, located rather than assumed. ``/bin/true`` exists on
#: Linux and not on macOS (it is /usr/bin/true there), so hardcoding the path
#: failed the whole parametrised suite on a Mac with FileNotFoundError -- a test
#: portability bug that reads exactly like six broken providers.
NOOP_BINARY = shutil.which("true") or "/usr/bin/true"


class TestAgyModelLargePrompt:
    def test_small_prompt_stays_inline(self, tmp_path):
        model = AgyModel(binary_path=_echo_argv_script(tmp_path), timeout_sec=10)
        out = model.generate("hello judge")
        assert "hello judge" in out
        assert "prompt.txt" not in out

    def test_large_prompt_spills_to_file(self, tmp_path):
        model = AgyModel(binary_path=_echo_argv_script(tmp_path), timeout_sec=10)
        big_prompt = "E" * (model._LARGE_PROMPT_BYTES + 1)
        out = model.generate(big_prompt)
        assert "prompt.txt" in out
        # The oversized text itself must never reach argv.
        assert "E" * 100 not in out

    def test_workspace_cleaned_up_after_large_prompt(self, tmp_path, monkeypatch):
        captured: dict[str, str] = {}

        def fake_mkdtemp(prefix=""):
            d = tmp_path / "agy_ws"
            d.mkdir()
            captured["dir"] = str(d)
            return str(d)

        monkeypatch.setattr("evalvitals.agent_runtime.judges.agy.tempfile.mkdtemp", fake_mkdtemp)
        model = AgyModel(binary_path=_echo_argv_script(tmp_path), timeout_sec=10)
        model.generate("F" * (model._LARGE_PROMPT_BYTES + 1))
        assert not os.path.exists(captured["dir"]), "temp workspace was not cleaned up"


class TestCliAgentBaseLargePrompt:
    class _EchoAgent(CliAgentBase):
        _provider_name = "echo_test"

        def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
            return ["/bin/sh", "-c", 'echo GOT_ARGS:"$@"', "--", prompt]

    def test_small_prompt_stays_inline(self, tmp_path):
        agent = self._EchoAgent(binary_path=NOOP_BINARY, timeout_sec=10)
        result = agent.run("short task", tmp_path)
        assert "short task" in result.raw_output
        assert not (tmp_path / "prompt.txt").exists()

    def test_large_prompt_spills_to_workdir(self, tmp_path):
        agent = self._EchoAgent(binary_path=NOOP_BINARY, timeout_sec=10)
        big_prompt = textwrap.dedent("evidence row\n") * 10_000  # well over 60KB
        result = agent.run(big_prompt, tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        assert prompt_file.exists()
        assert prompt_file.read_text(encoding="utf-8") == big_prompt
        assert "prompt.txt" in result.raw_output
        assert "evidence row" * 50 not in result.raw_output


@pytest.mark.parametrize(
    "provider_module,class_name",
    [
        ("evalvitals.agent_runtime.providers.claude_code", "ClaudeCodeAgent"),
        ("evalvitals.agent_runtime.providers.antigravity", "AntigravityAgent"),
        ("evalvitals.agent_runtime.providers.codex", "CodexAgent"),
        ("evalvitals.agent_runtime.providers.gemini_cli", "GeminiCliAgent"),
        ("evalvitals.agent_runtime.providers.kimi_cli", "KimiCliAgent"),
        ("evalvitals.agent_runtime.providers.opencode", "OpenCodeAgent"),
    ],
)
def test_every_real_provider_gets_the_spill_for_free(tmp_path, provider_module, class_name):
    """The fix lives once in CliAgentBase.run(); every subclass inherits it
    unmodified -- this just proves none of them override ``run()`` in a way
    that bypasses the spill."""
    import importlib

    cls = getattr(importlib.import_module(provider_module), class_name)
    agent = cls(binary_path=NOOP_BINARY, timeout_sec=10)
    big_prompt = "G" * (CliAgentBase._LARGE_PROMPT_BYTES + 1)
    # Exercise the real path: run() must write prompt.txt before _build_cmd
    # ever sees the oversized text. binary_path=NOOP_BINARY makes the actual
    # subprocess a no-op; only the pre-exec spill-to-file behavior is under test.
    agent.run(big_prompt, tmp_path)
    assert (tmp_path / "prompt.txt").read_text(encoding="utf-8") == big_prompt
