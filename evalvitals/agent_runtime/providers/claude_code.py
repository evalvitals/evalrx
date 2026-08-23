"""Claude Code CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalvitals.agent_runtime.cli_transcript import render_claude_stream
from evalvitals.agent_runtime.providers.base import CliAgentBase


class ClaudeCodeAgent(CliAgentBase):
    """Claude Code CLI backend (``claude -p``)."""

    _provider_name = "claude_code"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        # No --dangerously-skip-permissions. The narrow grant below already says
        # exactly what this agent may do, so the blanket flag added nothing --
        # and it is refused outright under root unless something sets
        # IS_SANDBOX=1, which our compose files do and a bare root box (Colab, a
        # plain container, CI) does not. There the coder exited in 0.7s having
        # written no files, ExperimentWriter aborted for want of a .py, and every
        # L2-and-above repair candidate was dead on arrival while the log said
        # only "CLI agent produced no .py files".
        #
        # Same reasoning as the judge in agent_runtime/judges/claude.py: ask for
        # the tools the work needs, not for every permission there is.
        allowed = "Bash Edit Write Read" + (" Skill" if self._allow_skills else "")
        cmd = [
            self._binary,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--allowed-tools",
            allowed,
            "--add-dir",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._max_budget_usd:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        cmd.extend(self._extra_args)
        return cmd

    def _postprocess_output(self, stdout: str) -> tuple[str, dict | None]:
        return render_claude_stream(stdout)
