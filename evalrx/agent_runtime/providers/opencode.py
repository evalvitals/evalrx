"""OpenCode CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalrx.agent_runtime.providers.base import CliAgentBase


class OpenCodeAgent(CliAgentBase):
    """OpenCode CLI backend (``opencode run``)."""

    _provider_name = "opencode"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "run",
            "--message",
            prompt,
            "--cwd",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd
