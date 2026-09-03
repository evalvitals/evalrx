"""Gemini CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalrx.agent_runtime.providers.base import CliAgentBase


class GeminiCliAgent(CliAgentBase):
    """Gemini CLI backend (``gemini -p``)."""

    _provider_name = "gemini_cli"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "-p",
            prompt,
            "--cwd",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd
