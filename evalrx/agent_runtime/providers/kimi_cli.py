"""Kimi CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalrx.agent_runtime.providers.base import CliAgentBase


class KimiCliAgent(CliAgentBase):
    """Kimi CLI backend (``kimi chat``)."""

    _provider_name = "kimi_cli"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "chat",
            "--message",
            prompt,
            "--workdir",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd
