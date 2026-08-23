"""OpenAI Codex CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalvitals.agent_runtime.providers.base import CliAgentBase
from evalvitals.agent_runtime.skills.installer import CodexSkillInstaller, SkillInstaller


class CodexAgent(CliAgentBase):
    """OpenAI Codex CLI backend (``codex exec``)."""

    _provider_name = "codex"

    def _make_skill_installer(self, skills: list[str]) -> SkillInstaller:
        return CodexSkillInstaller(skills)

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "exec",
            prompt,
            # Generated-code workspaces are intentionally isolated artifact
            # directories, not Git checkouts.  Codex otherwise exits before
            # doing any work with "Not inside a trusted directory".
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--json",
            "-C",
            str(workdir),
        ]
        if self._model:
            cmd += ["-m", self._model]
        cmd.extend(self._extra_args)
        return cmd
