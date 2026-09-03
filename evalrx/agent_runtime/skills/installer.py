"""Skill installation strategies for CLI coding-provider adapters."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillInstaller:
    """Vendor skill directories into a run-local Claude-compatible layout."""

    def __init__(self, skills: list[str] | tuple[str, ...]) -> None:
        self._skills = list(skills)

    @property
    def skills(self) -> list[str]:
        return list(self._skills)

    @property
    def enabled(self) -> bool:
        return bool(self._skills)

    def install(self, workdir: Path) -> None:
        if not self._skills:
            return
        dest_root = workdir / ".claude" / "skills"
        for src in self._skills:
            src_path = Path(src)
            if not src_path.exists() or not src_path.is_dir():
                logger.warning("skill dir not found, skipping: %s", src)
                continue
            try:
                shutil.copytree(src_path, dest_root / src_path.name, dirs_exist_ok=True)
            except OSError as exc:
                logger.warning("could not vendor skill %s: %s", src, exc)


class CodexSkillInstaller(SkillInstaller):
    """Vendor skills and expose them through AGENTS.md for Codex."""

    def install(self, workdir: Path) -> None:
        super().install(workdir)
        names = [Path(s).name for s in self._skills if Path(s).is_dir()]
        if not names:
            return
        section = "\n".join([
            "# Agent Skills (vendored)",
            "",
            "Before writing any figure/plot, read and APPLY these style guides:",
            "",
            *[f"- `.claude/skills/{n}/SKILL.md`" for n in names],
            "",
            "They govern chart-type choice and styling only -- never change the "
            "data, the analysis, or the required output format.",
            "",
        ])
        agents_md = workdir / "AGENTS.md"
        try:
            existing = (
                agents_md.read_text(encoding="utf-8").rstrip() + "\n\n"
                if agents_md.exists() else ""
            )
            agents_md.write_text(existing + section, encoding="utf-8")
        except OSError as exc:
            logger.warning("could not write AGENTS.md for codex skills: %s", exc)
