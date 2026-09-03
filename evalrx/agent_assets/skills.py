"""Bundled Agent Skills shipped with the package.

Bundled skills (eval-chart-style, nature-figure, evalrx-report-ui) are
applied BY DEFAULT on every skill-capable coding backend: claude/agy discover
them via the vendored ``<workdir>/.claude/skills/`` + the ``Skill`` tool, and
codex is pointed at the same vendored ``SKILL.md`` files through the workdir's
``AGENTS.md`` (it has no Skill tool). Pass ``--no-skills`` / empty ``skills``
to opt out.
"""

from __future__ import annotations

from pathlib import Path

BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
SKILL_BACKENDS = {"claude_code", "antigravity", "codex"}


def bundled_skill_paths() -> list[str]:
    """Absolute paths of bundled Agent-Skill directories containing ``SKILL.md``."""
    if not BUNDLED_SKILLS_DIR.is_dir():
        return []
    return [
        str(path)
        for path in sorted(BUNDLED_SKILLS_DIR.iterdir())
        if (path / "SKILL.md").is_file()
    ]
