from __future__ import annotations

from pathlib import Path

from evalrx.agent_runtime.cli_types import CliAgentConfig
from evalrx.agent_runtime.skills.prompt_policy import fences_hint, skills_hint
from evalrx.agent_runtime.skills.resolver import resolve_skill_paths
from evalrx.eval_agent.cli_skills import CodexSkillInstaller, SkillInstaller


def _make_skill(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.mkdir()
    (path / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return str(path)


def test_cli_skills_facade_reexports_installers():
    from evalrx.agent_runtime.skills.installer import (
        CodexSkillInstaller as RealCodexSkillInstaller,
    )
    from evalrx.agent_runtime.skills.installer import (
        SkillInstaller as RealSkillInstaller,
    )

    assert SkillInstaller is RealSkillInstaller
    assert CodexSkillInstaller is RealCodexSkillInstaller


def test_resolve_skill_paths_dedupes_bundled_and_explicit(tmp_path):
    explicit = _make_skill(tmp_path, "custom-skill")
    paths = resolve_skill_paths(
        provider="claude_code",
        explicit=(explicit,),
        use_bundled=True,
    )

    assert paths[-1] == explicit
    assert len(paths) == len(set(paths))
    assert any(Path(path).name == "nature-figure" for path in paths)


def test_resolve_skill_paths_skips_bundled_for_non_skill_backend(tmp_path):
    explicit = _make_skill(tmp_path, "custom-skill")
    assert resolve_skill_paths(
        provider="opencode",
        explicit=(explicit,),
        use_bundled=True,
    ) == (explicit,)


def test_prompt_policy_preserves_explorer_hint_contract():
    cfg = CliAgentConfig(
        provider="codex",
        skills=("/x/outcome-driver-analysis", "/x/eval-chart-style"),
    )
    hint = skills_hint(cfg)

    assert fences_hint(cfg) == ", written to a file named analysis.py"
    assert "ANALYSIS METHOD" in hint
    assert "FIGURE STYLING" in hint
    assert ".claude/skills/eval-chart-style/SKILL.md" in hint
    assert "must NOT be phrased as significance" in hint
