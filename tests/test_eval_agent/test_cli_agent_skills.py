"""Agent-Skill wiring for CLI coding backends.

Skills (e.g. a nature-figure plot-styling skill) are vendored into the sandbox
so an ``--add-dir`` Claude Code / agy run auto-discovers them, and the ``Skill``
tool is added to the allowlist so the agent may invoke them. Skills style the
agent-authored ``figures/*.png`` only — they never touch the host-rendered,
deterministic chart specs.
"""

from __future__ import annotations

from pathlib import Path

from evalvitals.analysis.explorer import _skills_hint
from evalvitals.eval_agent.cli_agent import (
    ClaudeCodeAgent,
    CliAgentConfig,
    create_cli_agent,
)


def _make_skill(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "SKILL.md").write_text(f"# {name}\nStyle figures nicely.\n", encoding="utf-8")
    (d / "style.mplstyle").write_text("axes.grid: True\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_skills_enabled_implied_by_skill_dirs():
    assert CliAgentConfig().skills_enabled is False
    assert CliAgentConfig(allow_skills=True).skills_enabled is True
    assert CliAgentConfig(skills=("/some/skill",)).skills_enabled is True


# ---------------------------------------------------------------------------
# vendoring: skills copied into <workdir>/.claude/skills/<name>/
# ---------------------------------------------------------------------------

def test_install_skills_vendors_into_workdir(tmp_path):
    skill = _make_skill(tmp_path, "nature-figure")
    workdir = tmp_path / "wd"
    agent = ClaudeCodeAgent(binary_path="claude", skills=[str(skill)])
    agent._install_skills(workdir)

    dest = workdir / ".claude" / "skills" / "nature-figure"
    assert (dest / "SKILL.md").exists()
    assert (dest / "style.mplstyle").exists()


def test_install_skills_missing_dir_is_skipped_not_fatal(tmp_path):
    workdir = tmp_path / "wd"
    agent = ClaudeCodeAgent(binary_path="claude", skills=["/no/such/skill"])
    agent._install_skills(workdir)  # must not raise
    assert not (workdir / ".claude" / "skills" / "skill").exists()


def test_no_skills_is_noop(tmp_path):
    workdir = tmp_path / "wd"
    ClaudeCodeAgent(binary_path="claude")._install_skills(workdir)
    assert not (workdir / ".claude").exists()


# ---------------------------------------------------------------------------
# allowlist: the Skill tool is added only when skills are enabled
# ---------------------------------------------------------------------------

def test_claude_cmd_adds_skill_tool_only_when_enabled(tmp_path):
    wd = tmp_path / "wd"

    plain = ClaudeCodeAgent(binary_path="claude")._build_cmd("hi", wd)
    allowed_plain = plain[plain.index("--allowed-tools") + 1]
    assert "Skill" not in allowed_plain

    withskill = ClaudeCodeAgent(binary_path="claude", allow_skills=True)._build_cmd("hi", wd)
    allowed = withskill[withskill.index("--allowed-tools") + 1]
    assert "Skill" in allowed
    assert "Bash" in allowed and "Write" in allowed  # base tools preserved


def test_create_cli_agent_threads_skills():
    cfg = CliAgentConfig(provider="claude_code", binary_path="claude",
                         skills=("/a/nature-figure",), allow_skills=True)
    agent = create_cli_agent(cfg)
    assert agent._skills == ["/a/nature-figure"]
    assert agent._allow_skills is True


# ---------------------------------------------------------------------------
# prompt hint: the explorer tells the agent it may use skills for figures
# ---------------------------------------------------------------------------

def test_skills_hint_empty_without_skills():
    assert _skills_hint(None) == ""
    assert _skills_hint(CliAgentConfig(provider="claude_code")) == ""


def test_skills_hint_names_the_skill_and_scopes_to_figures():
    cfg = CliAgentConfig(provider="claude_code", skills=("/x/nature-figure",))
    hint = _skills_hint(cfg)
    assert "/nature-figure" in hint
    assert "figures/" in hint
    # must NOT invite changing the data/analysis/result
    assert "styling only" in hint


def test_skills_hint_is_default_on_not_optional():
    cfg = CliAgentConfig(
        provider="claude_code", skills=("/x/eval-chart-style", "/x/nature-figure")
    )
    hint = _skills_hint(cfg)
    assert "BY DEFAULT" in hint and "BEFORE plotting" in hint
    assert "MAY invoke" not in hint  # the old opt-in wording agents ignored
    # division of labor is stated so both skills get applied
    assert "chart-TYPE" in hint and "publication-grade" in hint


def test_skills_hint_codex_points_at_vendored_files():
    cfg = CliAgentConfig(provider="codex", skills=("/x/eval-chart-style",))
    hint = _skills_hint(cfg)
    # codex has no Skill tool — it must be told to READ the vendored guide
    assert ".claude/skills/eval-chart-style/SKILL.md" in hint
    assert "Skill tool and follow" not in hint


def test_skills_hint_stages_analysis_before_figures():
    """outcome-driver-analysis is a methodology skill: the hint must direct it
    at the pre-code analysis stage, keep the descriptive contract, and still
    carry the figure-styling stage for the styling skills."""
    cfg = CliAgentConfig(
        provider="claude_code",
        skills=("/x/outcome-driver-analysis", "/x/eval-chart-style", "/x/nature-figure"),
    )
    hint = _skills_hint(cfg)
    # staged: analysis method first, figure styling after
    assert "ANALYSIS METHOD" in hint and "FIGURE STYLING" in hint
    assert hint.index("ANALYSIS METHOD") < hint.index("FIGURE STYLING")
    assert "BEFORE writing any analysis code" in hint
    # descriptive-contract + non-interactive + layout guards
    assert "must NOT be phrased as significance" in hint
    assert "never stop to ask" in hint or "never\nstop to ask" in hint
    assert "not its file layout" in hint
    # few-clusters guidance (3 checkpoints -> fixed effect, not random)
    assert "fixed effect" in hint
    # the styling stage names only the styling skills
    styling = hint[hint.index("FIGURE STYLING"):]
    assert "/eval-chart-style" in styling and "/nature-figure" in styling
    # arbitration: chart-type/palette conflicts resolve to eval-chart-style
    assert "take precedence" in styling
    assert "/outcome-driver-analysis" not in styling


def test_skills_hint_without_analysis_skill_has_no_analysis_stage():
    cfg = CliAgentConfig(provider="claude_code", skills=("/x/nature-figure",))
    hint = _skills_hint(cfg)
    assert "ANALYSIS METHOD" not in hint
    assert "FIGURE STYLING" in hint


# ---------------------------------------------------------------------------
# codex delivery: vendored skills surfaced through the workdir's AGENTS.md
# ---------------------------------------------------------------------------

def test_codex_install_skills_writes_agents_md(tmp_path):
    from evalvitals.eval_agent.cli_agent import CodexAgent

    skill = _make_skill(tmp_path, "eval-chart-style")
    workdir = tmp_path / "wd"
    CodexAgent(binary_path="codex", skills=[str(skill)])._install_skills(workdir)

    # vendored like every other backend
    assert (workdir / ".claude" / "skills" / "eval-chart-style" / "SKILL.md").exists()
    # and surfaced where codex actually looks
    agents = (workdir / "AGENTS.md").read_text(encoding="utf-8")
    assert ".claude/skills/eval-chart-style/SKILL.md" in agents
    assert "styling only" in agents or "styling" in agents


def test_codex_install_skills_appends_to_existing_agents_md(tmp_path):
    from evalvitals.eval_agent.cli_agent import CodexAgent

    skill = _make_skill(tmp_path, "nature-figure")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "AGENTS.md").write_text("# Existing project notes\n", encoding="utf-8")
    CodexAgent(binary_path="codex", skills=[str(skill)])._install_skills(workdir)

    agents = (workdir / "AGENTS.md").read_text(encoding="utf-8")
    assert "Existing project notes" in agents  # not clobbered
    assert ".claude/skills/nature-figure/SKILL.md" in agents


def test_codex_install_skills_noop_without_skills(tmp_path):
    from evalvitals.eval_agent.cli_agent import CodexAgent

    workdir = tmp_path / "wd"
    CodexAgent(binary_path="codex")._install_skills(workdir)
    assert not (workdir / "AGENTS.md").exists()
