"""Bundled (package-vendored) Agent Skills travel with the repo and are applied
by default on every skill-capable coding backend (claude/agy/codex).

This is what makes skills like eval-chart-style and nature-figure available to
anyone who clones the repo (or pip-installs it) — no per-machine ~/.claude
setup required.
"""

from __future__ import annotations

from pathlib import Path

from evalvitals.agent_assets.skills import SKILL_BACKENDS, bundled_skill_paths
from evalvitals.analysis import api as explore_api
from evalvitals.analysis import explore_run


def test_bundled_skill_set():
    paths = bundled_skill_paths()
    names = {Path(p).name for p in paths}
    assert "nature-figure" in names
    assert "evalvitals-report-ui" in names
    # The eval_viz_theme chart-type policy, codified as a skill for agents.
    assert "eval-chart-style" in names
    # The statistical-method protocol consulted BEFORE analysis code is written.
    assert "outcome-driver-analysis" in names
    oda = next(Path(p) for p in paths if Path(p).name == "outcome-driver-analysis")
    assert (oda / "SKILL.md").is_file()
    assert (oda / "references" / "model_selection.md").is_file()
    assert not (oda / ".DS_Store").exists()  # vendored clean
    nf = next(Path(p) for p in paths if Path(p).name == "nature-figure")
    assert (nf / "SKILL.md").is_file()
    assert (nf / "LICENSE").is_file()  # Apache-2.0 attribution preserved
    ecs = next(Path(p) for p in paths if Path(p).name == "eval-chart-style")
    body = (ecs / "SKILL.md").read_text(encoding="utf-8")
    assert "never" in body.lower() and "bar" in body.lower()  # the anti-mean-bar policy
    # Palette locked to eval_viz_theme's (the dataviz-validated light values) so
    # agent PNGs and host plotly charts stay on ONE palette:
    assert "#d03b3b" in body           # FAIL = status critical
    assert "#0ca30c" in body           # PASS/supported = status good
    assert "#fab219" in body           # inconclusive = status warning, not grey
    assert "#C0413B" not in body       # pre-redesign palette fully retired
    assert "#5B7A99" not in body
    # Non-outcome dimensions get real color range, not endless red/blue/grey:
    assert "#2a78d6" in body           # categorical slot 1 / accent present
    assert "ramp" in body.lower()      # ordered-dimension single-hue ramp rule
    assert "Nothing else" not in body  # the old two-colors-only lockdown is gone
    # The single chart authority: absorbed the stats-skill's result figures and
    # carries the arbitration clause other skills defer to.
    assert "ROC curve" in body and "calibration" in body
    assert "This policy wins" in body

    # outcome-driver-analysis defers chart HOW to the chart-style skill and no
    # longer references skills that are not bundled.
    oda_body = (oda / "SKILL.md").read_text(encoding="utf-8")
    assert "eval-chart-style" in oda_body          # explicit deference target
    assert "standalone fallback" in oda_body       # keeps solo usability
    assert "clear-technical-writing" not in oda_body  # dangling ref removed

    # nature-figure's canonical palette constants are synced to the same
    # dataviz-validated values (host's load_nature_style parses api.md).
    nf_api = (nf / "references" / "api.md").read_text(encoding="utf-8")
    assert '"blue_main":      "#2a78d6"' in nf_api
    assert "#0F4D92" not in nf_api                 # pre-sync palette retired


def test_skill_backends_cover_claude_agy_codex():
    assert {"claude_code", "antigravity", "codex"} <= SKILL_BACKENDS


def test_explore_applies_bundled_skills_on_claude_by_default(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)
    explore_run.run_explore(tmp_path, coder_provider="claude_code", out=tmp_path / "o")

    cfg = captured["cli_config"]
    assert any(Path(s).name == "nature-figure" for s in cfg.skills)
    assert any(Path(s).name == "eval-chart-style" for s in cfg.skills)
    assert cfg.allow_skills is True  # implied by bundled skills


def test_explore_applies_bundled_skills_on_codex_by_default(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)
    # codex reaches the same vendored SKILL.md files through AGENTS.md.
    explore_run.run_explore(tmp_path, coder_provider="codex", out=tmp_path / "o")
    cfg = captured["cli_config"]
    assert any(Path(s).name == "eval-chart-style" for s in cfg.skills)


def test_no_skills_flag_disables_bundled(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)
    explore_run.run_explore(
        tmp_path, coder_provider="claude_code", out=tmp_path / "o", use_bundled_skills=False
    )
    assert captured["cli_config"].skills == ()


def test_non_skill_backend_does_not_vendor_skills(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_api, "ExploratoryAnalysisAgent", _FakeAgent)
    # opencode has no skill discovery mechanism; we don't vendor skills there.
    explore_run.run_explore(tmp_path, coder_provider="opencode", out=tmp_path / "o")
    assert captured["cli_config"].skills == ()


def test_explorer_agent_defaults_bundled_skills():
    """Any flow that hands the explorer a bare CliAgentConfig (e.g. the fused
    pipeline via an example's build_codegen) gets the bundled skills without
    caller wiring; opt out via use_bundled_skills=False."""
    from evalvitals.analysis.explorer import ExploratoryAnalysisAgent
    from evalvitals.eval_agent.cli_agent import CliAgentConfig

    agent = ExploratoryAnalysisAgent(cli_config=CliAgentConfig(provider="claude_code"))
    names = {Path(s).name for s in agent._cli_config.skills}
    assert {"eval-chart-style", "nature-figure"} <= names
    assert agent._cli_config.allow_skills is True

    off = ExploratoryAnalysisAgent(
        cli_config=CliAgentConfig(provider="claude_code"), use_bundled_skills=False
    )
    assert off._cli_config.skills == ()

    # explicit skills are respected, not overridden
    custom = ExploratoryAnalysisAgent(
        cli_config=CliAgentConfig(provider="claude_code", skills=("/my/skill",))
    )
    assert custom._cli_config.skills == ("/my/skill",)

    # non-skill backends stay untouched
    oc = ExploratoryAnalysisAgent(cli_config=CliAgentConfig(provider="opencode"))
    assert oc._cli_config.skills == ()
