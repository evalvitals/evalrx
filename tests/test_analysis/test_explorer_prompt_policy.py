"""The explorer prompt and the eval-chart-style skill must agree with the host
renderer's chart-form policy (host picks composition / dot+CI / line from the
CSV shape; bars only for counts or many categories)."""

from __future__ import annotations

from evalrx.agent_assets.skills import BUNDLED_SKILLS_DIR
from evalrx.analysis.explorer import _framing_block
from evalrx.analysis.prompts.explorer import _ANALYSIS_CONTRACT, _GENERIC_FRAMING


def test_prompt_asks_for_n_and_numerators_and_forbids_two_group_mean_specs():
    assert "group->value plus n" in _ANALYSIS_CONTRACT
    assert "dot + 95% CI" in _ANALYSIS_CONTRACT
    assert "composition strip" in _ANALYSIS_CONTRACT
    assert "do NOT also" in _ANALYSIS_CONTRACT and "two-group" in _ANALYSIS_CONTRACT
    assert "BINNED fail-rate LINE" in _ANALYSIS_CONTRACT
    assert "group -> fail_rate, n, n_fail" in _GENERIC_FRAMING


def test_binary_battery_routes_numeric_signals_to_the_binned_line():
    text = _framing_block({"kind": "binary", "column": "label", "unique": 2})
    assert "n per bin" in text
    assert "do not emit a two-group" in text
    assert "n_fail per group" in text and "dot + 95% CI" in text


def test_skill_scope_note_applies_the_policy_to_host_specs_too():
    skill = (BUNDLED_SKILLS_DIR / "eval-chart-style" / "SKILL.md").read_text(encoding="utf-8")
    scope = skill.split("## Scope note for EvalRX sandboxes", 1)[1]
    assert "§0 applies to them too" in scope
    assert "dot + 95% CI" in scope and "composition strip" in scope
    assert "never gets an invented interval" in scope
    assert "binned fail-rate line" in scope
