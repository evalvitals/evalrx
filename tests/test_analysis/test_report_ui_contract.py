from __future__ import annotations

from evalrx.analysis.prompts.explorer import _ANALYSIS_CONTRACT
from evalrx.reporting.stages import STAGE_SPECS, stage_specs_as_dicts
from evalrx.viz.prompts import DASHBOARD_STORYBOARD_SYSTEM_PROMPT


def test_stage_specs_define_m1_to_m5_dashboard_roles():
    ids = [s.id for s in STAGE_SPECS]
    assert ids == ["M1", "M2", "M3", "M4", "M5"]
    rows = stage_specs_as_dicts()
    assert rows[0]["dashboard_role"].startswith("Problem Setting")
    assert "Analysis" in rows[1]["dashboard_role"]
    assert "Hypotheses" in rows[2]["dashboard_role"]


def test_dashboard_storyboard_prompt_is_three_panel_and_stage_aware():
    prompt = DASHBOARD_STORYBOARD_SYSTEM_PROMPT
    assert "Problem Setting" in prompt
    assert "Analysis" in prompt
    assert "Hypotheses & Artifacts" in prompt
    assert "M1" in prompt and "M2" in prompt and "M3-M5" in prompt
    assert "display_name" in prompt


def test_explorer_analysis_contract_requires_reader_friendly_m2_language():
    prompt = _ANALYSIS_CONTRACT

    assert "user who may understand the evaluation problem" in prompt
    assert "may NOT know" in prompt
    assert "machine-learning or statistics terms" in prompt
    assert "unit of analysis" in prompt
    assert "Use raw counts alongside percentages" in prompt
    assert "what the outcome column means in practical terms" in prompt
    assert "what remains uncertain" in prompt
    assert "Never imply that an observed relationship proves a cause" in prompt
