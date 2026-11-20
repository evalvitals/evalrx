from __future__ import annotations

from evalrx.analysis.prompts.explorer import _ANALYSIS_CONTRACT


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
