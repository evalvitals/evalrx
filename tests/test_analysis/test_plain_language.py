from __future__ import annotations

from evalvitals.analysis.plain_language import jargon_violation


def test_jargon_violation_flags_missing_text():
    assert jargon_violation("") == "missing"
    assert jargon_violation("   ") == "missing"


def test_jargon_violation_flags_verbatim_copy_of_technical_line():
    technical = "focus_share separates FAIL from PASS at AUC 0.82."
    assert jargon_violation(technical, technical) == "repeats the technical line verbatim"
    # case-insensitive
    assert jargon_violation(technical.upper(), technical) == "repeats the technical line verbatim"


def test_jargon_violation_flags_statistics_terms():
    reason = jargon_violation("This signal is collinear with the AUC.")
    assert reason is not None
    assert "jargon" in reason


def test_jargon_violation_flags_symbols():
    reason = jargon_violation("Focus share → more failures.")
    assert reason is not None
    assert "symbol" in reason


def test_jargon_violation_passes_clean_plain_english():
    assert jargon_violation(
        "When the model stares at one spot instead of scanning the image, "
        "it is much more likely to say an object is there when it isn't."
    ) is None
