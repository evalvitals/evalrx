"""Contract checks for the LLM band-probe harness.

The graders are where silent wrongness lives in this harness: a grader that
under-reports bins a usable dataset as 'floor' on a number that measured the
grader, not the model. ZebraLogic already shipped that bug once (a single-line
extractor feeding a full-grid grader), so each grader gets a test that would
have caught it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "examples" / "dataset_selection" / "llm_band_probe"


def _load(name: str):
    pytest.importorskip("requests")
    sys.path.insert(0, str(_HARNESS))
    try:
        spec = importlib.util.spec_from_file_location(name, _HARNESS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if str(_HARNESS) in sys.path:
            sys.path.remove(str(_HARNESS))


@pytest.fixture(scope="module")
def band():
    return _load("band_locate")


# ── spec table ────────────────────────────────────────────────────────────────
def test_spec_names_are_unique(band):
    names = [s.name for s in band.SPECS]
    assert len(names) == len(set(names))


def test_every_spec_has_a_chapter_and_a_budget(band):
    for spec in band.SPECS:
        assert spec.chapter and spec.dataset and spec.split
        assert spec.max_tokens >= 1024


def test_multiline_gold_specs_grade_the_raw_output(band):
    """extract_answer is structurally single-line, so a grader that parses a
    multi-line gold must be fed the whole generation."""
    for spec in band.SPECS:
        if spec.grader is band._grade_zebra:
            assert spec.grades_raw_output, f"{spec.name} would grade one line of a grid"


# ── graders ───────────────────────────────────────────────────────────────────
_ZEBRA_GOLD = (
    "House 1: Name=Arnold, Color=white\n"
    "House 2: Name=Peter, Color=yellow\n"
    "House 3: Name=Eric, Color=red"
)


def test_zebra_grader_accepts_a_correct_grid_in_either_layout(band):
    assert band._grade_zebra(f"reasoning\nAnswer:\n{_ZEBRA_GOLD}", _ZEBRA_GOLD)
    one_line = (
        "Answer: House 1: Name=Arnold, Color=white; House 2: Name=Peter, "
        "Color=yellow; House 3: Name=Eric, Color=red"
    )
    assert band._grade_zebra(one_line, _ZEBRA_GOLD)


def test_zebra_grader_binds_cells_to_houses(band):
    """Every value right but assigned to the wrong house is the whole puzzle."""
    swapped = (
        "House 1: Name=Peter, Color=yellow\n"
        "House 2: Name=Arnold, Color=white\n"
        "House 3: Name=Eric, Color=red"
    )
    assert not band._grade_zebra(swapped, _ZEBRA_GOLD)
    assert not band._grade_zebra("House 1: Name=Arnold, Color=white", _ZEBRA_GOLD)


def test_latex_grader_collapses_notation_but_not_meaning(band):
    same = [
        (r"Answer: $\dfrac{\pi}{3}$", r"\frac{\pi}{3}"),
        (r"so \boxed{\frac{\pi}{3}}", r"\frac{\pi}{3}"),
        (r"Answer: 2\sqrt {2}", r"2 \sqrt{2}"),
        ("Answer: 18", "18"),
        (r"Answer: \leftarrow x", r"\leftarrow x"),  # \left must not eat \leftarrow
    ]
    for pred, gold in same:
        assert band._grade_latex(pred, gold), (pred, gold)
    assert not band._grade_latex(r"Answer: \frac{\pi}{6}", r"\frac{\pi}{3}")
    # documented miss: this is a surface-form matcher, not a CAS
    assert not band._grade_latex("Answer: 0.5", r"\frac{1}{2}")


def test_alias_grader_accepts_any_listed_surface_form(band):
    gold = ["Alfredo Stroessner's Paraguay", "Alfredo Stroessner"]
    assert band._grade_aliases("Answer: Alfredo Stroessner", gold)
    assert not band._grade_aliases("Answer: Someone Else", gold)


# ── band classification ───────────────────────────────────────────────────────
def test_band_classification_uses_the_interval_not_the_point(band):
    lo, hi = band.wilson(30, 60)
    assert band.band_of(0.5, lo, hi) == "USABLE"
    assert band.band_of(58 / 60, *band.wilson(58, 60)) == "saturated"
    assert band.band_of(2 / 60, *band.wilson(2, 60)) == "floor"


def test_truncation_short_circuits_the_band(band):
    """A tag-less majority means the budget was measured, not the model."""
    lo, hi = band.wilson(30, 60)
    assert band.band_of(0.5, lo, hi, 0.0) == "USABLE"
    assert band.band_of(0.5, lo, hi, 0.5) == "budget_limited"


def test_wilson_interval_brackets_the_point_estimate(band):
    for k, n in [(0, 50), (1, 50), (25, 50), (49, 50), (50, 50)]:
        lo, hi = band.wilson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


# ── adapters (offline: shaped like the real rows, no network) ─────────────────
def test_polymath_adapter_strips_the_latex_delimiters(band):
    row = {"id": "medium-en-0", "question": "q", "answer": r"$\frac{\pi}{3}$"}
    assert band._adapter_polymath(row) == ("q", r"\frac{\pi}{3}")
    assert band._adapter_polymath({"question": "", "answer": "1"}) is None


def test_mc_adapter_drops_rows_with_more_options_than_letters(band):
    adapter = band._adapter_mc(("question",), "options", "answer_letter")
    row = {"question": "q", "options": [f"o{i}" for i in range(10)], "answer_letter": "J"}
    prompt, gold = adapter(row)
    assert gold == "J" and "J. o9" in prompt
    too_many = dict(row, options=[f"o{i}" for i in range(11)])
    assert adapter(too_many) is None


def test_musr_adapter_parses_string_repr_choices(band):
    row = {
        "narrative": "n", "question": "Who?",
        "choices": "['Mackenzie', 'Ana']", "answer_choice": "Ana",
    }
    prompt, gold = band._adapter_musr(row)
    assert gold == "B" and "A. Mackenzie" in prompt
