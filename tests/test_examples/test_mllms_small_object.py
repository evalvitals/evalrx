from __future__ import annotations

import importlib.util
from pathlib import Path

from evalvitals.core.case import FailureCase, Inputs, Label


def _load_example():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "m1_m4"
        / "mllms_small_object"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("mllms_small_object_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_textvqa_scorer_uses_main_answer_not_rejected_alternative():
    ex = _load_example()
    case = FailureCase(
        id="36304",
        inputs=Inputs(prompt="what number is after the word pool?"),
        expected={"any_of": ["318", "22"]},
    )
    observed = (
        'The number after the word "POOL" is **388**.\n\n'
        "The number **22** is painted on the side of the car, but that is separate."
    )

    assert ex._score_case(case, observed) == Label.FAIL


def test_textvqa_scorer_accepts_direct_bold_answer():
    ex = _load_example()
    case = FailureCase(
        id="36381",
        inputs=Inputs(prompt="what is the alcohol percentage on the bottle?"),
        expected={"any_of": ["4.8%", "11:37", "devassa", "4.0", "4.5", "4%"]},
    )

    assert ex._score_case(case, "The alcohol percentage is **4.5% vol.**") == Label.PASS
