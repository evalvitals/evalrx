"""Grading and loop signals of tools/generate_astra_edit_qwen35.py.

The script's generation path needs a GPU and a real checkpoint; what decides its
numbers on any machine is how an answer is read and compared, so that part is
tested here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture(scope="module")
def gen():
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(
            "generate_astra_edit_qwen35", TOOLS / "generate_astra_edit_qwen35.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


def test_answer_is_read_after_the_thinking_block(gen):
    text = "<think>maybe \\boxed{7}</think>\nThe answer is \\boxed{18}."
    assert gen.answer_text(text) == "The answer is \\boxed{18}."
    assert gen.is_correct(text, "18") is True
    # An answer boxed only inside an unfinished thinking block still counts as given:
    # there is no later text to prefer.
    assert gen.is_correct("<think> so \\boxed{18}", "18") is True


@pytest.mark.parametrize("text,expected", [
    ("\\boxed{18}", "18"),
    ("first \\boxed{1} then \\boxed{2}", "2"),
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("\\boxed {5}", "5"),
    ("\\boxed{12", None),
    ("\\boxed 5", None),
    ("no box", None),
])
def test_last_boxed(gen, text, expected):
    assert gen.last_boxed(text) == expected


@pytest.mark.parametrize("value,expected", [
    ("18", 18.0), ("-3.5", -3.5), ("$1,250$", 1250.0), ("\\$70000", 70000.0), ("45\\%", 45.0),
    ("18\\text{ dollars}", 18.0), ("\\frac{1}{2}", 0.5), ("\\dfrac{3}{4}", 0.75), ("3/4", 0.75),
    ("x=5", None), ("eighteen", None), ("", None), (None, None), ("1/0", None),
])
def test_to_number(gen, value, expected):
    assert gen.to_number(value) == expected


def test_is_correct_compares_numbers_not_strings(gen):
    assert gen.is_correct("\\boxed{18.0}", "18") is True
    assert gen.is_correct("\\boxed{\\$18}", "18") is True
    assert gen.is_correct("\\boxed{19}", "18") is False
    assert gen.is_correct("the answer is 18", "18") is False      # no box, no credit
    assert gen.is_correct("\\boxed{many}", "18") is False


def test_verbatim_loop_end(gen):
    block = "so the total is forty two dollars and we check again"
    looping = "We start. " + " ".join([block] * 6) + " so the tot"
    assert gen.verbatim_loop_end(looping) is True
    assert gen.verbatim_loop_end("a plain answer with no repetition at all, the end.") is False
    # Three repeats of a short block do not cover enough words.
    assert gen.verbatim_loop_end("x y z x y z x y z cut") is False
    assert gen.verbatim_loop_end("") is False
