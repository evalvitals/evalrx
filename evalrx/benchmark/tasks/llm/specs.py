"""The nine text-LLM task slices: where each comes from, how a row becomes a
prompt, and how an answer is graded.

These are the slices :mod:`evalrx.benchmark.tasks.llm` runs. The band survey in
``examples/dataset_selection/llm_band_probe`` measured them (and ~40 other
candidates it keeps for itself) and reuses these objects, so a slice is defined
once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from evalrx.analyzers.reasoning._text import answer_equal, extract_answer

MC_LETTERS = "ABCDEFGHIJ"


@dataclass
class Spec:
    """One candidate dataset slice to measure."""

    name: str
    chapter: str
    dataset: str
    config: str = "default"
    split: str = "test"
    question_field: str = "question"
    answer_field: str = "answer"
    # row -> (prompt, gold) ; None drops the row
    adapter: Optional[Callable[[dict], Optional[tuple]]] = None
    instruction: str = (
        "Solve the problem. Put the final answer on its own last line as "
        "'Answer: <answer>'."
    )
    # Gold shapes differ enough that a single grader would silently under-report
    # (list-wrapped LaTeX, answer aliases, full grids) -> allow an override.
    grader: Optional[Callable[[str, Any], bool]] = None
    #: Feed the grader the RAW generation instead of the extracted answer.
    #: extract_answer is structurally single-line (the tag regex does not span
    #: newlines), so a multi-line gold — a puzzle grid — graded on the extracted
    #: span can only ever see its first row and scores 0 for every model.
    grades_raw_output: bool = False
    max_tokens: int = 8192
    note: str = ""
    meta: dict = field(default_factory=dict)
    #: ``row -> keep?``, applied BEFORE the adapter.
    #:
    #: Several of these datasets carry their own difficulty ladder as a column
    #: (ZebraLogic ``size``, Enigmata ``task_name``). Measuring the pooled split
    #: averages a saturated rung with a floored one and reports a number that
    #: describes NEITHER — a pooled 0.22 on ZebraLogic is not "22% hard", it is
    #: "some rungs are free and some are impossible". The filter exists to scan
    #: the ladder rung by rung instead.
    row_filter: Optional[Callable[[dict], bool]] = None
    #: Over-fetch factor, counted in rows that PASS ``row_filter``.
    #: A filter keeping 1/25 of the split needs far more than the default, or
    #: the sweep silently reports a band on four items.
    fetch_multiplier: int = 3
    #: Windows spread across the split. A filtered spec needs the whole split
    #: swept, not a sample of it, because its rows may be contiguous.
    n_windows: int = 12
    #: datasets-server ``where`` clause, e.g. ``"discipline"='Law'``.
    #: Set it to address a NAMED subdivision of a split that is too large to
    #: use whole. Prefer this over ``row_filter`` whenever the predicate is
    #: expressible server-side: the filter runs before the download, so the
    #: sample is drawn from the slice instead of sieved out of the split.
    where: Optional[str] = None
    #: Append ``instruction`` to the prompt.
    #:
    #: Off for datasets that ship their OWN response-format section. Enigmata
    #: tells the model to print a fenced grid; appending "put the final answer
    #: on its own last line as 'Answer: <answer>'" on top of that is a second,
    #: contradictory format order, and a model that obeys either one is graded
    #: against the other. Adding an instruction is not free.
    append_instruction: bool = True


def field_text(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def mc_prompt(question: str, options: list[str]) -> str:
    lines = [question, ""]
    lines += [f"{MC_LETTERS[i]}. {opt}" for i, opt in enumerate(options[: len(MC_LETTERS)])]
    return "\n".join(lines)


# -- per-dataset adapters (field names verified against /first-rows) -----
def plain_adapter(q: str, a: str) -> Callable[[dict], Optional[tuple]]:
    def _fn(row: dict) -> Optional[tuple]:
        question, gold = field_text(row, q), field_text(row, a)
        return (question, gold) if question and gold else None

    return _fn


def cruxeval_adapter(row: dict) -> Optional[tuple]:
    code, inp, out = field_text(row, "code"), field_text(row, "input"), field_text(row, "output")
    if not code or not out:
        return None
    prompt = (
        "Predict the output of this Python function for the given input.\n\n"
        f"{code}\n\nInput: {inp}\n\n"
        "Give only the output value on the last line as 'Answer: <value>'."
    )
    return prompt, out.strip()


def mc_adapter(question_keys: tuple, options_key: str, answer_key: str):
    def _fn(row: dict) -> Optional[tuple]:
        question = field_text(row, *question_keys)
        options = row.get(options_key)
        gold = field_text(row, answer_key)
        if not question or not isinstance(options, (list, tuple)) or not gold:
            return None
        if len(options) > len(MC_LETTERS):
            return None  # would need a letter the prompt cannot label
        if gold in MC_LETTERS:
            letter = gold
        elif gold in options:
            letter = MC_LETTERS[list(options).index(gold)]
        else:
            return None
        return mc_prompt(question, list(options)), letter

    return _fn


def bbh_adapter(row: dict) -> Optional[tuple]:
    return plain_adapter("input", "target")(row)


#: Spacing/sizing commands and the two \frac aliases carry no meaning, so two
#: answers that differ only in them are the same answer.
_LATEX_NOISE = re.compile(
    # the (?![a-zA-Z]) guard keeps \\left from eating \\leftarrow
    r"\\(?:left|right|qquad|quad|displaystyle|text|mathrm)(?![a-zA-Z])|\\[!,;:]"
)


def _latex_key(value: Any) -> str:
    """Normalised surface form of a LaTeX answer.

    Deliberately NOT a CAS: it collapses notation that never changes meaning
    (\\dfrac vs \\frac, \\left(, stray spaces) and nothing else. Two answers that
    are algebraically equal but written differently (1/2 vs 0.5, \\sqrt2/2 vs
    1/\\sqrt2) still miss — which is why the specs using it say so.
    """
    text = str(value or "").strip().strip("$").strip()
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = _LATEX_NOISE.sub("", text)
    text = re.sub(r"\\cdot|\\times", "*", text)
    text = re.sub(r"[\s{}]", "", text)
    return text.lower()


ANSWER_MARK = re.compile(r"answer\s*:", re.IGNORECASE)


#: `4.5 \times 10^{33}`, `4.5 \cdot 10^33`, `4.5 x 10^33` -> `4.5e33`. Requiring a
#: literal `10^` after the operator is what keeps the bare `x` alternative from
#: eating a variable; nothing else in these answer sets looks like this.
_SCI_LATEX = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\\ast|[x*])\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?"
)


def _sci_float(value: Any) -> Optional[float]:
    """Parse a lone scalar written in program OR LaTeX scientific notation.

    Returns None unless the whole string is that one number, so a expression that
    merely contains a power of ten never reaches the tolerant comparison below.
    """
    text = str(value or "").strip().strip("$").strip()
    text = _LATEX_NOISE.sub("", text)
    text = _SCI_LATEX.sub(lambda m: f"{m.group(1)}e{int(m.group(2))}", text)
    text = re.sub(r"[\s{},]", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def _has_exponent(value: Any) -> bool:
    text = str(value or "")
    return bool(_SCI_LATEX.search(text) or re.search(r"\de-?\d", text, re.I))


def grade_latex(prediction: Any, gold: Any) -> bool:
    """Numeric equality first, then normalised LaTeX surface form.

    Scientific notation gets one extra pass.  Minerva-style physics sets write
    their golds in PROGRAM form (`4.5e33`) while a model writing mathematics
    writes `4.5 \\times 10^{33}`, so surface comparison marks correct answers
    wrong -- 23.5% of minervamath's golds look like this.  Collapsing the two
    notations is in the same spirit as _latex_key: it never changes meaning.

    The 1% relative tolerance is NOT applied to ordinary numbers.  These golds
    carry 2-3 significant figures (`4.5e33`, `8.7e8`), so a correct derivation
    rounds differently in the last digit; a competition answer of `100` gets no
    such licence, because there `99.5` is simply wrong.
    """
    if answer_equal(prediction, gold):
        return True
    # a newline is only a proxy for "this is a generation, not a bare answer";
    # a single-line "Answer: 4.5e33" needs the label stripped just as much
    text = str(prediction)
    pred_raw = (
        extract_answer(prediction)
        if "\n" in text or ANSWER_MARK.search(text) else prediction
    )
    if _has_exponent(pred_raw) or _has_exponent(gold):
        p, g = _sci_float(pred_raw), _sci_float(gold)
        if p is not None and g is not None:
            return math.isclose(p, g, rel_tol=1e-2, abs_tol=0.0)
    pred = _latex_key(pred_raw)
    return bool(pred) and pred == _latex_key(gold)


#: Keyed by task name, in the order :data:`evalrx.benchmark.tasks.llm.DATASETS` lists them.
SPECS: dict[str, Spec] = {
    "cruxeval_output": Spec(
        "cruxeval_output", "ch2-code", "cruxeval-org/cruxeval", split="test",
        adapter=cruxeval_adapter, max_tokens=20480, note="exact string, no sandbox",
    ),
    "bbh_causal_judgement": Spec(
        "bbh_causal_judgement", "ch4-basic", "lukaemon/bbh",
        config="causal_judgement", split="test", adapter=bbh_adapter,
        max_tokens=8192, note="187 items; Yes/No, no grader ambiguity",
    ),
    "supergpqa_economics": Spec(
        "supergpqa_economics", "ch4-basic", "m-a-p/SuperGPQA", split="train",
        where="\"discipline\"='Economics'",
        adapter=mc_adapter(("question",), "options", "answer_letter"),
        max_tokens=20480,
        note="873 items; largest whole discipline <1000, 65% `middle`",
    ),
    "bbh_word_sorting": Spec(
        "bbh_word_sorting", "ch4-basic", "lukaemon/bbh", config="word_sorting",
        split="test", adapter=bbh_adapter, max_tokens=4096,
    ),
    "bbh_tracking7": Spec(
        "bbh_tracking7", "ch4-basic", "lukaemon/bbh",
        config="tracking_shuffled_objects_seven_objects", split="test",
        adapter=bbh_adapter, max_tokens=4096,
    ),
    "bamboogle": Spec(
        "bamboogle", "ch4-basic", "chiayewken/bamboogle", split="test",
        adapter=plain_adapter("Question", "Answer"), max_tokens=20480,
    ),
    "minervamath": Spec(
        "minervamath", "ch1-math", "math-ai/minervamath", split="test",
        adapter=plain_adapter("question", "answer"), grader=grade_latex,
        max_tokens=40960,
    ),
    "supergpqa_law": Spec(
        "supergpqa_law", "ch4-basic", "m-a-p/SuperGPQA", split="train",
        where="\"discipline\"='Law'",
        adapter=mc_adapter(("question",), "options", "answer_letter"),
        max_tokens=20480,
        note="656 items; highest hard share (9%) among whole disciplines <1000",
    ),
    "supergpqa_medicine_hard": Spec(
        "supergpqa_medicine_hard", "ch4-basic", "m-a-p/SuperGPQA", split="train",
        where="\"discipline\"='Medicine' AND \"difficulty\"='hard'",
        adapter=mc_adapter(("question",), "options", "answer_letter"),
        max_tokens=20480,
        note="217 items; the only <1000 slice that is entirely `hard`",
    ),
}
