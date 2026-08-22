"""Answer parsers and graders shared by the task kinds.

``exact_or_numeric`` is the ChartQA/Spatial457 rule from
``examples/m1_m4/vlm_benchmark_common.py`` (label-blind final-answer
extraction, normalisation, relaxed numeric tolerance);
``multiple_choice_letter`` and ``yes_no`` are the MMAU / AudioCaps parsers from
the ``m1_m4`` audio examples. ``llm_graded`` delegates to the dataset's own
grader in ``examples/dataset_selection`` (see ``tasks/llm.py``).
"""

from __future__ import annotations

import re
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
_ANSWER_TAG = re.compile(r"(?:final\s+answer|answer)\s*(?:[:=]|\bis\b)\s*(.+)", re.IGNORECASE)


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("–", "-")
    text = _ARTICLES.sub(" ", text)
    text = re.sub(r"[^\w.%+\-]+", " ", text)
    return " ".join(text.split())


def _number(value: Any) -> float | None:
    match = _NUMBER.fullmatch(normalize_answer(value).replace(" ", ""))
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        # ChartQA treats a trailing percent sign as formatting: 6.8 == 6.8%.
        return float(raw.rstrip("%"))
    except ValueError:
        return None


def extract_final_answer(raw: str) -> str:
    """Label-blind final-answer extraction: the last ``Answer:``-tagged span,
    else the last non-empty line. Candidates may reason before answering."""
    raw = str(raw or "")
    tagged = [m.group(1).splitlines()[0].strip() for m in _ANSWER_TAG.finditer(raw)]
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return tagged[-1] if tagged else (lines[-1] if lines else raw)


def answer_matches(observed: str, expected: Any, *, numeric_tolerance: float) -> bool:
    golds = expected if isinstance(expected, list) else [expected]
    candidate = normalize_answer(extract_final_answer(observed))
    aliases = {"yes": "true", "no": "false"}
    candidate = aliases.get(candidate, candidate)
    for gold in golds:
        target = normalize_answer(gold)
        target = aliases.get(target, target)
        if candidate == target:
            return True
        candidate_number, target_number = _number(candidate), _number(target)
        if candidate_number is None or target_number is None:
            continue
        if target_number == 0:
            if abs(candidate_number) <= 1e-9:
                return True
        elif abs(candidate_number - target_number) <= numeric_tolerance * abs(target_number):
            return True
    return False


def parsed_choice(output: str, letters: str = "ABCD") -> str:
    """The option letter the model committed to: a tagged one wins, else the last bare letter."""
    text = str(output or "").upper()
    cls = f"[{letters}]"
    marked = re.findall(rf"(?:ANSWER|CHOICE|FINAL|OPTION)\s*[:=\-]?\s*\(?({cls})\)?\b", text)
    bare = re.findall(rf"\b\(?({cls})\)?\b", text)
    return marked[-1] if marked else (bare[-1] if bare else "")


def parsed_yes_no(output: str) -> str:
    m = re.search(r"\b(yes|no)\b", str(output or ""), re.IGNORECASE)
    return m.group(1).capitalize() if m else ""


def score_output(kind: str, output: str, gold: Any, *, numeric_tolerance: float = 0.0,
                 choices: list | None = None, dataset: str = "") -> bool:
    """``True`` iff *output* is correct for a case of task *kind*."""
    golds = gold if isinstance(gold, list) else [gold]
    if kind == "exact_or_numeric":
        return answer_matches(output, golds, numeric_tolerance=numeric_tolerance)
    if kind == "multiple_choice_letter":
        letters = "".join(str(c)[0] for c in (choices or [])) or "ABCD"
        return parsed_choice(output, letters) == str(golds[0]).strip().upper()
    if kind == "yes_no":
        return parsed_yes_no(output) == str(golds[0]).strip().capitalize()
    if kind == "llm_graded":
        from .tasks import llm as _llm

        return _llm.grade(dataset, output, gold)
    raise ValueError(f"unknown task kind {kind!r}")
