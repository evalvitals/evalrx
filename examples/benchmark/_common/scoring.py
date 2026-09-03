"""Answer parsers and graders shared by the task kinds.

``exact_or_numeric`` is the ChartQA/Spatial457 rule from
``examples/m1_m4/vlm_benchmark_common.py`` (label-blind final-answer
extraction, normalisation, relaxed numeric tolerance);
``multiple_choice_letter`` and ``yes_no`` are the MMAU / AudioCaps parsers from
the ``m1_m4`` audio examples. ``llm_graded`` delegates to the dataset's own
grader in ``examples/dataset_selection`` (see ``tasks/llm.py``);
``short_answer_em`` is SQuAD's normalisation (HotpotQA's official metric and
dspy's ``answer_exact_match``) on the extracted answer — unlike
``normalize_answer`` it REMOVES punctuation, so ``"Paris."`` == ``"Paris"``.
"""

from __future__ import annotations

import re
import string
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


# A leaked reasoning channel (Gemma 4 opens ``<|channel>thought`` on its own even
# with enable_thinking=False; ``skip_special_tokens`` leaves the bare word) is not
# a committed answer unless it carries an explicit answer tag. Measured on
# gemma-4-e2b/MMAU (2026-08-22): 100/256 outputs were such preambles cut at the
# token cap, and the old "last bare letter anywhere" fallback scraped a letter out
# of the option enumeration they were restating — 16 spurious PASSes, 55 fictional
# "chosen" letters on FAILs.
_THOUGHT_PREAMBLE = re.compile(r"^\s*(?:<\|channel>)?thought\b", re.IGNORECASE)


def _is_thought_preamble(text: str) -> bool:
    return bool(_THOUGHT_PREAMBLE.match(text))


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _committed_letters(line: str, cls: str) -> set[str]:
    """Option letters a line commits to. ``A`` is also an English word, so it only
    counts delimited — ``(A)``, ``A)``/``A.`` at line start, a whole-line ``a``/``A``,
    or closing the line — while the other letters also count standalone
    (``"I'd go with B"``). Several distinct letters = an enumeration or a hedge."""
    found = re.findall(rf"^\(?([{cls}{cls.lower()}])\)?[.:]?$", line)       # "b" / "(B)" / "B."
    found += re.findall(rf"\(([{cls}])\)", line)                              # "(B) Live music"
    found += re.findall(rf"^([{cls}])[.):]", line)                              # "B) Live" / "B. Live"
    found += re.findall(rf"(?<![A-Za-z(])([{cls}])[.):,]?\s*$", line)         # "... so it is B."
    standalone = cls.replace("A", "")
    if standalone:
        found += re.findall(rf"(?<![A-Za-z(])([{standalone}])(?![A-Za-z])", line)
    return {f.upper() for f in found}


def parsed_choice(output: str, letters: str = "ABCD") -> str:
    """The option letter the model committed to, or ``""`` when it committed to none.

    1. an answer tag anywhere (``Answer: B`` / ``Option (B)``) wins — last one;
    2. a leaked thought preamble without a tag is not an answer;
    3. else the last non-empty line (or, when that line names no option, the first
       one — ``"B\n\nbecause ..."``) must name exactly ONE distinct option letter;
       an enumeration (``"(A) dog, (B) cat"``) or a hedge names several -> ``""``.
    """
    text = str(output or "")
    cls = "".join(sorted(set(letters.upper())))
    marked = re.findall(rf"(?:answer|choice|final|option)\s*[:=\-]?\s*\(?([{cls}{cls.lower()}])\)?\b",
                        text, re.IGNORECASE)
    if marked:
        return marked[-1].upper()
    if _is_thought_preamble(text):
        return ""
    lines = _nonempty_lines(text)
    for line in (lines[-1:] + lines[:1]):
        found = _committed_letters(line, cls)
        if found:
            return found.pop() if len(found) == 1 else ""
    return ""


def parsed_yes_no(output: str) -> str:
    """``Yes``/``No`` the model committed to: the first one in the answer, ``""`` for
    an untagged leaked thought preamble (it restates the question, not an answer)."""
    text = str(output or "")
    if _is_thought_preamble(text) and not re.search(r"(?:answer|final)\s*[:=\-]?\s*(yes|no)\b", text, re.IGNORECASE):
        return ""
    m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else ""


_PUNCTUATION = set(string.punctuation)


def squad_normalize(text: Any) -> str:
    """SQuAD ``normalize_answer``: lower, strip punctuation, strip articles, fix
    whitespace — the exact chain HotpotQA's official eval and dspy's
    ``answer_exact_match`` grade with."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def squad_em(observed: str, golds: list) -> bool:
    candidate = squad_normalize(extract_final_answer(observed))
    return any(candidate == squad_normalize(g) for g in golds)


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
    if kind == "short_answer_em":
        return squad_em(output, golds)
    if kind == "llm_graded":
        from .tasks import llm as _llm

        return _llm.grade(dataset, output, gold)
    if kind == "chair_caption":
        from .tasks import chair as _chair

        return _chair.grade(output, gold)
    raise ValueError(f"unknown task kind {kind!r}")
