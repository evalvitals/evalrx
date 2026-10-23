"""Answer parsers and graders shared by the task kinds.

``exact_or_numeric`` is the ChartQA/Spatial457 rule from
``examples/m1_m5/vlm_benchmark_common.py`` (label-blind final-answer
extraction, normalisation, relaxed numeric tolerance);
``multiple_choice_letter`` and ``yes_no`` are the MMAU / AudioCaps parsers from
the ``m1_m5`` audio examples. ``llm_graded`` delegates to the dataset's own
grader in ``examples/dataset_selection`` (see ``tasks/llm.py``);
``short_answer_em`` is SQuAD's normalisation (HotpotQA's official metric and
dspy's ``answer_exact_match``) on the extracted answer — unlike
``normalize_answer`` it REMOVES punctuation, so ``"Paris."`` == ``"Paris"``.

:func:`answer_key` reads the answer an output commits to without the gold, and
:func:`same_answer` says when two such keys are one answer under the same
relaxed rule the grader uses: the triage checks compare answers with these two,
so an answer that moved by less than the task's tolerance has not moved.
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


#: ChartQA labels: a yes/no answer and a true/false label are one answer.
_ALIASES = {"yes": "true", "no": "false"}
#: An ``exact_or_numeric`` answer key that is a number (``f"{x:g}"``: plain or exponent form).
_KEY_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _within(candidate: float, target: float, tolerance: float) -> bool:
    """ChartQA's relaxed match: within *tolerance* (relative) of *target*; a zero target needs a zero."""
    if target == 0:
        return abs(candidate) <= 1e-9
    return abs(candidate - target) <= tolerance * abs(target)


def answer_matches(observed: str, expected: Any, *, numeric_tolerance: float) -> bool:
    golds = expected if isinstance(expected, list) else [expected]
    candidate = normalize_answer(extract_final_answer(observed))
    candidate = _ALIASES.get(candidate, candidate)
    for gold in golds:
        target = normalize_answer(gold)
        target = _ALIASES.get(target, target)
        if candidate == target:
            return True
        candidate_number, target_number = _number(candidate), _number(target)
        if candidate_number is None or target_number is None:
            continue
        if _within(candidate_number, target_number, numeric_tolerance):
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
    # Strip balanced presentation markup around a SINGLE choice, not arbitrary
    # prose. Preserve the thought-channel and ambiguous-answer guards below.
    text = re.sub(rf"(\*\*|__|`|\*|_)([{cls}{cls.lower()}])\1", r"(\2)", text)
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


#: Task kinds whose answers are free text graded by a model or a caption metric:
#: two outputs can agree without being equal strings, so no answer key is read.
_UNKEYED = frozenset({"llm_graded", "chair_caption"})


def answer_key(kind: str, output: str, *, choices: list | None = None) -> "str | None":
    """The answer *output* commits to, read the way :func:`score_output` reads it, without the gold.

    Two outputs with the same key give the same answer, so a triage check can tell
    whether an answer moved across samples or rewordings without grading either.
    ``""`` means the output commits to no readable answer; ``None`` means the
    task kind has no categorical answer to compare.
    """
    text = str(output if output is not None else "")
    if kind in _UNKEYED:
        return None
    if kind == "multiple_choice_letter":
        return parsed_choice(text, "".join(str(c)[0] for c in (choices or [])) or "ABCD")
    if kind == "yes_no":
        return parsed_yes_no(text).lower()
    if kind == "short_answer_em":
        return squad_normalize(extract_final_answer(text)) if text.strip() else ""
    if kind == "exact_or_numeric":
        if not text.strip():
            return ""
        candidate = normalize_answer(extract_final_answer(text))
        candidate = _ALIASES.get(candidate, candidate)
        number = _number(candidate)
        return f"{number:g}" if number is not None else candidate
    raise ValueError(f"unknown task kind {kind!r}")


def same_answer(candidate: "str | None", reference: "str | None", *, numeric_tolerance: float = 0.0) -> bool:
    """Whether two answer keys (:func:`answer_key`) commit to the same answer, the way
    :func:`score_output` holds a candidate against a gold: equal keys, or two numbers with the
    candidate within *numeric_tolerance* (relative) of the *reference*, a zero reference needing a
    zero. The reference is the answer the candidate is held against (the Stage-0 answer, the
    re-asked baseline), so the relation is read from it, not symmetric at the band's edge. A zero
    tolerance is key equality, and a non-numeric key only ever equals itself.
    """
    if candidate == reference:
        return True
    if not numeric_tolerance or not candidate or not reference:
        return False
    if not (_KEY_NUMBER.fullmatch(candidate) and _KEY_NUMBER.fullmatch(reference)):
        return False
    return _within(float(candidate), float(reference), float(numeric_tolerance))


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
