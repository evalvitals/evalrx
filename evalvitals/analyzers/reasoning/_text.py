"""Shared answer-handling helpers for the reasoning probes.

Every reasoning probe has to answer the same three questions — *what did the
model finally claim?*, *is that the gold answer?*, and *did the generation even
terminate?* — so the conventions live here once instead of drifting per probe.

Extraction follows the self-consistency / GSM8K convention (last ``Answer:`` tag
or ``\\boxed{}``, else the last non-empty line), and grading is numeric-aware:
for a numeric gold we compare against the LAST number in the prediction, which
is what "the answer is 18" means and what plain substring matching gets wrong
("18" is inside "180").
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from evalvitals.core.case import FailureCase

#: The separator is MANDATORY (":", "=", or the word "is"). With it optional the
#: bare word "answer" matches — "the answer to this is unclear" would extract
#: "to this is unclear" — and since extraction takes the LAST hit, one trailing
#: "I hope this answer helps" silently replaces a correct tagged answer.
_ANSWER_TAG = re.compile(
    r"(?:final\s+answer|answer)\s*(?:[:=]|\bis\b)\s*(.+)", re.IGNORECASE
)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_GIVE_UP = re.compile(
    r"(i (?:can(?:no|')t|am unable|do not know|don't know)"
    r"|cannot (?:be )?determine[d]?|not enough (?:information|context)"
    r"|insufficient (?:information|data)|unable to (?:answer|solve|determine)"
    r"|无法(?:确定|回答|判断)|不知道)",
    re.IGNORECASE,
)
#: Sentence/clause terminators that mark a *clean* end of generation.
_TERMINATORS = ".!?\"'`)]}\u3002\uff01\uff1f"


#: A span that is only the format placeholder the prompt asked for — models
#: restate "give it as 'Answer: <answer>'" and that echo, being LAST, would
#: otherwise replace the real answer.
_PLACEHOLDER = re.compile(r"^[<\[{(]\s*\w*\s*[>\]})]?['\".\s]*$")

#: A bare option label — ``(A)``, ``[B]``, ``C.`` — is a REAL multiple-choice
#: answer, but it is character-for-character the shape :data:`_PLACEHOLDER`
#: describes, so it needs an explicit exemption.  Without one the guard throws
#: the answer away and extraction keeps walking backwards into the chain-of-
#: thought, where it picks up whatever prose came before.  Measured on BBH
#: ``tracking_shuffled_objects_seven_objects`` / Qwen3.5-9B: 244 of 250 final
#: claims are a bare ``(X)``, 98 correct answers were scored FAIL, and the slice
#: read 0.592 instead of 0.984 — a mid-band dataset invented out of a saturated
#: one.  A one-letter *placeholder* (``<X>``) loses to a one-letter answer here
#: on purpose: the letter is a valid option either way, so the collision costs
#: nothing, while the reverse costs the whole measurement.
_OPTION_LABEL = re.compile(r"^[<\[{(]?\s*[A-Za-z]\s*[>\]})]?[.\s]*$")


def _is_placeholder(span: str) -> bool:
    """True when *span* is only the prompt's format hint echoed back."""
    return bool(_PLACEHOLDER.match(span)) and not _OPTION_LABEL.match(span)


def extract_answer(text: Any) -> str:
    r"""Last usable ``\boxed{}`` or ``Answer:``-tagged span, else the last non-empty line."""
    raw = str(text or "")
    # "Last usable" is by POSITION IN THE TEXT, across both conventions. Draining
    # every \boxed{} before looking at a single "Answer:" tag made an INTERMEDIATE
    # box outrank the final answer line whenever a chain boxed its working — on
    # minervamath / Qwen3.5-9B that mis-scored 14 of 272 (gold 2.45e6, the model
    # closed with "Answer: 2.45e6", extraction returned a mid-chain 7.353e14).
    candidates: list[tuple[int, str]] = []
    for match in _BOXED.finditer(raw):
        span = match.group(1).strip()
        if span:
            candidates.append((match.end(), span))
    for match in _ANSWER_TAG.finditer(raw):
        # the tag regex is line-greedy; keep only the first line of the span
        span = match.group(1)
        candidate = span.splitlines()[0].strip() if span.splitlines() else ""
        if candidate and not _is_placeholder(candidate):
            candidates.append((match.end(), candidate))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(value: Any) -> str:
    """Lowercase, drop thousands separators and decoration, collapse whitespace."""
    text = str(value if value is not None else "").lower().strip()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)          # 1,234 -> 1234
    text = re.sub(r"[$€£%\\]|\*\*|\bthe\b", " ", text)
    text = re.sub(r"[^\w\s./:+-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split()).strip(" .")


def as_number(value: Any) -> Optional[float]:
    """Parse *value* as a number if it is one **in its entirety**, else ``None``."""
    text = normalize_answer(value)
    if not text:
        return None
    match = _NUMBER.fullmatch(text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:  # pragma: no cover - regex already constrains this
        return None


def numbers_in(text: Any) -> list[float]:
    """Every number appearing in *text*, in order."""
    out: list[float] = []
    for token in _NUMBER.findall(str(text or "")):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:  # pragma: no cover - defensive
            continue
    return out


def answer_equal(prediction: Any, gold: Any, rel_tol: float = 1e-6) -> bool:
    """Numeric-aware answer equality.

    A numeric gold is matched against the LAST number in the prediction (the
    "the answer is 18" convention); short non-numeric golds — MC letters, yes/no
    — need a standalone token match so that ``"b"`` does not match "probably".
    """
    if isinstance(gold, (list, tuple, set, frozenset)):
        return any(answer_equal(prediction, item, rel_tol=rel_tol) for item in gold)
    pred = normalize_answer(prediction)
    exp = normalize_answer(gold)
    if not exp:
        return False
    if pred == exp:
        return True
    gold_num = as_number(exp)
    if gold_num is not None:
        found = numbers_in(pred)
        if not found:
            return False
        # Both ends, because the prediction reaches here in two shapes: a tagged
        # span that STARTS with the answer and may trail commentary ("620, since
        # the broken beds don't count") — first number — and a bare sentence
        # ending in it ("the answer is 18") — last number. Checking only one end
        # scores the commentary's numbers on half the cases.
        return any(
            abs(candidate - gold_num) <= rel_tol * max(1.0, abs(gold_num))
            for candidate in (found[0], found[-1])
        )
    if len(exp) <= 3:
        return re.search(rf"(?<!\w){re.escape(exp)}(?!\w)", pred) is not None
    return exp in pred


def default_grader(prediction: Any, case: "FailureCase") -> Optional[bool]:
    """Grade *prediction* against ``case.expected`` (``None`` = ungradable)."""
    if case.expected is None:
        return None
    tolerance = float((getattr(case, "metadata", {}) or {}).get("numeric_tolerance", 1e-6))
    return answer_equal(extract_answer(prediction), case.expected, rel_tol=tolerance)


# ----------------------------------------------------------------------
# Generation-shape helpers (termination / degeneration)
# ----------------------------------------------------------------------

def word_ngrams(text: Any, n: int) -> list[tuple[str, ...]]:
    words = str(text or "").lower().split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def repetition_score(text: Any, n: int = 8) -> float:
    """Fraction of word ``n``-grams that are repeats — 0 for clean text, →1 for a loop.

    Degenerate repetition is the failure mode that makes a truncated generation
    look like a reasoning failure, so it gets its own column rather than being
    folded into a length heuristic.
    """
    grams = word_ngrams(text, n)
    if not grams:
        return 0.0
    return round(1.0 - len(set(grams)) / len(grams), 4)


def has_answer_tag(text: Any) -> bool:
    raw = str(text or "")
    return bool(_BOXED.search(raw) or _ANSWER_TAG.search(raw))


def looks_truncated(text: Any) -> bool:
    """True when the text stops mid-thought: no terminator and no answer tag."""
    stripped = str(text or "").rstrip()
    if not stripped:
        return False
    return stripped[-1] not in _TERMINATORS and not has_answer_tag(stripped)


def looks_like_give_up(text: Any) -> bool:
    return bool(_GIVE_UP.search(str(text or "")))


# ----------------------------------------------------------------------
# Arithmetic verification
# ----------------------------------------------------------------------

_ARITH_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
#: ``lhs = rhs`` where the lhs is a pure-number expression carrying an operator.
_EQUATION = re.compile(
    r"(?<![\w=])((?:-?\d[\d,]*(?:\.\d+)?|[+\-*/×÷^()\s]){3,}?)\s*=\s*"
    # the number regex already absorbs a decimal tail, so a trailing '.' here is
    # sentence punctuation ("= 41.") and must not block the match
    r"(-?\d[\d,]*(?:\.\d+)?)(?!\.?\d)"
)


def safe_eval_arithmetic(expression: str) -> Optional[float]:
    """Evaluate a pure-number arithmetic expression, or ``None`` if it is not one.

    Parsed with :mod:`ast` and walked against an allow-list — model output is
    untrusted text and must never reach :func:`eval`.
    """
    cleaned = (
        expression.replace("×", "*").replace("÷", "/").replace("^", "**").strip()
    )
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)
    if not cleaned or not re.fullmatch(r"[\d.+\-*/()\s]+", cleaned):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> Optional[float]:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = _eval(node.operand)
            if inner is None:
                return None
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if left is None or right is None:
                return None
            try:
                return float(_ARITH_OPS[type(node.op)](left, right))
            except (ZeroDivisionError, OverflowError, ValueError):
                return None
        return None

    try:
        return _eval(tree)
    except RecursionError:  # pragma: no cover - pathological nesting
        return None


def find_equations(text: Any) -> list[tuple[str, float, float]]:
    """Every ``expr = value`` statement as ``(expr, stated_value, computed_value)``.

    Only statements whose left-hand side is a pure-number expression with an
    operator are returned — ``x = 5`` is a definition, not a claim we can check.
    """
    out: list[tuple[str, float, float]] = []
    for match in _EQUATION.finditer(str(text or "")):
        lhs, rhs = match.group(1).strip(), match.group(2)
        if not re.search(r"[+\-*/×÷^]", lhs) or len(numbers_in(lhs)) < 2:
            continue
        computed = safe_eval_arithmetic(lhs)
        stated = as_number(rhs)
        if computed is None or stated is None:
            continue
        out.append((f"{lhs} = {rhs}", stated, computed))
    return out
