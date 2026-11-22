"""Keyword contrasts for a free-text column, frozen as recipes.

"Which words turn up more on the failing side" is a natural first look at a
text column, and an easy one to fool oneself with: there are thousands of
words, and some will separate any two groups by chance. This module gives the
look the same discipline as any other candidate signal. The words are chosen
on the exploration rows only; each becomes a recipe ``has_word(column,
"word")`` with the direction it showed there stated as its expectation; and
the recipes are then re-tested on held-out rows, in the same correction family
as everything else.

Ranking uses the log-odds ratio with a Dirichlet prior, which does not let a
word seen three times outrank one seen three hundred:
  "Fightin' Words: Lexical Feature Selection and Evaluation for Identifying
  the Content of Political Conflict"
  Monroe, Colaresi & Quinn (2008), Political Analysis 16(4):372-403
  https://doi.org/10.1093/pan/mpn018
"""

from __future__ import annotations

import math
import re
from typing import Any

from evalrx.analysis.explorer import CandidateSignal

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def tokenize(text: Any) -> set[str]:
    """The distinct lower-cased words of *text* (three characters or longer)."""
    return {w for w in _WORD.findall(str(text).lower()) if len(w) >= 3}


def keyword_candidates(
    rows: "list[dict[str, Any]]",
    text_col: str,
    side: "list[float | None]",
    *,
    top_k: int = 5,
    min_rows: int = 5,
    prior: float = 0.5,
    side_names: "tuple[str, str]" = ("failing", "passing"),
) -> list[CandidateSignal]:
    """Up to *top_k* words over-represented on each side, as candidate signals.

    *side* is 1.0 for a row on the first side (by convention the failing one),
    0.0 for the other, None to leave the row out. A word must occur in at
    least *min_rows* rows. Counts are of rows containing the word, not of
    occurrences, so one long text cannot carry a word on its own.

    Each candidate's recipe expects the effect to have the sign it had here
    (more failures where the word occurs, or fewer). The candidates carry no
    verdict: they were picked on these very rows, so only a held-out re-test
    says anything about them.
    """
    docs = [
        (tokenize(r.get(text_col)), s) for r, s in zip(rows, side)
        if s is not None and r.get(text_col) not in (None, "")
    ]
    n1 = sum(1 for _, s in docs if s == 1.0)
    n0 = len(docs) - n1
    if n1 == 0 or n0 == 0:
        return []
    count1: dict[str, int] = {}
    count0: dict[str, int] = {}
    for words, s in docs:
        target = count1 if s == 1.0 else count0
        for w in words:
            target[w] = target.get(w, 0) + 1
    scored: list[tuple[float, str]] = []
    for w in set(count1) | set(count0):
        c1, c0 = count1.get(w, 0), count0.get(w, 0)
        if c1 + c0 < min_rows:
            continue
        # Log odds of a row containing the word, on each side, with a prior.
        l1 = math.log((c1 + prior) / (n1 - c1 + prior))
        l0 = math.log((c0 + prior) / (n0 - c0 + prior))
        variance = (1 / (c1 + prior) + 1 / (n1 - c1 + prior)
                    + 1 / (c0 + prior) + 1 / (n0 - c0 + prior))
        scored.append(((l1 - l0) / math.sqrt(variance), w))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picks = [(z, w) for z, w in scored[:top_k] if z > 0]
    picks += [(z, w) for z, w in scored[::-1][:top_k] if z < 0 and (z, w) not in picks]

    out: list[CandidateSignal] = []
    for z, w in picks:
        c1, c0 = count1.get(w, 0), count0.get(w, 0)
        toward = side_names[0] if z > 0 else side_names[1]
        name = f"kw_{re.sub(r'[^0-9a-z]+', '_', w)}"
        out.append(CandidateSignal(
            name=name,
            display_name=f'Text mentions "{w}"',
            rationale=(
                f'"{w}" occurs in {c1}/{n1} {side_names[0]} rows and {c0}/{n0} '
                f"{side_names[1]} rows of the exploration half (leans {toward}; "
                f"z = {z:+.1f}). Chosen on those rows, so untested."
            ),
            suggested_test="held-out re-test of the frozen keyword recipe",
            recipe={
                "name": name, "kind": "expr",
                "expr": f"has_word({text_col}, {w!r})",
                "expect": {"relation": ">" if z > 0 else "<", "value": 0.0},
            },
        ))
    return out
