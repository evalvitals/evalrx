"""The scalar every intervention is scored on: a next-token logit margin.

``logit[positive] - logit[negative]`` at one position equals the log-prob
difference (the softmax normaliser cancels), so it is comparable across edits
and differentiable for the gradient methods.

Two targets:

* ``"prediction"`` (default) — the clean run's own top-1 vs top-2 token. Label
  free and defined for every case, so the per-case columns are never computed
  differently for FAIL and PASS cases. It answers "what drove the answer the
  model actually gave".
* ``"gold"`` — the gold answer's first token vs the strongest other token of the
  clean run. Needs ``case.expected`` and a tokenizer; a case without either is
  skipped (counted, never guessed). It answers "what supported the right answer".

One answer has several first tokens: a BPE vocabulary holds ``"Yes"`` and
``" Yes"`` (and ``"yes"``) as different ids, and a model can split its
probability between them (Qwen3.5-0.8B on POPE gives " Yes" 0.80 and "Yes"
0.20 on one correct case). Comparing "Yes" with the strongest other token
then measures "Yes" against " Yes", not against "No". So the comparison
token is the strongest token outside the answer's surface forms
(:func:`answer_variants`: with or without a leading space, lower-case or
capitalised), the gold target is the surface form the clean run scores
highest, and :attr:`LogitMargin.variants` lists the forms for readers that
sum probabilities over them. Without a decoder to confirm that a form's first
token carries the answer's text, only the literal form is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from evalrx.analyzers._internals.words import decode_word

if TYPE_CHECKING:
    from evalrx.core.case import FailureCase
    from evalrx.core.model import Model

TARGETS = ("prediction", "gold")


@dataclass(frozen=True)
class LogitMargin:
    """``logits[position, positive] - logits[position, negative]``.

    ``variants``: token ids that start the same answer as ``positive`` (its
    surface forms, ``positive`` included); ``negative`` is never one of them.
    """

    positive: int
    negative: int
    position: int = -1
    kind: str = "prediction"
    variants: tuple[int, ...] = ()

    def __call__(self, logits: Any) -> Any:
        row = logits[self.position]
        return row[self.positive].float() - row[self.negative].float()

    def value(self, logits: Any) -> float:
        return float(self(logits).detach())

    def top1_changed(self, logits: Any, clean_top1: int) -> bool:
        return int(logits[self.position].argmax()) != int(clean_top1)


def gold_text(expected: Any) -> Optional[str]:
    """The gold answer as a string: a str, the first alias, or a lone ``all_of`` entry."""
    if isinstance(expected, dict):
        values = expected.get("all_of") or []
        expected = values[0] if values else None
    if isinstance(expected, (list, tuple)):
        expected = expected[0] if expected else None
    if expected is None:
        return None
    text = str(expected).strip()
    return text or None


def first_token_id(tokenizer: Any, text: str) -> Optional[int]:
    """Id of the first token of *text* as an answer would start (no special tokens)."""
    if tokenizer is None or not text:
        return None
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001 - a tokenizer that cannot encode → no gold target
        return None
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return int(ids[0]) if ids else None


def answer_variants(tokenizer: Any, text: "str | None") -> frozenset:
    """First-token ids of the surface forms *text* can start an answer with.

    Forms: the text as given, lower-case and capitalised (for more than one
    character, so the option letter "A" never takes in the article "a"), each
    with and without a leading space. A form counts only when the decoder
    shows its first token carries the start of the answer's text (a
    character-level vocabulary would otherwise make the bare space a "form" of
    every answer); with no decoder, only the literal form is returned.
    """
    base = str(text or "").strip()
    literal = first_token_id(tokenizer, base)
    if literal is None:
        return frozenset()
    forms = {base} | ({base.lower(), base[:1].upper() + base[1:]} if len(base) > 1 else set())
    ids = {literal}
    for form in forms:
        for spelled in (form, " " + form):
            tid = first_token_id(tokenizer, spelled)
            piece = decode_word(tokenizer, tid) if tid is not None else None
            if piece is not None and piece.strip() and form.startswith(piece.strip()):
                ids.add(int(tid))
    return frozenset(ids)


def _strongest_outside(row: Any, ids: "frozenset", positive: int) -> int:
    """The strongest token that is not a form of the answer (only *positive*
    is excluded when the forms would cover the whole vocabulary)."""
    masked = sorted(i for i in ids if i < row.numel())
    if len(masked) >= row.numel():
        masked = [positive]
    others = row.clone()
    others[masked] = float("-inf")
    return int(others.argmax())


def build_margin(
    target: str,
    clean_logits: Any,
    *,
    model: "Model | None" = None,
    case: "FailureCase | None" = None,
    position: int = -1,
) -> "tuple[LogitMargin | None, str]":
    """``(margin, "")`` or ``(None, skip_reason)``."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    row = clean_logits[position].detach().float()
    if row.numel() < 2:
        return None, "vocab_too_small"
    from evalrx.analyzers._internals.access import tokenizer_of

    tokenizer = tokenizer_of(model) if model is not None else None
    if target == "prediction":
        top1 = int(row.argmax())
        word = decode_word(tokenizer, top1) if tokenizer is not None else None
        variants = (answer_variants(tokenizer, word) if word and word.strip() else frozenset()) | {top1}
        return LogitMargin(top1, _strongest_outside(row, variants, top1), position, "prediction",
                           tuple(sorted(variants))), ""

    gold = gold_text(getattr(case, "expected", None))
    if gold is None:
        return None, "no_gold"
    variants = frozenset(t for t in answer_variants(tokenizer, gold) if t < row.numel())
    if not variants:
        return None, "gold_untokenizable"
    # the surface form this run gives the gold answer, fixed on the clean logits
    tid = max(variants, key=lambda t: float(row[t]))
    return LogitMargin(tid, _strongest_outside(row, variants, tid), position, "gold", tuple(sorted(variants))), ""
