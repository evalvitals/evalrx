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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from evalrx.core.case import FailureCase
    from evalrx.core.model import Model

TARGETS = ("prediction", "gold")


@dataclass(frozen=True)
class LogitMargin:
    """``logits[position, positive] - logits[position, negative]``."""

    positive: int
    negative: int
    position: int = -1
    kind: str = "prediction"

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


def build_margin(
    target: str,
    clean_logits: Any,
    *,
    model: "Model | None" = None,
    case: "FailureCase | None" = None,
    position: int = -1,
) -> "tuple[LogitMargin | None, str]":
    """``(margin, "")`` or ``(None, skip_reason)``."""
    import torch

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    row = clean_logits[position].detach().float()
    if row.numel() < 2:
        return None, "vocab_too_small"
    if target == "prediction":
        top2 = torch.topk(row, 2).indices
        return LogitMargin(int(top2[0]), int(top2[1]), position, "prediction"), ""

    from evalrx.analyzers._internals.access import tokenizer_of

    gold = gold_text(getattr(case, "expected", None))
    if gold is None:
        return None, "no_gold"
    tid = first_token_id(tokenizer_of(model) if model is not None else None, gold)
    if tid is None or tid >= row.numel():
        return None, "gold_untokenizable"
    others = row.clone()
    others[tid] = float("-inf")
    return LogitMargin(tid, int(others.argmax()), position, "gold"), ""
