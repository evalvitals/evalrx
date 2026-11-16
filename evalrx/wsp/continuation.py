"""Prompt + recorded generation as one token sequence, with positions grouped.

Some failures only exist during generation: a repetition loop starts thousands
of tokens after the prompt. To read or score the model there, the recorded
output of a case (``case.observed``) is appended to its prompt and the whole
sequence is run teacher-forced. This module builds that sequence and labels its
positions; the continuation audit and the continuation contrast both use it, so
they agree on what "a loop position" is.

Conventions. ``ids`` is the full sequence, ``start`` the index of the first
continuation token. A *target* ``i >= start`` is a continuation token; it is
predicted from the model state at position ``i - 1``. Prompt tokens are never
targets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from evalrx.core.model import Model

GROUPINGS = ("repetition", "none")


def repetition_mask(ids: "list[int]", n: int) -> "list[bool]":
    """``mask[i]`` is True when the *n* tokens before index *i* occurred earlier.

    The window ``ids[i-n:i]`` is compared with every earlier window of the same
    length, so ``mask[i]`` describes the context in which token *i* is
    predicted: inside a stretch the sequence has already produced once.
    """
    seen: set[tuple[int, ...]] = set()
    mask = [False] * len(ids)
    for i in range(n, len(ids)):
        window = tuple(ids[i - n:i])
        mask[i] = window in seen
        seen.add(window)
    return mask


def continuation_text(case: Any) -> "str | None":
    """The recorded output of *case* as text, or ``None`` when there is none."""
    observed = getattr(case, "observed", None)
    if observed is None:
        return None
    text = observed if isinstance(observed, str) else str(observed)
    return text if text.strip() else None


@dataclass
class ContinuationSequence:
    """One case's token sequence with its grouped targets."""

    case: Any
    ids: "list[int]"
    start: int
    targets: "list[int]"
    labels: "list[str]"

    def positions(self, group: "str | None" = None) -> "list[int]":
        """State positions (``target - 1``) of every target, or of one group's."""
        return [i - 1 for i, lab in zip(self.targets, self.labels)
                if group is None or lab == group]


def group_names(groups: str) -> "tuple[str, ...]":
    if groups not in GROUPINGS:
        raise ValueError(f"groups must be one of {GROUPINGS}, got {groups!r}")
    return ("loop", "productive") if groups == "repetition" else ("all",)


def encode_continuation(model: "Model", case: Any, *, groups: str = "repetition",
                        repeat_ngram: int = 32, max_tokens: int = 4096,
                        ) -> "ContinuationSequence | str":
    """Build the sequence for one case, or return the reason it cannot be built.

    The prompt is tokenised by the model handle (one short forward pass, so the
    handle's own template and special-token rules apply); the continuation is
    appended as plain tokens. Reasons: ``no_continuation``, ``forward_failed``,
    ``has_media``, ``continuation_untokenizable``,
    ``no_continuation_tokens_after_truncation``.
    """
    from evalrx.analyzers._internals.access import media_positions, tokenizer_of
    from evalrx.core.capability import Capability

    names = group_names(groups)
    text = continuation_text(case)
    if text is None:
        return "no_continuation"
    tokenizer = tokenizer_of(model)
    if tokenizer is None:
        raise ValueError("the model handle exposes no tokenizer to encode the continuation with")
    try:
        trace = model.forward(getattr(case, "inputs", case), capture={Capability.LOGITS})
    except Exception:  # noqa: BLE001 - an unreadable case is reported, not fatal
        return "forward_failed"
    if media_positions(trace)[1]:
        return "has_media"
    prompt_ids = [int(t) for t in trace.token_ids]
    try:
        cont = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001
        return "continuation_untokenizable"
    if cont and isinstance(cont[0], (list, tuple)):
        cont = cont[0]
    ids = (prompt_ids + [int(t) for t in cont])[: int(max_tokens)]
    start = len(prompt_ids)
    if len(ids) <= start:
        return "no_continuation_tokens_after_truncation"
    targets = list(range(max(start, 1), len(ids)))
    if groups == "repetition":
        mask = repetition_mask(ids, int(repeat_ngram))
        labels = [names[0] if mask[i] else names[1] for i in targets]
    else:
        labels = [names[0]] * len(targets)
    return ContinuationSequence(case=case, ids=ids, start=start, targets=targets, labels=labels)


def encode_continuations(model: "Model", cases: Iterable[Any], **kwargs: Any,
                         ) -> "tuple[list[ContinuationSequence], Counter]":
    """:func:`encode_continuation` over *cases*: ``(sequences, skip reasons)``."""
    sequences: list[ContinuationSequence] = []
    skipped: Counter = Counter()
    for case in cases:
        built = encode_continuation(model, case, **kwargs)
        if isinstance(built, str):
            skipped[built] += 1
        else:
            sequences.append(built)
    return sequences, skipped


@dataclass
class ContinuationStats:
    """Mean MLP input and output per position group and layer.

    ``groups[name][layer] = {"in": [d_mlp], "out": [d_model], "n": positions}``.
    """

    groups: "dict[str, dict[int, dict[str, Any]]]" = field(default_factory=dict)
    n_cases: int = 0
    skipped: "dict[str, int]" = field(default_factory=dict)

    def n_positions(self, group: str) -> int:
        layers = self.groups.get(group) or {}
        return next(iter(layers.values()))["n"] if layers else 0
