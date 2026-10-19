"""Decoded tokens as content words, and the prompt words specific to one case.

The lenses that read media tokens as words (``embed_lens``, ``patchscope``)
ask whether a case's own question word shows up there. Which prompt words are
the case's own is told apart from the task's template by a set of prompts: a
word found in at most ``max_share`` of the distinct prompts is specific
("snowboard"), one found in most of them is template ("image", "answer").

That set is the analysed batch (:func:`specific_words`), or a fixed reference
set of prompts (:func:`reference_specific_words`), with which a case's words
no longer depend on which other cases share its batch.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

# Function words the decoder emits for almost any state; kept out of word tallies.
STOP = frozenset("""a an and are as at be but by for from has have in into is it its of on or
that the their there this to was were with""".split())


def decode_word(tokenizer: Any, token_id: int) -> "str | None":
    """``tokenizer.decode([token_id])``, or ``None`` when there is no decoder (never a guess)."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 - no decoder -> no word
        return None
    return text if isinstance(text, str) else None


def content_word(tokenizer: Any, token_id: int, special: "set[str]") -> "str | None":
    """The token as a lower-case content word, or ``None`` (punctuation, function word, special)."""
    word = decode_word(tokenizer, token_id)
    if word is None or word in special:
        return None
    key = word.strip().lower()
    return key if key.isalpha() and len(key) >= 2 and key not in STOP else None


def specific_words(prompt_words: "dict[str, tuple[Any, set[str]]]", *, max_share: float = 0.5,
                   min_prompts: int = 3) -> "tuple[dict[str, set[str]], bool]":
    """Each case's prompt words found in at most *max_share* of the distinct prompts.

    Args:
        prompt_words: ``{case_id: (prompt_key, content_words)}``. Cases with the
                      same ``prompt_key`` (the question asked verbatim of several
                      media) count as one prompt, so a repeated question does
                      not make its own words look like template.
        max_share:    largest share of the distinct prompts a specific word may appear in.
        min_prompts:  fewest distinct prompts the template can be inferred from.

    Returns:
        ``({case_id: specific_words}, inferred)``; below *min_prompts* nothing can
        be told apart, so every set is empty and ``inferred`` is ``False``.
    """
    distinct: dict[Any, set[str]] = {}
    for key, words in prompt_words.values():
        distinct.setdefault(key, set(words))
    if len(distinct) < min_prompts:
        return {cid: set() for cid in prompt_words}, False
    share: dict[str, int] = {}
    for words in distinct.values():
        for word in words:
            share[word] = share.get(word, 0) + 1
    limit = max_share * len(distinct)
    return {cid: {w for w in words if share.get(w, 0) <= limit}
            for cid, (_, words) in prompt_words.items()}, True


def text_words(tokenizer: Any, text: str, special: "set[str]") -> set[str]:
    """Content words of *text* tokenized on its own (a prompt without the chat template around it)."""
    try:
        ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001 - a tokenizer that is not callable
        try:
            ids = tokenizer.encode(str(text), add_special_tokens=False)
        except Exception:  # noqa: BLE001 - no tokenizer at all -> no words
            return set()
    return {w for t in ids if (w := content_word(tokenizer, t, special)) is not None}


def reference_specific_words(texts: "dict[str, str]", reference: "Iterable[str]", tokenizer: Any,
                             special: "set[str]", *, max_share: float = 0.5,
                             min_prompts: int = 3) -> "tuple[dict[str, set[str]], bool]":
    """Each case's prompt words found in at most *max_share* of the distinct *reference* prompts.

    Args:
        texts:     ``{case_id: prompt}`` of the analysed cases (the prompt text alone).
        reference: prompts of a fixed reference set (repeats count once).

    Returns:
        ``({case_id: specific_words}, inferred)`` as :func:`specific_words`, except that
        the template is told from the reference, so a case's specific words are the
        same whatever else is in its batch.
    """
    distinct = {str(p): text_words(tokenizer, p, special) for p in dict.fromkeys(str(p) for p in reference)}
    if len(distinct) < min_prompts:
        return {cid: set() for cid in texts}, False
    share = Counter(w for words in distinct.values() for w in words)
    limit = max_share * len(distinct)
    return {cid: {w for w in text_words(tokenizer, text, special) if share.get(w, 0) <= limit}
            for cid, text in texts.items()}, True
