"""Dataset base — loaders produce a CaseBatch; plus simple answer verifiers.

A dataset is just a ``load() -> CaseBatch`` of :class:`FailureCase` (the unit
analyzers consume).  ``cases_from_records`` maps plain dict records (question /
answer / image / extra metadata) into cases; the verifiers give A/B strategies a
``case -> bool success`` signal without pulling in a heavy metric library.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from evalrx.core.case import CaseBatch, FailureCase, Inputs, Provenance, Source


# ---------------- verifiers ----------------
def normalize(s) -> str:
    return " ".join(str(s).strip().lower().split())


def exact_match(prediction, gold) -> bool:
    return normalize(prediction) == normalize(gold)


def contains_answer(prediction, gold) -> bool:
    """True if the (normalised) gold answer appears in the prediction."""
    return bool(str(gold).strip()) and normalize(gold) in normalize(prediction)


# ---------------- record -> cases ----------------
def cases_from_records(
    records: Iterable[dict],
    *,
    prompt_key: str = "question",
    answer_key: str = "answer",
    image_key: str = "image",
    tags: set[str] | None = None,
    source: Source = Source.DATASET,
) -> CaseBatch:
    """Build a :class:`CaseBatch` from dict records; the full record is kept in metadata."""
    out = CaseBatch()
    for rec in records:
        out.append(
            FailureCase(
                inputs=Inputs(prompt=str(rec.get(prompt_key, "")), image=rec.get(image_key)),
                expected=rec.get(answer_key),
                tags=set(tags or ()),
                provenance=Provenance(source=source),
                metadata=dict(rec),
            )
        )
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Dataset(ABC):
    """A benchmark loader: ``load() -> CaseBatch``."""

    @abstractmethod
    def load(self) -> CaseBatch:  # pragma: no cover - interface
        ...
