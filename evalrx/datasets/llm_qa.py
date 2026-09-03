"""LLM (text) QA loader → CaseBatch.

Text-only question answering.  Source-agnostic: in-memory records, a JSONL path,
or the built-in sample.  ``expected`` carries the gold answer; pair it with
:func:`evalrx.datasets.base.contains_answer` to turn a model output into a
success bool for A/B comparisons.  (For image+text QA use :mod:`evalrx.datasets.vlm_qa`.)
"""

from __future__ import annotations

from evalrx.core.case import CaseBatch
from evalrx.datasets.base import Dataset, cases_from_records, read_jsonl

_SAMPLE = [
    {"question": "What is the capital of France?", "answer": "Paris", "difficulty": "easy"},
    {"question": "What is 17 multiplied by 3?", "answer": "51", "difficulty": "easy"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "difficulty": "easy"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "difficulty": "medium"},
]


class LLMQADataset(Dataset):
    """Text question-answering loader."""

    def __init__(
        self,
        records: list[dict] | None = None,
        path: str | None = None,
        *,
        prompt_key: str = "question",
        answer_key: str = "answer",
    ) -> None:
        self._records = records
        self._path = path
        self._keys = dict(prompt_key=prompt_key, answer_key=answer_key)

    @classmethod
    def from_records(cls, records: list[dict], **keys) -> "LLMQADataset":
        return cls(records=records, **keys)

    @classmethod
    def from_jsonl(cls, path: str, **keys) -> "LLMQADataset":
        return cls(path=path, **keys)

    @classmethod
    def sample(cls) -> "LLMQADataset":
        return cls(records=_SAMPLE)

    def load(self) -> CaseBatch:
        records = self._records if self._records is not None else (
            read_jsonl(self._path) if self._path else _SAMPLE
        )
        return cases_from_records(records, tags={"llm_qa"}, **self._keys)


# Back-compat: the old combined name. Prefer LLMQADataset (text) / VLMQADataset (image+text).
PureQADataset = LLMQADataset
