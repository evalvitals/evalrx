"""Web-search-augmented QA loader → CaseBatch (multi-hop / retrieval-dependent QA).

Same shape as PureQA, but records may carry a ``context`` (retrieved passages) and
a ``requires_retrieval`` flag; cases are tagged ``web_search`` so analyzers /
A/B strategies can condition on them.  Records / JSONL / built-in sample.
"""

from __future__ import annotations

from evalrx.core.case import CaseBatch
from evalrx.datasets.base import Dataset, cases_from_records, read_jsonl

_SAMPLE = [
    {"question": "In what year did the author of 'The Origin of Species' publish it?",
     "answer": "1859", "requires_retrieval": True, "hops": 2},
    {"question": "What is the population of the capital of Japan?",
     "answer": "Tokyo", "requires_retrieval": True, "hops": 2},
]


class WebSearchQADataset(Dataset):
    """QA with (optional) retrieved web context."""

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
    def from_records(cls, records: list[dict], **keys) -> "WebSearchQADataset":
        return cls(records=records, **keys)

    @classmethod
    def from_jsonl(cls, path: str, **keys) -> "WebSearchQADataset":
        return cls(path=path, **keys)

    @classmethod
    def sample(cls) -> "WebSearchQADataset":
        return cls(records=_SAMPLE)

    def load(self) -> CaseBatch:
        records = self._records if self._records is not None else (
            read_jsonl(self._path) if self._path else _SAMPLE
        )
        return cases_from_records(records, tags={"web_search"}, **self._keys)
