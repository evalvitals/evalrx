"""Dataset loaders → CaseBatch, plus simple answer verifiers.

Text QA: ``LLMQADataset``.  Image+text QA: ``VLMQADataset`` and the
``TextVQASizeDataset`` / ``Spatial457Dataset`` / ``VQARADDataset`` benchmarks.
``PureQADataset`` is a back-compat alias of ``LLMQADataset``.
"""

from evalrx.datasets.base import (
    Dataset,
    cases_from_records,
    contains_answer,
    exact_match,
    normalize,
    read_jsonl,
)
from evalrx.datasets.gui_os import GUIOSDataset
from evalrx.datasets.llm_qa import LLMQADataset, PureQADataset
from evalrx.datasets.vlm_qa import (
    Spatial457Dataset,
    TextVQASizeDataset,
    VLMQADataset,
    VQARADDataset,
)
from evalrx.datasets.web_search_qa import WebSearchQADataset

__all__ = [
    "Dataset",
    "LLMQADataset",
    "VLMQADataset",
    "TextVQASizeDataset",
    "Spatial457Dataset",
    "VQARADDataset",
    "PureQADataset",
    "WebSearchQADataset",
    "GUIOSDataset",
    "cases_from_records",
    "read_jsonl",
    "exact_match",
    "contains_answer",
    "normalize",
]
