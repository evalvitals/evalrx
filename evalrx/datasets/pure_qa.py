"""Back-compat shim — ``pure_qa`` was split into ``llm_qa`` (text) and ``vlm_qa`` (image+text).

``PureQADataset`` remains importable here as an alias of
:class:`~evalrx.datasets.llm_qa.LLMQADataset`.  Prefer the explicit names:
``LLMQADataset`` for text QA, ``VLMQADataset`` / ``Spatial457Dataset`` for image+text.
"""

from __future__ import annotations

from evalrx.datasets.llm_qa import LLMQADataset, PureQADataset

__all__ = ["PureQADataset", "LLMQADataset"]
