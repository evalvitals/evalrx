"""Base class for black-box (API-only) models.

Black-box models have no parameter access, so they provide only output-level
capabilities (``GENERATE``, sometimes ``LOGITS``).  Capability matching means
attention/saliency/probing analyzers are simply *not offered* for them — the
agent sees an empty (or small) compatible-analyzer set and never tries.
"""

from __future__ import annotations

from typing import Any

from evalrx.core.capability import Capability
from evalrx.core.model import Model, Trace


class BlackboxModel(Model):
    """Base for API-based models (no local weights)."""

    capabilities = frozenset({Capability.GENERATE})

    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self.model_id = model_id
        self.api_key = api_key

    def forward(self, inputs: Any, capture: set[Capability]) -> Trace:
        raise NotImplementedError(
            "Black-box models expose no internals; only generate() is available."
        )
