"""Black-box API agent with tool calling.

Planned for Stage 2.
"""

from __future__ import annotations

from typing import Any

from evalrx.core.capability import Capability
from evalrx.core.model import Trace
from evalrx.models.base import BaseAgent


class BlackboxAgent(BaseAgent):
    """API-based agent model."""

    capabilities = frozenset({Capability.GENERATE})

    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self.model_id = model_id
        self.api_key = api_key

    def generate(self, inputs: Any, **kwargs) -> str:
        raise NotImplementedError("BlackboxAgent is planned for Stage 2.")

    def forward(self, inputs: Any, capture: set[Capability]) -> Trace:
        raise NotImplementedError("Black-box models expose no internals.")

    def step(self, observation: str, **kwargs) -> str:
        raise NotImplementedError("BlackboxAgent is planned for Stage 2.")
