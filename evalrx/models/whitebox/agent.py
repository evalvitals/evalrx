"""Local agent wrapper with tool calling (white-box).

Planned for Stage 2.
"""

from __future__ import annotations

from typing import Any

from evalrx.core.capability import Capability
from evalrx.core.model import Trace
from evalrx.models.base import BaseAgent
from evalrx.models.whitebox.base import WhiteboxModel


class WhiteboxAgent(WhiteboxModel, BaseAgent):
    """Local model that can call external tools over multiple steps."""

    capabilities = frozenset({Capability.GENERATE})

    def load(self) -> None:
        raise NotImplementedError("WhiteboxAgent is planned for Stage 2.")

    def generate(self, inputs: Any, **kwargs) -> str:
        raise NotImplementedError("WhiteboxAgent is planned for Stage 2.")

    def forward(self, inputs: Any, capture: set[Capability]) -> Trace:
        raise NotImplementedError("WhiteboxAgent is planned for Stage 2.")

    def step(self, observation: str, **kwargs) -> str:
        raise NotImplementedError("WhiteboxAgent is planned for Stage 2.")
