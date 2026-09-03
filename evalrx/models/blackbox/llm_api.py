"""Black-box LLM via API (OpenAI, Anthropic, Google, etc.).

Planned for Stage 2.
"""

from __future__ import annotations

from typing import Any

from evalrx.models.blackbox.base import BlackboxModel


class BlackboxLLM(BlackboxModel):
    """API-based LLM — no local weights, output-level capabilities only."""

    def generate(self, inputs: Any, **kwargs) -> str:
        raise NotImplementedError("BlackboxLLM is planned for Stage 2.")
