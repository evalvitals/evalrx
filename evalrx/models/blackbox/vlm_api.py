"""Black-box VLM via API (GPT-4V, Gemini Vision, etc.).

Planned for Stage 2.
"""

from __future__ import annotations

from typing import Any

from evalrx.models.blackbox.base import BlackboxModel


class BlackboxVLM(BlackboxModel):
    """API-based vision-language model."""

    def generate(self, inputs: Any, **kwargs) -> str:
        raise NotImplementedError("BlackboxVLM is planned for Stage 2.")
