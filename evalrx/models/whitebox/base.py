"""Base class for locally-deployed (white-box) models.

White-box models have full parameter access, so they can provide rich
capabilities (attention, hidden states, gradients).  Concrete subclasses
declare their ``capabilities`` and implement ``generate`` + ``forward``.
"""

from __future__ import annotations

from abc import abstractmethod

from evalrx.core.model import Model


class WhiteboxModel(Model):
    """Base for open-source models loaded with full parameter access.

    Subclasses must implement :meth:`load`, :meth:`generate`, and
    :meth:`forward`, and set the ``capabilities`` class attribute.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (called lazily on first use)."""
