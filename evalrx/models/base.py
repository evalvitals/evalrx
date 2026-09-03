"""Model base classes.

The abstract contract lives in :mod:`evalrx.core.model` (:class:`Model`,
:class:`Trace`).  This module re-exports it and adds the agent base.
Deployment-specific bases live alongside their models:
  - :class:`~evalrx.models.whitebox.base.WhiteboxModel` (local weights)
  - :class:`~evalrx.models.blackbox.base.BlackboxModel` (API-only)
"""

from __future__ import annotations

from abc import abstractmethod

from evalrx.core.model import Model, Trace

__all__ = ["Model", "Trace", "BaseAgent"]


class BaseAgent(Model):
    """Abstract base for agent-mode models that use tools over multiple steps."""

    @abstractmethod
    def step(self, observation: str, **kwargs) -> str:
        """Process one agent step and return the next action/response."""
