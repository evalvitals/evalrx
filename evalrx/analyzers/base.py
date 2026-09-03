"""Analyzer base re-export — every analyzer subclasses the core estimator contract.

Configure hyper-parameters in ``__init__``, declare ``requires`` (capabilities) and
``applies_to_modalities``, implement ``_run(model, cases) -> Result``.
"""

from __future__ import annotations

from evalrx.core.analyzer import Analyzer
from evalrx.core.result import Result

__all__ = ["Analyzer", "Result"]
