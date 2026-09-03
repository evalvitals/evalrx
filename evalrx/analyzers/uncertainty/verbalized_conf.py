"""Verbalized confidence — elicit and parse the model's stated confidence.

Black-box (``GENERATE`` only): ask the model to state a 0–100 confidence and parse
it.  A cheap, well-studied (often miscalibrated) uncertainty signal.

References:
- Teaching Models to Express Their Uncertainty in Words
  Lin, Hilton & Evans, TMLR 2022 — arXiv:2205.14334
- Just Ask for Calibration: ... Verbalized Confidence (RLHF LMs)
  Tian et al., EMNLP 2023 — arXiv:2305.14975
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model

_DEFAULT_ELICIT = (
    "\n\nAfter answering, state your confidence on a 0-100 scale "
    "on its own line as: Confidence: <number>"
)
_CONF = re.compile(r"confidence[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE)
_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


@register_analyzer("verbalized_confidence")
class VerbalizedConfidenceAnalyzer(Analyzer):
    """Elicit a stated confidence and parse it to a 0–1 float (None if unparseable)."""

    name = "verbalized_confidence"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, elicit: str = _DEFAULT_ELICIT) -> None:
        super().__init__(elicit=elicit)

    @staticmethod
    def _parse(text: str) -> Optional[float]:
        m = _CONF.search(text) or _PCT.search(text)
        if not m:
            return None
        val = float(m.group(1))
        return max(0.0, min(1.0, val / 100.0 if val > 1.0 else val))

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        raw = model.generate(str(case.inputs) + self.elicit)
        conf = self._parse(raw)
        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            artifacts={"raw": raw},
            findings={"verbalized_confidence": conf, "parsed": conf is not None, "raw_tail": raw[-120:]},
        )
