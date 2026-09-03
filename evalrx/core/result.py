"""Result — the structured, agent-readable output of every analyzer.

A result has two faces:
  - ``findings``: a light, JSON-serialisable dict that an LLM agent reads to
    decide what to do next (e.g. top attended tokens, entropy stats).
  - ``artifacts``: heavy objects (attention tensors, hidden states) for humans,
    plotting, and downstream numeric analysis — never sent to the agent verbatim.

Subclasses (e.g. ``AttentionResult``) add domain-specific convenience methods
backed by ``artifacts`` while keeping the uniform ``findings`` contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch


@dataclass
class Result:
    """Uniform analyzer output.

    Attributes:
        analyzer:  Registered name of the analyzer that produced this.
        model:     ``repr()`` of the analysed model.
        findings:  Light, JSON-serialisable summary (the agent-facing answer).
        artifacts: Heavy objects (tensors, arrays) — not serialised by default.
        cases:     The :class:`~evalrx.core.case.CaseBatch` analysed.
        metadata:  Free-form extra info.
    """

    analyzer: str
    model: str
    findings: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    cases: "CaseBatch | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human/LLM-readable one-screen summary built from ``findings``."""
        lines = [f"[{self.analyzer}] on {self.model}"]
        for key, value in self.findings.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (``findings`` only — artifacts are dropped)."""
        return {
            "analyzer": self.analyzer,
            "model": self.model,
            "findings": self.findings,
            "metadata": self.metadata,
            "n_cases": len(self.cases) if self.cases is not None else 0,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(analyzer={self.analyzer!r}, model={self.model!r})"
