"""The plain-language summary a report opens with."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReaderFinding:
    """One claim phrased for a reader who does not know the pipeline."""

    title: str
    summary: str
    why_it_matters: str
    evidence_level: str
    limitation: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "evidence_level": self.evidence_level,
            "limitation": self.limitation,
        }


@dataclass
class ReaderReport:
    """Stable, audience-first view model rendered before technical evidence."""

    headline: str
    question: str
    answer: str
    confidence: str
    what_we_did: list[str] = field(default_factory=list)
    key_findings: list[ReaderFinding] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "what_we_did": self.what_we_did,
            "key_findings": [finding.to_dict() for finding in self.key_findings],
            "open_questions": self.open_questions,
            "next_steps": self.next_steps,
        }
