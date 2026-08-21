"""Structured diagnostic report model for claim-first dashboards.

The raw EvalVitals artifacts are optimized for auditability: JSONL events,
candidate signal dicts, CSV tables, and rendered chart files.  This module
defines the semantic layer above those artifacts.  The dashboard should render
this model instead of re-interpreting raw logs ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ClaimStatus = Literal["supported", "inconclusive", "refuted", "descriptive"]
Confidence = Literal["high", "medium", "low", "unknown"]


@dataclass
class Evidence:
    id: str
    kind: str
    title: str
    summary: str = ""
    artifact: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "artifact": self.artifact,
        }


@dataclass
class Claim:
    id: str
    text: str
    status: ClaimStatus
    evidence_ids: list[str] = field(default_factory=list)
    counter_evidence_ids: list[str] = field(default_factory=list)
    interpretation: str = ""
    do_not_infer: str = ""
    downstream: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "evidence_ids": self.evidence_ids,
            "counter_evidence_ids": self.counter_evidence_ids,
            "interpretation": self.interpretation,
            "do_not_infer": self.do_not_infer,
            "downstream": self.downstream,
        }


@dataclass
class ReportStep:
    stage: str
    title: str
    summary: str
    artifact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "title": self.title,
            "summary": self.summary,
            "artifact_ids": self.artifact_ids,
        }


@dataclass
class DiagnosticReport:
    question: str = ""
    answer: str = ""
    confidence: Confidence = "unknown"
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    timeline: list[ReportStep] = field(default_factory=list)
    visual_decisions: list[dict[str, Any]] = field(default_factory=list)
    chart_readings: list[dict[str, Any]] = field(default_factory=list)
    dashboard_storyboard: list[dict[str, Any]] = field(default_factory=list)
    critique: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "claims": [c.to_dict() for c in self.claims],
            "evidence": [e.to_dict() for e in self.evidence],
            "timeline": [s.to_dict() for s in self.timeline],
            "visual_decisions": self.visual_decisions,
            "chart_readings": self.chart_readings,
            "dashboard_storyboard": self.dashboard_storyboard,
            "critique": self.critique,
            "caveats": self.caveats,
            "next_actions": self.next_actions,
        }


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
