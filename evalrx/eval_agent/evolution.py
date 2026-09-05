"""EvolutionStore — cross-run lesson accumulation with time-decay weighting.

Mirrors ``researchclaw/evolution.py`` adapted to the evalrx M1→M5 pipeline.

The store is JSONL-backed (append-only) so it accumulates lessons across many
diagnosis runs.  When building prompt overlays for the LLM agents, lessons are
weighted by a 30-day half-life exponential decay — older insights matter less.

Usage::

    store = EvolutionStore(run_dir / "evolution")
    lessons = extract_lessons(report)
    store.append_many(lessons)

    # Inject relevant context into M3 DiagnosisAgent prompt
    overlay = store.build_overlay("diagnosis", max_lessons=5)
"""

from __future__ import annotations

import json
import logging
import math
import re as _re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword tokenisation for relevance scoring
# ---------------------------------------------------------------------------

_TOKEN_RE = _re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> frozenset[str]:
    """Lowercase + split on non-alphanumeric characters; return a frozenset of tokens."""
    return frozenset(_TOKEN_RE.findall(text.lower()))

# Half-life for time-decay weighting (days)
HALF_LIFE_DAYS: float = 30.0
MAX_AGE_DAYS: float = 90.0

# Valid category tags matching the M1–M5 pipeline stages
CATEGORIES = frozenset({"probe", "analysis", "diagnosis", "surgery"})
SEVERITIES = frozenset({"info", "warning", "error"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LessonEntry:
    """A single lesson extracted from a diagnosis run.

    Attributes:
        run_id:      Identifier of the run that generated this lesson.
        cycle:       M1→M5 cycle number within that run.
        category:    Pipeline stage: ``"probe" | "analysis" | "diagnosis" | "surgery"``.
        severity:    ``"info" | "warning" | "error"``.
        description: Human-readable lesson text (injected into prompts).
        timestamp:   ISO 8601 UTC timestamp.
    """

    run_id: str
    cycle: int
    category: str
    severity: str
    description: str
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LessonEntry:
        return cls(
            run_id=str(data.get("run_id", "")),
            cycle=int(data.get("cycle", 0)),
            category=str(data.get("category", "surgery")),
            severity=str(data.get("severity", "info")),
            description=str(data.get("description", "")),
            timestamp=str(data.get("timestamp", _utcnow_iso())),
        )


# ---------------------------------------------------------------------------
# Time-decay weighting
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _time_weight(timestamp_iso: str) -> float:
    """30-day half-life exponential decay; returns 0.0 after 90 days."""
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return 0.5  # unknown age — treat as moderate recency

    if age_days > MAX_AGE_DAYS:
        return 0.0
    return math.exp(-age_days * math.log(2) / HALF_LIFE_DAYS)


# ---------------------------------------------------------------------------
# Lesson extraction
# ---------------------------------------------------------------------------


def extract_lessons(report: "AutoDiagnoseReport") -> list[LessonEntry]:
    """Auto-extract lessons from a completed :class:`AutoDiagnoseReport`.

    Detects:
    - INCONCLUSIVE hypotheses → surgery/warning
    - Loop unresolved after exhausting max_cycles → diagnosis/warning
    - HIGH or CRITICAL analysis severity with no resolution → analysis/info
    """
    from evalrx.eval_agent.hypothesis import HypothesisStatus

    lessons: list[LessonEntry] = []
    run_id = getattr(report, "_run_id", "unknown")
    cycles = getattr(report, "cycles", 0)
    ts = _utcnow_iso()

    # INCONCLUSIVE hypotheses
    for h in getattr(report, "final_hypotheses", []):
        if getattr(h, "status", None) == HypothesisStatus.INCONCLUSIVE:
            lessons.append(LessonEntry(
                run_id=run_id,
                cycle=cycles,
                category="surgery",
                severity="warning",
                description=(
                    f"Hypothesis INCONCLUSIVE: '{h.statement[:120]}' "
                    f"(failure_mode={h.predicted_failure_mode}). "
                    "Consider a more targeted experiment or more failure cases."
                ),
                timestamp=ts,
            ))

    # Loop exhausted without resolution
    if not getattr(report, "resolved", False):
        lessons.append(LessonEntry(
            run_id=run_id,
            cycle=cycles,
            category="diagnosis",
            severity="warning",
            description=(
                f"Loop exhausted {cycles} cycle(s) without resolving the failure. "
                "Hypotheses may need broader scope or additional analyzer coverage."
            ),
            timestamp=ts,
        ))

    # High-severity analysis finding without resolution
    analysis = getattr(report, "final_analysis", None)
    if analysis is not None:
        severity = getattr(analysis, "severity", "")
        if severity in ("HIGH", "CRITICAL") and not getattr(report, "resolved", False):
            lessons.append(LessonEntry(
                run_id=run_id,
                cycle=cycles,
                category="analysis",
                severity="info",
                description=(
                    f"Analysis severity={severity} but no hypothesis was confirmed. "
                    "The analyzer findings may point to a failure mode not yet hypothesised."
                ),
                timestamp=ts,
            ))

    return lessons


# ---------------------------------------------------------------------------
# EvolutionStore
# ---------------------------------------------------------------------------


class EvolutionStore:
    """JSONL-backed store for diagnosis lessons with 30-day half-life time decay.

    Args:
        store_dir: Directory where ``lessons.jsonl`` is written.
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lessons_path = self._dir / "lessons.jsonl"

    @property
    def lessons_path(self) -> Path:
        return self._lessons_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, lesson: LessonEntry) -> None:
        """Append a single lesson to the store."""
        try:
            with self._lessons_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(lesson.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.warning("EvolutionStore: failed to append lesson: %s", exc)

    def append_many(self, lessons: list[LessonEntry]) -> None:
        """Append multiple lessons in one write."""
        if not lessons:
            return
        try:
            with self._lessons_path.open("a", encoding="utf-8") as f:
                for lesson in lessons:
                    f.write(json.dumps(lesson.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.warning("EvolutionStore: failed to append lessons: %s", exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self) -> list[LessonEntry]:
        """Load all lessons from disk."""
        if not self._lessons_path.exists():
            return []
        entries: list[LessonEntry] = []
        try:
            with self._lessons_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(LessonEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        logger.debug("EvolutionStore: skipping malformed line")
        except OSError as exc:
            logger.warning("EvolutionStore: failed to read lessons: %s", exc)
        return entries

    def query_for_category(
        self,
        category: str,
        *,
        max_lessons: int = 5,
        context_keywords: list[str] | None = None,
    ) -> list[LessonEntry]:
        """Return the most relevant lessons for *category*.

        Scoring combines three signals so that a highly relevant older lesson
        can rank above a recent but unrelated one:

        1. **Recency** — 30-day exponential half-life.
        2. **Keyword relevance** — Jaccard-like token overlap between the lesson
           text and the query context supplied via *context_keywords* (e.g. the
           current failure modes, analysis narrative snippets).
        3. **Severity boost** — errors outrank warnings which outrank info.

        *context_keywords*: additional terms from the current run context.
        The query always includes the category name itself as a token.
        """
        all_lessons = self.load_all()
        relevant = [lesson for lesson in all_lessons if lesson.category == category]
        if not relevant:
            return []

        # Build query token set from category + caller-supplied keywords
        query_tokens = _tokenise(category)
        for kw in (context_keywords or []):
            query_tokens |= _tokenise(kw)

        scored: list[tuple[float, LessonEntry]] = []
        for lesson in relevant:
            weight = _time_weight(lesson.timestamp)
            if weight <= 0.0:
                continue

            # Keyword relevance: overlap between query and lesson text
            lesson_tokens = _tokenise(lesson.description)
            overlap = len(query_tokens & lesson_tokens)
            if overlap > 0:
                # Cap boost at 3× so recency still matters
                weight *= 1.0 + min(overlap * 0.4, 2.0)

            # Severity boost
            if lesson.severity == "error":
                weight *= 1.5
            elif lesson.severity == "warning":
                weight *= 1.2

            scored.append((weight, lesson))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_lessons]]

    def build_overlay(
        self,
        category: str,
        *,
        max_lessons: int = 5,
        context_keywords: list[str] | None = None,
        caller_run_id: str = "",
    ) -> str:
        """Generate a prompt overlay string for a given M1–M5 *category*.

        Returns an empty string when no relevant lessons qualify.
        The overlay is formatted for direct injection into LLM prompts.

        Side-effect: each call is appended to ``injection_log.jsonl`` so
        there is an immutable audit trail of which lessons were injected
        into which runs.  This makes it possible to bisect a regression
        back to the specific lesson that changed agent behaviour.

        Args:
            context_keywords: Extra terms from the current run used to boost
                relevance scoring (failure mode tags, narrative snippets, etc.).
            caller_run_id:    Identifier of the run requesting the overlay —
                used to link the injection record back to the loop run.
        """
        lessons = self.query_for_category(
            category, max_lessons=max_lessons, context_keywords=context_keywords
        )

        self._record_injection(category, lessons, caller_run_id)

        if not lessons:
            return ""

        lines = ["## Lessons from Prior Diagnosis Runs"]
        for i, lesson in enumerate(lessons, 1):
            icon = {"error": "[ERROR]", "warning": "[WARN]", "info": "[INFO]"}.get(
                lesson.severity, "[INFO]"
            )
            lines.append(f"{i}. {icon} {lesson.description}")
        return "\n".join(lines)

    def _record_injection(
        self,
        category: str,
        lessons: list[LessonEntry],
        caller_run_id: str,
    ) -> None:
        """Append one record to ``injection_log.jsonl`` (non-blocking).

        Each record captures which lessons were injected, when, into which
        run, and for which category.  The log is append-only so it acts as
        a provenance chain — a given mutation in agent behaviour can be
        traced back to the exact lesson that was active at the time.
        """
        log_path = self._dir / "injection_log.jsonl"
        record = {
            "ts": _utcnow_iso(),
            "caller_run_id": caller_run_id,
            "category": category,
            "n_lessons": len(lessons),
            "lessons": [
                {
                    "run_id": lesson.run_id,
                    "cycle": lesson.cycle,
                    "severity": lesson.severity,
                    "description": lesson.description[:120],
                    "timestamp": lesson.timestamp,
                }
                for lesson in lessons
            ],
        }
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.debug("EvolutionStore: could not write injection log: %s", exc)

    def count(self) -> int:
        """Return total number of lessons stored."""
        if not self._lessons_path.exists():
            return 0
        try:
            with self._lessons_path.open(encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0
