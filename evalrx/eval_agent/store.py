"""Store — the persistent memory that makes self-evolution possible.

Without memory, an agent re-explores the same ground forever. The store
accumulates the corpus the agent evolves over: cases it has constructed,
results it has produced, and hypotheses it has tested. The loop writes to it
each cycle and reads from it to decide what to try next.

This module defines the interface, an in-memory reference, and a JSONL-backed
persistent implementation (JsonlStore).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.case import FailureCase
    from evalrx.core.result import Result
    from evalrx.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)


class Store(ABC):
    """Persistent memory interface for the self-evolving loop."""

    @abstractmethod
    def add_case(self, case: "FailureCase") -> None: ...

    @abstractmethod
    def add_result(self, result: "Result") -> None: ...

    @abstractmethod
    def add_hypothesis(self, hypothesis: "Hypothesis") -> None: ...

    @abstractmethod
    def query(self, **filters: Any) -> list[Any]:
        """Retrieve stored items matching *filters* (tags, label, status, …)."""

    @abstractmethod
    def summarize(self) -> dict[str, Any]:
        """Aggregate view the agent reads to spot gaps and patterns."""


class InMemoryStore(Store):
    """Minimal non-persistent reference store (Stage-2 will add durability)."""

    def __init__(self) -> None:
        self.cases: list[FailureCase] = []
        self.results: list[Result] = []
        self.hypotheses: list[Hypothesis] = []

    def add_case(self, case: "FailureCase") -> None:
        self.cases.append(case)

    def add_result(self, result: "Result") -> None:
        self.results.append(result)

    def add_hypothesis(self, hypothesis: "Hypothesis") -> None:
        self.hypotheses.append(hypothesis)

    def query(self, kind: str | None = None, **filters: Any) -> list[Any]:
        """Retrieve stored items. ``kind`` in {cases, results, hypotheses}; filters:
        ``label`` / ``tags`` (cases), ``status`` (hypotheses), ``analyzer`` (results)."""
        out: list[Any] = []
        if kind in (None, "cases"):
            for c in self.cases:
                if filters.get("label") is not None and getattr(c, "label", None) != filters["label"]:
                    continue
                tags = filters.get("tags")
                if tags is not None and not set(tags).issubset(getattr(c, "tags", set())):
                    continue
                out.append(c)
        if kind in (None, "results"):
            for r in self.results:
                if filters.get("analyzer") is not None and getattr(r, "analyzer", None) != filters["analyzer"]:
                    continue
                out.append(r)
        if kind in (None, "hypotheses"):
            for h in self.hypotheses:
                if filters.get("status") is not None and getattr(h, "status", None) != filters["status"]:
                    continue
                out.append(h)
        return out

    def summarize(self) -> dict[str, Any]:
        return {
            "n_cases": len(self.cases),
            "n_results": len(self.results),
            "n_hypotheses": len(self.hypotheses),
        }


class JsonlStore(Store):
    """Persistent JSONL-backed store.

    Files written under *store_dir*:
      - ``hypotheses.jsonl`` — one serialised :class:`Hypothesis` per line
      - ``results.jsonl``    — one scalar-only result summary per line
      - ``cases.jsonl``      — one case summary per line

    Hypotheses survive process restart; cases and results store lightweight
    summaries only (no numpy arrays or heavy objects).

    Thread safety: appends use ``O_APPEND`` which is atomic on Linux for
    small writes; reads reload from disk each call.
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hyp_path = self._dir / "hypotheses.jsonl"
        self._res_path = self._dir / "results.jsonl"
        self._case_path = self._dir / "cases.jsonl"

    # ------------------------------------------------------------------
    # Store interface
    # ------------------------------------------------------------------

    def add_case(self, case: "FailureCase") -> None:
        record: dict[str, Any] = {
            "id": getattr(case, "id", ""),
            "label": getattr(case, "label", None),
            "tags": list(getattr(case, "tags", [])),
            "prompt_snippet": str(getattr(getattr(case, "inputs", None), "prompt", ""))[:200],
        }
        self._append(self._case_path, record)

    def add_result(self, result: "Result") -> None:
        findings = getattr(result, "findings", {}) or {}
        scalar_findings: dict[str, Any] = {}
        for k, v in findings.items():
            try:
                scalar_findings[k] = float(v) if hasattr(v, "__float__") else str(v)
            except Exception:
                scalar_findings[k] = str(v)
        record: dict[str, Any] = {
            "analyzer": getattr(result, "analyzer", ""),
            "findings": scalar_findings,
            "metadata": {
                k: str(v) for k, v in (getattr(result, "metadata", {}) or {}).items()
                if isinstance(k, str)
            },
        }
        self._append(self._res_path, record)

    def add_hypothesis(self, hypothesis: "Hypothesis") -> None:
        from evalrx.eval_agent.hypothesis import hypothesis_to_dict
        self._append(self._hyp_path, hypothesis_to_dict(hypothesis))

    def query(self, kind: str | None = None, **filters: Any) -> list[Any]:
        """Retrieve stored items matching *filters*.

        Returns reconstructed :class:`Hypothesis` objects for ``kind="hypotheses"``;
        raw dicts for cases and results.
        """
        out: list[Any] = []

        if kind in (None, "cases"):
            for rec in self._load(self._case_path):
                if filters.get("label") is not None and rec.get("label") != filters["label"]:
                    continue
                tags = filters.get("tags")
                if tags is not None and not set(tags).issubset(set(rec.get("tags", []))):
                    continue
                out.append(rec)

        if kind in (None, "results"):
            for rec in self._load(self._res_path):
                if filters.get("analyzer") is not None and rec.get("analyzer") != filters["analyzer"]:
                    continue
                out.append(rec)

        if kind in (None, "hypotheses"):
            from evalrx.eval_agent.hypothesis import hypothesis_from_dict
            for rec in self._load(self._hyp_path):
                h = hypothesis_from_dict(rec)
                if filters.get("status") is not None:
                    wanted = filters["status"]
                    # Accept both enum and string comparisons
                    h_status = h.status.value if h.status else None
                    if wanted != h.status and wanted != h_status:
                        continue
                out.append(h)

        return out

    def summarize(self) -> dict[str, Any]:
        n_hyp = sum(1 for _ in self._load(self._hyp_path))
        n_res = sum(1 for _ in self._load(self._res_path))
        n_cas = sum(1 for _ in self._load(self._case_path))
        return {
            "n_cases": n_cas,
            "n_results": n_res,
            "n_hypotheses": n_hyp,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.warning("JsonlStore: failed to append to %s: %s", path, exc)

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.debug("JsonlStore: skipping malformed line in %s", path)
        except OSError as exc:
            logger.warning("JsonlStore: failed to read %s: %s", path, exc)
        return records
