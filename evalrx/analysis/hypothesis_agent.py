"""M3 — HypothesisAgent: propose falsifiable hypotheses from an exploratory
analysis report (an ``ExploratoryAnalysisAgent`` / M2 output).

This is a lightweight, standalone counterpart to the diagnosis loop's
``DiagnosisAgent`` (``evalrx.eval_agent.stages.diagnosis``): that one is
hard-bound to the loop's own ``AnalysisReport`` type and only accepts a judge
``Model`` (Gemini by default) — it cannot read an ``ExploratoryAnalysisReport``
without raising or silently producing nothing (there is no adapter between the
two report shapes). This module reads the standalone explorer's own report
directly and supports the same ``judge``/``cli_config`` backend flexibility as
``ExploratoryAnalysisAgent``, so it can run with whatever ``--backend`` the
``evalrx explore`` CLI was given.

Proposal only — no validation. A hypothesis here is a candidate explanation to
investigate further, not a conclusion; there is no confirm/test phase wired up
for it (see ``StatsAnalysisAgent``/``HypothesisTester`` in the diagnosis loop
for that, a separate, currently out-of-scope system).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from evalrx.agent_runtime.judge_call import JudgeError, ask_judge
from evalrx.analysis.plain_language import jargon_violation
from evalrx.analysis.prompts.hypothesis_agent import (
    AGENT_COLUMN_MARKERS as _AGENT_COLUMN_MARKERS,
)
from evalrx.analysis.prompts.hypothesis_agent import (
    AGENT_TRAJECTORY_HINT as _AGENT_TRAJECTORY_HINT,
)
from evalrx.analysis.prompts.hypothesis_agent import PROPOSE_PROMPT as _PROPOSE_PROMPT


def _is_agent_report(report: dict) -> bool:
    """True when the exploratory report shows agent-trajectory column families.

    Detection scans the question, takeaway titles/analysis, and candidate
    signal names/rationales for the marker substrings — the same places the
    propose prompt quotes, so the hint fires exactly when the model will see
    trajectory columns.
    """
    pieces: list[str] = [str(report.get("question") or "")]
    for t in report.get("takeaways") or []:
        if isinstance(t, dict):
            pieces.append(str(t.get("title", "")))
            pieces.append(str(t.get("analysis", "")))
    for s in report.get("candidate_signals") or []:
        if isinstance(s, dict):
            pieces.append(str(s.get("name", "")))
            pieces.append(str(s.get("display_name", "")))
            pieces.append(str(s.get("rationale", "")))
    text = " ".join(pieces)
    return any(marker in text for marker in _AGENT_COLUMN_MARKERS)

if TYPE_CHECKING:
    from evalrx.agent_runtime.cli_types import CliAgentConfig
    from evalrx.core.model import Judge

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """One M3-proposed, falsifiable candidate explanation. Proposed only —
    ``test_design`` names how it *could* be checked, not a verdict."""

    statement: str = ""
    plain_statement: str = ""
    basis: str = ""
    test_design: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "plain_statement": self.plain_statement,
            "basis": self.basis,
            "test_design": self.test_design,
        }


class HypothesisAgent:
    """M3: propose hypotheses from an ``ExploratoryAnalysisReport`` (as a dict).

    Args:
        judge:      LLM-like object with ``generate(prompt) -> str``.
        cli_config: Optional CLI coding-agent backend (same as
                    ``ExploratoryAnalysisAgent``'s ``cli_config``); used
                    instead of ``judge`` when its provider isn't ``"llm"``.
        timeout_sec: Timeout for a single generation call.
    """

    def __init__(
        self,
        judge: "Judge | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        timeout_sec: int = 90,
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._timeout_sec = timeout_sec

    @property
    def available(self) -> bool:
        return self._backend() is not None

    def propose(self, report: dict[str, Any], *, max_items: int = 8) -> list[Hypothesis]:
        """Read *report* (an ``ExploratoryAnalysisReport.to_dict()``-shaped
        dict) and return 0-3 proposed hypotheses. Never raises — a backend
        failure, or no valid answer after one retry, yields no hypotheses."""
        judge = self._backend()
        if judge is None:
            return []

        takeaways = [t for t in report.get("takeaways") or [] if isinstance(t, dict)][:max_items]
        takeaways_text = "\n".join(
            f"- {t.get('title', '')}: {t.get('analysis', '')}" for t in takeaways
        ) or "(none recorded)"
        observations = [str(o) for o in report.get("observations") or []][:max_items]
        observations_text = "\n".join(f"- {o}" for o in observations) or "(none recorded)"
        signals = [s for s in report.get("candidate_signals") or [] if isinstance(s, dict)][:max_items]
        signals_text = "\n".join(
            f"- {s.get('display_name') or s.get('name')}: {s.get('rationale', '')}" for s in signals
        ) or "(none recorded)"

        if not takeaways and not observations and not signals:
            return []

        prompt = _PROPOSE_PROMPT.format(
            question=report.get("question") or "Explore this dataset.",
            takeaways_text=takeaways_text,
            observations_text=observations_text,
            signals_text=signals_text,
        )
        if _is_agent_report(report):
            prompt += "\n\n" + _AGENT_TRAJECTORY_HINT
        try:
            reply = ask_judge(judge, prompt, _Proposal, role="explore_hypotheses",
                              check=_plain_language_problems)
        except JudgeError as exc:
            logger.warning("HypothesisAgent: no hypotheses: %s", exc)
            return []
        return [
            Hypothesis(statement=d.hypothesis, plain_statement=d.plain_statement,
                       basis=d.basis, test_design=d.test)
            for d in reply.output.hypotheses
        ]

    def _backend(self) -> "Any | None":
        if self._cli_config is not None and self._cli_config.provider != "llm":
            from evalrx.agent_runtime.judges.cli_agent import CliAgentJudge

            return CliAgentJudge(self._cli_config, timeout_sec=self._timeout_sec)
        return self._judge


class _Draft(BaseModel):
    hypothesis: str = Field(min_length=10, description="one falsifiable technical sentence")
    plain_statement: str = Field("", description="the same claim in one everyday sentence")
    basis: str = Field("", description="the takeaways or signals it rests on")
    test: str = Field("", description="the evidence that would confirm or refute it")


class _Proposal(BaseModel):
    hypotheses: list[_Draft] = Field(max_length=3)


def _plain_language_problems(proposal: _Proposal) -> str | None:
    """Feedback for plain statements that fail the jargon check, or ``None``."""
    problems = [
        f"- hypothesis {i}'s plain_statement {reason}: {d.plain_statement!r}"
        for i, d in enumerate(proposal.hypotheses, start=1)
        for reason in [jargon_violation(d.plain_statement, d.hypothesis)]
        if reason
    ]
    if not problems:
        return None
    return (
        "Some plain_statement fields fail the plain-language check: each must be a "
        "jargon-free, one-sentence, everyday paraphrase of its hypothesis, with no "
        "statistics terms, acronyms or symbols. Rewrite only those and keep every "
        "other field as it was.\n" + "\n".join(problems)
    )
