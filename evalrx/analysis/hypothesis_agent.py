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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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


class ExperimentPlan(BaseModel):
    """An experiment the existing table cannot stand in for, in four parts.

    A hypothesis about a cause usually cannot be settled by re-counting rows;
    it needs something changed and re-run. Free text hides whether a plan is
    complete, so the four parts are separate: a plan with no stated way to
    fail is visible as such.
    """

    data_needed: str = Field("", description="rows or inputs to collect or re-run")
    operation: str = Field("", description="what is changed or applied to them")
    expected_result: str = Field("", description="what is seen if the hypothesis is right")
    falsified_by: str = Field("", description="what result would show it is wrong")

    def is_empty(self) -> bool:
        return not any((self.data_needed, self.operation, self.expected_result,
                        self.falsified_by))


class HeldOutPrediction(BaseModel):
    """A numeric prediction about one candidate signal's held-out effect.

    The host checks it directly against the held-out rows, so the part of a
    hypothesis that is a number is not left to a model to grade.
    """

    signal: str = Field(description="name of a candidate signal that has a recipe")
    relation: Literal["nonzero", ">", "<", "abs<"] = "nonzero"
    value: float = 0.0


@dataclass
class Hypothesis:
    """One M3-proposed, falsifiable candidate explanation. Proposed only —
    ``test_design`` names how it *could* be checked, not a verdict.

    ``predictions`` are the parts of the claim the held-out rows can decide
    (checked by the host); ``experiment`` is the plan for the part they
    cannot (an intervention), or None when the hypothesis needs none.
    """

    statement: str = ""
    plain_statement: str = ""
    basis: str = ""
    test_design: str = ""
    predictions: list[dict[str, Any]] = field(default_factory=list)
    experiment: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "statement": self.statement,
            "plain_statement": self.plain_statement,
            "basis": self.basis,
            "test_design": self.test_design,
        }
        # Optional parts appear only when stated, so a hypothesis without them
        # serialises exactly as it did before they existed.
        if self.predictions:
            out["predictions"] = self.predictions
        if self.experiment:
            out["experiment"] = self.experiment
        return out


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
        signals_text = "\n".join(_signal_line(s) for s in signals) or "(none recorded)"

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
        known = {str(s.get("name")) for s in signals if isinstance(s.get("recipe"), dict)}
        return [
            Hypothesis(
                statement=d.hypothesis, plain_statement=d.plain_statement,
                basis=d.basis, test_design=d.test,
                # A prediction about a signal with no recipe cannot be checked
                # on held-out rows, so it is not carried forward as one.
                predictions=[p.model_dump() for p in d.predictions if p.signal in known],
                experiment=(None if d.experiment is None or d.experiment.is_empty()
                            else d.experiment.model_dump()),
            )
            for d in reply.output.hypotheses
        ]

    def _backend(self) -> "Any | None":
        if self._cli_config is not None and self._cli_config.provider != "llm":
            from evalrx.agent_runtime.judges.cli_agent import CliAgentJudge

            return CliAgentJudge(self._cli_config, timeout_sec=self._timeout_sec)
        return self._judge


def _signal_line(signal: dict[str, Any]) -> str:
    """One prompt line per candidate signal; names the recipe when there is one,
    since only a signal with a recipe can carry a held-out prediction."""
    label = signal.get("display_name") or signal.get("name")
    line = f"- {label}: {signal.get('rationale', '')}"
    recipe = signal.get("recipe")
    if isinstance(recipe, dict) and recipe.get("expr"):
        line += f" [signal name: {signal.get('name')}; recipe: {recipe.get('expr')}]"
    return line


class _Draft(BaseModel):
    hypothesis: str = Field(min_length=10, description="one falsifiable technical sentence")
    plain_statement: str = Field("", description="the same claim in one everyday sentence")
    basis: str = Field("", description="the takeaways or signals it rests on")
    test: str = Field("", description="the evidence that would confirm or refute it")
    predictions: list[HeldOutPrediction] = Field(
        default_factory=list, max_length=4,
        description="numeric predictions about named candidate signals on held-out rows",
    )
    experiment: ExperimentPlan | None = Field(
        None, description="the experiment needed when held-out rows cannot decide the claim",
    )


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
