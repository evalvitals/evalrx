"""M3 — DiagnosisAgent: synthesize analysis findings into falsifiable hypotheses.

The agent reads the structured :class:`~evalrx.eval_agent.analysis.AnalysisReport`
produced by M2, formats it as a prompt, and asks the judge for a typed
:class:`Proposal` through :func:`~evalrx.agent_runtime.judge_call.ask_judge`.
A second call, the adversarial critic, returns a typed :class:`Review`.

The judge model defaults to **Gemini** when ``GEMINI_API_KEY`` is set in the
environment and no explicit judge is passed.

Usage::

    # Gemini default (set GEMINI_API_KEY in env)
    agent = DiagnosisAgent()
    diag  = agent.diagnose(analysis_report)

    # Explicit judge (any Model with Capability.GENERATE)
    agent = DiagnosisAgent(judge=my_model)
    diag  = agent.diagnose(analysis_report)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from evalrx.agent_runtime.judge_call import JudgeError, ask_judge
from evalrx.analysis.plain_language import jargon_violation
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.prompts.diagnosis import _DIAGNOSE_PROMPT, _VALIDATE_PROMPT

if TYPE_CHECKING:
    from evalrx.analysis.analysis_module import AnalysisReport
    from evalrx.core.model import Judge

def _format_prior_section(prior_cycles: list[dict]) -> str:
    """Render prior cycle history as a context block for the diagnosis prompt."""
    if not prior_cycles:
        return ""
    lines = [
        "\nPrior investigation cycles — do NOT repeat these hypotheses; propose "
        "distinct or refined ones that address what remains unexplained:",
    ]
    for c in prior_cycles:
        lines.append(f"\nCycle {c['cycle']} (severity={c['severity']}):")
        for h in c.get("hypotheses", []):
            lines.append(
                f"  [{h['status'].upper()}] {h['statement']} "
                f"(failure_mode: {h['failure_mode']})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class ExploreContext:
    """Descriptive mechanism notes from the LAMBDA explorer (Step 1).

    This is **never authoritative**: it is read-only, enters ONLY the M3
    hypothesis-proposal prompt (and the dashboard), and never the M2 confirmatory
    family, M4 testing, or the fix gate. It informs *which* hypotheses M3
    proposes — not *whether* any of them is true. Every chart/observation here is
    free-form EDA on a HELD-OUT explore split and stays UNCONFIRMED until the
    downstream M4+M5+e-BH machinery tests it.

    Attributes:
        observations: Free-text EDA observations.
        charts:       Chart dicts ``{title, kind, description, figure_path}`` —
                      ``figure_path`` (when present) is attached to M3 as an image.
        caveats:      The explorer's own warnings about its findings.
        source:       Provenance tag (default ``"lambda_explorer"``).
    """

    observations: list[str] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    source: str = "lambda_explorer"

    @classmethod
    def from_report(cls, data: dict[str, Any] | None) -> "ExploreContext | None":
        """Build from an explore/fused report dict (``fused_report.json`` etc.).

        Returns ``None`` when *data* is empty or carries no descriptive content,
        so callers can pass it through unconditionally.
        """
        if not data or not isinstance(data, dict):
            return None
        observations = [str(x) for x in (data.get("observations") or [])]
        charts = [dict(c) for c in (data.get("charts") or []) if isinstance(c, dict)]
        caveats = [str(x) for x in (data.get("caveats") or [])]
        if not (observations or charts or caveats):
            return None
        return cls(
            observations=observations,
            charts=charts,
            caveats=caveats,
            source=str(data.get("source") or "lambda_explorer"),
        )

    @property
    def figure_paths(self) -> list[str]:
        return [
            str(c["figure_path"])
            for c in self.charts
            if isinstance(c, dict) and c.get("figure_path")
        ]

    @property
    def is_empty(self) -> bool:
        return not (self.observations or self.charts or self.caveats)


def _format_explore_section(ctx: "ExploreContext | None") -> str:
    """Render an :class:`ExploreContext` as a strongly-labelled, UNCONFIRMED
    block for the diagnosis prompt. Returns ``""`` when there is nothing to add."""
    if ctx is None or ctx.is_empty:
        return ""
    lines = [
        "",
        "EXPLORATORY MECHANISM NOTES (free-form EDA on a HELD-OUT explore split — "
        "DESCRIPTIVE, UNCONFIRMED; use ONLY to decide WHICH hypotheses to propose, "
        "NEVER as evidence; every claim must still be tested downstream):",
    ]
    if ctx.observations:
        lines.append("  observations:")
        for obs in ctx.observations[:12]:
            lines.append(f"    - {obs}")
    if ctx.charts:
        lines.append("  charts (attached as images when rendered):")
        for c in ctx.charts[:12]:
            title = str(c.get("title") or c.get("name") or "chart")
            desc = str(c.get("description") or c.get("kind") or "")
            tag = "" if c.get("figure_path") else " [text-only, not rendered]"
            lines.append(f"    - [{title}]{tag} {desc}")
    if ctx.caveats:
        lines.append("  caveats (the explorer's own warnings):")
        for cav in ctx.caveats[:8]:
            lines.append(f"    - {cav}")
    return "\n".join(lines) + "\n"


def _format_failure_modes_section(report: "Any | None") -> str:
    """Render a :class:`~evalrx.analysis.failure_modes.FailureModeReport`
    as a labelled block for the diagnosis prompt. Descriptive only — clustering
    groups FAIL cases by similarity, it does not itself confirm a mechanism.
    Returns ``""`` when there is nothing to add (no report, or zero clusters)."""
    context = getattr(report, "as_hypothesis_context", lambda: "")()
    if not context:
        return ""
    return "\n" + context + "\n"


def _extract_referenced(raw: str, ctx: "ExploreContext | None") -> list[str]:
    """Best-effort, DISPLAY-ONLY list of explore artifacts M3's output referenced.

    Matches chart titles/names mentioned in the judge output as a whole phrase
    (word boundaries, min length) to avoid crediting a chart just because a short
    generic title like "rate"/"loss" happens to be a substring of unrelated prose.
    Used only for provenance/dashboard display — no effect on hypothesis survival.
    """
    if ctx is None or ctx.is_empty:
        return []
    import re

    low = str(raw).lower()
    referenced: list[str] = []
    for c in ctx.charts:
        for key in ("title", "name"):
            label = str(c.get(key) or "").strip()
            # Require a reasonably specific label matched on word boundaries.
            if len(label) >= 6 and label not in referenced:
                if re.search(rf"\b{re.escape(label.lower())}\b", low):
                    referenced.append(label)
                    break
    return referenced


@dataclass
class DiagnosisResult:
    """Output of :class:`DiagnosisAgent`.

    Attributes:
        model_name:           ``repr()`` of the analysed model.
        hypotheses:           Proposed :class:`~evalrx.eval_agent.hypothesis.Hypothesis`
                              objects for M5.
        findings_summary:     The findings dict forwarded to the judge.
        raw_judge_output:     Verbatim LLM response (useful for debugging).
        referenced_charts:    Explore chart titles M3 cited (provenance only).
        explore_context_used: Whether an :class:`ExploreContext` was supplied.
        failure_modes_used:   Whether a non-empty ``FailureModeReport`` was supplied.
    """

    model_name: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    findings_summary: dict[str, Any] = field(default_factory=dict)
    raw_judge_output: str = ""
    prompt: str = ""
    referenced_charts: list[str] = field(default_factory=list)
    explore_context_used: bool = False
    failure_modes_used: bool = False
    #: The adversarial critic's verbatim verdicts (``KEEP:``/``REJECT:`` +
    #: ``REASON:`` lines) and its tally. The critic ANNOTATES hypotheses
    #: (``metadata["critic"]``, ``metadata["critic_reason"]``) and orders the
    #: kept ones first — it never removes a proposal: with M4 testing on the
    #: held-out split, the data is the arbiter, and a critic that rejects every
    #: claim used to end the run with "0 hypotheses" and no M4/M5/fix at all.
    critic_raw_output: str = ""
    critic_prompt: str = ""
    n_critic_kept: int = 0
    n_critic_rejected: int = 0
    # M3 is two separate model calls: a proposer and an adversarial reviewer.
    # The proposer's full list and the reviewer's per-hypothesis decisions are
    # kept as well, so a rejected proposal stays distinguishable from a
    # parser/model failure in the report and in Langfuse. ``review_prompt`` /
    # ``review_raw`` mirror ``critic_prompt`` / ``critic_raw_output``;
    # ``review_decisions`` (statement, decision, reason) is derived from the
    # critic annotations — decision keep | reject | unparsed |
    # review_unavailable — and the reviewer never removes a hypothesis.
    proposed_hypotheses: list[Hypothesis] = field(default_factory=list)
    review_prompt: str = ""
    review_raw: str = ""
    review_decisions: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)


logger = logging.getLogger(__name__)


class HypothesisDraft(BaseModel):
    """One hypothesis as the judge proposes it."""

    hypothesis: str = Field(min_length=10, description="one-sentence falsifiable technical claim")
    plain_statement: str = Field("", description="the same claim in one everyday sentence")
    failure_mode: str = Field(pattern=r"^[a-z][a-z0-9_]*$",
                              description="snake_case tag naming the mechanism")
    test: str = Field("", description="the evidence that verifies the claim")
    expected_association: Literal["higher_on_failures", "lower_on_failures"] | None = None


class Proposal(BaseModel):
    """M3's answer: up to three hypotheses, none when there is no problem."""

    hypotheses: list[HypothesisDraft] = Field(max_length=3)


class ReviewDecision(BaseModel):
    number: int = Field(ge=1, description="the hypothesis number in the list")
    decision: Literal["keep", "reject"]
    reason: str = Field(min_length=1)


class Review(BaseModel):
    """The critic's answer: one decision per numbered hypothesis."""

    decisions: list[ReviewDecision]


def _plain_language_problems(proposal: Proposal) -> str | None:
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
        "Some plain_statement fields fail the plain-language check. Rewrite only "
        "those, for an engineer who runs evaluations but does no statistics: no "
        "statistics terms, acronyms or symbols, not a copy of the hypothesis, and "
        "every number with what it amounts to. Keep every other field as it was.\n"
        + "\n".join(problems)
    )


def _unmeasured_test_problems(proposal: Proposal, stats_results: list[Any]) -> str | None:
    """Feedback for tests that name only evidence M2 did not measure, or ``None``.

    M4 does not judge such a hypothesis: its test is unmet until a later cycle
    measures what it names. Asking once here keeps a run from spending a
    hypothesis on a signal name the judge made up or on a case id.
    """
    from evalrx.eval_agent.stages.hypothesis_tester import (
        testable_signals,
        unmeasured_identifiers,
    )

    problems = [
        f"- hypothesis {i}'s test names {', '.join(sorted(missing)[:6])}"
        for i, d in enumerate(proposal.hypotheses, start=1)
        for missing in [unmeasured_identifiers(d.test, stats_results)]
        if missing
    ]
    if not problems:
        return None
    usable = testable_signals(stats_results)
    return (
        "Some tests name identifiers that have no result to check them against "
        "this cycle. Rewrite only those tests: name one of the signals below "
        "exactly as written, or, when none of them tests the claim, describe the "
        "evidence in plain words without identifiers or case ids. Keep every "
        "other field as it was.\n" + "\n".join(problems)
        + "\nSignals with a result: " + (", ".join(usable[:30]) or "none")
    )


def _to_hypothesis(draft: HypothesisDraft, model_name: str) -> Hypothesis:
    return Hypothesis(
        statement=draft.hypothesis,
        target_model=model_name,
        predicted_failure_mode=draft.failure_mode,
        plain_statement=draft.plain_statement,
        test_design=draft.test,
        expected_association=draft.expected_association or "",
    )


def _format_label_context(cases: "Any | None") -> str:
    """Descriptive label summary for the M3 critic (and nothing else).

    The critic used to see only the findings JSON — per-case rows with
    ``labelled_fail`` and ``extracted_answer`` but no gold — and rejected a
    correct "answers Yes regardless of the audio" hypothesis for "no
    ground-truth present/absent field in the evidence" (audiocaps 2026-08-20)
    while the proposer had read exactly that breakdown in the explore notes.
    Counts only; on a yes/no (true/false) batch a gold x answer table whose
    cells carry the FAIL count. Empty string when *cases* is ``None`` or
    carries no labels, so callers can concatenate it unconditionally.
    """
    if cases is None:
        return ""
    try:
        from evalrx.analyzers.reasoning._text import (
            binary_direction,
            binary_gold,
            extract_answer,
        )
        from evalrx.core.case import Label
    except Exception:  # pragma: no cover - defensive import guard
        return ""
    rows = [c for c in cases if getattr(c, "label", None) in (Label.PASS, Label.FAIL)]
    if not rows:
        return ""
    n_fail = sum(1 for c in rows if c.label == Label.FAIL)
    lines = [f"  labelled cases: {len(rows)} ({n_fail} FAIL / {len(rows) - n_fail} PASS)"]
    golds = [(c, binary_gold(getattr(c, "expected", None))) for c in rows]
    binary = [(c, g) for c, g in golds if g is not None]
    if len(binary) >= max(4, int(0.8 * len(rows))):
        table: dict[tuple[str, str], list[int]] = {}
        for c, g in binary:
            text = str(getattr(c, "observed", "") or "")
            a = binary_direction(extract_answer(text)) or binary_direction(text[-200:], last=True)
            key = (g, a or "unparsed")
            cell = table.setdefault(key, [0, 0])
            cell[0] += 1
            cell[1] += int(c.label == Label.FAIL)
        lines.append("  yes/no task -- gold x answer (n, of which FAIL):")
        for g in ("yes", "no"):
            for a in ("yes", "no", "unparsed"):
                n, f = table.get((g, a), [0, 0])
                if n:
                    lines.append(f"    gold={g:<3} answered={a:<8} n={n:<4} FAIL={f}")
        n_yes = sum(n for (g, a), (n, f) in table.items() if a == "yes")
        lines.append(f"  answered yes on {n_yes}/{len(binary)} binary cases")
    return "LABEL SUMMARY (descriptive counts from the case batch, not a test result):\n" + "\n".join(lines) + "\n"


def _validate_hypotheses(
    hypotheses: list[Hypothesis],
    findings_json: str,
    judge: "Judge",
    *,
    capture: "dict[str, Any] | None" = None,
    context: str = "",
) -> list[Hypothesis]:
    """Adversarial review of *hypotheses* by a second judge call.

    The same judge is asked to find reasons to *reject* each hypothesis,
    breaking the loop where the model that proposed a claim then waves it
    through.

    The critic ANNOTATES, it does not filter: every hypothesis comes back with
    ``metadata["critic"]`` set to ``"keep"`` / ``"reject"`` (``"unparsed"``
    for one the critic gave no decision for) and ``metadata["critic_reason"]``,
    kept hypotheses first. Dropping rejected ones used to end runs at "0
    hypotheses" with no M4, M5 or fix; M4 tests on held-out data, so the data
    decides and the verdict travels with the hypothesis as provenance.

    *capture*, when given, receives ``prompt``, ``raw``, ``error``,
    ``exchanges``, ``decisions``, ``n_kept`` and ``n_rejected``. When the
    critic call fails, the hypotheses come back unannotated and every
    decision reads ``review_unavailable``.

    *context* is the evidence the proposer worked from beyond the findings
    JSON, so the critic judges against what the proposer saw.
    """
    if not hypotheses:
        return hypotheses
    capture = {} if capture is None else capture
    context_section = (
        "\nContext the proposer worked from (analyst conclusion, evidence chain, "
        "statistical verdicts, exploratory notes, label summary):\n"
        + context.strip() + "\n"
    ) if context and context.strip() else ""
    prompt = _VALIDATE_PROMPT.format(
        context_section=context_section,
        findings_json=findings_json,
        hypotheses_text="\n".join(
            f"{i}. {h.statement}  (failure_mode: {h.predicted_failure_mode})"
            for i, h in enumerate(hypotheses, start=1)
        ),
    )
    capture["prompt"] = prompt
    try:
        reply = ask_judge(judge, prompt, Review, role="hypothesis_critic")
    except JudgeError as exc:
        logger.warning("DiagnosisAgent: the critic gave no usable review (%s); "
                       "keeping %d hypothesis(es) unreviewed", exc, len(hypotheses))
        capture.update(
            raw=exc.raw, error=str(exc), exchanges=exc.exchanges, n_kept=0, n_rejected=0,
            decisions=[{
                "statement": h.statement, "decision": "review_unavailable",
                "reason": "The adversarial review produced no usable answer; proposal retained.",
            } for h in hypotheses],
        )
        return hypotheses

    for d in reply.output.decisions:
        if d.number <= len(hypotheses):
            hypotheses[d.number - 1].metadata.update(critic=d.decision, critic_reason=d.reason)
    for h in hypotheses:
        h.metadata.setdefault("critic", "unparsed")
    n_kept = sum(1 for h in hypotheses if h.metadata["critic"] == "keep")
    n_rejected = sum(1 for h in hypotheses if h.metadata["critic"] == "reject")
    capture.update(raw=reply.raw, error=None, exchanges=reply.exchanges,
                   n_kept=n_kept, n_rejected=n_rejected, decisions=_critic_decisions(hypotheses))
    if n_rejected and not n_kept:
        logger.info("DiagnosisAgent: the critic rejected all %d hypothesis(es); "
                    "kept as flagged leads, M4 testing decides", len(hypotheses))
    order = {"keep": 0, "unparsed": 1, "reject": 2}
    return sorted(hypotheses, key=lambda h: order.get(h.metadata["critic"], 1))


def _critic_decisions(hypotheses: list[Hypothesis]) -> list[dict[str, Any]]:
    """Per-hypothesis review record derived from the critic's annotations: the
    ``{statement, decision, reason}`` shape the run log, report and Langfuse
    consume. The critic never removes a hypothesis, so this is a record, not a
    filter."""
    return [{
        "statement": h.statement,
        "decision": str(h.metadata.get("critic", "unparsed")),
        "reason": str(h.metadata.get("critic_reason", "")),
    } for h in hypotheses]


def _default_judge() -> "Judge":
    """Return a judge model without requiring an explicit API key.

    Resolution order:
    1. ``agy`` CLI (antigravity) — if the binary is on PATH, no key needed.
    2. Gemini — if ``GEMINI_API_KEY`` is set.
    3. Raise with instructions.
    """
    import os
    import shutil

    if shutil.which("agy"):
        from evalrx.agent_runtime.judges import AgyModel
        return AgyModel()

    if os.getenv("GEMINI_API_KEY"):
        from evalrx.models.backends.api.gemini_model import GeminiModel
        return GeminiModel()

    raise ValueError(
        "DiagnosisAgent requires a judge model. "
        "Options: install antigravity (agy) — no API key needed, "
        "or set GEMINI_API_KEY to use Gemini "
        "(install with: pip install 'evalrx[gemini]'), "
        "or pass judge= explicitly."
    )


class DiagnosisAgent:
    """Proposes hypotheses from an :class:`~evalrx.eval_agent.analysis.AnalysisReport`.

    Args:
        judge:    Any :class:`~evalrx.core.model.Model` with
                  ``Capability.GENERATE``.  Defaults to a ``GeminiModel``
                  when ``GEMINI_API_KEY`` is in the environment.
        api_key:  Gemini API key — used only when *judge* is ``None`` and no
                  ``GEMINI_API_KEY`` env var is set.
        model_id: Gemini model identifier (default: ``"gemini-2.0-flash"``).
    """

    def __init__(
        self,
        judge: "Judge | None" = None,
        api_key: str | None = None,
        model_id: str = "gemini-2.0-flash",
    ) -> None:
        if judge is None and api_key is not None:
            from evalrx.models.backends.api.gemini_model import GeminiModel

            judge = GeminiModel(model_id=model_id, api_key=api_key)
        self._judge = judge  # None → resolved lazily on first call

    @property
    def judge(self) -> "Judge":
        if self._judge is None:
            self._judge = _default_judge()
        return self._judge

    @judge.setter
    def judge(self, value: "Judge") -> None:
        self._judge = value

    def diagnose(
        self,
        analysis: "AnalysisReport",
        prior_cycles: list[dict] | None = None,
        explore_context: "ExploreContext | None" = None,
        failure_modes: "Any | None" = None,
        cases: "Any | None" = None,
    ) -> DiagnosisResult:
        """Synthesize *analysis* into a set of falsifiable hypotheses.

        Args:
            analysis:     The M2 report: an
                          :class:`~evalrx.analysis.analysis_module.AnalysisReport`
                          (``StatsAnalysisAgent`` returns a subclass).
            prior_cycles: Summary of previous M1→M5 cycles, from the caller.
                          Each entry is ``{"cycle": int, "severity": str,
                          "hypotheses": [{"statement", "failure_mode", "status"}]}``.
                          Injected into the prompt so the judge avoids re-proposing
                          already-tested hypotheses.
            failure_modes: Optional :class:`~evalrx.analysis.failure_modes.FailureModeReport`
                          (clustered FAIL cases). Descriptive only, like
                          *explore_context* — informs which hypotheses M3
                          proposes, never a claim itself. ``None`` (default)
                          adds nothing to the prompt and costs no extra call.
            cases:        The labelled case batch M1/M2 ran on. Used ONLY to
                          give the critic a label summary (PASS/FAIL counts;
                          gold x answer table on a yes/no batch) — see
                          :func:`_format_label_context`. ``None`` adds nothing.

        Returns:
            :class:`DiagnosisResult` with zero or more hypotheses.
        """
        self._model_calls: list[dict[str, Any]] = []

        summary = {name: r.findings for name, r in analysis.raw_results.items()}

        # Prefer M2's LLM conclusion + evidence chain (StatsAnalysisReport) over
        # the bare threshold narrative, so a real failure mode surfaced by the
        # analyst is not dropped just because no numeric threshold fired.
        conclusion = getattr(analysis, "conclusion", "") or analysis.narrative or "(no conclusion)"
        evidence_chain = getattr(analysis, "evidence_chain", None) or []
        evidence_section = ""
        if evidence_chain:
            evidence_section = (
                "\nEvidence chain:\n"
                + "\n".join(f"  - {step}" for step in evidence_chain)
                + "\n"
            )
        stats_results = getattr(analysis, "stats_results", None) or []
        stats_section = ""
        stats_lines = [
            f"  - {r.summary}"
            for r in stats_results
            if getattr(r, "ok", False) and getattr(r, "summary", "")
        ]
        if stats_lines:
            stats_section = "\nStatistical test results:\n" + "\n".join(stats_lines) + "\n"

        # Evidence sources M3 can reference in TEST designs: per-case signal
        # keys and strategy contrasts that already exist in this cycle.
        signals: list[str] = []
        for r in stats_results:
            cfg = getattr(r, "config", None) or {}
            sig = cfg.get("signal")
            if sig and sig not in signals:
                signals.append(sig)
            for s in cfg.get("strategies", []) or []:
                tag = f"strategy contrast vs '{s}'"
                if s != "baseline" and tag not in signals:
                    signals.append(tag)
        available_signals_section = ""
        if signals:
            available_signals_section = (
                "AVAILABLE EVIDENCE (per-case signals / contrasts already measured):\n"
                + "\n".join(f"  - {s}" for s in signals)
                + "\n\n"
            )

        protocol = getattr(analysis, "protocol", None)
        protocol_section = (
            protocol.prompt_text() if protocol is not None
            else "(not supplied; infer cautiously from the stored findings)"
        )

        prompt = _DIAGNOSE_PROMPT.format(
            prior_section=_format_prior_section(prior_cycles or []),
            model_name=analysis.model_name,
            severity=analysis.severity,
            protocol_section=protocol_section,
            conclusion=conclusion,
            evidence_section=evidence_section,
            stats_section=stats_section,
            explore_section=_format_explore_section(explore_context),
            failure_modes_section=_format_failure_modes_section(failure_modes),
            available_signals_section=available_signals_section,
            findings_json=json.dumps(summary, indent=2, default=str),
        )
        from pathlib import Path as _Path
        # M2's confirmatory figures, then the explorer's (UNCONFIRMED) charts.
        # The explore charts are attached to the M3 prompt ONLY — they never reach
        # M2's confirmatory family, M4, or the fix gate.
        _figs = [_Path(f) for f in getattr(analysis, "figures", []) if _Path(f).exists()]
        if explore_context is not None:
            _figs += [_Path(f) for f in explore_context.figure_paths if _Path(f).exists()]
        # A failed or unreadable proposal raises JudgeError with every exchange;
        # the loop logs them and retries M3 or stops, instead of this stage
        # standing in a guess for the judge's answer.
        def _problems(proposal: Proposal) -> str | None:
            found = [text for text in (
                _plain_language_problems(proposal),
                _unmeasured_test_problems(proposal, stats_results) if signals else None,
            ) if text]
            return "\n\n".join(found) or None

        reply = ask_judge(
            self.judge, prompt, Proposal, role="diagnosis_judge",
            images=_figs, check=_problems,
        )
        self._model_calls.extend(e.record() for e in reply.exchanges)
        raw = reply.raw
        proposed_hypotheses = [_to_hypothesis(d, analysis.model_name) for d in reply.output.hypotheses]
        hypotheses = list(proposed_hypotheses)

        # Adversarial validation: run a second critic call at temperature=0 to
        # prune hypotheses the generator produced without sufficient evidence.
        # This prevents the self-evaluation loop where the same model that
        # proposed a hypothesis then approves it uncritically.
        critic: dict[str, Any] = {}
        if hypotheses:
            findings_json_str = json.dumps(summary, indent=2, default=str)
            # The critic reviews against what the proposer saw, not less:
            # conclusion + evidence chain + stats verdicts + explore notes,
            # plus a label summary the proposer never had either.
            critic_context = "\n".join(part.strip("\n") for part in (
                f"Analysis conclusion: {conclusion}",
                f"Experiment protocol:\n{protocol_section}",
                evidence_section,
                stats_section,
                _format_explore_section(explore_context),
                _format_label_context(cases),
            ) if part and part.strip())
            hypotheses = _validate_hypotheses(
                hypotheses, findings_json_str, self.judge, capture=critic,
                context=critic_context,
            )
            self._model_calls.extend(e.record() for e in critic.get("exchanges", ()))

        # The judge saw no problem, but M2 has medium/high findings: one
        # hypothesis per finding so M4 still tests them. This guards against
        # self-diagnosis bias when the judge is the model under test.
        if not hypotheses and analysis.findings:
            from evalrx.analysis.analysis_module import _SEVERITY_ORDER

            for finding in analysis.findings:
                if _SEVERITY_ORDER.get(finding.severity, 0) >= 2:  # medium or high
                    hypotheses.append(Hypothesis(
                        statement=(
                            f"The model {finding.message} "
                            f"({finding.analyzer}.{finding.metric}={finding.value:.3g})"
                        ),
                        target_model=analysis.model_name,
                        predicted_failure_mode=finding.analyzer,
                    ))

        return DiagnosisResult(
            model_name=analysis.model_name,
            hypotheses=hypotheses,
            findings_summary=summary,
            raw_judge_output=str(raw),
            prompt=prompt,
            referenced_charts=_extract_referenced(str(raw), explore_context),
            explore_context_used=bool(explore_context is not None and not explore_context.is_empty),
            failure_modes_used=bool(getattr(failure_modes, "clusters", None)),
            critic_raw_output=str(critic.get("raw", "") or ""),
            critic_prompt=str(critic.get("prompt", "") or ""),
            n_critic_kept=int(critic.get("n_kept", 0) or 0),
            n_critic_rejected=int(critic.get("n_rejected", 0) or 0),
            # The same review in the proposer/reviewer record shape: the full
            # proposal list and a per-hypothesis decision derived from the
            # critic annotations (the critic never removes a hypothesis).
            proposed_hypotheses=proposed_hypotheses,
            review_prompt=str(critic.get("prompt", "") or ""),
            review_raw=str(critic.get("raw", "") or ""),
            review_decisions=list(critic.get("decisions") or []),
            model_calls=list(self._model_calls),
        )
