"""M2 — StatsAnalysisAgent: statistically analyze probe results.

Extends :class:`~evalvitals.analysis.analysis_module.AnalysisModule` with:

- **Protocol-aware interpretation**: focuses the analysis narrative on what
  the user's experiment was actually testing.
- **Richer output** (:class:`StatsAnalysisReport`): adds ``conclusion``,
  ``evidence_chain``, and ``qualitative_findings`` on top of the base
  :class:`~evalvitals.analysis.analysis_module.AnalysisReport` fields so M3
  (DiagnosisAgent) receives richer context.
- **LLM-guided path** (``judge=`` set): the LLM synthesizes a protocol-specific
  narrative from the raw findings.  The threshold-rules pass always runs first;
  the LLM *enriches*, not replaces, its output.

Two analysis paths:

a. **Basic** (default, no judge): apply built-in threshold rules + build
   conclusion/evidence_chain from the flagged anomalies.  Fast, deterministic,
   no API key needed.

b. **LLM-guided** (``judge=`` + ``protocol=`` at call time): judge reads the
   protocol + raw findings and writes a protocol-specific conclusion.
   Falls back to the basic path if the LLM call fails.

:class:`StatsAnalysisReport` inherits from :class:`AnalysisReport` so it is
accepted everywhere an ``AnalysisReport`` is expected (e.g. ``DiagnosisAgent``).

Usage::

    # Basic path
    agent  = StatsAnalysisAgent()
    report = agent.analyze(probe_results, model_name="qwen3-vl-8b")

    # Protocol-guided (richer conclusion)
    agent  = StatsAnalysisAgent(judge=gemini)
    report = agent.analyze(probe_results, protocol=protocol)
    print(report.conclusion)
    print(report.evidence_chain)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.analysis.analysis_module import AnalysisModule, AnalysisReport
from evalvitals.analysis.prompts.stats_agent import _ANALYSIS_PROMPT, _TOOL_SELECT_PROMPT
from evalvitals.analysis.stats_tool_agent import StatsToolAgent as LegacyStatsToolAgent
from evalvitals.analysis.stats_tools import (
    STATS_TOOL_CATALOG,
    StatsInput,
    StatsToolResult,
    build_stats_input,
    build_stats_input_from_records,
    default_plan,
    describe_data,
    fdr_correct,
    has_testable_data,
    plot_effects,
    run_stats_tool,
)

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig
    from evalvitals.analysis.stats_tool_generator import (
        GeneratedStatsTool,
        StatsToolGenerator,
    )
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

logger = logging.getLogger(__name__)


def _select_tools_via_llm(
    judge: "Model",
    protocol: "ExperimentProtocol",
    inp: StatsInput,
    valid_names: set[str],
) -> set[str]:
    """Ask the judge to choose tool names from the catalog. Returns a name set.

    Raises on any failure so the caller can fall back to the deterministic plan.
    """
    import inspect

    catalog = "\n".join(f"  - {name}: {desc}" for name, desc in STATS_TOOL_CATALOG.items())
    prompt = _TOOL_SELECT_PROMPT.format(
        protocol_text=protocol.prompt_text(),
        tool_catalog=catalog,
        data_shape=json.dumps(describe_data(inp), indent=2),
    )
    sig = inspect.signature(judge.generate)
    if "temperature" in sig.parameters:
        raw = judge.generate(prompt, temperature=0)
    else:
        raw = judge.generate(prompt)

    cleaned = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL).strip()
    match = re.search(r"\{[^{}]*\}", cleaned)
    if match:
        data = json.loads(match.group())
        names = {n.lower() for n in data.get("tools", []) if n.lower() in valid_names}
        if names:
            return names
    # soft fallback: scan for any known tool name
    low = cleaned.lower()
    return {n for n in valid_names if n in low}


#: A section label as an LLM actually writes it. The original matcher required
#: a literal "CONCLUSION:" at line start; a judge that wrote the markdown
#: heading "## CONCLUSION" entered no section at all, so a 4,158-character
#: analysis fell through to the base narrative's first line
#: ("Model: EndpointModel(qwen3.5-2b)") and M3 had nothing to hypothesise from —
#: the run then reported stopped_by=no_hypotheses as if that were a finding.
#: Earlier runs parsed only because the judge happened to use the colon form.
_SECTION_HEADER = re.compile(
    r"^[#*\s]*(CONCLUSION|EVIDENCE_CHAIN|QUALITATIVE)\s*[:：]?[*#\s]*(.*)$",
    re.IGNORECASE,
)


def _parse_llm_analysis(
    raw: str,
    base: AnalysisReport,
) -> tuple[str, list[str], list[str]]:
    """Parse LLM output into (conclusion, evidence_chain, qualitative_findings).

    Returns "" for conclusion when the judge's response didn't contain a
    parseable CONCLUSION: section (empty/error response, quota message,
    wrong format, ...) -- NOT a synthesized fallback string. The only caller
    (_analyze_llm_guided) already built a real, statistically-grounded
    conclusion via _build_conclusion() before this parse ever runs and only
    overwrites it `if conclusion:` — inventing a fallback here (this used to
    return base.narrative's first line, which is just "Model: <repr>") meant
    a failed/malformed judge call silently threw away the good deterministic
    conclusion and replaced it with a useless one, with no error anywhere.
    Measured live: a judge call that hit its account's spend limit returned
    the CLI's plain-text quota message as `raw`, which parses to nothing
    here — before this fix, that overwrote a real "statistically supported:
    signal X flips on FAIL, effect +0.83" conclusion with "Model: <object
    repr>", and M3 saw the latter.
    """
    conclusion = ""
    evidence: list[str] = []
    qualitative: list[str] = []
    section: str | None = None

    for line in raw.splitlines():
        s = line.strip()
        upper = s.upper()
        header = _SECTION_HEADER.match(s)
        if header:
            section = header.group(1).upper()
            section = {"CONCLUSION": "conclusion",
                       "EVIDENCE_CHAIN": "evidence",
                       "QUALITATIVE": "qualitative"}[section]
            rest = header.group(2).strip()
            if section == "conclusion" and rest:
                conclusion = rest
        elif s.startswith("- "):
            content = s[2:].strip()
            if section == "evidence":
                evidence.append(content)
            elif section == "qualitative":
                qualitative.append(content)
        elif section == "conclusion" and s and not upper.startswith(("EVIDENCE", "QUALITATIVE")):
            conclusion = (conclusion + " " + s).strip()

    return conclusion, evidence, qualitative


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class StatsAnalysisReport(AnalysisReport):
    """M2 output: a richer, protocol-aware analysis report.

    Inherits all :class:`~evalvitals.analysis.analysis_module.AnalysisReport`
    fields so it is accepted everywhere that type is expected.

    Additional attributes:
        conclusion:             NL paragraph summarising what was found.
        evidence_chain:         Step-by-step derivation of the conclusion.
        qualitative_findings:   Patterns or anomalies noted in free text.
        stats_tool:             ``"threshold_rules"`` (basic), ``"llm_guided"``
                                (basic + LLM narrative), or ``"selected_tools"``
                                (statistical tools from the catalog were run).
        stats_results:          Per-tool :class:`StatsToolResult` verdicts.
        stats_plan:             ``[{tool, config, rationale}]`` — which tools
                                were selected/generated and why (M2's analog of
                                M1's ProbingSchema).
        corrected_rejections:   e-BH FDR correction across all tool e-values.
        figures:                Paths to generated figures (forest plot, …).
        stats_tool_results:     Backward-compatible JSON-safe stats summaries.
        visualizations:         Backward-compatible figure/spec list.
        protocol:               The protocol that guided this analysis, if any.
        llm_fallback_reason:    Why the LLM-guided path did not produce this
                                report. Empty when it was never attempted (no
                                judge) or when it succeeded — non-empty ONLY
                                when it was tried and raised.
    """

    conclusion: str = ""
    evidence_chain: list[str] = field(default_factory=list)
    qualitative_findings: list[str] = field(default_factory=list)
    stats_tool: str = "threshold_rules"
    stats_results: list[StatsToolResult] = field(default_factory=list)
    stats_plan: list[dict[str, Any]] = field(default_factory=list)
    corrected_rejections: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    stats_tool_results: list[dict[str, Any]] = field(default_factory=list)
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    protocol: "ExperimentProtocol | None" = None
    # When True the report carries effect sizes + charts but NO validity verdict:
    # e-BH FDR correction was NOT run (``corrected_rejections`` is deferred). The
    # analysis phase uses this so the dashboard shows distributions only; the
    # confirm phase recomputes e-BH and flips this off. See VLDiagnoseLoop.
    descriptive_only: bool = False
    # Judge I/O for the LLM-guided path (empty on the threshold-rules path).
    # Surfaced so RunLogger can persist exactly what the M2 judge was shown.
    llm_prompt: str = ""
    llm_raw: str = ""
    # Why the LLM path did not produce this report. Non-empty ONLY after it was
    # attempted and raised, so "" never has to be read as "it worked".
    llm_fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "conclusion": self.conclusion,
            "evidence_chain": self.evidence_chain,
            "qualitative_findings": self.qualitative_findings,
            "stats_tool": self.stats_tool,
            "llm_fallback_reason": self.llm_fallback_reason,
            "stats_results": [r.to_dict() for r in self.stats_results],
            "stats_plan": self.stats_plan,
            "corrected_rejections": self.corrected_rejections,
            "figures": self.figures,
            "stats_tool_results": self.stats_tool_results,
            "visualizations": self.visualizations,
            "descriptive_only": self.descriptive_only,
        })
        return d


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StatsAnalysisAgent:
    """M2: statistically analyze probe results guided by the experiment protocol.

    Args:
        judge:       Optional LLM for the LLM-guided analysis path.
                     Any :class:`~evalvitals.core.model.Model` with
                     ``Capability.GENERATE``.  Pass ``None`` to use
                     threshold rules only.
        extra_rules: Additional ``{analyzer_name: [_Rule, …]}`` entries
                     merged with the built-in rules.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        extra_rules: dict | None = None,
        stats_tool_agent: Any | None = None,
        enable_stats_tools: bool = True,
        figure_dir: "str | None" = None,
        max_signal_tools: int | None = None,
        allow_codegen: bool = False,
        codegen_config: "CliAgentConfig | None" = None,
        tool_generator: "StatsToolGenerator | None" = None,
        run_logger: "Any | None" = None,
    ) -> None:
        self._base = AnalysisModule(extra_rules)
        self._judge = judge
        self._stats_tool_agent = stats_tool_agent or LegacyStatsToolAgent()
        self._enable_stats_tools = enable_stats_tools
        self._figure_dir = figure_dir
        self._max_signal_tools = max_signal_tools
        # Tier (b): generate a bespoke stats tool in a sandbox when no catalog
        # tool fits.  Disabled unless allow_codegen=True and a backend exists.
        self._allow_codegen = allow_codegen
        self._codegen_config = codegen_config
        self._tool_generator = tool_generator
        # Tier (c): generated tools are cached here and re-run on later cycles
        # without a second LLM call.
        self._generated_tools: list[GeneratedStatsTool] = []
        # Optional RunLogger forwarded to the tool generator so its code-writing
        # attempts surface as "tool_codegen" events.  The loop sets this if unset.
        self.run_logger = run_logger

    def analyze(
        self,
        results: "dict[str, Result]",
        model_name: str = "",
        protocol: "ExperimentProtocol | None" = None,
        data: "CaseBatch | None" = None,
        extra_figures: "list | None" = None,
        confirmatory: bool = True,
    ) -> StatsAnalysisReport:
        """Analyze *results* into a :class:`StatsAnalysisReport`.

        Args:
            results:    ``{analyzer_name: Result}`` dict from M1.
            model_name: Human-readable model identifier.
            protocol:   Experiment protocol for context-aware interpretation.
                        When ``None``, the analysis is protocol-agnostic.
            data:       Labeled cases from M1.  Required to run the statistical
                        tool layer (per-case fail/pass labels).  When ``None``
                        or unlabeled, M2 falls back to threshold rules only.

        Returns:
            :class:`StatsAnalysisReport` — backward-compatible with
            :class:`~evalvitals.analysis.analysis_module.AnalysisReport`.
        """
        base = self._base.analyze(results, model_name)
        legacy_tool_results = self._run_legacy_stats_tools(results, protocol)
        inp = build_stats_input(results, data)

        return self._analyze_from_input(
            inp=inp,
            base=base,
            protocol=protocol,
            results=results,
            extra_figures=extra_figures,
            legacy_tool_results=legacy_tool_results,
            confirmatory=confirmatory,
        )

    def analyze_input(
        self,
        inp: StatsInput,
        *,
        model_name: str = "",
        protocol: "ExperimentProtocol | None" = None,
        extra_figures: "list | None" = None,
    ) -> StatsAnalysisReport:
        """Analyze a pre-normalized :class:`StatsInput`.

        This is the standalone path for callers that have already prepared
        labels, per-case signals, scalar metrics, or strategy groups and do not
        need to run M1 analyzers first.
        """
        base = AnalysisReport(
            model_name=model_name,
            findings=[],
            severity="none",
            narrative=_build_input_narrative(model_name, inp),
            raw_results={},
        )
        return self._analyze_from_input(
            inp=inp,
            base=base,
            protocol=protocol,
            results={},
            extra_figures=extra_figures,
            legacy_tool_results=[],
        )

    def analyze_records(
        self,
        records: "Any",
        *,
        id_col: str = "case_id",
        label_col: str = "label",
        signal_cols: "list[str] | tuple[str, ...] | None" = None,
        scalar_cols: "list[str] | tuple[str, ...] | None" = None,
        signal_prefix: str = "",
        model_name: str = "",
        protocol: "ExperimentProtocol | None" = None,
        extra_figures: "list | None" = None,
    ) -> StatsAnalysisReport:
        """Analyze plain row dictionaries without constructing M1 ``Result`` objects."""
        inp = build_stats_input_from_records(
            records,
            id_col=id_col,
            label_col=label_col,
            signal_cols=signal_cols,
            scalar_cols=scalar_cols,
            signal_prefix=signal_prefix,
        )
        return self.analyze_input(
            inp,
            model_name=model_name,
            protocol=protocol,
            extra_figures=extra_figures,
        )

    def _analyze_from_input(
        self,
        *,
        inp: StatsInput,
        base: AnalysisReport,
        protocol: "ExperimentProtocol | None",
        results: "dict[str, Result]",
        extra_figures: "list | None",
        legacy_tool_results: list[Any],
        confirmatory: bool = True,
    ) -> StatsAnalysisReport:
        """Shared implementation for loop-driven and standalone M2 analysis.

        ``confirmatory=False`` runs the tools (effect sizes + charts) but DEFERS
        the validity verdict: e-BH FDR correction is skipped and the report is
        marked ``descriptive_only`` so the dashboard shows distributions without
        a supported/not-supported claim. The confirm phase recomputes e-BH.
        """

        # ── Statistical tool layer (select → run → [FDR-correct] → plot) ──
        stats_results, stats_plan, corrected, figures = self._run_stats_tools_input(
            inp, protocol, confirmatory=confirmatory
        )

        # Merge in analyzer heatmap PNGs forwarded from the run logger.
        if extra_figures:
            from pathlib import Path as _Path
            figures = figures + [
                str(p) for p in extra_figures
                if _Path(p).exists() and str(p).endswith(".png")
            ]

        report: "StatsAnalysisReport | None" = None
        fallback_reason = ""
        if self._judge is not None and protocol is not None:
            try:
                report = self._analyze_llm_guided(
                    base, results, protocol,
                    stats_results, stats_plan, corrected, figures,
                    legacy_tool_results,
                )
            except Exception as exc:
                # A judge that never answered and a judge that found nothing
                # produce the SAME downstream run: threshold narrative, M3 with
                # nothing to hypothesise from, stopped_by=no_hypotheses, rc=0.
                # The reason has to travel with the report — a logger.warning
                # here reached neither the console nor run_log.jsonl, so an
                # OSError(E2BIG) on the prompt left no trace anywhere in the run.
                fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning("LLM-guided M2 analysis failed, falling back: %s", exc)
        if report is None:
            report = self._to_stats_report(
                base, protocol, stats_results, stats_plan, corrected, figures,
                legacy_tool_results,
            )
            report.llm_fallback_reason = fallback_reason
        report.descriptive_only = not confirmatory
        return report

    # ------------------------------------------------------------------
    # Statistical tool layer
    # ------------------------------------------------------------------

    def _run_stats_tools(
        self,
        results: "dict[str, Result]",
        data: "CaseBatch | None",
        protocol: "ExperimentProtocol | None",
    ) -> tuple[list[StatsToolResult], list[dict[str, Any]], dict[str, Any], list[str]]:
        """Build input, select and run tools, FDR-correct, and plot.

        Returns ``(stats_results, stats_plan, corrected, figures)``.  All empty
        when there is no labeled/grouped data to test (pure backward compat).
        """
        inp = build_stats_input(results, data)
        return self._run_stats_tools_input(inp, protocol)

    def _run_stats_tools_input(
        self,
        inp: StatsInput,
        protocol: "ExperimentProtocol | None",
        confirmatory: bool = True,
    ) -> tuple[list[StatsToolResult], list[dict[str, Any]], dict[str, Any], list[str]]:
        """Run the statistical tool layer over already-normalized input.

        ``confirmatory=False`` skips the e-BH FDR correction (the validity
        verdict) — the per-tool effect sizes / CIs are still computed (the
        charts need them), but the family-level reject decision is deferred to
        the confirm phase.
        """
        if not self._enable_stats_tools or not has_testable_data(inp):
            return [], [], {}, []

        plan = self._select_plan(inp, protocol)

        stats_results: list[StatsToolResult] = []
        for tool, cfg, rationale in plan:
            try:
                r = run_stats_tool(tool, inp, cfg)
            except Exception as exc:  # never let one tool sink the analysis
                logger.warning("stats tool %r failed: %s", tool, exc)
                r = StatsToolResult(
                    tool=tool, config=cfg, ok=False,
                    error=str(exc), summary=f"{tool}: {exc}",
                )
            r.details.setdefault("rationale", rationale)
            stats_results.append(r)

        stats_plan = [
            {"tool": t, "config": c, "rationale": rat} for t, c, rat in plan
        ]

        # Tier (b)/(c): reuse previously generated tools, then generate a new one
        # if the catalog produced nothing usable for this data.
        gen_results, gen_plan = self._run_generated_tools(inp, protocol, stats_results)
        stats_results.extend(gen_results)
        stats_plan.extend(gen_plan)

        # e-BH FDR correction is the validity verdict — deferred in descriptive
        # (analysis-phase) mode; the confirm phase recomputes it.
        corrected = (
            fdr_correct(stats_results) if confirmatory
            else {"method": "e-BH", "deferred": True, "n_tested": 0, "rejected_tools": [],
                  "note": "validity deferred to the confirm phase (descriptive_only)"}
        )

        figures: list[str] = []
        if self._figure_dir:
            fig = plot_effects(
                stats_results, os.path.join(self._figure_dir, "m2_effects.png")
            )
            if fig:
                figures.append(fig)

        return stats_results, stats_plan, corrected, figures

    def _run_legacy_stats_tools(
        self,
        results: "dict[str, Result]",
        protocol: "ExperimentProtocol | None",
    ) -> list[Any]:
        """Run the older lightweight M2 tools for backward-compatible reports."""
        if not self._enable_stats_tools or self._stats_tool_agent is None:
            return []
        try:
            return list(self._stats_tool_agent.analyze(results, protocol=protocol))
        except Exception as exc:
            logger.warning("legacy M2 stats-tool analysis failed: %s", exc)
            return []

    def _run_generated_tools(
        self,
        inp: StatsInput,
        protocol: "ExperimentProtocol | None",
        catalog_results: list[StatsToolResult],
    ) -> tuple[list[StatsToolResult], list[dict[str, Any]]]:
        """Reuse cached generated tools (tier c) and generate a new one (tier b).

        Generation fires only when ``allow_codegen`` is set, a backend exists,
        and no catalog or cached tool produced a usable (``ok``) result.
        """
        if not self._allow_codegen:
            return [], []
        generator = self._get_generator()
        if generator is None:
            return [], []

        results: list[StatsToolResult] = []
        plan: list[dict[str, Any]] = []

        # Tier (c): re-run cached generated tools on this cycle's data.
        for tool in self._generated_tools:
            r = generator.run_cached(tool, inp)
            results.append(r)
            plan.append({"tool": f"generated:{tool.name}", "config": {},
                         "rationale": f"reuse generated tool ({tool.source})"})

        usable = any(r.ok for r in catalog_results) or any(r.ok for r in results)
        if usable:
            return results, plan

        # Tier (b): nothing fit — synthesise a new tool.
        need = (
            (protocol.prompt_text() + " " if protocol else "")
            + "No built-in statistical tool fit this data; write a test for "
            "whether the analyzer signals predict case FAIL."
        )
        name = f"custom{len(self._generated_tools) + 1}"
        result, tool = generator.generate(need, inp, name=name)
        results.append(result)
        plan.append({"tool": f"generated:{name}", "config": {},
                     "rationale": "generated: no catalog tool fit the data"})
        if tool is not None:
            self._generated_tools.append(tool)
        return results, plan

    def _get_generator(self) -> "StatsToolGenerator | None":
        """Lazily build the tool generator from the judge / codegen config."""
        if self._tool_generator is not None:
            if self.run_logger is not None and getattr(self._tool_generator, "run_logger", None) is None:
                try:
                    self._tool_generator.run_logger = self.run_logger
                except Exception:  # noqa: BLE001
                    pass
            return self._tool_generator
        if self._judge is None and self._codegen_config is None:
            return None
        from evalvitals.analysis.stats_tool_generator import StatsToolGenerator
        gen = StatsToolGenerator(
            judge=self._judge, cli_config=self._codegen_config, run_logger=self.run_logger
        )
        if not gen.available:
            return None
        self._tool_generator = gen
        return gen

    def _select_plan(
        self,
        inp: StatsInput,
        protocol: "ExperimentProtocol | None",
    ) -> list[tuple[str, dict, str]]:
        """Select tools: LLM picks from the catalog, deterministic plan otherwise.

        The deterministic plan is always the backbone (it guarantees coverage);
        when a judge + protocol are present, the LLM *narrows* it to the chosen
        tools.  Any failure or empty selection falls back to the full plan.
        """
        plan = default_plan(inp, max_signals=self._max_signal_tools)
        if self._judge is None or protocol is None or not plan:
            return plan
        try:
            chosen = _select_tools_via_llm(
                self._judge, protocol, inp, set(STATS_TOOL_CATALOG)
            )
        except Exception as exc:
            logger.debug("LLM stats-tool selection failed, using default plan: %s", exc)
            return plan
        if not chosen:
            return plan
        # Paired/omnibus tools (mcnemar, friedman, attention_decoding) only enter
        # the plan when their inputs exist — they carry causal / tensor-level
        # verdicts and must never be silently dropped by name-based narrowing.
        _mandatory = {"mcnemar_evalue", "friedman_nemenyi", "attention_decoding"}
        filtered = [p for p in plan if p[0] in chosen or p[0] in _mandatory]
        return filtered or plan

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def _to_stats_report(
        self,
        base: AnalysisReport,
        protocol: "ExperimentProtocol | None",
        stats_results: list[StatsToolResult] | None = None,
        stats_plan: list[dict[str, Any]] | None = None,
        corrected: dict[str, Any] | None = None,
        figures: list[str] | None = None,
        legacy_tool_results: list[Any] | None = None,
    ) -> StatsAnalysisReport:
        """Basic path: wrap AnalysisReport + stats-tool results into a report."""
        stats_results = stats_results or []
        stats_plan = stats_plan or []
        corrected = corrected or {}
        figures = figures or []
        legacy_tool_results = legacy_tool_results or []

        chain: list[str] = []
        if base.findings:
            chain.append(
                f"Applied threshold rules across {len(base.raw_results)} "
                f"analyzer(s): {', '.join(sorted(base.raw_results))}"
            )
            for f in base.findings:
                cmp = ">" if f.direction == "above" else "<"
                chain.append(
                    f"  [{f.severity.upper()}] {f.analyzer}.{f.metric}="
                    f"{f.value:.3g} {cmp} {f.threshold}: {f.message}"
                )
            chain.append(f"Overall severity: {base.severity}")
        elif base.raw_results:
            chain.append(
                f"All metrics within normal ranges across "
                f"{len(base.raw_results)} analyzer(s)."
            )

        # Append statistical verdicts to the evidence chain.
        for r in stats_results:
            if r.ok:
                chain.append(f"[stats:{r.tool}] {r.summary}")
            else:
                chain.append(f"[stats:{r.tool}] N/A ({r.error})")
        if corrected.get("rejected_tools"):
            chain.append(
                f"FDR (e-BH) survivors: {', '.join(corrected['rejected_tools'])}"
            )

        qualitative = [str(f) for f in base.findings]
        conclusion = _build_conclusion(base, protocol, stats_results)
        if any(r.tool.startswith("generated:") for r in stats_results):
            stats_tool = "generated"
        elif stats_results:
            stats_tool = "selected_tools"
        else:
            stats_tool = "threshold_rules"

        return StatsAnalysisReport(
            # AnalysisReport fields
            model_name=base.model_name,
            findings=base.findings,
            severity=base.severity,
            narrative=base.narrative,
            raw_results=base.raw_results,
            # StatsAnalysisReport extras
            conclusion=conclusion,
            evidence_chain=chain,
            qualitative_findings=qualitative,
            stats_tool=stats_tool,
            stats_results=stats_results,
            stats_plan=stats_plan,
            corrected_rejections=corrected,
            figures=figures,
            stats_tool_results=(
                _serialize_legacy_tool_results(legacy_tool_results)
                or _compat_stats_tool_results(stats_results)
            ),
            visualizations=(
                _legacy_visualizations(legacy_tool_results)
                or [{"type": "figure", "path": p} for p in figures]
            ),
            protocol=protocol,
        )

    def _analyze_llm_guided(
        self,
        base: AnalysisReport,
        results: "dict[str, Result]",
        protocol: "ExperimentProtocol",
        stats_results: list[StatsToolResult],
        stats_plan: list[dict[str, Any]],
        corrected: dict[str, Any],
        figures: list[str],
        legacy_tool_results: list[Any],
    ) -> StatsAnalysisReport:
        """LLM-guided path: judge synthesises protocol-specific insights.

        The judge now also sees the statistical verdicts so its conclusion is
        grounded in effect sizes + e-values, not just raw findings.
        """
        findings_json = json.dumps(
            {r: res.findings for r, res in results.items()},
            indent=2,
            default=str,
        )
        stats_block = _format_stats_for_prompt(stats_results, corrected)
        prompt = _ANALYSIS_PROMPT.format(
            protocol_text=protocol.prompt_text(),
            task_domain=protocol.task_domain or "general",
            narrative=base.narrative + stats_block,
            findings_json=findings_json,
        )
        import inspect as _inspect
        from pathlib import Path as _Path
        _sig = _inspect.signature(self._judge.generate)  # type: ignore[union-attr]
        if "images" in _sig.parameters and figures:
            _imgs = [_Path(f) for f in figures if _Path(f).exists()]
            raw = self._judge.generate(prompt, images=_imgs or None)  # type: ignore[union-attr]
        else:
            raw = self._judge.generate(prompt)  # type: ignore[union-attr]
        conclusion, evidence, qualitative = _parse_llm_analysis(str(raw), base)

        report = self._to_stats_report(
            base, protocol, stats_results, stats_plan, corrected, figures,
            legacy_tool_results,
        )
        if conclusion:
            report.conclusion = conclusion
        if evidence:
            report.evidence_chain = evidence
        if qualitative:
            report.qualitative_findings = qualitative
        report.stats_tool = "llm_guided"
        report.llm_prompt = prompt
        report.llm_raw = str(raw)
        return report


def _build_conclusion(
    base: AnalysisReport,
    protocol: "ExperimentProtocol | None",
    stats_results: list[StatsToolResult] | None = None,
) -> str:
    """Build a short NL conclusion from the base report + stats + optional protocol."""
    domain = f"For {protocol.task_domain} tasks: " if (protocol and protocol.task_domain) else ""

    # Prefer a statistically supported verdict when one exists.
    supported = [
        r for r in (stats_results or [])
        if r.ok and r.reject and r.effect is not None
    ]
    stat_msg = ""
    if supported:
        top = max(supported, key=lambda r: abs(r.effect or 0.0))
        stat_msg = f" Statistically supported: {top.summary}"

    if not base.raw_results and base.narrative:
        if stat_msg:
            return f"{domain}Standalone statistical analysis found a supported signal.{stat_msg}"
        return f"{domain}Standalone statistical analysis found no supported signal."

    if not base.findings:
        if stat_msg:
            return f"{domain}No threshold violations, but a statistical test fired.{stat_msg}"
        msg = "No threshold violations detected; all metrics within normal ranges."
        return f"{domain}{msg}"

    top_f = base.findings[0]
    msg = (
        f"Primary anomaly: {top_f.message} "
        f"({top_f.analyzer}.{top_f.metric}={top_f.value:.3g}, severity={top_f.severity})."
    )
    if len(base.findings) > 1:
        msg += f" {len(base.findings) - 1} additional finding(s) also flagged."
    return f"{domain}{msg}{stat_msg}"


def _build_input_narrative(model_name: str, inp: StatsInput) -> str:
    """Human-readable summary for standalone StatsInput analysis."""
    d = describe_data(inp)
    lines = [f"Model: {model_name}" if model_name else "Model: unspecified"]
    lines.append("Input: standalone statistical dataset")
    lines.append(
        f"Labeled cases: {d['n_labeled']} "
        f"({d['n_fail']} FAIL, {d['n_pass']} PASS)"
    )
    if d["per_case_signals"]:
        lines.append(f"Per-case signals: {', '.join(d['per_case_signals'])}")
    else:
        lines.append("Per-case signals: none")
    if d["scalar_metrics"]:
        lines.append(f"Scalar metrics: {', '.join(d['scalar_metrics'])}")
    if d["n_strategy_groups"]:
        lines.append(f"Strategy groups: {d['n_strategy_groups']}")
    return "\n".join(lines)


def _compat_stats_tool_results(stats_results: list[StatsToolResult]) -> list[dict[str, Any]]:
    """Return the legacy JSON shape used by older logs/examples."""
    out: list[dict[str, Any]] = []
    for result in stats_results:
        payload = result.to_dict()
        payload["name"] = result.tool
        payload["conclusion"] = result.summary or result.error or ""
        payload["visualizations"] = (
            [{"type": "figure", "path": result.figure_path}]
            if result.figure_path
            else []
        )
        out.append(payload)
    return out


def _serialize_legacy_tool_results(tool_results: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for result in tool_results:
        if hasattr(result, "to_dict"):
            payloads.append(result.to_dict())
        elif isinstance(result, dict):
            payloads.append(dict(result))
    return payloads


def _legacy_visualizations(tool_results: list[Any]) -> list[dict[str, Any]]:
    visualizations: list[dict[str, Any]] = []
    for result in tool_results:
        visualizations.extend(getattr(result, "visualizations", []) or [])
    return visualizations


def _multiplicity_note(result: StatsToolResult) -> str:
    """Per-result multiplicity verdict, or "" for a result outside every family.

    ``correction_method`` is set by :func:`~evalvitals.stats.multiplicity.correct_results`
    on family members only, so its absence means "descriptive, never corrected" —
    which must not be rendered as a failure to survive.
    """
    method = result.correction_method
    if not method:
        return ""
    bits = ["survived" if result.fdr_corrected else "NOT survived"]
    if result.p_value is not None:
        bits.append(f"p={result.p_value:.3g}")
    if result.e_value is not None:
        bits.append(f"e={result.e_value:.3g}")
    n_signal = (result.details or {}).get("n_signal")
    if n_signal is not None:
        bits.append(f"n_signal={n_signal}")
    return f"   [{method}: {', '.join(bits)}]"


def _format_stats_for_prompt(
    stats_results: list[StatsToolResult],
    corrected: dict[str, Any],
) -> str:
    """Render statistical verdicts as a block appended to the LLM narrative.

    Every line carries its OWN multiplicity verdict, because two separate things
    otherwise mislead the reader in the same direction:

    * ``summary`` is baked when the tool RUNS, before any correction, so a line
      ending "-> REJECT H0" is the uncorrected verdict. On qwen3.5-2b /
      bbh_word_sorting, 23 of 42 ``signal_label_assoc`` results printed
      REJECT H0 while BH kept 13; of the other ten, one rested on a single case
      at permutation p = 1.000.
    * The footer used to list ``rejected_tools`` — a set of TOOL NAMES. Every
      per-signal test shares the name ``signal_label_assoc``, so a 42-signal
      family collapsed to the single word "signal_label_assoc" and no reader
      could tell WHICH signals survived. The judge on that run reconciled the
      mismatch the only way left to it, concluding the tool had not run at all,
      and discarded the entire family. ``rejected_result_keys`` was already in
      the same dict, at per-signal granularity.

    ``reject`` itself is deliberately not touched here: for the BH family it
    keeps the tool's raw CI verdict for the M1-M5 loop (see
    :mod:`evalvitals.stats.multiplicity`). This renderer is the place that has
    to make the difference between raw and corrected legible.
    """
    if not stats_results:
        return ""
    lines = ["", "Statistical test results (effect-sized, FDR-aware):"]
    for r in stats_results:
        if r.ok:
            lines.append(f"  - {r.summary}{_multiplicity_note(r)}")
        else:
            lines.append(f"  - {r.tool}: not run ({r.error})")
    n_tested = int(corrected.get("n_tested") or 0)
    if n_tested:
        keys = list(corrected.get("rejected_result_keys") or [])
        method = corrected.get("method") or "FDR"
        lines.append(
            f"  After {method} correction, {len(keys)} of {n_tested} corrected "
            f"test(s) survive" + (":" if keys else " — none.")
        )
        lines.extend(f"    * {key}" for key in keys)
    return "\n".join(lines)
