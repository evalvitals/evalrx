"""VLDiagnoseLoop — the current M1→M2→M3→M5 diagnosis loop.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ M1 · ProbeAgent         protocol-guided analyzer selection + execute │
    │ M2 · StatsAnalysisAgent protocol-aware stats analysis                │
    │ M3 · DiagnosisAgent     "AI scientist" hypothesis generation         │
    │ M5 · HypothesisTester   stats test + protocol consistency check      │
    └──────────────────────────────────────────────────────────────────────┘
                            ↑_________________________________│
             stop when M5 finds a verified, protocol-consistent hypothesis

M4 (SurgeryAgent) runs separately via ``VLDiagnoseLoop.run_m4()`` once the
loop stops — propose a fix for the best verified hypothesis (Plan A), or
propose + execute a fix (Plan B).

See also:
  - :class:`~evalvitals.eval_agent.agentic.AgenticDiagnoseLoop` — the same
    M1-M5 stages driven by a judge-decided action loop instead of a fixed
    cycle.
  - :mod:`~evalvitals.eval_agent.legacy` — ``SelfEvolveLoop`` and
    ``AutoDiagnoseLoop`` (the pre-2026-06-05 M1→M2→M3→M4 architecture), kept
    for existing callers.

Usage::

    from evalvitals.eval_agent import VLDiagnoseLoop
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    protocol = ExperimentProtocol(
        description="QwenVL often confuses left/right positions in spatial tasks.",
        task_domain="spatial reasoning",
        target_modalities=frozenset({"text", "image"}),
    )
    loop   = VLDiagnoseLoop(model=vlm, protocol=protocol)
    report = loop.run(failure_cases)
    fix    = loop.run_m4(report, failure_cases)   # separate fix-proposal step
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.loop_reports import VLDiagnoseReport
from evalvitals.eval_agent.run_metadata import (
    _attach_run_logger,
    _coerce_explore_context,
    _log_generated_tools,
    _make_intervention_result_from_test,
    _run_config,
)
from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

logger = logging.getLogger(__name__)

_STOPPED_BY_CRITERIA  = "criteria_met"
_STOPPED_BY_MAX       = "max_cycles"
_STOPPED_BY_BUDGET    = "budget"
_STOPPED_BY_NO_HYPS   = "no_hypotheses"
_STOPPED_BY_NO_PROBE  = "no_probe_results"
# run_analysis() ran M1->M2->M3 and proposed hypotheses without confirming them
# (M5 deferred to run_confirm()). Not a failure — the analysis dashboard is ready.
_STOPPED_BY_ANALYSIS  = "analysis_complete"


def _diagnose_with_optional_context(
    diag_agent: "Any", stats_report: "Any", prior_cycles: "Any", explore_context: "Any | None",
    failure_modes: "Any | None" = None,
) -> "Any":
    """Call ``diag_agent.diagnose`` passing ``explore_context``/``failure_modes``
    only when the agent accepts them, so custom/legacy diagnosis agents keep
    working unchanged."""
    import inspect as _inspect

    kwargs: dict[str, Any] = {"prior_cycles": prior_cycles or None}
    if explore_context is not None or failure_modes is not None:
        try:
            params = _inspect.signature(diag_agent.diagnose).parameters
        except (TypeError, ValueError):
            params = {}
        if explore_context is not None and "explore_context" in params:
            kwargs["explore_context"] = explore_context
        if failure_modes is not None and "failure_modes" in params:
            kwargs["failure_modes"] = failure_modes
    return diag_agent.diagnose(stats_report, **kwargs)


def _propose_and_validate(agent: "Any", model: "Any", data: "Any", hypotheses: "Any",
                          **kwargs: Any) -> "Any":
    """Call ``agent.propose_and_validate`` passing ``prior_attempts`` /
    ``context`` only when the agent accepts them (custom/stub fix agents keep
    working unchanged — same pattern as ``_diagnose_with_optional_context``)."""
    import inspect as _inspect

    try:
        params = _inspect.signature(agent.propose_and_validate).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
    passed = {k: v for k, v in kwargs.items()
              if v is not None and (accepts_any or k in params)}
    return agent.propose_and_validate(model, data, hypotheses, **passed)


def _unverified_hypotheses(report: "Any") -> "list[Any]":
    """Best-first UNVERIFIED hypotheses: M5-tested and not refuted, highest
    confidence first; then untested proposals from the last cycle."""
    seen: set[str] = set()
    out: list[Any] = []
    tested = list(getattr(report, "all_test_results", None) or [])
    ranked = sorted(
        tested,
        key=lambda tr: float(getattr(tr, "confidence", 0.0) or 0.0),
        reverse=True,
    )
    for tr in ranked:
        h = getattr(tr, "hypothesis", None)
        if h is None or _hyp_key(h) in seen:
            continue
        seen.add(_hyp_key(h))  # tested (in any way) -> never re-added as "untested"
        status = getattr(tr, "status", None)
        status_s = str(getattr(status, "value", status) or "").lower()
        if status_s == "refuted":
            continue
        out.append(h)
    for h in reversed(list(getattr(report, "final_hypotheses", None) or [])):
        if _hyp_key(h) not in seen:
            seen.add(_hyp_key(h))
            out.append(h)
    return out


def _m4_supported_key(report: "Any") -> "str | None":
    """Key of the hypothesis M4's experiment SUPPORTED, if any."""
    iv = getattr(report, "fix_proposal", None)
    status = getattr(iv, "status", None)
    status_s = str(getattr(status, "value", status) or "").lower()
    hyp = getattr(iv, "hypothesis", None)
    if iv is None or hyp is None or status_s != "supported":
        return None
    return _hyp_key(hyp)


def _hyp_key(hypothesis: "Any") -> str:
    """Identity of a hypothesis for matching across M5/M4 results."""
    hid = str(getattr(hypothesis, "id", "") or "")
    return hid or str(getattr(hypothesis, "statement", hypothesis))


def _m4_refuted(report: "Any") -> "tuple[set[str], list[str]]":
    """Hypotheses M4's intervention experiment REFUTED, with a one-line why.

    Reads ``report.fix_proposal`` (an ``InterventionResult`` from ``run_m4``).
    Returns ``(keys, notes)``; both empty when M4 did not run or did not refute.
    """
    iv = getattr(report, "fix_proposal", None)
    if iv is None:
        return set(), []
    status = getattr(iv, "status", None)
    status_s = str(getattr(status, "value", status) or "").lower()
    if status_s != "refuted":
        return set(), []
    hyp = getattr(iv, "hypothesis", None)
    if hyp is None:
        return set(), []
    statement = str(getattr(hyp, "statement", hyp))
    evidence = getattr(iv, "evidence", None) or {}
    scalars = []
    if isinstance(evidence, dict):
        for k, v in evidence.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scalars.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
            if len(scalars) >= 8:
                break
    why = f" [experiment: {', '.join(scalars)}]" if scalars else ""
    return {_hyp_key(hyp)}, [f"{statement}{why}"]


def _fix_context_from_report(
    report: "Any",
    *,
    example_cases: "Any | None",
    explore_context: "Any | None",
    protocol: "Any | None",
    refuted: "list[str]",
    refuted_ids: "set[str] | None" = None,
) -> "Any":
    """Build the :class:`~evalvitals.eval_agent.stages.fix_agent.FixContext`
    the proposer sees: M5 verdicts + M2 conclusion/tests + exploratory notes,
    the M4-refuted hypotheses, and (when a confirm split is in play) the
    EXPLORE cases in full."""
    from evalvitals.eval_agent.stages.fix_agent import FixContext

    lines: list[str] = []
    refuted_ids = set(refuted_ids or ())
    verified = [
        tr for tr in list(getattr(report, "verified_hypotheses", None) or [])
        if _hyp_key(getattr(tr, "hypothesis", None)) not in refuted_ids
    ]
    if verified:
        lines.append("  M5 verified hypotheses (statistical tests on the diagnosis split):")
        for tr in verified[:6]:
            stmt = str(getattr(getattr(tr, "hypothesis", None), "statement", ""))[:220]
            verdict = str(getattr(tr, "verdict", "") or "")[:300]
            conf = getattr(tr, "confidence", None)
            grade = getattr(tr, "evidence_grade", "")
            conf_s = f" conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
            lines.append(f"    - {stmt}{conf_s} grade={grade}")
            if verdict:
                lines.append(f"        evidence: {verdict}")
    stats = getattr(report, "final_stats_report", None) or getattr(report, "final_analysis", None)
    if stats is not None:
        conclusion = str(getattr(stats, "conclusion", "") or getattr(stats, "narrative", "") or "")
        if conclusion:
            lines.append("  M2 conclusion:")
            lines.append("    " + conclusion.strip()[:1200])
        chain = list(getattr(stats, "evidence_chain", None) or [])
        if chain:
            lines.append("  M2 evidence chain:")
            lines += [f"    - {str(step)[:300]}" for step in chain[:8]]
        results = list(getattr(stats, "stats_results", None) or [])
        sig = [r for r in results if getattr(r, "ok", False) and getattr(r, "reject", False)]
        shown = sig[:8] or [r for r in results if getattr(r, "ok", False)][:5]
        if shown:
            lines.append("  M2 statistical tests" + (" (rejecting H0):" if sig else ":"))
            lines += [f"    - {str(getattr(r, 'summary', ''))[:300]}" for r in shown]
    if explore_context is not None and not getattr(explore_context, "is_empty", True):
        obs = list(getattr(explore_context, "observations", None) or [])
        cav = list(getattr(explore_context, "caveats", None) or [])
        if obs:
            lines.append("  Exploratory notes (free-form EDA, UNCONFIRMED — hints only):")
            lines += [f"    - {str(o)[:300]}" for o in obs[:10]]
        if cav:
            lines.append("  Explorer caveats:")
            lines += [f"    - {str(c)[:200]}" for c in cav[:5]]
    task_bits = []
    for attr in ("description", "task_domain", "failure_patterns"):
        value = str(getattr(protocol, attr, "") or "").strip()
        if value:
            task_bits.append(value)
    return FixContext(
        example_cases=example_cases,
        evidence="\n".join(lines),
        refuted=list(refuted),
        task_note=" — ".join(task_bits),
    )


class VLDiagnoseLoop:
    """M1→M2→M3→M5 failure-analysis loop for VL tasks (Plan A architecture).

    M4 (**SurgeryAgent**) is intentionally excluded from the inner loop.
    Call :meth:`run_m4` on the returned :class:`VLDiagnoseReport` to obtain
    a fix proposal based on the best verified hypothesis candidates.

    Inner loop::

        for cycle in range(max_cycles):
            probe_results  = M1.probe(model, data, protocol)   # guided by protocol
            explore_notes  = explorer.explore_records(...)      # optional, descriptive
            stats_report   = M2.analyze(probe_results, protocol)
            diag           = M3.diagnose(stats_report, explore_context=explore_notes)
            test_results   = M5.test(diag.hypotheses, stats_report, data, protocol)
            if M5.stopping_criteria_met(test_results, protocol): break

    Stopping criteria: at least one M5-verified hypothesis that is also
    consistent with the user's experiment protocol.

    **Decoupled two-phase use** (analysis → deferred confirm + fix)::

        # Phase 1 — analyse + propose, build the dashboard. No M5, no fix.
        report = loop.run_analysis(data)        # M1 → M2 → M3, stop
        save(report.final_hypotheses, report.final_stats_report)

        # Phase 2 — later, reuse the saved artifacts to confirm + repair.
        report = loop.run_confirm(data, hypotheses, stats_report=stats)  # M5
        loop.run_fix(report, data)              # tiered repair

    :meth:`run` is the all-in-one path (M1→M2→M3→M5 + stopping). Use
    :meth:`run_analysis` + :meth:`run_confirm` when you want the analysis
    dashboard *before* deciding whether to confirm hypotheses and repair.

    Args:
        model:              The model under evaluation.
        protocol:           Experiment protocol — the human prior that guides
                            M1 analyzer selection, M2 narrative, and M5
                            consistency checks.
        probe_agent:        M1.  Defaults to ``ProbeAgent()``.
        stats_agent:        M2.  Defaults to ``StatsAnalysisAgent()``.
        diagnosis_agent:    M3.  ``None`` lazily resolves ``DiagnosisAgent()``
                            on first use.
        hypothesis_tester:  M5.  Defaults to ``HypothesisTester()``.
        surgery_agent:      M4 — used only by :meth:`run_m4`, never inside
                            the main loop.  Defaults to ``SurgeryAgent()``.
        store:              Persistent memory.
        m5_holdout:         When a confirm split is in play (default on),
                            M5 is taken OUT of the cycle entirely: the cycles
                            only mine (M1→explore→M2→M3), and after the loop
                            every proposed hypothesis is tested ONCE on the
                            held-out confirm split (the last cycle's analyzers
                            re-run there, pinned). The loop neither stops
                            early nor keeps cycling based on M5 verdicts —
                            the same discipline the fix gate follows. Without
                            a confirm split M5 stays in-cycle as before.
        max_cycles:         Hard cap on M1→M5 iterations (default 1: one
                            diagnosis pass, then the caller moves on to
                            M4/fix — with fix-on-unverified enabled the
                            extra cycles rarely verified anything and
                            tripled the wall-clock; raise it to keep
                            mining when a cycle's M5 designs feed the
                            next cycle's M1).
        run_logger:         Optional :class:`~evalvitals.eval_agent.run_logger.RunLogger`.
        token_budget:       Stop early when accumulated token usage reaches
                            this limit (0 = unlimited).
        analysis_only:      Run only M1→M2 and stop before hypothesis generation.
        explorer:           Optional :class:`~evalvitals.analysis.explorer.ExploratoryAnalysisAgent`.
                            When given, every cycle runs a free-form EDA step
                            between M1 and M2 over the same per-case table M2
                            sees (M1 analyzer signals + labels). Its output is
                            DESCRIPTIVE ONLY: observations/charts/caveats go to
                            M3 as an ``ExploreContext`` (which hypotheses to
                            propose) and to disk for the dashboard — never into
                            M2's confirmatory family, M5, or the fix gate. The
                            catalog M2 is unchanged. Best-effort: an explorer
                            failure logs a warning and the cycle continues.
        explore_dir:        Where the explore step persists
                            ``exploratory_report.json`` + ``tables/`` +
                            ``figures/`` (rendered chart PNGs, which M3 is shown).
                            Default: ``<ctx.root>/explore`` for a RunContext-
                            backed logger, ``<run>/explore`` beside a standalone
                            ``logs*/`` dir, else ``<run_dir>/explore`` — always
                            inside the run, where the dashboard looks; with no
                            run_logger, nothing is persisted.
        explore_question:   The question handed to the explorer. Defaults to
                            :func:`~evalvitals.eval_agent.prompts.explore_step.default_explore_question`
                            built from *protocol*.
        verbose:            When ``True``, print live M1-M5 stage narration to
                            stdout (equivalent to calling
                            ``evalvitals.enable_console_logging()`` yourself).
                            Separate from ``run_logger`` — this surfaces the
                            existing ``logger.info()``/``.warning()`` calls,
                            not RunLogger's structured JSONL event stream.
    """

    def __init__(
        self,
        model: "Model",
        protocol: "ExperimentProtocol",
        probe_agent: "Any | None" = None,
        stats_agent: "StatsAnalysisAgent | None" = None,
        diagnosis_agent: "Any | None" = None,
        hypothesis_tester: "HypothesisTester | None" = None,
        surgery_agent: "Any | None" = None,
        fix_agent: "Any | None" = None,
        store: Store | None = None,
        max_cycles: int = 1,
        run_logger: "Any | None" = None,
        token_budget: int = 0,
        analysis_only: bool = False,
        confirm_split: float = 0.0,
        confirm_split_seed: int = 0,
        m5_holdout: bool = True,
        signal_recipes: "list | None" = None,
        bridge_analyzer_name: str = "explored",
        explore_report: "Any | None" = None,
        explorer: "Any | None" = None,
        explore_dir: "str | Path | None" = None,
        explore_question: str = "",
        verbose: bool = False,
    ) -> None:
        from evalvitals.analysis.stats_agent import StatsAnalysisAgent
        from evalvitals.eval_agent.stages.fix_agent import FixAgent
        from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester
        from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
        from evalvitals.eval_agent.stages.surgery import SurgeryAgent

        if verbose:
            # M1-M5 already narrate every stage transition via logger.info()/
            # .warning() (this module's `logger`, plus probe_agent/diagnosis/
            # hypothesis_tester/fix_agent's own) -- it's just invisible by
            # default. This is the one-line equivalent of a caller doing
            # `evalvitals.enable_console_logging()` themselves; it does NOT
            # touch RunLogger's separate structured JSONL event stream.
            from evalvitals.logging_utils import enable_console_logging

            enable_console_logging()

        self.model = model
        self.protocol = protocol
        self.probe_agent = probe_agent or ProbeAgent()
        self.stats_agent = stats_agent or StatsAnalysisAgent()
        self.diagnosis_agent = diagnosis_agent  # None = lazy default on first call
        self.hypothesis_tester = hypothesis_tester or HypothesisTester()
        self.surgery_agent = surgery_agent or SurgeryAgent()
        self.fix_agent = fix_agent or FixAgent(run_logger=run_logger)
        self.store = store or InMemoryStore()
        self.max_cycles = max_cycles
        self.run_logger = run_logger
        self.token_budget = token_budget
        self.analysis_only = analysis_only
        # Held-out CONFIRM split (leak #3): fraction of the batch reserved, away
        # from M1-M5 hypothesis generation, for the post-loop fix/surgery to
        # validate on — so the deployed fix is confirmed on data the loop never
        # mined. 0.0 = off (current behavior); the split is deterministic
        # (stratified by label+probe_type, seeded), so run() and run_m4/run_fix
        # derive the identical partition from the same input batch.
        self.confirm_split = float(confirm_split)
        self.confirm_split_seed = int(confirm_split_seed)
        self.m5_holdout = bool(m5_holdout)
        # Operationalization bridge (off by default): pre-registered SignalRecipes
        # are compiled over the analyzer per_case signals each cycle into a synthetic
        # "<bridge_analyzer_name>" analyzer Result, so LAMBDA-discovered composite
        # signals enter M2's family via the standard findings["per_case"] contract.
        # Leak-free by construction — recipes must be discovered out-of-band (e.g.
        # the fused pipeline's held-out split), never by peeking at these labels.
        self._signal_recipes = list(signal_recipes or [])
        self._bridge_analyzer_name = bridge_analyzer_name
        # Step-1 explorer mechanism notes (charts/observations/caveats). Descriptive,
        # UNCONFIRMED: passed to M3's hypothesis-proposal prompt ONLY — never to the
        # M2 confirmatory family, M5 testing, or the fix gate. Accepts an
        # ExploreContext, a report dict (fused_report.json), or None.
        self._explore_context = _coerce_explore_context(explore_report)
        # In-cycle explore step (off by default): a free-form EDA pass over the
        # M1 per-case table, run between M1 and M2. Same standing as a Step-1
        # explore_report — descriptive notes for M3 + files for the dashboard.
        # AgenticDiagnoseLoop overrides `self.explorer` after this constructor
        # and drives it through its own judge-decided `explore_data` tool
        # instead of the fixed in-cycle call (see run()/run_analysis()).
        self.explorer = explorer
        self._explore_dir = Path(explore_dir) if explore_dir is not None else None
        self._explore_question = str(explore_question or "")
        self._tokens_used: int = 0
        self._run_id: str = ""

    def publish_report(
        self,
        *,
        model: "Any | None" = None,
        example_dir: "str | Path | None" = None,
    ) -> "Any":
        """Compose the completed-run UI, reusing the M3 judge when available."""
        if self.run_logger is None:
            raise RuntimeError("publish_report needs a run_logger with a durable run directory")
        if model is None:
            model = getattr(self.diagnosis_agent, "judge", None)
        from evalvitals.reporting.dynamic import publish_report

        return publish_report(
            self.run_logger.run_dir,
            example_dir=example_dir,
            model=model,
            run_logger=self.run_logger,
        )

    @staticmethod
    def _strat_key(case: "Any") -> "tuple":
        """Stratify the explore/confirm split by label + probe_type when present
        (keeps the no-free-lunch control mix, e.g. present-detections, in both
        partitions). Falls back to label alone for generic batches."""
        label = getattr(getattr(case, "label", None), "value", "?")
        probe = (getattr(case, "metadata", {}) or {}).get("probe_type")
        return (label, probe)

    def _split_explore_confirm(self, data: "CaseBatch"):
        """Deterministic, stratified (explore, confirm) partition.

        Returns ``(explore_batch, confirm_batch)``. When ``confirm_split <= 0``
        (or the batch is too small to split), returns ``(data, None)`` — a
        no-op, so existing runs are byte-for-byte unchanged.
        """
        from evalvitals.core.case import CaseBatch
        from evalvitals.stats.subset_sampling import stratified_subset

        cases = list(data)
        frac = self.confirm_split
        if frac <= 0.0 or len(cases) < 4:
            return data, None
        n_confirm = round(len(cases) * frac)
        if n_confirm <= 0 or n_confirm >= len(cases):
            return data, None
        confirm = stratified_subset(cases, self._strat_key, n_confirm,
                                    seed=self.confirm_split_seed)
        confirm_ids = {id(c) for c in confirm}
        explore = [c for c in cases if id(c) not in confirm_ids]
        return CaseBatch(explore), CaseBatch(confirm)

    def _bridge_signals(self, probe_results: "dict[str, Any]", data: "Any | None") -> None:
        """Compile pre-registered signal recipes into a synthetic analyzer Result
        and inject it into *probe_results* so M2/M3/M5 see the bridged composite
        signals through the standard findings["per_case"] contract. No-op when no
        recipes are configured. Never raises into the loop."""
        if not self._signal_recipes:
            return
        # Never silently overwrite a real analyzer that already used this key.
        name = self._bridge_analyzer_name
        if name in probe_results:
            base, n = name, 1
            while name in probe_results:
                name, n = f"{base}_bridge{n}", n + 1
            logger.warning(
                "bridge analyzer name %r collides with a real analyzer; "
                "injecting under %r instead", base, name,
            )
        try:
            from evalvitals.analysis.operationalize import bridge_recipes_to_result

            synth = bridge_recipes_to_result(
                self._signal_recipes, probe_results, data,
                model_repr=repr(self.model),
                analyzer_name=name,
            )
        except Exception as exc:  # bridging must never sink the loop
            logger.warning("signal bridge failed: %s", exc)
            return
        if synth is None:
            return
        probe_results[name] = synth
        self.store.add_result(synth)
        # The bridged signals ARE the discovered candidates — they must be tested,
        # not optional. default_plan caps at the stats agent's max_signal_tools and
        # appends bridged signals LAST, so a low cap silently drops them. Raise the
        # cap to cover the expanded family (more multiplicity = more conservative,
        # never less — e-BH still controls FDR over whatever is tested).
        agent = getattr(self, "stats_agent", None)
        if agent is not None and hasattr(agent, "_max_signal_tools"):
            try:
                from evalvitals.analysis.stats_tools import build_stats_input

                n_signals = len(build_stats_input(probe_results, data).per_case)
                current = getattr(agent, "_max_signal_tools", None)
                if current is not None:
                    agent._max_signal_tools = max(int(current), n_signals)
            except Exception as exc:  # never let the cap-bump break the loop
                logger.debug("bridge: could not raise stats signal cap: %s", exc)
        logger.info(
            "bridged %d signal row(s) into M2 as analyzer %r",
            len(synth.findings.get("per_case", [])), name,
        )

    # ──────────────────────────────────────────────────────────────────
    # Stage helpers (shared by run / run_analysis / run_confirm)
    # ──────────────────────────────────────────────────────────────────

    def _get_diagnosis_agent(self) -> "Any":
        """Resolve M3 lazily so the default Gemini fallback matches
        AutoDiagnoseLoop — a DiagnosisAgent() built eagerly would raise if
        GEMINI_API_KEY is absent even when the caller passed diagnosis_agent=None."""
        if self.diagnosis_agent is not None:
            return self.diagnosis_agent
        from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
        return DiagnosisAgent()

    def _do_m1(
        self, cycle: int, data: "Any", all_hypotheses: "list[Any]",
        timings: "dict[str, float]", *, log: bool = True,
    ) -> "tuple[dict[str, Any], list]":
        """M1: protocol-guided probing + signal bridge.

        Returns ``(probe_results, artifact_pngs)``; ``probe_results`` is empty
        when M1 produced nothing (caller decides whether to stop). When
        ``log`` is False the probe events are not written — used by
        :meth:`run_confirm` when it only needs to regenerate the stats the
        tester reads (M1/M2 already belong to the analysis phase's log)."""
        prior_modes = list(dict.fromkeys(
            h.predicted_failure_mode for h in all_hypotheses
            if getattr(h, "predicted_failure_mode", None)
        ))
        _t0 = time.monotonic()
        probe_results = self.probe_agent.probe(
            self.model,
            data,
            protocol=self.protocol,
            prior_hypotheses=all_hypotheses or None,
            hint_failure_modes=prior_modes or None,
        )
        _dt = time.monotonic() - _t0
        timings["m1"] = timings.get("m1", 0.0) + _dt
        artifact_pngs: list = []
        if log and self.run_logger:
            artifact_pngs = self.run_logger.log_probe(
                cycle, probe_results, schema=self.probe_agent.last_schema,
                judge_prompt=getattr(self.probe_agent, "last_selection_prompt", ""),
                judge_raw=getattr(self.probe_agent, "last_selection_raw", ""),
                duration_sec=_dt,
            ) or []
            _log_generated_tools(self.run_logger, cycle, "m1_probe", self.probe_agent)
        if not probe_results:
            return {}, []
        for r in probe_results.values():
            self.store.add_result(r)
        # Operationalization bridge: pre-registered recipes -> synthetic
        # "explored" analyzer Result (no-op when none configured).
        self._bridge_signals(probe_results, data)
        return probe_results, artifact_pngs

    def _explore_out_dir(self) -> "Path | None":
        """Where the explore step persists its report/tables/figures.

        Explicit ``explore_dir`` wins. Otherwise it is derived from the run
        logger so the files land INSIDE the run, beside the log, where the
        dashboard's ``_find_explore_report`` looks (``<root>/*/exploratory_report.json``):

        - a :class:`~evalvitals.eval_agent.run_context.RunContext`-backed logger
          → ``<ctx.root>/explore`` (the context owns the whole run directory;
          ``run_log.jsonl`` sits directly under root);
        - a standalone ``RunLogger("<run>/logs")`` / ``logs_confirm`` … →
          ``<run>/explore`` (a sibling of the ``logs*/`` dir, the llm_benchmark
          layout — under the log dir it would be two levels down for a
          ``logs_confirm/`` carrier log and the dashboard would miss it);
        - any other standalone run dir → ``<run_dir>/explore``.

        ``None`` (nothing persisted, context in memory only) without a logger."""
        if self._explore_dir is not None:
            return self._explore_dir
        if self.run_logger is None:
            return None
        ctx = getattr(self.run_logger, "_context", None)
        ctx_root = getattr(ctx, "root", None) if ctx is not None else None
        if ctx_root is not None:
            return Path(ctx_root) / "explore"
        run_dir = getattr(self.run_logger, "run_dir", None)
        if run_dir is None:
            return None
        run_dir = Path(run_dir)
        if run_dir.name.startswith("logs"):
            return run_dir.parent / "explore"
        return run_dir / "explore"

    def _do_explore(
        self, cycle: int, probe_results: "dict[str, Any]", data: "Any",
        timings: "dict[str, float]", *, log: bool = True,
    ) -> "Any | None":
        """Optional in-cycle explore step: free-form EDA over M1's per-case table.

        Runs between M1 and M2 when an ``explorer`` is configured. The explorer
        sees exactly what M2 sees — ``build_stats_input`` → ``per_case_to_records``
        (M1 analyzer per-case signals + PASS/FAIL labels) — writes tables and
        chart specs, and the host renders the charts. The result feeds:

        - M3, as an :class:`~evalvitals.eval_agent.stages.diagnosis.ExploreContext`
          (observations / rendered charts / caveats — descriptive, UNCONFIRMED,
          used only to decide WHICH hypotheses to propose);
        - the dashboard, via ``exploratory_report.json`` + ``tables/`` +
          ``figures/`` under :meth:`_explore_out_dir` (one directory, rewritten
          each cycle — it always holds what the *latest* M3 was shown; the
          per-cycle ``explore`` run-log events keep every cycle's counts and
          observations).

        It never touches M2's confirmatory family, M5, or the fix gate: the
        explorer's candidate-signal verdicts are host-adjudicated IN-SAMPLE
        (labelled so) and dropped from the M3 context by construction. Held-out
        confirmation of explorer recipes is the fused pipeline's job, not this
        step's. Best-effort: any failure logs a warning and returns ``None`` —
        an explorer outage must not cost the M2/M3 that already have their data.

        Returns the explorer report (or ``None``)."""
        if self.explorer is None:
            return None
        from evalvitals.analysis.operationalize import per_case_to_records
        from evalvitals.analysis.stats_tools import build_stats_input
        from evalvitals.eval_agent.stages.diagnosis import ExploreContext

        _t0 = time.monotonic()
        report: Any = None
        out_dir = self._explore_out_dir()
        try:
            inp = build_stats_input(probe_results, data)
            records = per_case_to_records(inp.per_case, inp.labels)
            if not records:
                logger.info("explore: M1 produced no per-case signals — skipping.")
                return None
            question = self._explore_question
            if not question:
                from evalvitals.eval_agent.prompts.explore_step import default_explore_question

                question = default_explore_question(self.protocol)
            report = self.explorer.explore_records(
                records, question=question, outcome_col="label",
            )
            # Host firewall: recompute every candidate verdict from sufficient
            # statistics with the M2 core (in-sample here — labelled as such so
            # nothing downstream mistakes it for a held-out result).
            try:
                from evalvitals.analysis.adjudicate import adjudicate_report

                adjudicate_report(report, split_label="in_sample")
            except Exception as exc:  # adjudication is a verdict layer, not the data
                logger.warning("explore: host adjudication failed: %s", exc)
            if out_dir is not None:
                from evalvitals.analysis.explore_run import write_report_artifacts

                # Renders chart specs from the copied tables/ CSVs, so the
                # persisted report carries each chart's absolute figure_path —
                # the PNGs M3 is shown and the dashboard displays.
                write_report_artifacts(report, out_dir)
            ctx = ExploreContext.from_report(
                report.to_dict() if hasattr(report, "to_dict") else None
            )
            if ctx is not None:
                ctx.source = "loop_explorer"
                self._explore_context = ctx
            elif getattr(report, "ok", False):
                logger.info("explore: report carried no observations/charts — "
                            "M3 keeps its previous explore context.")
        except Exception as exc:  # the explorer must never sink the loop
            logger.warning(
                "explore step failed at cycle %d (%s) — M2/M3 continue without "
                "explorer notes.", cycle, exc,
            )
        _dt = time.monotonic() - _t0
        timings["explore"] = timings.get("explore", 0.0) + _dt
        if log and self.run_logger is not None:
            try:
                self.run_logger.log_explore(
                    cycle, report, out_dir=out_dir, duration_sec=_dt,
                )
            except Exception as exc:  # logging must never break the run
                logger.warning("explore: could not log the explore event: %s", exc)
        return report

    def _do_m2(
        self, cycle: int, probe_results: "dict[str, Any]", data: "Any",
        artifact_pngs: "list", timings: "dict[str, float]", *, log: bool = True,
        confirmatory: bool = True,
    ) -> "Any":
        """M2: protocol-aware rigorous stats analysis (effect sizes + charts).

        ``confirmatory=False`` defers the e-BH validity verdict (the analysis
        phase shows distributions only); the confirm phase recomputes it."""
        _t0 = time.monotonic()
        stats_report = self.stats_agent.analyze(
            probe_results,
            model_name=repr(self.model),
            protocol=self.protocol,
            data=data,
            extra_figures=artifact_pngs,
            confirmatory=confirmatory,
        )
        _dt = time.monotonic() - _t0
        timings["m2"] = timings.get("m2", 0.0) + _dt
        if log and self.run_logger:
            self.run_logger.log_analysis(cycle, stats_report, duration_sec=_dt)
            _log_generated_tools(self.run_logger, cycle, "m2_stats", self.stats_agent)
        return stats_report

    def _do_m3(
        self, cycle: int, stats_report: "Any", prior_cycles: "list[Any]",
        timings: "dict[str, float]", *, log: bool = True,
        failure_modes: "Any | None" = None,
    ) -> "Any | None":
        """M3: hypothesis generation. Returns the diagnosis result, or ``None``
        when M3 could not run (judge unavailable / timeout / quota) — the caller
        stops gracefully on ``None`` (regression guard: an M3 timeout must not
        kill the whole run after M1+M2 succeeded). Also accrues token usage.

        ``failure_modes`` (optional clustered ``FailureModeReport``) is passed
        through to the diagnosis agent only when it accepts the parameter —
        ``None`` (the default, always the case for ``VLDiagnoseLoop``) adds
        nothing to the prompt and costs no extra call."""
        try:
            diag_agent = self._get_diagnosis_agent()
            # Retain the lazily resolved agent so report publication can reuse
            # the same judge without asking callers to inject it a second time.
            self.diagnosis_agent = diag_agent
        except Exception as exc:
            logger.warning("Could not resolve DiagnosisAgent: %s", exc)
            return None

        _t0 = time.monotonic()
        try:
            diag = _diagnose_with_optional_context(
                diag_agent, stats_report, prior_cycles, self._explore_context,
                failure_modes,
            )
        except Exception as exc:  # judge timeout/quota must not kill the loop
            logger.warning(
                "M3 diagnosis failed at cycle %d (%s) — stopping with the "
                "evidence collected so far.", cycle, exc,
            )
            return None
        _dt = time.monotonic() - _t0
        timings["m3"] = timings.get("m3", 0.0) + _dt

        _tok = getattr(diag, "tokens_used", None)
        if _tok is None:
            _tok = max(1, len(diag.raw_judge_output) // 4)
        self._tokens_used += _tok

        if log and self.run_logger:
            # Log only the figures that actually existed (the same existence
            # filter M3 applies) so the audit trail reflects what the judge saw.
            _explore_figs = None
            if self._explore_context is not None:
                from pathlib import Path as _P

                _explore_figs = [
                    f for f in self._explore_context.figure_paths if _P(f).exists()
                ]
            self.run_logger.log_diagnosis(
                cycle, diag, duration_sec=_dt, explore_figures=_explore_figs or None
            )
        return diag

    @staticmethod
    def _finalize_confirmatory_stats(stats_report: "Any") -> None:
        """Promote a descriptive (analysis-phase) stats report to confirmatory.

        The analysis phase runs M2 with the e-BH validity verdict DEFERRED
        (``descriptive_only=True``). The confirm phase recomputes e-BH FDR
        correction over the report's stats results so M5 and the dashboard see
        the family-level reject decision. No-op when already confirmatory."""
        if stats_report is None:
            return
        corr = getattr(stats_report, "corrected_rejections", None) or {}
        if getattr(stats_report, "descriptive_only", False) or corr.get("deferred"):
            from evalvitals.analysis.stats_tools import fdr_correct

            stats_report.corrected_rejections = fdr_correct(
                list(getattr(stats_report, "stats_results", None) or [])
            )
            stats_report.descriptive_only = False

    def _do_m5(
        self, cycle: int, hypotheses: "list[Any]", stats_report: "Any",
        data: "Any", timings: "dict[str, float]", *, log: bool = True,
        split_label: "str | None" = None,
    ) -> "list[Any]":
        """M5: statistical test + protocol consistency for each hypothesis,
        writing the verdict back onto ``hypothesis.status``.

        ``split_label`` tags each result's evidence with which data split the
        test read (``"confirm_holdout"`` for the held-out pass) BEFORE the
        result is logged, so the marker reaches the run log too."""
        _t0 = time.monotonic()
        test_results = self.hypothesis_tester.test(
            hypotheses,
            stats_report,
            data,
            protocol=self.protocol,
        )
        _dt = time.monotonic() - _t0
        timings["m5"] = timings.get("m5", 0.0) + _dt
        for tr in test_results:
            tr.hypothesis.status = tr.status
            if split_label and isinstance(getattr(tr, "evidence", None), dict):
                tr.evidence["split"] = split_label
        if log and self.run_logger:
            # Reuse the surgery log slot for M5 results (backward compat).
            for tr in test_results:
                _iv = _make_intervention_result_from_test(tr)
                self.run_logger.log_surgery(
                    cycle, tr.hypothesis, _iv, duration_sec=_dt,
                    judge_prompt=getattr(tr, "judge_prompt", None) or None,
                    judge_raw=getattr(tr, "judge_raw", None) or None,
                )
        return test_results

    def _m5_holdout_pass(
        self,
        hypotheses: "list[Any]",
        confirm: "Any",
        analyzer_names: "list[str]",
        timings: "dict[str, float]",
    ) -> "tuple[list[Any], str]":
        """The M5 pass — on the held-out confirm split.

        With a confirm split in play this is the ONLY hypothesis test: M3's
        hypotheses are taken straight to data M1/M2/M3 never mined. The same
        analyzers as the last explore cycle are re-run on the confirm split
        (pinned via ``ProbeAgent.probe(analyzers=)`` — a fresh judge selection
        could miss the designated signals); with no names to pin (a reloaded
        run without a stats report) the probe agent selects normally.

        Returns ``(test_results, status)`` with status ``"confirmed"`` (the
        pass ran) or ``"failed"`` (the confirm re-probe produced nothing — no
        hypothesis can be verified this run).
        """
        pinned = [n for n in analyzer_names if not n.startswith("generated:")]
        logger.info(
            "M5 (held-out): testing %d hypothesis(es) on the %d-case confirm "
            "split (analyzers: %s)",
            len(hypotheses), len(list(confirm)),
            ", ".join(pinned) or "<probe agent's own selection>",
        )
        _t0 = time.monotonic()
        try:
            if pinned:
                probe_results = self.probe_agent.probe(
                    self.model, confirm, protocol=self.protocol,
                    prior_hypotheses=hypotheses or None, analyzers=pinned,
                )
            else:
                probe_results = self.probe_agent.probe(
                    self.model, confirm, protocol=self.protocol,
                    prior_hypotheses=hypotheses or None,
                )
        except TypeError:
            # A custom probe agent that predates the ``analyzers=`` hook.
            probe_results = self.probe_agent.probe(
                self.model, confirm, protocol=self.protocol,
                prior_hypotheses=hypotheses or None,
            )
        timings["m5_holdout_m1"] = timings.get("m5_holdout_m1", 0.0) + (
            time.monotonic() - _t0)
        if not probe_results:
            logger.warning(
                "M5 (held-out): the confirm-split re-probe produced no results "
                "— no hypothesis can be verified this run."
            )
            return [], "failed"
        self._bridge_signals(probe_results, confirm)
        if self.run_logger:
            self.run_logger.current_cycle = -1
            self.run_logger.log_probe(
                -1, probe_results,
                schema=getattr(self.probe_agent, "last_schema", None),
            )
        stats_confirm = self._do_m2(
            -1, probe_results, confirm, [], timings, confirmatory=True
        )
        results = self._do_m5(-1, hypotheses, stats_confirm, confirm, timings,
                              split_label="confirm_holdout")
        return results, "confirmed"

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(self, data: "CaseBatch") -> VLDiagnoseReport:
        """Drive the M1→M2→M3→M5 loop to convergence.

        Args:
            data: Cases to analyse (should carry :class:`~evalvitals.core.case.Label`
                  values for the M5 statistical tests to work).

        Returns:
            :class:`VLDiagnoseReport` with the final state.
            Call :meth:`run_m4` on this to get a fix proposal (Plan A).
        """
        all_hypotheses: list[Any] = []
        all_test_results: list[Any] = []
        final_stats_report = None
        stopped_by = _STOPPED_BY_MAX
        prior_cycles: list[dict[str, Any]] = []
        last_analyzer_names: list[str] = []
        self._tokens_used = 0

        # Per-stage wall-clock totals (seconds) for the loop_end cost profile.
        timings: dict[str, float] = {}

        # Held-out CONFIRM split (leak #3): M1-M5 see only EXPLORE; the post-loop
        # fix/surgery validate on the frozen CONFIRM partition (run_m4/run_fix
        # re-derive the same deterministic split from the same input batch).
        explore, confirm = self._split_explore_confirm(data)
        if confirm is not None:
            logger.info(
                "confirm split: explore=%d cases, confirm=%d held out (frac=%.2f)",
                len(list(explore)), len(list(confirm)), self.confirm_split,
            )
            data = explore

        # With a confirm split (and m5_holdout on), M5 runs ONCE after the
        # loop, on the held-out split — the cycles only mine (M1→M3). Without
        # one there is no held-out data, so M5 stays in-cycle as before.
        holdout_mode = confirm is not None and self.m5_holdout

        # Forward the RunLogger into the agents so the probe / stats tool
        # generators record their tool-synthesis attempts ("tool_codegen" events).
        _attach_run_logger(self.run_logger, self.probe_agent, self.stats_agent)
        if self.run_logger is not None:
            self.run_logger.log_run_start(
                _run_config(self, data, loop_name="VLDiagnoseLoop")
            )
            self.run_logger.log_cases(data)

        for cycle in range(self.max_cycles):
            if self.token_budget > 0 and self._tokens_used >= self.token_budget:
                logger.warning(
                    "Token budget %d exhausted after %d cycles", self.token_budget, cycle
                )
                stopped_by = _STOPPED_BY_BUDGET
                break

            if self.run_logger is not None:
                self.run_logger.current_cycle = cycle

            # ── M1: protocol-guided probing (+ signal bridge) ─────────
            probe_results, artifact_pngs = self._do_m1(
                cycle, data, all_hypotheses, timings
            )
            if not probe_results:
                logger.info("M1 produced no probe results — stopping.")
                stopped_by = _STOPPED_BY_NO_PROBE
                break
            last_analyzer_names = list(probe_results)

            # ── explore (optional): free-form EDA on M1's table → M3 notes ──
            self._do_explore(cycle, probe_results, data, timings)

            # ── M2: protocol-aware stats analysis ────────────────────
            stats_report = self._do_m2(
                cycle, probe_results, data, artifact_pngs, timings
            )
            final_stats_report = stats_report

            if self.analysis_only:
                stopped_by = _STOPPED_BY_NO_HYPS
                break

            # ── M3: hypothesis generation ("AI scientist") ────────────
            diag = self._do_m3(cycle, stats_report, prior_cycles, timings)
            if diag is None or not diag.hypotheses:
                if diag is not None:
                    logger.info("M3 produced no hypotheses at cycle %d.", cycle)
                stopped_by = _STOPPED_BY_NO_HYPS
                break

            for h in diag.hypotheses:
                self.store.add_hypothesis(h)
            all_hypotheses.extend(diag.hypotheses)

            # ── M5: hypothesis testing (stats + protocol consistency) ─
            # In holdout mode M5 is deferred to the single held-out pass
            # after the loop: testing here would re-use the explore data
            # the hypotheses were mined from, and its verdict must not
            # steer the loop (no early stop / no extra cycles keyed on
            # SUPPORTED-or-not).
            if not holdout_mode:
                test_results = self._do_m5(
                    cycle, diag.hypotheses, stats_report, data, timings
                )
                all_test_results.extend(test_results)

                # ── Stopping criteria ────────────────────────────────────
                if self.hypothesis_tester.stopping_criteria_met(test_results, self.protocol):
                    logger.info(
                        "Stopping criteria met at cycle %d: verified, protocol-consistent "
                        "hypothesis found.",
                        cycle,
                    )
                    stopped_by = _STOPPED_BY_CRITERIA
                    break

            # Build prior-cycles context for next M3 call
            prior_cycles.append({
                "cycle": cycle,
                "severity": stats_report.severity,
                "hypotheses": [
                    {
                        "statement": h.statement,
                        "failure_mode": h.predicted_failure_mode,
                        "status": h.status.value if h.status else "pending",
                    }
                    for h in diag.hypotheses
                ],
            })

        # The M5 pass: with a confirm split in play the hypotheses are tested
        # ONCE, on the held-out split — the same discipline run_fix already
        # applies to candidates. Without one, the in-cycle results stand.
        m5_holdout_status: "str | None" = None
        if holdout_mode and all_hypotheses:
            all_test_results, m5_holdout_status = self._m5_holdout_pass(
                all_hypotheses, confirm, last_analyzer_names, timings,
            )
        verified = self.hypothesis_tester.best_hypotheses(all_test_results)

        report = VLDiagnoseReport(
            cycles=cycle + 1 if self.max_cycles > 0 else 0,  # type: ignore[possibly-undefined]
            resolved=bool(verified),
            stopped_by=stopped_by,
            verified_hypotheses=verified,
            final_hypotheses=all_hypotheses,
            all_test_results=all_test_results,
            final_stats_report=final_stats_report,
            store=self.store,
            m5_holdout=m5_holdout_status,
            _run_id=self._run_id,
        )
        if self.run_logger:
            self.run_logger.log_loop_end(
                report, tokens_used=self._tokens_used, timings=timings
            )
        return report

    def run_analysis(self, data: "CaseBatch") -> VLDiagnoseReport:
        """Phase 1 — analyse + propose, WITHOUT confirming (no M5, no fix).

        Runs a single **M1 → M2 → M3** pass: select+execute analyzers (M1),
        rigorous protocol-aware stats + charts (M2, e-BH adjudication kept),
        and propose root-cause hypotheses (M3) — then stop. This is the path
        that feeds the dashboard *before* deciding whether to confirm and
        repair.

        The returned :class:`VLDiagnoseReport` carries:
          - ``final_hypotheses``   — the M3 proposals (UNCONFIRMED), and
          - ``final_stats_report`` — the M2 report,
        with ``all_test_results`` / ``verified_hypotheses`` left empty (M5 has
        not run). Persist ``final_hypotheses`` (via
        :func:`~evalvitals.eval_agent.hypothesis.hypothesis_to_dict`) and
        ``final_stats_report``, then hand them to :meth:`run_confirm` for the
        deferred confirmation + repair phase.

        Unlike :meth:`run`, there is no M5 stopping signal, so this is a single
        pass (one M1→M2→M3), not a multi-cycle loop; ``max_cycles`` is ignored.
        """
        self._tokens_used = 0
        timings: dict[str, float] = {}

        # Same deterministic explore/confirm partition as run() / run_fix() —
        # M1-M3 see only EXPLORE so the deferred fix can validate on the
        # untouched CONFIRM partition. No-op when confirm_split=0.
        explore, confirm = self._split_explore_confirm(data)
        if confirm is not None:
            logger.info(
                "confirm split: explore=%d cases, confirm=%d held out (frac=%.2f)",
                len(list(explore)), len(list(confirm)), self.confirm_split,
            )
            data = explore

        _attach_run_logger(self.run_logger, self.probe_agent, self.stats_agent)
        if self.run_logger is not None:
            self.run_logger.current_cycle = 0
            self.run_logger.log_run_start(
                _run_config(self, data, loop_name="VLDiagnoseLoop.analysis")
            )
            self.run_logger.log_cases(data)

        all_hypotheses: list[Any] = []
        final_stats_report = None
        stopped_by = _STOPPED_BY_ANALYSIS

        probe_results, artifact_pngs = self._do_m1(0, data, all_hypotheses, timings)
        if not probe_results:
            logger.info("M1 produced no probe results — stopping.")
            stopped_by = _STOPPED_BY_NO_PROBE
        else:
            # Optional explore step (descriptive EDA → M3 notes + dashboard files).
            self._do_explore(0, probe_results, data, timings)
            # Descriptive M2: effect sizes + charts, but DEFER the e-BH validity
            # verdict to run_confirm so the analysis dashboard shows no
            # "supported/not-supported" claim (Q2: no validity before confirm).
            final_stats_report = self._do_m2(
                0, probe_results, data, artifact_pngs, timings, confirmatory=False
            )
            diag = self._do_m3(0, final_stats_report, [], timings)
            if diag is None or not diag.hypotheses:
                # Stats + dashboard are still valid; just no hypotheses to confirm.
                if diag is not None:
                    logger.info("M3 produced no hypotheses.")
                stopped_by = _STOPPED_BY_NO_HYPS
            else:
                for h in diag.hypotheses:
                    self.store.add_hypothesis(h)
                all_hypotheses.extend(diag.hypotheses)

        report = VLDiagnoseReport(
            cycles=1,
            stopped_by=stopped_by,
            verified_hypotheses=[],
            final_hypotheses=all_hypotheses,
            all_test_results=[],
            final_stats_report=final_stats_report,
            store=self.store,
            _run_id=self._run_id,
        )
        if self.run_logger:
            self.run_logger.log_loop_end(
                report, tokens_used=self._tokens_used, timings=timings
            )
        return report

    def run_confirm(
        self,
        data: "CaseBatch",
        hypotheses: "list[Any]",
        *,
        stats_report: "Any | None" = None,
    ) -> VLDiagnoseReport:
        """Phase 2a — confirm previously-proposed hypotheses with M5.

        Runs **M5** (:class:`~evalvitals.eval_agent.stages.hypothesis_tester.HypothesisTester`)
        on ``hypotheses`` — typically reloaded from :meth:`run_analysis`'s output
        via :func:`~evalvitals.eval_agent.hypothesis.hypothesis_from_dict` — so
        the *same* hypotheses the dashboard showed are the ones confirmed.

        ``stats_report`` is the M2 report M5 reads its rigorous evidence from
        (effect + CI + e-value, FDR-corrected). Pass the
        ``final_stats_report`` persisted by :meth:`run_analysis` to confirm
        against the *exact* statistics the dashboard displayed; when omitted,
        M1→M2 are re-run silently (not logged — they belong to the analysis
        phase) to regenerate it.

        Returns a :class:`VLDiagnoseReport` with ``all_test_results`` and
        ``verified_hypotheses`` populated. Feed it into :meth:`run_m4` /
        :meth:`run_fix` for the repair step.
        """
        self._tokens_used = 0
        timings: dict[str, float] = {}
        hypotheses = list(hypotheses or [])

        # Mirror run()'s partition: the screening M5 reads the supplied (or
        # regenerated) EXPLORE stats the hypotheses were mined from; the
        # held-out confirmation below re-tests the screened ones on CONFIRM
        # (run_m4/run_fix re-derive the same partition for their validation).
        explore, confirm = self._split_explore_confirm(data)
        if confirm is not None:
            data = explore

        _attach_run_logger(self.run_logger, self.probe_agent, self.stats_agent)
        if self.run_logger is not None:
            self.run_logger.current_cycle = 0
            self.run_logger.log_run_start(
                _run_config(self, data, loop_name="VLDiagnoseLoop.confirm")
            )
            self.run_logger.log_cases(data)

        # Regenerate the stats the tester needs only when not supplied. The M1/M2
        # events are NOT logged here — they were recorded in the analysis phase,
        # and re-logging them would double up the dashboard's analysis story.
        if stats_report is None:
            probe_results, artifact_pngs = self._do_m1(
                0, data, hypotheses, timings, log=False
            )
            if not probe_results:
                logger.warning(
                    "run_confirm: probe produced no results and no stats_report "
                    "was supplied — cannot confirm."
                )
                report = VLDiagnoseReport(
                    cycles=1, stopped_by=_STOPPED_BY_NO_PROBE,
                    final_hypotheses=hypotheses, store=self.store, _run_id=self._run_id,
                )
                if self.run_logger:
                    self.run_logger.log_loop_end(
                        report, tokens_used=self._tokens_used, timings=timings
                    )
                return report
            stats_report = self._do_m2(
                0, probe_results, data, artifact_pngs, timings, log=False
            )

        # The confirm phase OWNS the validity verdict: if the reused report came
        # from the descriptive analysis phase (e-BH deferred), compute e-BH now,
        # flip descriptive_only off, and log the confirmatory M2 so the dashboard
        # surfaces the signal validity it withheld before confirmation.
        self._finalize_confirmatory_stats(stats_report)
        if self.run_logger and stats_report is not None:
            self.run_logger.log_analysis(0, stats_report)

        test_results: list[Any] = []
        m5_holdout_status: "str | None" = None
        if hypotheses:
            for h in hypotheses:
                self.store.add_hypothesis(h)
            if confirm is not None and self.m5_holdout:
                # The one M5 pass, on the held-out split (never the explore
                # stats the hypotheses were mined from). The analyzer set to
                # re-run there is recovered from the supplied/regenerated
                # stats report's signal keys; with none recoverable the probe
                # agent selects on the confirm split itself.
                analyzer_names = _analyzer_names_from_stats(stats_report)
                test_results, m5_holdout_status = self._m5_holdout_pass(
                    hypotheses, confirm, analyzer_names, timings,
                )
            else:
                test_results = self._do_m5(0, hypotheses, stats_report, data, timings)
        else:
            logger.info("run_confirm: no hypotheses to confirm.")

        verified = self.hypothesis_tester.best_hypotheses(test_results)
        stopped_by = _STOPPED_BY_CRITERIA if verified else _STOPPED_BY_MAX

        report = VLDiagnoseReport(
            cycles=1,
            resolved=bool(verified),
            stopped_by=stopped_by,
            verified_hypotheses=verified,
            final_hypotheses=hypotheses,
            all_test_results=test_results,
            final_stats_report=stats_report,
            store=self.store,
            m5_holdout=m5_holdout_status,
            _run_id=self._run_id,
        )
        if self.run_logger:
            self.run_logger.log_loop_end(
                report, tokens_used=self._tokens_used, timings=timings
            )
        return report

    def run_m4(
        self,
        report: VLDiagnoseReport,
        data: "CaseBatch",
        *,
        allow_unverified: bool = False,
    ) -> "Any | None":
        """Plan A: run the M4 intervention experiment on the best hypothesis.

        Called *after* :meth:`run` to avoid polluting the inner loop with
        fix-execution noise.  Operates on the highest-confidence verified
        hypothesis from :attr:`VLDiagnoseReport.verified_hypotheses`; with
        ``allow_unverified=True`` and no verified hypothesis it falls back to
        the best *unverified* one (see :func:`_unverified_hypotheses`) — the
        experiment is then a genuine test of a lead M5 could not decide, and
        its verdict (supported / refuted) is what ``run_fix`` reads.

        Args:
            report: Returned by :meth:`run`.
            data:   Original case batch (needed by the surgery agent).
            allow_unverified: Fall back to the best unverified hypothesis when
                    M5 verified none (default False: verified only).

        Returns:
            :class:`~evalvitals.eval_agent.surgery.InterventionResult` or
            ``None`` if there is no hypothesis to act on.
        """
        if report.verified_hypotheses:
            best_hyp = report.verified_hypotheses[0].hypothesis
            unverified = False
        elif allow_unverified:
            candidates = _unverified_hypotheses(report)
            if not candidates:
                logger.info("run_m4: no hypotheses at all to act on.")
                return None
            best_hyp = candidates[0]
            unverified = True
            logger.info(
                "run_m4: no verified hypothesis — experimenting on the best UNVERIFIED "
                "one (allow_unverified=True): %s", str(getattr(best_hyp, "statement", best_hyp))[:120],
            )
        else:
            logger.info("run_m4: no verified hypotheses to act on.")
            return None

        # Confirm the fix on the held-out partition (leak #3): the loop generated
        # the hypothesis on EXPLORE, so M4 must operate on CONFIRM — data it never
        # mined. Deterministic re-split of the same batch; no-op when off.
        _, confirm = self._split_explore_confirm(data)
        if confirm is not None:
            data = confirm

        results: dict[str, Any] = (
            report.final_stats_report.raw_results
            if report.final_stats_report is not None
            else {}
        )
        iv = self.surgery_agent.operate(
            best_hyp,
            self.model,
            results,
            data,
        )
        try:
            iv.evidence = dict(getattr(iv, "evidence", None) or {})
            iv.evidence["hypothesis_was_verified"] = not unverified
        except Exception:  # evidence is informational
            pass
        report.fix_proposal = iv
        # M4 runs *after* the loop, so log its experiment separately — the
        # generated script(s), the run output, the agent's thinking and a
        # snapshot of the workspace.  ``cycle=-1`` marks it as post-loop.
        if self.run_logger is not None:
            try:
                self.run_logger.log_experiment(-1, best_hyp, iv, module="m4")
            except Exception as exc:  # logging must never break the fix step
                logger.warning("run_m4: log_experiment failed: %s", exc)
        return iv

    def run_fix(
        self,
        report: VLDiagnoseReport,
        data: "CaseBatch",
        max_tier: "str | Any | None" = None,
        fix_agent: "Any | None" = None,
        auto_escalate: bool = False,
    ) -> "Any":
        """Post-loop fix module: tiered, validated repair attempts.

        By default the allowed tier is fixed (default L2) and there is no
        automatic escalation — the returned
        :class:`~evalvitals.eval_agent.stages.fix_agent.FixOutcome` carries a
        recommendation when nothing validates.

        When ``auto_escalate=True`` the agent steps through the intervention
        ladder L0/L1/L2 → L3a → L3b, stopping as soon as a candidate validates.
        Each escalation round receives the full history of prior failed
        attempts so the judge can generate fundamentally different strategies
        rather than repeating what already failed.

        Args:
            report:         Returned by :meth:`run` (uses ``verified_hypotheses``,
                            falling back to the last cycle's proposals).
            data:           Original case batch (validated with paired McNemar
                            against the unmodified baseline).
            max_tier:       Ceiling tier: "L0", "L1", "L2", "L3a", "L3b", "L4".
                            Defaults to L3b when ``auto_escalate=True``, or the
                            agent's configured tier otherwise.
            fix_agent:      Per-call override of :attr:`fix_agent`.
            auto_escalate:  When True, step through tiers automatically,
                            feeding prior failure context to each round.
        """
        from evalvitals.eval_agent.stages.fix_tiers import FixTier, parse_tier

        # Validate the fix on the held-out partition (leak #3): the hypotheses
        # were generated on EXPLORE, so the deployed repair must be confirmed on
        # CONFIRM — cases the loop never used to pick the fix. Deterministic
        # re-split of the same batch; no-op when confirm_split=0.
        explore, confirm = self._split_explore_confirm(data)
        if confirm is not None:
            data = confirm

        # Tier escalation is feedback-driven: the next tier is authored after
        # observing which CONFIRM cases the previous tier repaired/broke.  That
        # invalidates the holdout just as surely as tuning a prompt on it.  A
        # held-out run therefore pre-registers one repair family at the caller's
        # configured ceiling; a failed attempt needs a fresh split/run seed.
        if confirm is not None and auto_escalate:
            logger.warning(
                "run_fix: disabling adaptive tier escalation on held-out "
                "confirmation data; use a fresh split for another attempt"
            )
            auto_escalate = False

        agent = fix_agent or self.fix_agent
        # M4's intervention experiment (run_m4) can REFUTE the very hypothesis
        # M5 verified. A refuted hypothesis must not reach the proposer as
        # "verified": on qwen3.5-2b/bbh_tracking7 the fix judge/coder were told
        # the M4-refuted grading-mismatch hypothesis was verified and half the
        # coded pipeline's design served it. Refuted ones are dropped from the
        # list and passed to the proposer as "do not build on these".
        refuted_ids, refuted_notes = _m4_refuted(report)
        hypotheses = [
            tr.hypothesis for tr in report.verified_hypotheses
            if _hyp_key(tr.hypothesis) not in refuted_ids
        ]
        hypotheses_note = ""
        if not hypotheses:
            # No verified hypothesis: the fix still runs on the best UNVERIFIED
            # leads (M5-tested, non-refuted, highest confidence first; else the
            # last cycle's proposals). They reach the proposer flagged as
            # leads, not facts — the fix gate is the candidate validation, not
            # the hypothesis, so this is safe; an M4 experiment that supported
            # one of them upgrades it in the note.
            hypotheses = [
                h for h in _unverified_hypotheses(report)
                if _hyp_key(h) not in refuted_ids
            ][:3]
            supported = _m4_supported_key(report)
            hypotheses_note = (
                "UNVERIFIED: M5 found no statistically significant evidence for these "
                "hypotheses (they are the best-scoring leads, not established mechanisms)"
                + ("; the M4 intervention experiment SUPPORTED the first one"
                   if supported and hypotheses and _hyp_key(hypotheses[0]) == supported else "")
                + ". Treat them as hints about WHERE to intervene; the candidate "
                "validation, not the hypothesis, decides."
            )
        context = _fix_context_from_report(
            report,
            example_cases=explore if confirm is not None else None,
            explore_context=self._explore_context,
            protocol=self.protocol,
            refuted=refuted_notes,
            refuted_ids=refuted_ids,
        )
        context.hypotheses_note = hypotheses_note

        if auto_escalate:
            _LADDER = [
                FixTier.L2_SCAFFOLD,
                FixTier.L3A_INTERNALS_READ,
                FixTier.L3B_INTERNALS_WRITE,
            ]
            ceiling = (
                parse_tier(max_tier) if max_tier is not None
                else FixTier.L3B_INTERNALS_WRITE
            )
            # Suppress per-round log_fix so we can emit one combined outcome.
            agent_logger = getattr(agent, "run_logger", None)
            agent.run_logger = None

            all_attempted: "list" = []
            all_prior: "list" = []
            last_outcome = None

            try:
                for tier in _LADDER:
                    if tier > ceiling:
                        break
                    agent.max_tier = tier
                    logger.info("run_fix: trying tier %s (%d prior attempt(s))",
                                tier.label, len(all_prior))
                    outcome = _propose_and_validate(
                        agent, self.model, data, hypotheses,
                        prior_attempts=all_prior if all_prior else None,
                        context=context,
                    )
                    all_attempted.extend(outcome.attempted)
                    all_prior.extend(v for v in outcome.attempted if not v.fixed)
                    last_outcome = outcome
                    if outcome.fixed:
                        logger.info("run_fix: fixed at tier %s", tier.label)
                        break
                    logger.info("run_fix: tier %s exhausted — escalating", tier.label)
            finally:
                agent.run_logger = agent_logger

            # Merge all rounds into one combined outcome and emit once. The
            # merged set spans every escalated tier, so it is a LARGER best-of-N
            # family than any single tier — re-apply e-BH FDR control over the
            # whole union (mirrors FixAgent.propose_and_validate) instead of an
            # uncorrected max, or auto-escalation would re-open the multiplicity
            # leak it was meant to respect.
            if last_outcome is not None:
                last_outcome.attempted = all_attempted
                last_outcome.max_tier = ceiling
                tested = [v for v in all_attempted if v.e_value is not None]
                survivors = agent._ebh_survivors(tested)
                last_outcome.ebh_survivors = sorted(
                    v.candidate.name for v in tested if id(v) in survivors)
                winners = [v for v in all_attempted
                           if v.fixed and id(v) in survivors]
                if winners:
                    last_outcome.best = max(
                        winners, key=lambda v: (v.effect or 0.0, -v.n_broken)
                    )
                    last_outcome.fixed = True
                else:
                    last_outcome.best = None
                    last_outcome.fixed = False
                try:
                    if agent_logger is not None:
                        agent_logger.log_fix(last_outcome)
                except Exception as exc:
                    logger.debug("run_fix: combined log_fix failed: %s", exc)

            report.fix_outcome = last_outcome
            return last_outcome

        # Non-escalating path.  With a held-out split this is a genuine
        # two-stage repair experiment: the agent may iterate and select on
        # EXPLORE, then exactly one frozen candidate is tested on CONFIRM.
        # Selection statistics are descriptive only; the final paired gate is
        # computed from CONFIRM alone.
        if max_tier is not None:
            agent.max_tier = parse_tier(max_tier)
        if confirm is not None:
            from evalvitals.eval_agent.stages.fix_agent import FixOutcome

            agent_logger = getattr(agent, "run_logger", None)
            agent.run_logger = None
            try:
                selection = _propose_and_validate(
                    agent, self.model, explore, hypotheses, context=context
                )
            finally:
                agent.run_logger = agent_logger

            # Custom/legacy agents may return an opaque application-specific
            # result rather than FixOutcome.  They cannot participate in the
            # built-in frozen-candidate confirmation protocol, but preserving
            # their return contract is preferable to turning a diagnostic run
            # into an AttributeError.
            if not hasattr(selection, "attempted"):
                report.fix_outcome = selection
                return selection

            executed = [
                validation
                for validation in selection.attempted
                if validation.n_pairs > 0 and validation.effect is not None
            ]
            improving = [
                validation
                for validation in executed
                if validation.n_fixed > validation.n_broken
            ]
            selected = max(
                improving,
                key=lambda validation: (
                    validation.effect or 0.0,
                    -validation.n_broken,
                    validation.n_fixed,
                ),
                default=None,
            )
            audit = [
                {
                    "name": validation.candidate.name,
                    "tier": validation.candidate.tier.label,
                    "n_pairs": validation.n_pairs,
                    "n_fixed": validation.n_fixed,
                    "n_broken": validation.n_broken,
                    "effect": validation.effect,
                    "verdict": validation.verdict,
                }
                for validation in selection.attempted
            ]
            if selected is None:
                outcome = FixOutcome(
                    max_tier=agent.max_tier,
                    repair_rounds=selection.repair_rounds,
                    selection_attempted=audit,
                    recommendation={
                        "recommend_tier": agent.max_tier.label,
                        "reason": (
                            "no EXPLORE candidate had positive net repairs; "
                            "CONFIRM was left untouched"
                        ),
                    },
                )
            else:
                validation = agent.validate_candidate(
                    self.model,
                    confirm,
                    selected.candidate,
                )
                survivors = agent._ebh_survivors(
                    [validation] if validation.e_value is not None else []
                )
                survived = id(validation) in survivors
                outcome = FixOutcome(
                    max_tier=agent.max_tier,
                    attempted=[validation],
                    best=validation if validation.fixed and survived else None,
                    fixed=bool(validation.fixed and survived),
                    repair_rounds=selection.repair_rounds,
                    ebh_survivors=[validation.candidate.name] if survived else [],
                    selection_attempted=audit,
                    selected_on_explore=selected.candidate.name,
                )
                if not outcome.fixed:
                    outcome.recommendation = {
                        "recommend_tier": agent.max_tier.label,
                        "reason": (
                            "the candidate selected on EXPLORE did not validate "
                            "on untouched CONFIRM"
                        ),
                    }
                outcome.refine_signal = agent._refine_signal([validation], confirm)
            try:
                if agent_logger is not None:
                    agent_logger.log_fix(outcome)
            except Exception as exc:
                logger.debug("run_fix: held-out log_fix failed: %s", exc)
            report.fix_outcome = outcome
            return outcome

        outcome = _propose_and_validate(agent, self.model, data, hypotheses, context=context)
        report.fix_outcome = outcome
        return outcome


def _analyzer_names_from_stats(stats_report: "Any") -> "list[str]":
    """Analyzer names whose signals the M2 report actually tested.

    Recovered from each result's signal key (``"analyzer.metric"``) so a
    reloaded report (confirm-only mode) can pin the same analyzer set for the
    held-out M5 re-probe without the original probe schema.
    """
    names: "list[str]" = []
    for r in list(getattr(stats_report, "stats_results", None) or []):
        cfg = getattr(r, "config", None) or {}
        det = getattr(r, "details", None) or {}
        sig = str(cfg.get("signal") or det.get("signal") or "")
        if "." in sig:
            name = sig.split(".", 1)[0]
            if name and name not in names:
                names.append(name)
    return names
