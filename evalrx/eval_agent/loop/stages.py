"""The stages of a diagnosis run: M1 probe, explore, M2 stats, M3 diagnosis, M4 test.

:class:`Stages` runs one stage at a time and records it (run log, contract
payloads, timings). It holds the stage dependencies only — the model, the
protocol, the agents, the store, the logger — never the loop; the loop builds
one per run (:meth:`~evalrx.eval_agent.loop.diagnose.VLDiagnoseLoop.stages`)
and a policy decides which stage runs next. :class:`StageMemory` is what
outlives one run on the same loop (the explorer notes M3 and the fix stage
read, the default M3 agent, the token count).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalrx.eval_agent.loop.metadata import (
    _log_generated_tools,
    _make_intervention_result_from_test,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evalrx.eval_agent.store import Store

logger = logging.getLogger(__name__)


@dataclass
class StageMemory:
    """What outlives one run on the same loop."""

    #: Explorer notes M3 is shown: a Step-1 ``explore_report`` or the latest
    #: in-cycle explore step. The fix stage reads it too.
    explore_context: Any = None
    #: The default DiagnosisAgent, once built (when none was configured).
    diagnosis_agent: Any = None
    #: Judge tokens spent in the current run (reset at each run's start).
    tokens_used: int = 0


@dataclass
class Stages:
    """Runs and records one stage at a time. Built by the loop for each run."""

    model: Any
    protocol: Any
    probe_agent: Any
    stats_agent: Any
    hypothesis_tester: Any
    store: "Store"
    run_logger: Any
    emitter: Any
    memory: StageMemory
    diagnosis_agent: Any = None
    diagnosis_retries: int = 0
    explorer: Any = None
    explore_dir: "Path | None" = None
    explore_question: str = ""
    signal_recipes: list[Any] = field(default_factory=list)
    bridge_analyzer_name: str = "explored"

    def _emit(self, name: str, build: "Any") -> None:
        """Emit one stage payload; never raises into the pipeline."""
        if self.emitter is not None:
            self.emitter.emit(name, build)

    def bridge_signals(self, probe_results: "dict[str, Any]", data: "Any | None") -> None:
        """Compile pre-registered signal recipes into a synthetic analyzer Result
        and inject it into *probe_results* so M2/M3/M4 see the bridged composite
        signals through the standard findings["per_case"] contract. No-op when no
        recipes are configured. Never raises into the loop."""
        if not self.signal_recipes:
            return
        # Never silently overwrite a real analyzer that already used this key.
        name = self.bridge_analyzer_name
        if name in probe_results:
            base, n = name, 1
            while name in probe_results:
                name, n = f"{base}_bridge{n}", n + 1
            logger.warning(
                "bridge analyzer name %r collides with a real analyzer; "
                "injecting under %r instead", base, name,
            )
        try:
            from evalrx.analysis.operationalize import bridge_recipes_to_result

            synth = bridge_recipes_to_result(
                self.signal_recipes, probe_results, data,
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
        # not optional. The plan caps at the stats agent's max_signal_tools and
        # appends bridged signals LAST, so a low cap silently drops them. Raise the
        # cap to cover the expanded family (more multiplicity = more conservative,
        # never less — e-BH still controls FDR over whatever is tested).
        agent = getattr(self, "stats_agent", None)
        if agent is not None and hasattr(agent, "_max_signal_tools"):
            try:
                from evalrx.analysis.stats_tools import build_stats_input

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

    def diagnosis_agent_or_default(self) -> "Any":
        """M3's agent: the configured one, else a default DiagnosisAgent built once
        per loop (built lazily: DiagnosisAgent() raises without its judge's
        credentials even when M3 never runs)."""
        if self.diagnosis_agent is not None:
            return self.diagnosis_agent
        if self.memory.diagnosis_agent is None:
            from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent

            self.memory.diagnosis_agent = DiagnosisAgent()
        return self.memory.diagnosis_agent

    def probe(
        self, cycle: int, data: "Any", all_hypotheses: "list[Any]",
        timings: "dict[str, float]", *, log: bool = True,
        analyzers: "list[str] | None" = None,
        analyzer_params: "dict[str, dict[str, Any]] | None" = None,
    ) -> "tuple[dict[str, Any], list]":
        """M1: protocol-guided probing + signal bridge.

        Returns ``(probe_results, artifact_pngs)``; ``probe_results`` is empty
        when M1 produced nothing (caller decides whether to stop). When
        ``log`` is False the probe events are not written — used by
        :meth:`run_confirm` when it only needs to regenerate the stats the
        tester reads (M1/M2 already belong to the analysis phase's log).

        ``analyzers`` / ``analyzer_params`` name the probes and their arguments
        explicitly, bypassing selection (see :meth:`ProbeAgent.probe`); the
        agentic loop's ``run_probe`` tool passes them through."""
        prior_modes = list(dict.fromkeys(
            h.predicted_failure_mode for h in all_hypotheses
            if getattr(h, "predicted_failure_mode", None)
        ))
        # Named calls run exactly what was named (no cap, no backfill).
        named: dict[str, Any] = {"analyzers": analyzers, "exact": True} if analyzers is not None else {}
        if analyzer_params:
            named["analyzer_params"] = analyzer_params
        _t0 = time.monotonic()
        probe_results = self.probe_agent.probe(
            self.model,
            data,
            protocol=self.protocol,
            prior_hypotheses=all_hypotheses or None,
            hint_failure_modes=prior_modes or None,
            **named,
        )
        _dt = time.monotonic() - _t0
        timings["m1"] = timings.get("m1", 0.0) + _dt
        artifact_pngs: list = []
        if log and self.run_logger:
            artifact_pngs = self.run_logger.log_probe(
                cycle, probe_results, schema=self.probe_agent.last_schema,
                cases=data,
                judge_prompt=getattr(self.probe_agent, "last_selection_prompt", ""),
                judge_raw=getattr(self.probe_agent, "last_selection_raw", ""),
                duration_sec=_dt,
            ) or []
            _log_generated_tools(self.run_logger, cycle, "m1_probe", self.probe_agent)
        self._emit_m1(cycle, probe_results, data, _dt)
        if not probe_results:
            return {}, []
        for r in probe_results.values():
            self.store.add_result(r)
        # Operationalization bridge: pre-registered recipes -> synthetic
        # "explored" analyzer Result (no-op when none configured).
        self.bridge_signals(probe_results, data)
        return probe_results, artifact_pngs

    def _emit_m1(self, cycle: int, probe_results: "dict[str, Any]", data: "Any",
                 duration_sec: float) -> None:
        """Emit M1's ProbeOutput, including how analyzer routing was decided.

        The three modality sets are recorded separately because they can differ
        and the difference is the diagnosis: an omni model on an audio benchmark
        declares four modalities, fills one, and is routed on one.
        """
        if self.emitter is None:
            return
        from evalrx.contract.emit import from_probe_results
        from evalrx.core.case import probed_modalities

        selector = self.probe_agent.selector if hasattr(self.probe_agent, "selector") else None
        declared = sorted(getattr(self.model, "modalities", frozenset({"text"})) or {"text"})
        probed = sorted(probed_modalities(data))
        routed = sorted(
            selector.routed_slots(self.model, data) if selector is not None else probed
        )
        # Trajectories in the batch, not declared TOOL_CALLS: the api backend
        # reports tool_calls for every chat model, which made a single-turn text
        # run record itself as an agent run.
        from evalrx.eval_agent.stages.probe import _carries_trajectories

        is_agent = bool(data is not None and _carries_trajectories(data))
        self._emit(f"c{cycle}.m1", lambda: from_probe_results(
            probe_results,
            trace_id=self.emitter.trace_id, cycle=cycle, model=self.model,
            model_modalities=declared, probed_modalities=probed, routed_on=routed,
            is_agent=is_agent,
            selector="llm_judge" if getattr(self.probe_agent, "judge", None) else "static_strategy",
            generated=list(getattr(self.probe_agent, "generated_probes", []) or []),
            failed_analyzers=dict(getattr(self.probe_agent, "_failed_analyzers", {}) or {}),
            duration_sec=duration_sec,
        ))

    def explore_out_dir(self) -> "Path | None":
        """Where the explore step persists its report/tables/figures.

        Explicit ``explore_dir`` wins. Otherwise it is derived from the run
        logger so the files land INSIDE the run, beside the log, where the
        report reader looks (``reporting.run_data``: ``<logs>/explore``, else a
        sibling ``explore/``):

        - a :class:`~evalrx.eval_agent.runlog.context.RunContext`-backed logger
          → ``ctx.explore_dir``, an ephemeral tree under ``ctx.runtime_root``
          (captured into ``M2/log.json`` and deleted at ``finalize()``;
          writing straight to ``<ctx.root>/explore`` would leak real files
          outside the JSON-only run artifact);
        - a standalone ``RunLogger("<run>/logs")`` / ``logs_confirm`` … →
          ``<run>/explore`` (a sibling of the ``logs*/`` dir, the llm_benchmark
          layout — under the log dir it would be two levels down for a
          ``logs_confirm/`` carrier log and the report would miss it);
        - any other standalone run dir → ``<run_dir>/explore``.

        ``None`` (nothing persisted, context in memory only) without a logger."""
        if self.explore_dir is not None:
            return self.explore_dir
        if self.run_logger is None:
            return None
        ctx = getattr(self.run_logger, "_context", None)
        if ctx is not None and hasattr(ctx, "explore_dir"):
            return ctx.explore_dir
        run_dir = getattr(self.run_logger, "run_dir", None)
        if run_dir is None:
            return None
        run_dir = Path(run_dir)
        if run_dir.name.startswith("logs"):
            return run_dir.parent / "explore"
        return run_dir / "explore"

    def explore(
        self, cycle: int, probe_results: "dict[str, Any]", data: "Any",
        timings: "dict[str, float]", *, log: bool = True, question: str | None = None,
    ) -> "Any | None":
        """Optional in-cycle explore step: free-form EDA over M1's per-case table.

        Runs between M1 and M2 when an ``explorer`` is configured. The explorer
        sees exactly what M2 sees — ``build_stats_input`` → ``per_case_to_records``
        (M1 analyzer per-case signals + PASS/FAIL labels) — writes tables and
        chart specs, and the host renders the charts. The result feeds:

        - M3, as an :class:`~evalrx.eval_agent.stages.diagnosis.ExploreContext`
          (observations / rendered charts / caveats — descriptive, UNCONFIRMED,
          used only to decide WHICH hypotheses to propose);
        - the report, via ``exploratory_report.json`` + ``tables/`` +
          ``figures/`` under :meth:`explore_out_dir` (one directory, rewritten
          each cycle — it always holds what the *latest* M3 was shown; the
          per-cycle ``explore`` run-log events keep every cycle's counts and
          observations).

        It never touches M2's confirmatory family, M4, or the fix gate: the
        explorer's candidate-signal verdicts are host-adjudicated IN-SAMPLE
        (labelled so) and dropped from the M3 context by construction. Held-out
        confirmation of explorer recipes is the fused pipeline's job, not this
        step's. Best-effort: any failure logs a warning and returns ``None`` —
        an explorer outage must not cost the M2/M3 that already have their data.

        Returns the explorer report (or ``None``)."""
        if self.explorer is None:
            return None
        from evalrx.analysis.operationalize import per_case_to_records
        from evalrx.analysis.stats_tools import build_stats_input
        from evalrx.eval_agent.stages.diagnosis import ExploreContext

        _t0 = time.monotonic()
        report: Any = None
        out_dir = self.explore_out_dir()
        try:
            inp = build_stats_input(probe_results, data)
            records = per_case_to_records(inp.per_case, inp.labels)
            if not records:
                logger.info("explore: M1 produced no per-case signals — skipping.")
                return None
            question = question or self.explore_question
            if not question:
                from evalrx.eval_agent.prompts.explore_step import default_explore_question

                question = default_explore_question(self.protocol)
            report = self.explorer.explore_records(
                records, question=question, outcome_col="label",
            )
            # Host firewall: recompute every candidate verdict from sufficient
            # statistics with the M2 core (in-sample here — labelled as such so
            # nothing downstream mistakes it for a held-out result).
            try:
                from evalrx.analysis.adjudicate import adjudicate_report

                adjudicate_report(report, split_label="in_sample")
            except Exception as exc:  # adjudication is a verdict layer, not the data
                logger.warning("explore: host adjudication failed: %s", exc)
            if out_dir is not None:
                from evalrx.analysis.explore_run import write_report_artifacts

                # Renders chart specs from the copied tables/ CSVs, so the
                # persisted report carries each chart's absolute figure_path —
                # the PNGs M3 is shown and the report displays.
                write_report_artifacts(report, out_dir)
            ctx = ExploreContext.from_report(
                report.to_dict() if hasattr(report, "to_dict") else None
            )
            if ctx is not None:
                ctx.source = "loop_explorer"
                self.memory.explore_context = ctx
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

    def analyze(
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
        if self.emitter is not None:
            from evalrx.contract.emit import from_stats_report

            self._emit(f"c{cycle}.m2", lambda: from_stats_report(
                stats_report, trace_id=self.emitter.trace_id, cycle=cycle,
                raw_results_ref=f"contract/c{cycle}.m1.json", duration_sec=_dt,
            ))
        return stats_report

    def diagnose(
        self, cycle: int, stats_report: "Any", prior_cycles: "list[Any]",
        timings: "dict[str, float]", *, log: bool = True,
        failure_modes: "Any | None" = None, cases: "Any | None" = None,
    ) -> "Any | None":
        """M3: hypothesis generation. Returns the diagnosis result, or ``None``
        when M3 could not run (judge unavailable / timeout / quota) — the caller
        stops gracefully on ``None`` (regression guard: an M3 timeout must not
        kill the whole run after M1+M2 succeeded). Also accrues token usage.

        ``failure_modes`` (optional clustered ``FailureModeReport``) is passed
        through to the diagnosis agent only when it accepts the parameter —
        ``None`` (the default, always the case for ``VLDiagnoseLoop``) adds
        nothing to the prompt and costs no extra call."""
        _t0 = time.monotonic()
        for attempt in range(self.diagnosis_retries + 1):
            try:
                diag_agent = self.diagnosis_agent_or_default()
                diag = _diagnose_with_optional_context(
                    diag_agent, stats_report, prior_cycles, self.memory.explore_context,
                    failure_modes, cases=cases,
                )
                break
            except Exception as exc:
                retrying = attempt < self.diagnosis_retries
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("M3 attempt %d failed (%s); retrying=%s",
                               attempt + 1, error, retrying)
                if self.run_logger is not None:
                    # A JudgeError carries what the judge was asked and answered.
                    for call in (e.record() for e in getattr(exc, "exchanges", ())):
                        self.run_logger.log_model_exchange(
                            "M3", cycle=cycle, role=call["role"], operation=call["operation"],
                            inputs=call["inputs"], output=call["output"], error=call["error"],
                            duration_sec=call["duration_sec"], metadata=call["metadata"],
                        )
                    self.run_logger.log_model_exchange(
                        stage="M3", cycle=cycle, role="diagnosis", operation="diagnose",
                        inputs=None, output=None, error=error,
                        metadata={"attempt": attempt + 1, "retrying": retrying},
                        duration_sec=time.monotonic() - _t0,
                    )
                if not retrying:
                    timings["m3"] = timings.get("m3", 0.0) + time.monotonic() - _t0
                    return None
        _dt = time.monotonic() - _t0
        timings["m3"] = timings.get("m3", 0.0) + _dt

        _tok = getattr(diag, "tokens_used", None)
        if _tok is None:
            _tok = max(1, len(diag.raw_judge_output) // 4)
        self.memory.tokens_used += _tok

        if log and self.run_logger:
            # Log only the figures that actually existed (the same existence
            # filter M3 applies) so the audit trail reflects what the judge saw.
            _explore_figs = None
            if self.memory.explore_context is not None:
                from pathlib import Path as _P

                _explore_figs = [
                    f for f in self.memory.explore_context.figure_paths if _P(f).exists()
                ]
            self.run_logger.log_diagnosis(
                cycle, diag, duration_sec=_dt, explore_figures=_explore_figs or None
            )
        if self.emitter is not None:
            from evalrx.contract.emit import from_diagnosis

            self._emit(f"c{cycle}.m3", lambda: from_diagnosis(
                diag, trace_id=self.emitter.trace_id, cycle=cycle, duration_sec=_dt,
            ))
        return diag

    def test_hypotheses(
        self, cycle: int, hypotheses: "list[Any]", stats_report: "Any",
        data: "Any", timings: "dict[str, float]", *, log: bool = True,
        split_label: "str | None" = None,
    ) -> "list[Any]":
        """M4: statistical test + protocol consistency for each hypothesis,
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
        timings["m4"] = timings.get("m4", 0.0) + _dt
        for tr in test_results:
            tr.hypothesis.status = tr.status
            if split_label and isinstance(getattr(tr, "evidence", None), dict):
                tr.evidence["split"] = split_label
        if log and self.run_logger:
            # M4 results are logged through the surgery slot.
            for tr in test_results:
                _iv = _make_intervention_result_from_test(tr)
                self.run_logger.log_surgery(
                    cycle, tr.hypothesis, _iv, duration_sec=_dt,
                    validation_cases=data,
                    judge_prompt=getattr(tr, "judge_prompt", None) or None,
                    judge_raw=getattr(tr, "judge_raw", None) or None,
                )
        if log and self.emitter is not None:
            from evalrx.contract.emit import from_test_results

            # M4 has no stage log of its own — it reuses the per-hypothesis
            # surgery slot — so this is the first place the whole verdict set
            # exists as one object a reader can load.
            split = "confirm" if split_label == "confirm_holdout" else "explore"
            self._emit(f"c{cycle}.m4", lambda: from_test_results(
                test_results, trace_id=self.emitter.trace_id, cycle=cycle,
                split=split, duration_sec=_dt,
                stopping_criteria_met=bool(
                    self.hypothesis_tester.stopping_criteria_met(test_results, self.protocol)
                ),
            ))
        return test_results

    def test_on_holdout(
        self,
        hypotheses: "list[Any]",
        confirm: "Any",
        analyzer_names: "list[str]",
        timings: "dict[str, float]",
    ) -> "tuple[list[Any], str]":
        """The M4 pass — on the held-out confirm split.

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
            "M4 (held-out): testing %d hypothesis(es) on the %d-case confirm "
            "split (analyzers: %s)",
            len(hypotheses), len(list(confirm)),
            ", ".join(pinned) or "<probe agent's own selection>",
        )
        # Stamped BEFORE probe() (not just before log_probe below): ProbeAgent
        # tags every model call it makes with whatever current_cycle reads AT
        # CALL TIME, and probe() is where those calls happen. Setting this
        # only right before log_probe(-1, ...) tagged this whole confirm-split
        # re-probe's model calls with the previous explore cycle's number.
        if self.run_logger:
            self.run_logger.current_cycle = -1
        _t0 = time.monotonic()
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
        timings["m4_holdout_m1"] = timings.get("m4_holdout_m1", 0.0) + (
            time.monotonic() - _t0)
        if not probe_results:
            logger.warning(
                "M4 (held-out): the confirm-split re-probe produced no results "
                "— no hypothesis can be verified this run."
            )
            return [], "failed"
        self.bridge_signals(probe_results, confirm)
        if self.run_logger:
            self.run_logger.log_probe(
                -1, probe_results,
                schema=getattr(self.probe_agent, "last_schema", None),
                cases=confirm,
            )
        stats_confirm = self.analyze(
            -1, probe_results, confirm, [], timings, confirmatory=True
        )
        results = self.test_hypotheses(-1, hypotheses, stats_confirm, confirm, timings,
                              split_label="confirm_holdout")
        return results, "confirmed"


def finalize_confirmatory_stats(stats_report: "Any") -> None:
    """Promote a descriptive (analysis-phase) stats report to confirmatory.

    The analysis phase runs M2 with the e-BH validity verdict DEFERRED
    (``descriptive_only=True``). The confirm phase recomputes e-BH FDR
    correction over the report's stats results so M4 and the report see
    the family-level reject decision. No-op when already confirmatory."""
    if stats_report is None:
        return
    corr = getattr(stats_report, "corrected_rejections", None) or {}
    if getattr(stats_report, "descriptive_only", False) or corr.get("deferred"):
        from evalrx.analysis.stats_tools import fdr_correct

        stats_report.corrected_rejections = fdr_correct(
            list(getattr(stats_report, "stats_results", None) or [])
        )
        stats_report.descriptive_only = False


def _diagnose_with_optional_context(
    diag_agent: "Any", stats_report: "Any", prior_cycles: "Any", explore_context: "Any | None",
    failure_modes: "Any | None" = None, cases: "Any | None" = None,
) -> "Any":
    """Call ``diag_agent.diagnose`` passing ``explore_context``/``failure_modes``/
    ``cases`` only when the agent accepts them, so custom/legacy diagnosis
    agents keep working unchanged. ``cases`` (the explore-split batch M1/M2
    ran on) only feeds the critic's label summary -- never the proposer."""
    import inspect as _inspect

    kwargs: dict[str, Any] = {"prior_cycles": prior_cycles or None}
    optional = {"explore_context": explore_context, "failure_modes": failure_modes,
                "cases": cases}
    if any(v is not None for v in optional.values()):
        params: "Mapping[str, _inspect.Parameter]"
        try:
            params = _inspect.signature(diag_agent.diagnose).parameters
        except (TypeError, ValueError):
            params = {}
        for name, value in optional.items():
            if value is not None and name in params:
                kwargs[name] = value
    return diag_agent.diagnose(stats_report, **kwargs)
