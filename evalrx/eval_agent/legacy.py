"""Legacy loop controllers, kept for backward compatibility.

``SelfEvolveLoop``
    Original Stage-1 skeleton.

``AutoDiagnoseLoop``
    Concrete M1→M2→M3→M5 implementation (original architecture):

    ┌───────────────────────────────────────────────────────────────────┐
    │ M1 · ProbeAgent      select analyzers + execute (direct / Docker) │
    │ M2 · AnalysisModule  interpret results → structured report        │
    │ M3 · DiagnosisAgent  Gemini reads report → hypotheses             │
    │ M5 · SurgeryAgent    operate + verify; stop or refocus data       │
    └───────────────────────────────────────────────────────────────────┘
                            ↑__________________________│  (repeat)

Prefer :class:`~evalrx.eval_agent.loop.VLDiagnoseLoop` (M1→M2→M3→M4, M5
post-loop) or :class:`~evalrx.eval_agent.agentic.AgenticDiagnoseLoop`
(judge-decided) for new work — this module is the pre-2026-06-05 architecture,
kept for existing callers and its resumable run_dir/checkpoint infrastructure.

Run-directory infrastructure (mirrors AutoResearchClaw):

When ``run_dir`` is supplied:
  - ``run_dir/artifacts/{run_id}/``  — per-run artifact staging
  - ``run_dir/checkpoint.json``      — atomic resume state (temp+rename)
  - ``run_dir/heartbeat.json``       — per-cycle liveness signal
  - ``run_dir/evolution/``           — JSONL lesson store (auto-created)

Git integration:
  - When the repo is detected, each resolved run is committed on
    ``eval/{run_id}``; unresolved runs are discarded with git reset.

Usage::

    loop   = AutoDiagnoseLoop(model=my_model, run_dir=Path("./runs"))
    report = loop.run(failure_cases)
    print(report.resolved, report.final_hypotheses)

    # Resume an interrupted run
    report = AutoDiagnoseLoop.resume(Path("./runs"), model=my_model, data=cases)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalrx.eval_agent.checkpoint import read_checkpoint, write_checkpoint, write_heartbeat
from evalrx.eval_agent.hypothesis import HypothesisStatus
from evalrx.eval_agent.loop_reports import AutoDiagnoseReport
from evalrx.eval_agent.run_metadata import _attach_run_logger, _log_generated_tools, _run_config
from evalrx.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalrx.analysis.analysis_module import AnalysisModule
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model
    from evalrx.eval_agent.evolution import EvolutionStore
    from evalrx.eval_agent.git_manager import ExperimentGitManager
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisGenerator
    from evalrx.eval_agent.run_logger import RunLogger
    from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalrx.eval_agent.stages.probe_agent import ProbeAgent
    from evalrx.eval_agent.stages.surgery import SurgeryAgent

logger = logging.getLogger(__name__)


class SelfEvolveLoop:
    """Original Stage-1 skeleton — kept for backward compatibility.

    Args:
        generator: Proposes and mutates hypotheses.
        store:     Persistent memory; defaults to an in-memory store.
    """

    def __init__(
        self,
        generator: "HypothesisGenerator | None" = None,
        store: Store | None = None,
        runner: Any = None,  # kept for signature compat
    ) -> None:
        from evalrx.core.experiment import ExperimentRunner

        self.generator = generator
        self.store = store or InMemoryStore()
        self.runner = runner if runner is not None else ExperimentRunner()

    def step(self) -> list:
        if self.generator is None:
            raise ValueError("SelfEvolveLoop needs a generator (e.g. ManualHypothesisGenerator).")
        proposed = self.generator.propose(self.store.summarize())
        for h in proposed:
            self.store.add_hypothesis(h)
        return list(proposed)

    def run(self, max_cycles: int = 10) -> list:
        history: list = []
        for _ in range(max_cycles):
            proposed = self.step()
            history.append(proposed)
            if not proposed:
                break
        return history


# ──────────────────────────────────────────────────────────────────────────────
# AutoDiagnoseLoop — M1 → M2 → M3 → M5
# ──────────────────────────────────────────────────────────────────────────────


class AutoDiagnoseLoop:
    """Automated M1→M2→M3→M5 diagnosis loop.

    Args:
        model:            The model under evaluation.
        probe_agent:      M1 — selects and executes analyzers (direct or Docker).
                          Defaults to ``ProbeAgent()``.
        analysis_module:  M2 — interprets raw Results into an AnalysisReport.
                          Defaults to ``AnalysisModule()``.
        diagnosis_agent:  M3 — reads the AnalysisReport and proposes hypotheses via
                          Gemini (default when ``GEMINI_API_KEY`` is set).
                          Pass ``None`` to run in *analysis-only* mode (M1+M2 only).
        surgery_agent:    M5 — operates on each hypothesis to verify or refute it.
                          Defaults to ``SurgeryAgent()``.
        store:            Persistent memory.  Defaults to ``InMemoryStore()``.
        max_cycles:       Hard cap on M1→M5 iterations.
        run_logger:       Optional :class:`~evalrx.eval_agent.run_logger.RunLogger`
                          that writes a JSONL event log and saves analyzer artifacts.
        run_dir:          Optional root directory for run infrastructure.
                          When set, enables checkpoints, heartbeat, and the
                          ``EvolutionStore``.
        git_manager:      Optional :class:`~evalrx.eval_agent.git_manager.ExperimentGitManager`.
                          ``None`` → auto-detect git repo when *run_dir* is given.
        evolution_store:  Optional :class:`~evalrx.eval_agent.evolution.EvolutionStore`.
                          ``None`` → auto-create under ``run_dir/evolution/`` when
                          *run_dir* is given.
    """

    def __init__(
        self,
        model: "Model",
        probe_agent: "ProbeAgent | None" = None,
        analysis_module: "AnalysisModule | None" = None,
        diagnosis_agent: "DiagnosisAgent | None" = None,
        surgery_agent: "SurgeryAgent | None" = None,
        store: Store | None = None,
        max_cycles: int = 5,
        run_logger: "RunLogger | None" = None,
        # --- run-directory infrastructure (new) ---
        run_dir: "Path | None" = None,
        git_manager: "ExperimentGitManager | None" = None,
        evolution_store: "EvolutionStore | None" = None,
        # --- token / cost budget ---
        token_budget: int = 0,
        _run_id_override: str | None = None,
    ) -> None:
        from evalrx.analysis.analysis_module import AnalysisModule
        from evalrx.eval_agent.stages.probe_agent import ProbeAgent
        from evalrx.eval_agent.stages.surgery import SurgeryAgent

        self.model = model
        self.probe_agent = probe_agent or ProbeAgent()
        self.analysis_module = analysis_module or AnalysisModule()
        self.diagnosis_agent = diagnosis_agent
        self.surgery_agent = surgery_agent or SurgeryAgent()
        self.store = store or InMemoryStore()
        self.max_cycles = max_cycles
        self.run_logger = run_logger
        self.token_budget = token_budget
        self._tokens_used: int = 0
        self._timings: dict[str, float] = {}

        # --- run-directory setup ---
        self._run_dir: Path | None = None
        self._artifacts_dir: Path | None = None
        self._checkpoint_path: Path | None = None
        self._heartbeat_path: Path | None = None
        self._run_id: str = ""

        if run_dir is not None:
            run_dir = Path(run_dir)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._run_id = _run_id_override or ts
            self._run_dir = run_dir
            self._artifacts_dir = run_dir / "artifacts" / ts
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path = run_dir / "checkpoint.json"
            self._heartbeat_path = run_dir / "heartbeat.json"

        # EvolutionStore: auto-create when run_dir is set
        self.evolution_store: "EvolutionStore | None" = evolution_store
        if self.evolution_store is None and run_dir is not None:
            try:
                from evalrx.eval_agent.evolution import EvolutionStore
                self.evolution_store = EvolutionStore(run_dir / "evolution")
            except Exception as exc:
                logger.debug("Could not create EvolutionStore: %s", exc)

        # ExperimentGitManager: auto-detect when run_dir is set
        self.git_manager: "ExperimentGitManager | None" = git_manager
        if self.git_manager is None and run_dir is not None:
            try:
                from evalrx.eval_agent.git_manager import ExperimentGitManager
                _gm = ExperimentGitManager(run_dir)
                if _gm.is_git_repo():
                    self.git_manager = _gm
            except Exception as exc:
                logger.debug("Git manager auto-detect failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def publish_report(
        self,
        *,
        model: "Any | None" = None,
        example_dir: "str | Path | None" = None,
    ) -> "Any":
        """Compose the completed-run UI, reusing the diagnosis judge."""
        if self.run_logger is None:
            raise RuntimeError("publish_report needs a run_logger with a durable run directory")
        if model is None:
            model = getattr(self.diagnosis_agent, "judge", None)
        from evalrx.reporting.dynamic import publish_report

        return publish_report(
            self.run_logger.run_dir,
            example_dir=example_dir,
            model=model,
            run_logger=self.run_logger,
        )

    def run(self, data: "CaseBatch") -> AutoDiagnoseReport:
        """Drive the M1→M2→M3→M5 loop until resolved or *max_cycles* reached.

        Args:
            data: Cases to analyse.  Refocused to unexplained cases when M5
                  returns a SUPPORTED finding with ``new_data``.

        Returns:
            :class:`AutoDiagnoseReport` with the final state.
        """
        all_hypotheses: list[Any] = []
        final_results: dict[str, Any] = {}
        final_analysis = None

        # Determine start cycle from checkpoint (resume support).
        # Only honor the checkpoint when run_id matches — this means the
        # caller explicitly requested a resume of a specific prior run
        # (via AutoDiagnoseLoop.resume() or _run_id_override).  A fresh
        # AutoDiagnoseLoop always gets a new run_id and starts from 0.
        start_cycle = 0
        if self._checkpoint_path is not None:
            cp = read_checkpoint(self._checkpoint_path)
            if cp is not None and cp.get("run_id") == self._run_id:
                start_cycle = cp.get("last_completed_cycle", -1) + 1
                if start_cycle > 0:
                    logger.info(
                        "Resuming from checkpoint: skipping cycles 0–%d",
                        start_cycle - 1,
                    )

        prior_cycles: list[dict[str, Any]] = []
        outstanding_modes: list[str] = []
        completed_cycles = 0
        # Per-cycle max confidence scores: detect flat progress early
        _confidence_history: list[float] = []
        self._tokens_used = 0

        # Per-stage wall-clock totals (seconds), surfaced in loop_end.
        self._timings = {}

        # Forward the RunLogger so the probe generators emit tool_codegen events.
        _attach_run_logger(self.run_logger, self.probe_agent)
        if self.run_logger is not None:
            self.run_logger.log_run_start(
                _run_config(self, data, loop_name="AutoDiagnoseLoop")
            )
            self.run_logger.log_cases(data)

        for cycle in range(start_cycle, self.max_cycles):
            if self.run_logger is not None:
                self.run_logger.current_cycle = cycle
            # Budget guard: stop before consuming more tokens than allowed
            if self.token_budget > 0 and self._tokens_used >= self.token_budget:
                logger.warning(
                    "Token budget %d exhausted (%d used) — stopping loop after %d cycles",
                    self.token_budget, self._tokens_used, completed_cycles,
                )
                break
            completed_cycles = cycle + 1

            # ── M1: probe ───────────────────────────────────────────────
            _t0 = time.monotonic()
            probe_results = self.probe_agent.probe(
                self.model, data,
                hint_failure_modes=outstanding_modes or None,
            )
            _dt = time.monotonic() - _t0
            self._timings["m1"] = self._timings.get("m1", 0.0) + _dt
            if self.run_logger:
                self.run_logger.log_probe(
                    cycle, probe_results, schema=self.probe_agent.last_schema,
                    judge_prompt=getattr(self.probe_agent, "last_selection_prompt", ""),
                    judge_raw=getattr(self.probe_agent, "last_selection_raw", ""),
                    duration_sec=_dt,
                    failed_analyzers=getattr(self.probe_agent, "_failed_analyzers", None) or None,
                )
                _log_generated_tools(self.run_logger, cycle, "m1_probe", self.probe_agent)
            if not probe_results:
                break
            final_results = probe_results
            for r in probe_results.values():
                self.store.add_result(r)

            # ── M2: analyze ─────────────────────────────────────────────
            _t0 = time.monotonic()
            analysis = self.analysis_module.analyze(probe_results, repr(self.model))
            _dt = time.monotonic() - _t0
            self._timings["m2"] = self._timings.get("m2", 0.0) + _dt
            final_analysis = analysis
            if self.run_logger:
                self.run_logger.log_analysis(cycle, analysis, duration_sec=_dt)

            if self.diagnosis_agent is None:
                self._on_cycle_complete(cycle, all_hypotheses)
                break  # analysis-only mode

            # ── M3: diagnose ─────────────────────────────────────────────
            _t0 = time.monotonic()
            diag = self.diagnosis_agent.diagnose(
                analysis, prior_cycles=prior_cycles or None
            )
            _dt = time.monotonic() - _t0
            self._timings["m3"] = self._timings.get("m3", 0.0) + _dt
            # Track token usage from the diagnosis LLM call.
            # Use response metadata when available; otherwise estimate from
            # output length (1 token ≈ 4 chars) so the budget is always counted.
            _diag_tokens = getattr(diag, "tokens_used", None)
            if _diag_tokens is None:
                _diag_tokens = max(1, len(diag.raw_judge_output) // 4)
            self._tokens_used += _diag_tokens
            if self.run_logger:
                self.run_logger.log_diagnosis(cycle, diag, duration_sec=_dt)
            if not diag.hypotheses:
                self._on_cycle_complete(cycle, all_hypotheses)
                break
            for h in diag.hypotheses:
                self.store.add_hypothesis(h)
            all_hypotheses.extend(diag.hypotheses)

            # ── M5: surgery ──────────────────────────────────────────────
            outstanding_modes = []
            cycle_max_confidence = 0.0
            for h in diag.hypotheses:
                _t0 = time.monotonic()
                iv = self.surgery_agent.operate(h, self.model, probe_results, data)
                _dt = time.monotonic() - _t0
                self._timings["m5"] = self._timings.get("m5", 0.0) + _dt
                h.status = iv.status
                cycle_max_confidence = max(cycle_max_confidence, iv.confidence_score)
                if self.run_logger:
                    self.run_logger.log_surgery(cycle, h, iv, duration_sec=_dt)
                    # When surgery wrote and ran an experiment, also persist it
                    # (script, output, thinking, workspace snapshot).
                    if getattr(iv, "experiment", None):
                        self.run_logger.log_experiment(cycle, h, iv, module="m5")
                if iv.fixed:
                    report = AutoDiagnoseReport(
                        cycles=completed_cycles,
                        resolved=True,
                        final_hypotheses=all_hypotheses,
                        final_results=final_results,
                        final_analysis=final_analysis,
                        store=self.store,
                        _run_id=self._run_id,
                    )
                    self._on_cycle_complete(cycle, all_hypotheses)
                    self._on_loop_end(report)
                    return report
                if iv.status == HypothesisStatus.INCONCLUSIVE:
                    outstanding_modes.append(h.predicted_failure_mode)
                if iv.new_data is not None and len(iv.new_data) > 0:
                    data = iv.new_data

            # Track confidence per cycle and stop early when progress stalls:
            # if the last 2 cycles both show confidence < 0.1 the hypotheses
            # are not gaining traction — continuing burns budget without insight.
            _confidence_history.append(cycle_max_confidence)
            if len(_confidence_history) >= 2 and all(
                c < 0.1 for c in _confidence_history[-2:]
            ):
                logger.info(
                    "Stopping early: confidence flat at %.3f for 2 consecutive cycles",
                    cycle_max_confidence,
                )
                break

            # Record cycle for M3 context
            prior_cycles.append({
                "cycle": cycle,
                "severity": analysis.severity,
                "hypotheses": [
                    {
                        "statement": h.statement,
                        "failure_mode": h.predicted_failure_mode,
                        "status": h.status.value if h.status else "pending",
                    }
                    for h in diag.hypotheses
                ],
            })

            # Write checkpoint + heartbeat after each full cycle
            self._on_cycle_complete(cycle, all_hypotheses)

        report = AutoDiagnoseReport(
            cycles=completed_cycles,
            resolved=False,
            final_hypotheses=all_hypotheses,
            final_results=final_results,
            final_analysis=final_analysis,
            store=self.store,
            _run_id=self._run_id,
        )
        self._on_loop_end(report)
        return report

    @classmethod
    def resume(
        cls,
        run_dir: Path,
        model: "Model",
        data: "CaseBatch",
        **kwargs: Any,
    ) -> "AutoDiagnoseReport":
        """Resume a loop from a checkpoint.

        Reads ``run_dir/checkpoint.json``, skips already-completed cycles,
        and continues from ``last_completed_cycle + 1``.

        Args:
            run_dir: Directory that contains ``checkpoint.json``.
            model:   The model under evaluation.
            data:    Original :class:`CaseBatch` to continue from.
            **kwargs: Forwarded to :class:`AutoDiagnoseLoop.__init__`.
                      ``run_dir`` is set automatically — do not pass it.
        """
        # Read the stored run_id so the checkpoint match succeeds and
        # the loop correctly skips already-completed cycles.
        run_id_override: str | None = None
        cp = read_checkpoint(Path(run_dir) / "checkpoint.json")
        if cp is not None:
            run_id_override = cp.get("run_id")
        instance = cls(
            model=model, run_dir=run_dir, _run_id_override=run_id_override, **kwargs
        )
        return instance.run(data)

    # ──────────────────────────────────────────────────────────────────
    # Per-cycle and end-of-loop hooks
    # ──────────────────────────────────────────────────────────────────

    def _on_cycle_complete(
        self, cycle: int, hypotheses: list["Hypothesis"]
    ) -> None:
        """Write checkpoint and heartbeat after each completed cycle."""
        statuses = [
            h.status.value if h.status else "pending"
            for h in hypotheses
        ]
        if self._checkpoint_path is not None:
            write_checkpoint(
                self._checkpoint_path, cycle=cycle, run_id=self._run_id,
                hypothesis_statuses=statuses,
            )
        if self._heartbeat_path is not None:
            write_heartbeat(self._heartbeat_path, cycle=cycle, run_id=self._run_id)

    def _on_loop_end(self, report: AutoDiagnoseReport) -> None:
        """Append lessons to EvolutionStore and commit/discard via git."""
        if self.run_logger:
            self.run_logger.log_loop_end(
                report,
                tokens_used=self._tokens_used,
                timings=getattr(self, "_timings", None) or None,
            )

        # EvolutionStore lesson extraction
        if self.evolution_store is not None:
            try:
                from evalrx.eval_agent.evolution import extract_lessons
                lessons = extract_lessons(report)
                self.evolution_store.append_many(lessons)
                logger.debug(
                    "EvolutionStore: appended %d lesson(s) for run %s",
                    len(lessons), self._run_id,
                )
            except Exception as exc:
                logger.warning("Failed to record evolution lessons: %s", exc)

        # Git integration
        if self.git_manager is not None and self._run_id:
            hyp_statuses = {
                h.statement[:80]: (h.status.value if h.status else "pending")
                for h in report.final_hypotheses
            }
            try:
                if report.resolved:
                    self.git_manager.commit_experiment(
                        self._run_id,
                        report.cycles,
                        hyp_statuses,
                        "resolved",
                    )
                    logger.info(
                        "Git: committed resolved run %s on eval/%s",
                        self._run_id, self._run_id,
                    )
                else:
                    self.git_manager.discard_experiment(
                        self._run_id, "not resolved"
                    )
            except Exception as exc:
                logger.warning("Git integration failed: %s", exc)
