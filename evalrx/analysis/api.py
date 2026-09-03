"""Standalone library entry point for exploratory data analysis.

``explore()`` is the single-call surface for ``evalrx.analysis`` as a
capability independent of the eval-agent diagnosis loop: given a results path
or a list of records, it runs the M2 exploratory agent, host-adjudicates any
candidate signal that attached host-checkable sufficient statistics, and
(best-effort) proposes M3 falsifiable hypotheses from the takeaways.

``evalrx.analysis.explore_run.run_explore`` is the CLI-facing counterpart:
it calls this function, then prints a summary and optionally opens the
dashboard — that is the only difference. Both write the same artifacts when
given an output directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalrx.agent_runtime.cli_types import CliAgentConfig
from evalrx.agent_runtime.skills.resolver import resolve_skill_paths
from evalrx.analysis.adjudicate import adjudicate_report
from evalrx.analysis.explorer import ExploratoryAnalysisAgent, ExploratoryAnalysisReport
from evalrx.analysis.hypothesis_agent import HypothesisAgent

logger = logging.getLogger(__name__)


@dataclass
class ExploreRunResult:
    """Outcome of one :func:`explore` call."""

    report: ExploratoryAnalysisReport
    out_dir: Path | None = None
    ok: bool = False
    hypotheses: list[dict[str, Any]] = field(default_factory=list)


def explore(
    source: "str | Path | list[dict[str, Any]]",
    *,
    question: str = "Explore this dataset and surface the patterns that matter.",
    out: "str | Path | None" = None,
    provider: str = "antigravity",
    model: str = "",
    binary: str = "",
    outcome_col: str | None = None,
    propose_hypotheses: bool = True,
    skills: "list[str] | tuple[str, ...]" = (),
    allow_skills: bool = False,
    use_bundled_skills: bool = True,
    max_rows: int = 2000,
    max_files: int = 200,
    include_tool_calls: bool = False,
    timeout_sec: int = 120,
    max_attempts: int = 2,
    progress_sink: Any = None,
) -> ExploreRunResult:
    """Run one exploratory analysis over a path or in-memory records.

    *source* is either a file/directory path (JSON/JSONL results) or a list of
    row dicts already in memory. When *out* is omitted, nothing is persisted
    to disk — the report and any figures/tables stay in the sandbox workdir
    only. Pass *out* to also write ``exploratory_report.json``, rendered
    figures, and tables (the same artifacts ``evalrx explore`` writes).

    *outcome_col* optionally names the target/label column explicitly; omit
    it to auto-detect by name heuristics, or fall back to unsupervised EDA
    when the data has no recognizable outcome.

    *propose_hypotheses* runs M3 (:class:`~evalrx.analysis.hypothesis_agent.HypothesisAgent`)
    on M2's takeaways after a successful explore; set False to skip it.
    """
    skill_dirs = resolve_skill_paths(
        provider=provider,
        explicit=skills or (),
        use_bundled=use_bundled_skills,
    )
    cli_config = CliAgentConfig(
        provider=provider,
        binary_path=binary,
        model=model,
        timeout_sec=timeout_sec,
        skills=skill_dirs,
        allow_skills=allow_skills or bool(skill_dirs),
    )
    agent = ExploratoryAnalysisAgent(
        cli_config=cli_config,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        use_bundled_skills=use_bundled_skills,
        progress_sink=progress_sink,
    )

    if progress_sink is not None:
        progress_sink.emit("m2", "started", "Starting exploratory analysis (M2)")
    if isinstance(source, (str, Path)):
        report = agent.explore_path(
            source,
            question=question,
            max_rows=max_rows,
            max_files=max_files,
            include_tool_calls=include_tool_calls,
            outcome_col=outcome_col,
        )
    else:
        report = agent.explore_records(source, question=question, outcome_col=outcome_col)

    # Host adjudication: any candidate that attached host-checkable `sufficient`
    # stats gets a verdict recomputed by the validated core (the explorer never
    # decides). A standalone run has no held-out split, so verdicts are IN-SAMPLE.
    adjudicate_report(report, split_label="in_sample")

    out_dir: Path | None = None
    if out is not None:
        out_dir = Path(out).resolve()
        # Persist M2 independently so the workbench can render it while M3 is
        # still running.  The final report keeps its existing stable filename.
        from evalrx.analysis.explore_run import write_report_artifacts

        write_report_artifacts(report, out_dir, report_filename="partial_report.json")
    if progress_sink is not None:
        progress_sink.emit(
            "m2", "completed" if report.ok else "failed",
            "Exploratory analysis completed" if report.ok else (report.error or "Exploratory analysis failed"),
            artifact_refs=[out_dir / "partial_report.json"] if out_dir else (),
            metrics={"attempts": report.attempts},
        )

    # M3: propose falsifiable hypotheses from M2's takeaways — proposal only,
    # no validation. Only worth trying when M2 actually produced something to
    # reason over.
    if propose_hypotheses and report.ok and (report.takeaways or report.observations):
        hyp_agent = HypothesisAgent(cli_config=cli_config, timeout_sec=timeout_sec)
        if progress_sink is not None:
            progress_sink.emit("m3", "started", "Proposing falsifiable hypotheses (M3)")
        try:
            report.hypotheses = [h.to_dict() for h in hyp_agent.propose(report.to_dict())]
        except Exception as exc:  # noqa: BLE001 — M3 is best-effort, never blocks the result
            logger.warning("hypothesis generation failed: %s", exc)
        if progress_sink is not None:
            progress_sink.emit("m3", "completed", f"Proposed {len(report.hypotheses)} hypotheses")

    if out is not None:
        out_dir = Path(out).resolve()
        # Lazy import: write_report_artifacts lives in explore_run.py, which
        # imports `explore` from this module at load time.
        from evalrx.analysis.explore_run import write_report_artifacts

        write_report_artifacts(report, out_dir)
    if progress_sink is not None:
        progress_sink.emit(
            "persist", "completed", "Analysis artifacts are ready",
            artifact_refs=[out_dir / "exploratory_report.json"] if out_dir else (),
        )

    return ExploreRunResult(
        report=report,
        out_dir=out_dir,
        ok=report.ok,
        hypotheses=report.hypotheses,
    )
