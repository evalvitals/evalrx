"""General-purpose exploratory data analysis agent (LAMBDA-style).

A coding agent writes Python over arbitrary tabular records — not just M1
pass/fail diagnosis logs — EvalVitals runs it locally in a sandbox, and the
result is parsed into a structured exploratory report. The prompt adapts its
framing to whatever the data's outcome column actually is (binary,
multi-class, continuous, or none — see :func:`_framing_block`), so a dataset
with no FAIL/PASS label gets unsupervised EDA instead of an invented split.

M1 integration is one caller among others, not a special case: the diagnosis
loop's per-case records already carry a ``label`` column, which the name
heuristic in :func:`evalvitals.analysis.profile.profile_records` recognizes
as the outcome automatically, so it still gets the same binary FAIL/PASS
framing it always did (callers with an arbitrarily-named target can pass
``outcome_col=`` explicitly instead of relying on the heuristic). Findings
from this module are candidates only; use ``StatsAnalysisAgent`` for
confirmatory effect/CI/e-value/FDR verdicts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from evalvitals.agent_runtime.sandbox import ExperimentSandbox, SandboxResult
from evalvitals.analysis.plain_language import jargon_violation
from evalvitals.analysis.profile import describe_outcome, profile_records
from evalvitals.analysis.prompts.explorer import (
    GENERATE_PROMPT_RAW_FOLDER as _GENERATE_PROMPT_RAW_FOLDER,
)
from evalvitals.analysis.prompts.explorer import (
    GENERATE_PROMPT_RECORDS as _GENERATE_PROMPT_RECORDS,
)
from evalvitals.analysis.prompts.explorer import (
    GENERIC_FRAMING as _GENERIC_FRAMING,
)
from evalvitals.analysis.prompts.explorer import (
    RECORDS_FILENAME,
)
from evalvitals.analysis.prompts.explorer import (
    REPAIR_PROMPT as _REPAIR_PROMPT,
)
from evalvitals.analysis.prompts.explorer import (
    RESULT_MARKER as _RESULT_MARKER,
)
from evalvitals.analysis.prompts.explorer import (
    fences_hint as _fences_hint,
)
from evalvitals.analysis.prompts.explorer import (
    skills_hint as _skills_hint,
)

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)


@dataclass
class Takeaway:
    """One finding paired with its supporting evidence — the primary UI unit.

    The dashboard renders each takeaway as: plain_title (falling back to
    title) -> chart(s)/table(s) -> analysis, so a reader never sees an
    orphaned chart or a claim with no evidence next to it. ``plain_title`` is
    the jargon-free headline aimed at a non-technical reader; ``title`` keeps
    the precise technical wording/numbers as a companion line. Always
    descriptive: this agent does not generate or validate hypotheses, so a
    takeaway is a description of the data, not a claim of causation or
    statistical confirmation.
    """

    title: str = ""
    plain_title: str = ""
    analysis: str = ""
    chart_names: list[str] = field(default_factory=list)
    table_names: list[str] = field(default_factory=list)
    caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "plain_title": self.plain_title,
            "analysis": self.analysis,
            "chart_names": self.chart_names,
            "table_names": self.table_names,
            "caveat": self.caveat,
        }


@dataclass
class CandidateSignal:
    """A signal worth testing later with a confirmatory pipeline
    (:class:`~evalvitals.analysis.stats_agent.StatsAnalysisAgent`, or the
    diagnosis loop) — not something this agent tests or confirms itself.

    The explorer PROPOSES this signal; it has no adjudication authority. When the
    explorer attaches host-adjudicable ``sufficient`` statistics, the host
    (:func:`evalvitals.analysis.adjudicate.adjudicate_report`) recomputes the
    verdict with the validated, multiplicity-aware core and fills in the
    ``effect`` / ``ci`` / ``e_value`` / ``reject`` / ``host_adjudicated`` fields.
    Any ``reject`` / ``e_value`` the explorer self-declares is IGNORED — proposing
    is not validating; only ``sufficient`` (or, in Phase B, ``recipe``) is read.
    """

    name: str
    display_name: str = ""
    rationale: str = ""
    suggested_test: str = ""
    # --- proposed by the explorer (no authority over the verdict) ---
    sufficient: dict[str, Any] | None = None  # host-adjudicable sufficient stats
    recipe: dict[str, Any] | None = None       # Phase B operationalization recipe
    # --- filled in by the host adjudication pass (authoritative) ---
    effect: float | None = None
    ci: tuple[float, float] | None = None
    e_value: float | None = None
    reject: bool | None = None
    underpowered: bool = False
    host_adjudicated: bool = False
    fdr_corrected: bool = False
    descriptive_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "rationale": self.rationale,
            "suggested_test": self.suggested_test,
            "sufficient": self.sufficient,
            "recipe": self.recipe,
            "effect": self.effect,
            "ci": list(self.ci) if self.ci is not None else None,
            "e_value": self.e_value,
            "reject": self.reject,
            "underpowered": self.underpowered,
            "host_adjudicated": self.host_adjudicated,
            "fdr_corrected": self.fdr_corrected,
            "descriptive_only": self.descriptive_only,
        }


@dataclass
class ExploratoryAnalysisReport:
    """Output of the exploratory analysis agent: a data profile, a ranked list
    of takeaways (each paired with its supporting chart/table), and the raw
    charts/tables/code behind them. Purely descriptive — this agent does not
    generate or validate hypotheses itself; ``hypotheses`` (if present) is
    filled in by a separate downstream stage (M3, see
    ``evalvitals.analysis.hypothesis_agent.HypothesisAgent``), not by this
    class — proposal only, still no validation; see ``StatsAnalysisAgent``
    for confirmatory testing.
    """

    question: str = ""
    plain_question: str = ""
    ok: bool = False
    observations: list[str] = field(default_factory=list)
    takeaways: list[Takeaway] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    visual_plan: list[dict[str, Any]] = field(default_factory=list)
    chart_readings: list[dict[str, Any]] = field(default_factory=list)
    dashboard_storyboard: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    candidate_signals: list[CandidateSignal] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    charts: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    critique: list[str] = field(default_factory=list)
    recommended_confirmatory_tests: list[str] = field(default_factory=list)
    data_profile: dict[str, Any] = field(default_factory=dict)
    # Host adjudication family metadata (method/alpha/split/n_in_family/rejected),
    # filled by evalvitals.analysis.adjudicate; empty until adjudicated.
    adjudication: dict[str, Any] = field(default_factory=dict)
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    attempts: int = 0
    workdir: str = ""
    raw_outputs: list[str] = field(default_factory=list)
    agent_audits: list[dict[str, Any]] = field(default_factory=list)
    # UNTRUNCATED raw CLI streams, one per attempt (the truncated renderings
    # above exist for compact UIs; these are the full audit trail).
    raw_streams: list[str] = field(default_factory=list)

    @property
    def candidate_signal_names(self) -> list[str]:
        return [s.name for s in self.candidate_signals if s.name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plain_question": self.plain_question,
            "ok": self.ok,
            "observations": self.observations,
            "takeaways": [t.to_dict() for t in self.takeaways],
            "hypotheses": self.hypotheses,
            "visual_plan": self.visual_plan,
            "chart_readings": self.chart_readings,
            "dashboard_storyboard": self.dashboard_storyboard,
            "claims": self.claims,
            "candidate_signals": [s.to_dict() for s in self.candidate_signals],
            "plots": self.plots,
            "tables": self.tables,
            "charts": self.charts,
            "caveats": self.caveats,
            "critique": self.critique,
            "recommended_confirmatory_tests": self.recommended_confirmatory_tests,
            "data_profile": self.data_profile,
            "adjudication": self.adjudication,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "attempts": self.attempts,
            "workdir": self.workdir,
            "agent_audits": self.agent_audits,
        }


class ExploratoryAnalysisAgent:
    """Backend-only exploratory analysis agent.

    Args:
        judge:      LLM-like object with ``generate(prompt) -> str`` used to
                    write the initial analysis script.
        inspector:  Optional LLM-like object used for repair prompts. When not
                    provided, ``judge`` repairs its own code.
        cli_config: Optional CLI coding-agent backend. If set to a non-``llm``
                    provider, it writes ``analysis.py`` locally.
        sandbox:    Local execution sandbox. A durable temp sandbox is created
                    when omitted so generated code and figures remain available.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        inspector: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: ExperimentSandbox | None = None,
        timeout_sec: int = 60,
        max_attempts: int = 2,
        use_bundled_skills: bool = True,
        progress_sink: Any = None,
    ) -> None:
        self._judge = judge
        self._inspector = inspector
        # Figure-styling skills are on by default for every skill-capable CLI
        # backend: when the caller configured none, vendor the package-bundled
        # ones (eval-chart-style, nature-figure, evalvitals-report-ui) so agent
        # figures follow the house chart-type policy without per-caller wiring.
        if use_bundled_skills and cli_config is not None and not cli_config.skills:
            from evalvitals.agent_runtime.skills.resolver import resolve_skill_paths

            bundled = resolve_skill_paths(
                provider=cli_config.provider,
                explicit=(),
                use_bundled=True,
            )
            if bundled:
                from dataclasses import replace

                cli_config = replace(cli_config, skills=bundled, allow_skills=True)
        self._cli_config = cli_config
        self._sandbox = sandbox or ExperimentSandbox(cleanup=False)
        self._timeout_sec = timeout_sec
        self._max_attempts = max(1, max_attempts)
        self._progress_sink = progress_sink
        self._last_agent_audit: dict[str, Any] | None = None
        self._last_raw_stream: str = ""

    @property
    def available(self) -> bool:
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    def explore_records(
        self,
        records: Any,
        *,
        question: str = "Explore this dataset and surface the patterns that matter.",
        outcome_col: str | None = None,
    ) -> ExploratoryAnalysisReport:
        """Run local exploratory analysis over plain records.

        ``outcome_col`` optionally names the target/label column explicitly
        (e.g. M1 always passes ``"label"``). When omitted, the outcome (if
        any) is auto-detected by name heuristics; arbitrary datasets with no
        recognizable outcome column fall back to unsupervised EDA instead of
        having a FAIL/PASS split forced onto them.
        """
        rows = _records_to_rows(records)
        profile = _profile_rows(rows, outcome_col=outcome_col)
        self._write_input(rows)
        return self._run_explore_loop(
            question, profile, lambda: self._write_code(question, profile)
        )

    def explore_path(
        self,
        path: str | Path,
        *,
        question: str = "Explore this dataset and surface the patterns that matter.",
        max_rows: int = 2000,
        max_files: int = 200,
        include_tool_calls: bool = False,
        outcome_col: str | None = None,
    ) -> ExploratoryAnalysisReport:
        """Run exploratory analysis directly over *path* (a file or directory).

        This is the no-code entrypoint behind the CLI. With a CLI coding-agent
        backend (``cli_config``, the ``evalvitals explore`` default), the raw
        source is handed to the agent as-is and IT loads/organizes whatever
        shape it finds — no host-side JSON-shape parsing, so this works for an
        arbitrary M1 output layout, not just the ones the host loader
        recognizes. With a plain LLM ``judge`` (no filesystem access), the host
        must still pre-flatten records via :func:`load_records_from_path`
        before handing them to :meth:`explore_records`.
        """
        folder_scan = scan_folder(path, max_files=max_files, include_tool_calls=include_tool_calls)
        is_agentic = self._cli_config is not None and self._cli_config.provider != "llm"
        if is_agentic:
            report = self._explore_raw_folder(
                path,
                question=question,
                folder_scan=folder_scan,
                max_files=max_files,
                include_tool_calls=include_tool_calls,
                outcome_col=outcome_col,
            )
        else:
            rows = load_records_from_path(
                path,
                max_rows=max_rows,
                max_files=max_files,
                include_tool_calls=include_tool_calls,
            )
            report = self.explore_records(rows, question=question, outcome_col=outcome_col)
            report.data_profile.setdefault("loaded_rows", len(rows))
        report.data_profile.setdefault("source_path", str(Path(path)))
        report.data_profile.setdefault("folder_scan", folder_scan)
        return report

    def _explore_raw_folder(
        self,
        path: str | Path,
        *,
        question: str,
        folder_scan: dict[str, Any],
        max_files: int,
        include_tool_calls: bool,
        outcome_col: str | None,
    ) -> ExploratoryAnalysisReport:
        raw_input_dir = self._copy_raw_input(path, max_files=max_files, include_tool_calls=include_tool_calls)
        profile: dict[str, Any] = {"folder_scan": folder_scan, "raw_input_dir": raw_input_dir}
        return self._run_explore_loop(
            question,
            profile,
            lambda: self._write_code_raw_folder(question, folder_scan, raw_input_dir, outcome_col),
        )

    def _copy_raw_input(self, path: str | Path, *, max_files: int, include_tool_calls: bool) -> str:
        """Copy the raw source into the sandbox untouched, so the CLI agent
        reads and organizes it itself — the host does no JSON-shape parsing
        for this path. Mirrors the same file selection ``load_records_from_path``
        would use (extension filter, ``tool_calls_*`` skip, ``max_files`` cap),
        but copies the original bytes instead of parsing them."""
        root = Path(path)
        dest_root = Path(self._sandbox.workdir) / "raw_input"
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True)
        if root.is_file():
            shutil.copy2(root, dest_root / root.name)
        else:
            # A workbench DatasetBundle contains normalized records.json plus
            # optional original media.  Keep the old JSON selection semantics
            # for regular result folders, but include known table/media files
            # so a CLI coding agent can inspect the same bundle the UI reports.
            from evalvitals.analysis.workbench import SUPPORTED_SUFFIXES

            files = [
                p for p in sorted(root.rglob("*"))
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_SUFFIXES
                and (include_tool_calls or not p.name.startswith("tool_calls_"))
            ]
            # Normalized rows are the primary analysis input and must not be
            # crowded out by a large media directory.
            files.sort(key=lambda p: (p.name != RECORDS_FILENAME, str(p)))
            files = files[:max_files]
            for f in files:
                dest = dest_root / f.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        return "raw_input"

    def _run_explore_loop(
        self,
        question: str,
        profile: dict[str, Any],
        write_first: "Callable[[], tuple[str, str]]",
    ) -> ExploratoryAnalysisReport:
        """Shared write/run/repair-attempt loop behind :meth:`explore_records`
        and :meth:`_explore_raw_folder` — they differ only in how the first
        attempt's code is written (pre-loaded records vs. a raw folder); repair
        attempts reuse :meth:`_repair_code` either way."""
        if not self.available:
            return ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                error="no code-writing backend configured (judge or cli_config)",
                workdir=str(self._sandbox.workdir),
            )

        raw_outputs: list[str] = []
        raw_streams: list[str] = []
        agent_audits: list[dict[str, Any]] = []
        code = ""
        last_result: SandboxResult | None = None
        last_error = ""
        # A working report we're retrying only to fix a plain_title, kept in
        # case the retry itself produces something worse (crashes, unparseable
        # JSON) — a jargon-y headline must never cost the whole analysis.
        best_report: ExploratoryAnalysisReport | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                if self._progress_sink is not None:
                    self._progress_sink.emit(
                        "m2_codegen", "started", "Generating analysis code", attempt=attempt
                    )
                if attempt == 1:
                    code, raw = write_first()
                else:
                    code, raw = self._repair_code(
                        question, profile, code, last_result, last_error
                    )
                raw_outputs.append(raw)
                raw_streams.append(self._last_raw_stream)
                self._last_raw_stream = ""
                if self._last_agent_audit is not None:
                    agent_audits.append(self._last_agent_audit)
                    self._last_agent_audit = None
                if code.strip() and self._progress_sink is not None:
                    self._progress_sink.emit(
                        "m2_codegen", "completed", "Analysis code is ready", attempt=attempt
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = f"code writing failed: {exc}"
                logger.warning("ExploratoryAnalysisAgent: %s", last_error)
                if self._progress_sink is not None:
                    self._progress_sink.emit("m2_codegen", "failed", last_error, attempt=attempt)
                break

            if not code.strip():
                last_error = "backend produced no code"
                if self._progress_sink is not None:
                    self._progress_sink.emit(
                        "m2_codegen", "failed", last_error, attempt=attempt
                    )
                continue

            if self._progress_sink is not None:
                self._progress_sink.emit(
                    "m2_execute", "started", "Running generated analysis", attempt=attempt
                )
            last_result = self._sandbox.run(code, timeout_sec=self._timeout_sec)
            if self._progress_sink is not None:
                self._progress_sink.emit(
                    "m2_execute", "completed" if last_result.ok else "failed",
                    "Generated analysis finished" if last_result.ok else "Generated analysis needs repair",
                    attempt=attempt,
                    metrics={"elapsed_sec": last_result.elapsed_sec, "returncode": last_result.returncode},
                )
            report, last_error = _report_from_sandbox(
                question=question,
                profile=profile,
                code=code,
                result=last_result,
                attempts=attempt,
                workdir=Path(self._sandbox.workdir),
            )
            report.raw_outputs = raw_outputs
            report.raw_streams = raw_streams
            report.agent_audits = agent_audits
            if report.ok:
                violations = _plain_language_violations(report)
                if not violations or attempt == self._max_attempts:
                    if violations:
                        report.critique = [
                            *report.critique,
                            "Plain-language check failed and repair attempts were "
                            "exhausted: " + "; ".join(violations),
                        ]
                    return report
                best_report = report
                last_error = (
                    "The analysis itself is fine — do not change it. Only the "
                    "plain-language check on \"plain_title\"/\"plain_question\" "
                    "failed: " + "; ".join(violations) + ". Rewrite each flagged "
                    "field as one jargon-free, everyday sentence (numbers are "
                    "fine; acronyms/stats terms/symbols are not), keep "
                    "everything else the same, and reprint the full result JSON."
                )

        if best_report is not None:
            # The plain-language retry crashed or produced something
            # unparseable — fall back to the last working analysis rather
            # than losing every takeaway/chart/hypothesis over a headline.
            best_report.critique = [
                *best_report.critique,
                "A plain-language rewrite attempt failed (" + last_error + "); "
                "kept the original working analysis, whose plain_title/"
                "plain_question violation may be unfixed.",
            ]
            return best_report

        stdout = last_result.stdout if last_result is not None else ""
        stderr = last_result.stderr if last_result is not None else ""
        return ExploratoryAnalysisReport(
            question=question,
            ok=False,
            data_profile=profile,
            code=code,
            stdout=stdout,
            stderr=stderr,
            error=last_error or "exploratory analysis failed",
            attempts=min(self._max_attempts, len(raw_outputs)),
            workdir=str(self._sandbox.workdir),
            raw_outputs=raw_outputs,
            raw_streams=raw_streams,
            agent_audits=agent_audits,
        )

    def _write_input(self, rows: list[dict[str, Any]]) -> None:
        path = Path(self._sandbox.workdir) / RECORDS_FILENAME
        path.write_text(json.dumps(rows, default=str), encoding="utf-8")

    def _write_code(self, question: str, profile: dict[str, Any]) -> tuple[str, str]:
        prompt = _GENERATE_PROMPT_RECORDS.format(
            question=question,
            input_filename=RECORDS_FILENAME,
            data_profile=json.dumps(profile, indent=2, default=str),
            framing=_framing_block(profile.get("outcome") or {}),
            marker=_RESULT_MARKER,
            skills_hint=_skills_hint(self._cli_config),
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_writer(prompt, use_inspector=False)

    def _write_code_raw_folder(
        self,
        question: str,
        folder_scan: dict[str, Any],
        raw_input_dir: str,
        outcome_col: str | None,
    ) -> tuple[str, str]:
        outcome_hint = (
            f'- the caller named "{outcome_col}" as the target/outcome column — '
            "use it as the outcome if it exists in your tidy table, regardless "
            "of any other auto-detection heuristic you might otherwise apply\n"
            if outcome_col else ""
        )
        prompt = _GENERATE_PROMPT_RAW_FOLDER.format(
            question=question,
            input_filename=RECORDS_FILENAME,
            raw_input_dir=raw_input_dir,
            folder_scan=json.dumps(folder_scan, indent=2, default=str),
            outcome_hint=outcome_hint,
            framing=_GENERIC_FRAMING,
            marker=_RESULT_MARKER,
            skills_hint=_skills_hint(self._cli_config),
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_cli_writer(prompt)

    def _repair_code(
        self,
        question: str,
        profile: dict[str, Any],
        code: str,
        result: SandboxResult | None,
        error: str,
    ) -> tuple[str, str]:
        prompt = _REPAIR_PROMPT.format(
            question=question,
            data_profile=json.dumps(profile, indent=2, default=str),
            code=code,
            stdout=(result.stdout if result is not None else "")[-2000:],
            stderr=(result.stderr if result is not None else "")[-2000:],
            error=error,
            input_filename=RECORDS_FILENAME,
            marker=_RESULT_MARKER,
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_writer(prompt, use_inspector=True)

    def _run_writer(self, prompt: str, *, use_inspector: bool) -> tuple[str, str]:
        if self._cli_config is not None and self._cli_config.provider != "llm":
            return self._run_cli_writer(prompt)
        model = self._inspector if use_inspector and self._inspector is not None else self._judge
        raw = model.generate(prompt)  # type: ignore[union-attr]
        raw_text = str(raw)
        return _extract_code(raw_text), raw_text

    def _run_cli_writer(self, prompt: str) -> tuple[str, str]:
        from evalvitals.agent_runtime.codegen import CodegenRunner

        result = CodegenRunner(self._cli_config).write_code(  # type: ignore[arg-type]
            prompt,
            workdir=Path(self._sandbox.workdir),
            timeout_sec=self._timeout_sec,
            preferred_filenames=("analysis.py",),
            include_error_in_raw=True,
        )
        self._last_agent_audit = result.audit
        # Snapshot the untruncated raw stream NOW: the next repair attempt in
        # the loop overwrites agent_raw_stream.txt in the shared workdir.
        self._last_raw_stream = ""
        if getattr(result, "raw_stream_path", ""):
            try:
                stream_p = Path(self._sandbox.workdir) / result.raw_stream_path
                self._last_raw_stream = stream_p.read_text(encoding="utf-8")
            except OSError:
                pass
        return result.code, result.raw_output


def _records_to_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []
    if hasattr(records, "to_dict"):
        try:
            data = records.to_dict(orient="records")
            if isinstance(data, list):
                return [dict(r) for r in data if isinstance(r, dict)]
        except TypeError:
            pass
    rows: list[dict[str, Any]] = []
    for row in list(records):
        if isinstance(row, dict):
            rows.append(dict(row))
        elif hasattr(row, "_asdict"):
            rows.append(dict(row._asdict()))
        elif hasattr(row, "__dict__"):
            rows.append(dict(vars(row)))
        else:
            rows.append({"value": row})
    return rows


def _discover_json_files(
    root: Path, *, max_files: int, include_tool_calls: bool
) -> tuple[list[Path], int]:
    """Find the JSON/JSONL files ``load_records_from_path`` would read.

    Returns ``(files_used, n_matching_before_filter_and_cap)`` — the second
    value counts every ``*.json`` file on disk (including ``tool_calls_*``),
    so :func:`scan_folder` can show how many were excluded/capped, not just
    how many survived."""
    if root.is_file():
        return [root], 1
    all_json = sorted(root.rglob("*.json"))
    files = all_json
    if not include_tool_calls:
        files = [p for p in files if not p.name.startswith("tool_calls_")]
    return files[:max_files], len(all_json)


def scan_folder(
    path: str | Path,
    *,
    max_files: int = 200,
    include_tool_calls: bool = False,
    max_listing: int = 60,
    max_scan_entries: int = 200_000,
) -> dict[str, Any]:
    """Filesystem-level summary of what a folder actually contains.

    This is deliberately independent of :class:`~evalvitals.analysis.profile.
    DatasetProfile` (which describes parsed row/column structure): it answers
    "what did the agent see on disk" — file/dir counts, extension mix, and how
    many of the discovered JSON files were actually sampled — since that
    differs from folder to folder and is useful to see before any row-level
    parsing happens.

    Uses a single ``os.walk`` pass rather than ``Path.rglob`` + per-entry
    ``is_file()``/``is_dir()`` checks: real run directories can hold tens of
    thousands of files (one JSON per tool call), and ``rglob`` results lose
    the cached direntry type, so each check becomes its own ``stat()`` — on
    networked storage that made this function slower than the actual row
    loader it's meant to summarize. ``max_scan_entries`` is a hard backstop so
    an unexpectedly huge tree degrades (partial counts, flagged as capped)
    instead of hanging.
    """
    root = Path(path)
    if root.is_file():
        return {
            "root": str(root),
            "is_file": True,
            "n_files_total": 1,
            "n_dirs": 0,
            "extensions": {root.suffix.lower() or "(no extension)": 1},
            "json_files_found": 1,
            "json_files_used": 1,
            "entries": [root.name],
            "truncated": False,
            "scan_capped": False,
        }

    n_files = 0
    n_dirs = 0
    n_json_found = 0
    n_json_after_filter = 0
    ext_counts: dict[str, int] = {}
    entries: list[str] = []
    scan_capped = False
    scanned = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        rel_dir = Path(dirpath).relative_to(root)

        for name in dirnames:
            n_dirs += 1
            scanned += 1
            if len(entries) < max_listing:
                rel = name if rel_dir == Path(".") else str(rel_dir / name)
                entries.append(rel + "/")

        for name in filenames:
            n_files += 1
            scanned += 1
            ext = Path(name).suffix.lower() or "(no extension)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            if name.endswith(".json"):
                n_json_found += 1
                if include_tool_calls or not name.startswith("tool_calls_"):
                    n_json_after_filter += 1
            if len(entries) < max_listing:
                rel = name if rel_dir == Path(".") else str(rel_dir / name)
                entries.append(rel)

        if scanned >= max_scan_entries:
            scan_capped = True
            break

    entries.sort()
    return {
        "root": str(root),
        "is_file": False,
        "n_files_total": n_files,
        "n_dirs": n_dirs,
        "extensions": dict(sorted(ext_counts.items(), key=lambda kv: -kv[1])),
        "json_files_found": n_json_found,
        "json_files_used": min(n_json_after_filter, max_files),
        "entries": entries[:max_listing],
        "truncated": (n_files + n_dirs) > max_listing,
        "scan_capped": scan_capped,
    }


def load_records_from_path(
    path: str | Path,
    *,
    max_rows: int = 2000,
    max_files: int = 200,
    include_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Load a bounded sample of JSON/JSONL records from a file or directory."""
    root = Path(path)
    files, _ = _discover_json_files(root, max_files=max_files, include_tool_calls=include_tool_calls)
    per_file_limit = max(1, max_rows // max(1, len(files)))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        n_from_file = 0
        for row in _load_json_records(file_path):
            flat = _flatten_record(row)
            flat["_source_file"] = str(file_path)
            flat["_source_dir"] = file_path.parent.name
            rows.append(flat)
            n_from_file += 1
            if len(rows) >= max_rows:
                return rows
            if n_from_file >= per_file_limit:
                break
    return rows


_RECORD_LIST_KEYS = ("cases", "records", "results", "rows", "items", "data", "examples", "samples")


def _load_json_records(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            unpacked = _unpack_record_container(parsed)
            if unpacked is not None:
                return unpacked
        return [parsed]
    except json.JSONDecodeError:
        pass
    records: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _unpack_record_container(parsed: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Recognize a common M1-output shape: one JSON object per run/model, with
    a list of per-case dicts under a conventional key (``cases``, ``results``,
    ...) alongside scalar run metadata (e.g. ``model``, ``seed``). Unpack it
    into one row per case, each carrying the file's scalar metadata merged in
    (per-case fields win on collision) — so a directory of these files loads
    as a flat, analyzable table without a bespoke pre-processing script."""
    list_key = next(
        (
            key
            for key in _RECORD_LIST_KEYS
            if isinstance(parsed.get(key), list)
            and parsed[key]
            and all(isinstance(item, dict) for item in parsed[key])
        ),
        None,
    )
    if list_key is None:
        return None
    meta = {
        key: value
        for key, value in parsed.items()
        if key != list_key and (isinstance(value, (str, int, float, bool)) or value is None)
    }
    return [{**meta, **item} for item in parsed[list_key]]


def _flatten_record(record: Any, *, max_text: int = 500) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {"value": _shorten(record, max_text)}

    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = _shorten(value, max_text)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_key = f"{key}.{sub_key}"
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    out[flat_key] = _shorten(sub_value, max_text)
        elif isinstance(value, list):
            out[f"{key}._len"] = len(value)
            if key == "model_answer":
                out["model_answer_text"] = _shorten(_join_text_blocks(value), max_text)
        else:
            out[str(key)] = _shorten(value, max_text)

    if "is_correct" in out:
        out.setdefault("label", "pass" if bool(out["is_correct"]) else "fail")
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        out.setdefault("attempts_count", len(attempts))
    trace = record.get("trace")
    if isinstance(trace, list):
        out.setdefault("trace_steps", len(trace))
    return out


def _join_text_blocks(value: list[Any]) -> str:
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _shorten(value: Any, max_text: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_text:
        return value
    return value[:max_text] + f"... [truncated {len(value) - max_text} chars]"


def _profile_rows(
    rows: list[dict[str, Any]],
    *,
    outcome_col: str | None = None,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Profile rows for the explorer prompt, including outcome-kind detection.

    Delegates role/dtype inference to :func:`profile_records` (the shared,
    domain-agnostic profiler) rather than re-deriving it from raw values, so
    the prompt can adapt to binary / categorical / continuous / no-outcome
    data instead of always assuming a FAIL/PASS split.
    """
    profile = profile_records(rows, outcome_col=outcome_col)
    outcome = describe_outcome(profile)
    numeric_columns = [
        c.name for c in profile.columns.values()
        if c.dtype == "numeric" and c.role == "predictor"
    ]
    categorical_columns = [
        c.name for c in profile.columns.values()
        if c.dtype in ("categorical", "boolean") and c.role == "predictor"
    ]
    label_like = [outcome["column"]] if outcome["present"] else []
    return {
        "n_rows": profile.n_rows,
        "grain": profile.grain,
        "columns": {name: c.to_dict() for name, c in profile.columns.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "label_like_columns": label_like,
        "id_columns": profile.id_columns,
        "group_columns": profile.group_columns,
        "time_columns": profile.time_columns,
        "outcome": outcome,
        "warnings": profile.warnings,
        "sample_rows": rows[:sample_size],
    }


def _framing_block(outcome: dict[str, Any]) -> str:
    """Build the prompt's outcome-specific framing + standard chart battery.

    This is what stops the explorer from forcing a FAIL/PASS story onto data
    that has a multi-class, continuous, or absent outcome. ``outcome`` is the
    ``describe_outcome()`` dict already embedded in the data profile.
    """
    kind = outcome.get("kind", "none")
    col = outcome.get("column")
    unique = outcome.get("unique", 0)

    if kind == "binary":
        return f"""\
OUTCOME: "{col}" is a BINARY outcome (host detected {unique} distinct values).
Call the two groups FAIL and PASS (map the two outcome values to whichever
reads as the negative/positive case) and tell the FAIL-vs-PASS story.

As a minimum, consider this standard battery when the columns exist:
  1. Class balance: count of FAIL vs PASS overall (and per group column if present).
  2. Per numeric signal — how it separates FAIL vs PASS: distribution view or
     group summary, AND a binned fail-rate curve (bin -> fail_rate).
  3. Top discriminators: a ranked bar of each signal's FAIL-vs-PASS separation
     (e.g. standardized mean difference / |meanFAIL - meanPASS| / s), largest first.
  4. Fail rate by each categorical group column (bar).
  5. Signal correlations: a correlation table and, when helpful, a heatmap PNG.
  6. 1-2 scatter plots of the most discriminative signal pairs, coloured by outcome."""

    if kind == "categorical":
        return f"""\
OUTCOME: "{col}" is a CATEGORICAL outcome with {unique} distinct classes. Tell
the per-class story — do NOT collapse it into a binary FAIL/PASS split.

As a minimum, consider this standard battery when the columns exist:
  1. Class balance: count per class overall (and per group column if present).
  2. Per numeric signal — how its distribution differs across classes (box/violin
     per class), AND a ranked bar of each signal's cross-class separation (e.g.
     between-class variance ratio or ANOVA-style F statistic), largest first.
  3. Class composition by each categorical group column (grouped/stacked bar).
  4. Signal correlations: a correlation table and, when helpful, a heatmap PNG.
  5. 1-2 scatter plots of the most discriminative signal pairs, coloured by class."""

    if kind == "continuous":
        return f"""\
OUTCOME: "{col}" is a CONTINUOUS outcome ({unique} distinct values seen). This
is a correlation/regression-style story — there is no FAIL/PASS split to
invent; do not binarize the outcome unless the question explicitly asks for it.

As a minimum, consider this standard battery when the columns exist:
  1. Outcome distribution (histogram).
  2. Per numeric signal vs outcome: scatter with a trend line, AND a binned
     mean-outcome curve (bin -> mean {col}).
  3. Top associates: a ranked bar of each signal's correlation magnitude with
     the outcome (Pearson and/or Spearman), largest first.
  4. Outcome distribution by each categorical group column (box/violin per group).
  5. Signal correlations among predictors: a correlation table and heatmap PNG.
  6. 1-2 scatter plots of the most associated signal pairs, coloured/sized by outcome."""

    return """\
OUTCOME: no recognizable outcome/target column was found. This is UNSUPERVISED
exploration — describe the dataset's structure; do NOT invent a FAIL/PASS or
any other label that is not actually in the data.

As a minimum, consider this standard battery when the columns exist:
  1. Missingness overview: non-null rate per column (bar).
  2. Per numeric column: distribution (histogram/box).
  3. Per categorical column: value counts (bar); skip columns with very high
     cardinality (say >20 distinct values) as a bar chart and note it in caveats.
  4. Signal correlations: a correlation table and, when helpful, a heatmap PNG
     among the numeric columns.
  5. 1-2 scatter plots of the most correlated numeric pairs.
  6. If a group or time column exists, contrast numeric distributions across
     its groups/periods; otherwise skip this item."""


def _plain_language_violations(report: ExploratoryAnalysisReport) -> list[str]:
    """Host-side check behind the prompt's ``plain_title``/``plain_question``
    requirement — the prompt already asks for jargon-free headlines, but that
    alone doesn't hold, so this is what actually gates a retry (see
    ``_run_explore_loop``)."""
    violations = []
    for i, t in enumerate(report.takeaways, start=1):
        reason = jargon_violation(t.plain_title, t.title)
        if reason:
            violations.append(f'takeaway {i}\'s "plain_title" {reason}: {t.plain_title!r}')
    # The caller's own `question` is often already plain (short, jargon-free)
    # — don't force a pointless rewrite of a fine question just because
    # plain_question is empty or a verbatim copy in that case.
    if jargon_violation(report.question) is not None:
        reason = jargon_violation(report.plain_question, report.question)
        if reason:
            violations.append(f'"plain_question" {reason}: {report.plain_question!r}')
    return violations


def _report_from_sandbox(
    *,
    question: str,
    profile: dict[str, Any],
    code: str,
    result: SandboxResult,
    attempts: int,
    workdir: Path,
) -> tuple[ExploratoryAnalysisReport, str]:
    if not result.ok:
        err = (result.stderr or "").strip() or "sandbox run failed"
        return (
            ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                error=err,
                attempts=attempts,
                workdir=str(workdir),
            ),
            err,
        )

    parsed, err = _parse_result_json(result.stdout)
    if err:
        return (
            ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                error=err,
                attempts=attempts,
                workdir=str(workdir),
            ),
            err,
        )

    plots = _normalize_plot_paths(parsed.get("plots", []), workdir)
    takeaways = [
        Takeaway(
            title=str(item.get("title", "")),
            plain_title=str(item.get("plain_title", "")),
            analysis=str(item.get("analysis", "")),
            chart_names=[str(x) for x in item.get("chart_names", []) or []],
            table_names=[str(x) for x in item.get("table_names", []) or []],
            caveat=str(item.get("caveat", "")),
        )
        for item in parsed.get("takeaways", []) or []
        if isinstance(item, dict) and (item.get("title") or item.get("plain_title"))
    ]
    signals = [
        CandidateSignal(
            name=str(item.get("name", "")),
            display_name=str(item.get("display_name", "")),
            rationale=str(item.get("rationale", "")),
            suggested_test=str(item.get("suggested_test", "")),
            sufficient=item["sufficient"] if isinstance(item.get("sufficient"), dict) else None,
            recipe=item["recipe"] if isinstance(item.get("recipe"), dict) else None,
        )
        for item in parsed.get("candidate_signals", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    return (
        ExploratoryAnalysisReport(
            question=question,
            plain_question=str(parsed.get("plain_question", "")),
            ok=True,
            observations=[str(x) for x in parsed.get("observations", []) or []],
            takeaways=takeaways,
            visual_plan=_normalize_visual_plan(parsed.get("visual_plan", [])),
            chart_readings=_normalize_list_of_dicts(parsed.get("chart_readings", [])),
            dashboard_storyboard=_normalize_list_of_dicts(parsed.get("dashboard_storyboard", [])),
            claims=_normalize_list_of_dicts(parsed.get("claims", [])),
            candidate_signals=signals,
            plots=plots,
            tables=dict(parsed.get("tables", {}) or {}),
            charts=[dict(x) for x in parsed.get("charts", []) or [] if isinstance(x, dict)],
            caveats=[str(x) for x in parsed.get("caveats", []) or []],
            critique=[str(x) for x in parsed.get("critique", []) or []],
            recommended_confirmatory_tests=[
                str(x) for x in parsed.get("recommended_confirmatory_tests", []) or []
            ],
            data_profile=profile,
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            attempts=attempts,
            workdir=str(workdir),
        ),
        "",
    )


def _parse_result_json(stdout: str) -> tuple[dict[str, Any], str]:
    marker_line = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_RESULT_MARKER):
            marker_line = stripped[len(_RESULT_MARKER):]
    if marker_line is None:
        return {}, f"no {_RESULT_MARKER} line in output"
    try:
        parsed = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        return {}, f"unparseable {_RESULT_MARKER} JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"{_RESULT_MARKER} payload must be a JSON object"
    return parsed, ""


def _normalize_visual_plan(raw: Any) -> list[dict[str, Any]]:
    """Keep agent plot-type decisions auditable without making old reports invalid."""
    plan: list[dict[str, Any]] = []
    for idx, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key in (
            "name",
            "display_name",
            "question",
            "data_shape",
            "plot_kind",
            "fallback_kind",
            "rationale",
            "status",
            "disposition",
            "not_promoted_reason",
        ):
            if item.get(key) is not None:
                cleaned[key] = str(item.get(key))
        cols = item.get("required_columns")
        if isinstance(cols, list):
            cleaned["required_columns"] = [str(c) for c in cols]
        elif cols is not None:
            cleaned["required_columns"] = [str(cols)]
        if "name" not in cleaned:
            cleaned["name"] = f"visual_{idx}"
        plan.append(cleaned)
    return plan


def _normalize_list_of_dicts(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[str(key)] = value
            elif isinstance(value, list):
                cleaned[str(key)] = [str(x) for x in value]
            elif isinstance(value, dict):
                cleaned[str(key)] = {str(k): str(v) for k, v in value.items()}
            else:
                cleaned[str(key)] = str(value)
        out.append(cleaned)
    return out


def _normalize_plot_paths(raw: Any, workdir: Path) -> list[str]:
    paths: list[str] = []
    for item in raw or []:
        text = str(item)
        path = Path(text)
        if not path.is_absolute():
            candidate = workdir / path
            if candidate.exists():
                path = candidate
        paths.append(str(path))
    figures_dir = workdir / "figures"
    if figures_dir.exists():
        for png in sorted(figures_dir.glob("*.png")):
            p = str(png)
            if p not in paths:
                paths.append(p)
    return paths


def _extract_code(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()
