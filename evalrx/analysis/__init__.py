"""Public data-analysis API: exploratory analysis, hypothesis proposal, and
statistical confirmation.

This package exposes EvalRX' data-analysis layer as a standalone
capability — ``ExploratoryAnalysisAgent`` (M2) for descriptive EDA (no
hypothesis generation or validation itself), ``HypothesisAgent`` (M3) for
proposing falsifiable hypotheses from M2's takeaways (proposal only, no
validation), and ``StatsAnalysisAgent`` for confirmatory effect/CI/e-value/FDR
verdicts. The eval-agent loops still use the same implementation internally,
but callers do not need to import from ``evalrx.eval_agent.stages``.
"""

from evalrx.analysis.adjudicate import adjudicate_report, adjudicate_signals
from evalrx.analysis.api import ExploreRunResult, explore
from evalrx.analysis.explorer import (
    CandidateSignal,
    ExploratoryAnalysisAgent,
    ExploratoryAnalysisReport,
    Takeaway,
    load_records_from_path,
    scan_folder,
)
from evalrx.analysis.failure_modes import FailureMode, FailureModeReport, cluster_failures
from evalrx.analysis.fused_pipeline import (
    FusedReport,
    FusedSignal,
    run_fused_analysis,
)
from evalrx.analysis.hypothesis_agent import Hypothesis, HypothesisAgent
from evalrx.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    bridge_recipes_to_result,
    compile_recipe,
    compile_recipes,
    per_case_finding,
    per_case_to_records,
    safe_ident,
)
from evalrx.analysis.planner import AnalysisPlanItem, plan_stats_input, ranked_signal_names
from evalrx.analysis.profile import (
    ColumnProfile,
    DatasetProfile,
    describe_outcome,
    profile_records,
    profile_stats_input,
)
from evalrx.analysis.run_codebase import CodebaseRunResult, run_codebase, run_codebase_cli
from evalrx.analysis.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
from evalrx.analysis.stats_tools import (
    STATS_TOOL_CATALOG,
    EvidenceResult,
    StatsInput,
    StatsToolResult,
    build_stats_input,
    build_stats_input_from_records,
    default_plan,
    fdr_correct,
    run_stats_tool,
)
from evalrx.analysis.workbench import (
    DatasetBundle,
    EventSink,
    ThreadStore,
    UploadLimits,
    extract_archive,
    ingest_directory,
)
from evalrx.reporting.compiler import compile_diagnostic_report
from evalrx.reporting.model import Claim, DiagnosticReport, Evidence, ReportStep

__all__ = [
    "explore",
    "ExploreRunResult",
    "run_codebase",
    "run_codebase_cli",
    "CodebaseRunResult",
    "cluster_failures",
    "FailureMode",
    "FailureModeReport",
    "StatsAnalysisAgent",
    "StatsAnalysisReport",
    "ExploratoryAnalysisAgent",
    "ExploratoryAnalysisReport",
    "Takeaway",
    "CandidateSignal",
    "Hypothesis",
    "HypothesisAgent",
    "adjudicate_report",
    "adjudicate_signals",
    "SignalRecipe",
    "compile_recipe",
    "compile_recipes",
    "per_case_finding",
    "per_case_to_records",
    "bridge_recipes_to_result",
    "safe_ident",
    "RecipeError",
    "ColumnProfile",
    "DatasetProfile",
    "describe_outcome",
    "profile_records",
    "profile_stats_input",
    "AnalysisPlanItem",
    "ranked_signal_names",
    "plan_stats_input",
    "run_fused_analysis",
    "FusedReport",
    "FusedSignal",
    "DiagnosticReport",
    "Claim",
    "Evidence",
    "ReportStep",
    "compile_diagnostic_report",
    "load_records_from_path",
    "scan_folder",
    "StatsInput",
    "StatsToolResult",
    "EvidenceResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
    "DatasetBundle",
    "EventSink",
    "ThreadStore",
    "UploadLimits",
    "extract_archive",
    "ingest_directory",
]
