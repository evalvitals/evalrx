"""Self-evolving evaluation agent — automated failure discovery and diagnosis.

Two loops are available:

  AutoDiagnoseLoop   M1 → M2 → M3 → M5 (legacy, four-stage sweep)
  VLDiagnoseLoop     M1 → M2 → M3 → M4 inner loop, M5 called post-loop (Plan A)
                     Stops when M4 finds a statistically supported,
                     protocol-consistent hypothesis.

Stage modules live in ``stages/``; shared infrastructure stays at the top level.
The CLI-agent runtime (sandboxing, code generation, providers, judge models,
skills) lives in ``evalrx.agent_runtime`` — shared with ``evalrx.analysis``
so neither package depends on the other. ``cli_agent.py`` / ``cli_skills.py``
remain here as compatibility facades over that runtime.

Top-level (shared / orchestration):
  loop.py              AutoDiagnoseLoop, VLDiagnoseLoop, SelfEvolveLoop
  run_logger.py        RunLogger — per-cycle JSONL log + artifact sink
  hypothesis.py        Hypothesis, HypothesisStatus, serialization helpers
  cli_agent.py         compatibility facade over agent_runtime CLI providers/judges
  cli_skills.py        compatibility facade over agent_runtime.skills.installer
  store.py             Store / InMemoryStore / JsonlStore — persistent memory
  orchestrator.py      thin facade over the loop (pre-registered A/B)
  ab_runner.py         A/B execution across prompting strategies
  report.py            DiagnosticReport — final diagnostic conclusions
  evolution.py         EvolutionStore — JSONL lesson store, 30-day half-life decay
  preregister.py       pre-registration helpers (DataSplit, PreregisteredHypothesis)
  git_manager.py       git-native experiment versioning (eval/{run_id} branches)

evalrx.agent_runtime (shared CLI-agent runtime, imports neither analysis nor eval_agent):
  cli_types.py         CliAgentConfig / CliAgentResult
  cli_runtime.py       SubprocessRunner, ProcessRun, collect_py_files
  sandbox.py           ExperimentSandbox, SandboxProtocol
  experiment_harness.py immutable evaluation harness injected into projects
  factory.py           sandbox factory (subprocess / docker backends)
  providers/           CLI coding-provider adapters and registry
  judges/              CLI-backed judge model wrappers (agy / Claude Code)
  codegen/             shared code-generation runner used by stages
  skills/              skill resolution, installation, and prompt policy

stages/ (M1–M4 implementation):
  probe.py             M1 — StrategyProbe: model-kind detection + analyzer ranking
  probe_agent.py       M1 — ProbeAgent: execute ranked analyzers (direct or Docker);
                              protocol-guided via ExperimentProtocol.probe_hints();
                              tier(b) generates a bespoke probe when no analyzer fits
  probe_generator.py   M1 tier(b) — ProbeGenerator: host collects model outputs,
                              an LLM/CLI writes a probe over them, run in a sandbox,
                              parse PROBE_RESULT_JSON into a Result (per_case findings)
  protocol.py          M1 — ExperimentProtocol (NL description → probe hints);
                              ProbingSchema (records M1 selection rationale)
  analysis.py          M2 — AnalysisModule: threshold rules → AnalysisReport
  stats_agent.py       M2 — StatsAnalysisAgent: extends AnalysisModule with a
                              statistical-tool layer (select tools from the catalog,
                              run them, e-BH FDR-correct, plot) + LLM-guided
                              conclusion/evidence chain (StatsAnalysisReport)
  stats_tools.py       M2 — statistical tool catalog wrapping evalrx.stats
                              (signal/label association, McNemar+e-value, Friedman,
                              single-rate e-value, rank corr) + StatsInput/fdr_correct
  stats_tool_agent.py  M2 — legacy deterministic exploratory stats tools
  stats_tool_generator.py M2 tier(b) — StatsToolGenerator: LLM/CLI writes a new
                              stats script, runs it in a sandbox, parses a
                              STATS_RESULT_JSON contract (never mutates repo source)
  diagnosis.py         M3 — DiagnosisAgent: judge reads report → Hypothesis list
  case_discovery.py    Data — run candidate prompts and label PASS/FAIL cases
  surgery.py           M5 — SurgeryAgent: correlate / param-sweep / ExperimentWriter
                              → InterventionResult (SUPPORTED / REFUTED / INCONCLUSIVE)
  experiment_writer.py M5 — multi-phase LLM/CLI agent writes + executes fix scripts
  fix_tiers.py         Fix — FixTier intervention-space ladder (L0 runtime /
                              L1 prompt / L2 scaffold / L3a read / L3b write /
                              L4 params)
                              + hypothesis -> minimum-tier routing
  fix_tools.py         Fix — L2 tool catalog (zoom/contrast/equalize/upscale)
                              + PipelineSpec executor around the unchanged model
  fix_agent.py         Fix — FixAgent: tiered candidates -> paired McNemar
                              validation -> FixOutcome (+ tier recommendation)
  fix_pipeline.py      Fix — L2 coded pipelines: sandboxed agent code with
                              bridged model access (model_generate/model_attend)
  fix_internals.py     Fix — L3a attention-guided crop, L3b intervention
                              primitives (visual embedding boost); L4
                              FinetuneSpec + run_lora_repair (v1: LoRA on
                              target="llm" only, trained on a caller-supplied
                              finetune_pool; other recipe shapes recorded only)
  hypothesis_tester.py M4 — HypothesisTester: statistical test + protocol consistency;
                              stopping_criteria_met() drives the VLDiagnoseLoop exit
"""

from evalrx.agent_runtime.cli_types import CliAgentConfig, CliAgentResult
from evalrx.agent_runtime.factory import SandboxConfig, SandboxFactoryConfig, create_sandbox
from evalrx.agent_runtime.judges import AgyModel, ClaudeModel, CodexModel
from evalrx.agent_runtime.providers import create_cli_agent
from evalrx.agent_runtime.sandbox import (
    ExperimentSandbox,
    SandboxProtocol,
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)
from evalrx.analysis.analysis_module import AnalysisModule, AnalysisReport
from evalrx.analysis.planner import AnalysisPlanItem, plan_stats_input, ranked_signal_names
from evalrx.analysis.profile import (
    ColumnProfile,
    DatasetProfile,
    describe_outcome,
    profile_records,
    profile_stats_input,
)
from evalrx.analysis.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
from evalrx.analysis.stats_tool_agent import StatsToolAgent
from evalrx.analysis.stats_tool_generator import (
    GeneratedStatsTool,
    StatsToolGenerator,
)
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
from evalrx.eval_agent.ab_runner import ABResult, ABRunner
from evalrx.eval_agent.agentic import (
    AgenticDiagnoseLoop,
    ToolOutcome,
    ToolRegistry,
    ToolSpec,
)
from evalrx.eval_agent.evolution import EvolutionStore, LessonEntry, extract_lessons
from evalrx.eval_agent.git_manager import ExperimentGitManager
from evalrx.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisGenerator,
    HypothesisStatus,
    ManualHypothesisGenerator,
    hypothesis_from_dict,
    hypothesis_to_dict,
)
from evalrx.eval_agent.legacy import AutoDiagnoseLoop, SelfEvolveLoop
from evalrx.eval_agent.log_schema import (
    SCHEMA_PATH,
    build_schema,
    iter_log_errors,
    load_schema,
    validate_event,
)
from evalrx.eval_agent.loop import VLDiagnoseLoop
from evalrx.eval_agent.loop_reports import AutoDiagnoseReport, VLDiagnoseReport
from evalrx.eval_agent.nl_runner import scaffold_from_description
from evalrx.eval_agent.orchestrator import EvalOrchestrator
from evalrx.eval_agent.preregister import (
    DataSplit,
    PreregisteredHypothesis,
    PreregistrationLog,
    Split,
)
from evalrx.eval_agent.report import DiagnosticReport
from evalrx.eval_agent.run_context import RunContext
from evalrx.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION, RunLogger
from evalrx.eval_agent.stages.case_discovery import (
    CaseDiscoveryAgent,
    CaseDiscoveryReport,
)
from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent, DiagnosisResult
from evalrx.eval_agent.stages.experiment_writer import (
    ExperimentWriter,
    ExperimentWriterConfig,
    ExperimentWriterResult,
    SolutionNode,
    build_model_context,
)
from evalrx.eval_agent.stages.fix_agent import (
    FixAgent,
    FixCandidate,
    FixOutcome,
    FixValidation,
)
from evalrx.eval_agent.stages.fix_internals import (
    INTERNALS_PRIMITIVES,
    FinetuneSpec,
    InternalsPrimitive,
)
from evalrx.eval_agent.stages.fix_tiers import FixTier, parse_tier, route_min_tier
from evalrx.eval_agent.stages.fix_tools import PipelineSpec
from evalrx.eval_agent.stages.hypothesis_tester import HypothesisTester, HypothesisTestResult
from evalrx.eval_agent.stages.probe import ModelKind, StrategyProbe
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from evalrx.eval_agent.stages.probe_candidate_generator import VLMProbeCandidateGenerator
from evalrx.eval_agent.stages.probe_generator import GeneratedProbe, ProbeGenerator
from evalrx.eval_agent.stages.probe_search_agent import ProbeSearchAgent
from evalrx.eval_agent.stages.protocol import ExperimentProtocol, ProbingSchema
from evalrx.eval_agent.stages.surgery import InterventionResult, SurgeryAgent
from evalrx.eval_agent.stages.whitebox_probe_generator import (
    GeneratedWhiteboxProbe,
    WhiteboxProbeGenerator,
)
from evalrx.eval_agent.store import InMemoryStore, JsonlStore, Store

__all__ = [
    # Judge
    "AgyModel",
    "ClaudeModel",
    "CodexModel",
    # M1
    "ProbeAgent",
    "StrategyProbe",
    "ModelKind",
    # M1 tier (b) probe generation
    "ProbeGenerator",
    "GeneratedProbe",
    "WhiteboxProbeGenerator",
    "GeneratedWhiteboxProbe",
    # ProbeLLM-style hierarchical MCTS probe search
    "ProbeSearchAgent",
    "VLMProbeCandidateGenerator",
    # Fix module (post-loop tiered repair)
    "FixAgent",
    "FixCandidate",
    "FixOutcome",
    "FixValidation",
    "FixTier",
    "parse_tier",
    "route_min_tier",
    "PipelineSpec",
    "INTERNALS_PRIMITIVES",
    "InternalsPrimitive",
    "FinetuneSpec",
    # M2
    "AnalysisModule",
    "AnalysisReport",
    # M3
    "DiagnosisAgent",
    "DiagnosisResult",
    # Case discovery / labeling
    "CaseDiscoveryAgent",
    "CaseDiscoveryReport",
    # M5
    "SurgeryAgent",
    "InterventionResult",
    # Loop
    "AutoDiagnoseLoop",
    "AutoDiagnoseReport",
    "SelfEvolveLoop",
    "VLDiagnoseLoop",
    "VLDiagnoseReport",
    # Agentic loop (judge-decided M1-M4, alternative to VLDiagnoseLoop's fixed cycle)
    "AgenticDiagnoseLoop",
    "ToolSpec",
    "ToolOutcome",
    "ToolRegistry",
    # Protocol
    "ExperimentProtocol",
    "ProbingSchema",
    # M2 stats agent
    "StatsAnalysisAgent",
    "StatsAnalysisReport",
    "StatsToolAgent",
    # M2 stats tools
    "StatsInput",
    "StatsToolResult",
    "EvidenceResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
    "ColumnProfile",
    "DatasetProfile",
    "describe_outcome",
    "profile_records",
    "profile_stats_input",
    "AnalysisPlanItem",
    "ranked_signal_names",
    "plan_stats_input",
    # M2 tier (b) code generation
    "StatsToolGenerator",
    "GeneratedStatsTool",
    # M4 hypothesis tester
    "HypothesisTester",
    "HypothesisTestResult",
    # Shared
    "EvalOrchestrator",
    "Hypothesis",
    "HypothesisGenerator",
    "ManualHypothesisGenerator",
    "HypothesisStatus",
    "hypothesis_to_dict",
    "hypothesis_from_dict",
    "Store",
    "InMemoryStore",
    "JsonlStore",
    "ABRunner",
    "ABResult",
    "DataSplit",
    "Split",
    "PreregisteredHypothesis",
    "PreregistrationLog",
    "DiagnosticReport",
    "RunContext",
    "RunLogger",
    # run_log.jsonl published schema
    "RUN_LOG_SCHEMA_VERSION",
    "build_schema",
    "load_schema",
    "validate_event",
    "iter_log_errors",
    "SCHEMA_PATH",
    # Experiment execution
    "ExperimentWriter",
    "ExperimentWriterConfig",
    "ExperimentWriterResult",
    "SolutionNode",
    "build_model_context",
    "ExperimentSandbox",
    "SandboxResult",
    "SandboxProtocol",
    "parse_metrics",
    "validate_entry_point",
    "validate_entry_point_resolved",
    # Sandbox factory
    "SandboxConfig",
    "SandboxFactoryConfig",
    "create_sandbox",
    # NL scaffold
    "scaffold_from_description",
    # Git versioning
    "ExperimentGitManager",
    # Evolution store
    "EvolutionStore",
    "LessonEntry",
    "extract_lessons",
    # CLI agents
    "CliAgentConfig",
    "CliAgentResult",
    "create_cli_agent",
]
