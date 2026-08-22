"""Self-evolving evaluation agent — automated failure discovery and diagnosis.

Two loops are available:

  AutoDiagnoseLoop   M1 → M2 → M3 → M4 (legacy, four-stage sweep)
  VLDiagnoseLoop     M1 → M2 → M3 → M5 inner loop, M4 called post-loop (Plan A)
                     Stops when M5 finds a statistically supported,
                     protocol-consistent hypothesis.

Stage modules live in ``stages/``; shared infrastructure stays at the top level.
The CLI-agent runtime (sandboxing, code generation, providers, judge models,
skills) lives in ``evalvitals.agent_runtime`` — shared with ``evalvitals.analysis``
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

evalvitals.agent_runtime (shared CLI-agent runtime, imports neither analysis nor eval_agent):
  cli_types.py         CliAgentConfig / CliAgentResult
  cli_runtime.py       SubprocessRunner, ProcessRun, collect_py_files
  sandbox.py           ExperimentSandbox, SandboxProtocol
  experiment_harness.py immutable evaluation harness injected into projects
  factory.py           sandbox factory (subprocess / docker backends)
  providers/           CLI coding-provider adapters and registry
  judges/              CLI-backed judge model wrappers (agy / Claude Code)
  codegen/             shared code-generation runner used by stages
  skills/              skill resolution, installation, and prompt policy

stages/ (M1–M5 implementation):
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
  stats_tools.py       M2 — statistical tool catalog wrapping evalvitals.stats
                              (signal/label association, McNemar+e-value, Friedman,
                              single-rate e-value, rank corr) + StatsInput/fdr_correct
  stats_tool_agent.py  M2 — legacy deterministic exploratory stats tools
  stats_tool_generator.py M2 tier(b) — StatsToolGenerator: LLM/CLI writes a new
                              stats script, runs it in a sandbox, parses a
                              STATS_RESULT_JSON contract (never mutates repo source)
  diagnosis.py         M3 — DiagnosisAgent: judge reads report → Hypothesis list
  case_discovery.py    Data — run candidate prompts and label PASS/FAIL cases
  surgery.py           M4 — SurgeryAgent: correlate / param-sweep / ExperimentWriter
                              → InterventionResult (SUPPORTED / REFUTED / INCONCLUSIVE)
  experiment_writer.py M4 — multi-phase LLM/CLI agent writes + executes fix scripts
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
  hypothesis_tester.py M5 — HypothesisTester: statistical test + protocol consistency;
                              stopping_criteria_met() drives the VLDiagnoseLoop exit
"""

from evalvitals.agent_runtime.cli_types import CliAgentConfig, CliAgentResult
from evalvitals.agent_runtime.factory import SandboxConfig, SandboxFactoryConfig, create_sandbox
from evalvitals.agent_runtime.judges import AgyModel, ClaudeModel, CodexModel
from evalvitals.agent_runtime.providers import create_cli_agent
from evalvitals.agent_runtime.sandbox import (
    ExperimentSandbox,
    SandboxProtocol,
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)
from evalvitals.analysis.analysis_module import AnalysisModule, AnalysisReport
from evalvitals.analysis.planner import AnalysisPlanItem, plan_stats_input, ranked_signal_names
from evalvitals.analysis.profile import (
    ColumnProfile,
    DatasetProfile,
    describe_outcome,
    profile_records,
    profile_stats_input,
)
from evalvitals.analysis.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
from evalvitals.analysis.stats_tool_agent import StatsToolAgent
from evalvitals.analysis.stats_tool_generator import (
    GeneratedStatsTool,
    StatsToolGenerator,
)
from evalvitals.analysis.stats_tools import (
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
from evalvitals.eval_agent.ab_runner import ABResult, ABRunner
from evalvitals.eval_agent.agentic import (
    AgenticDiagnoseLoop,
    ToolOutcome,
    ToolRegistry,
    ToolSpec,
)
from evalvitals.eval_agent.evolution import EvolutionStore, LessonEntry, extract_lessons
from evalvitals.eval_agent.git_manager import ExperimentGitManager
from evalvitals.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisGenerator,
    HypothesisStatus,
    ManualHypothesisGenerator,
    hypothesis_from_dict,
    hypothesis_to_dict,
)
from evalvitals.eval_agent.legacy import AutoDiagnoseLoop, SelfEvolveLoop
from evalvitals.eval_agent.log_schema import (
    SCHEMA_PATH,
    build_schema,
    iter_log_errors,
    load_schema,
    validate_event,
)
from evalvitals.eval_agent.loop import VLDiagnoseLoop
from evalvitals.eval_agent.loop_reports import AutoDiagnoseReport, VLDiagnoseReport
from evalvitals.eval_agent.nl_runner import scaffold_from_description
from evalvitals.eval_agent.orchestrator import EvalOrchestrator
from evalvitals.eval_agent.preregister import (
    DataSplit,
    PreregisteredHypothesis,
    PreregistrationLog,
    Split,
)
from evalvitals.eval_agent.report import DiagnosticReport
from evalvitals.eval_agent.run_context import RunContext
from evalvitals.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION, RunLogger
from evalvitals.eval_agent.stages.case_discovery import (
    CaseDiscoveryAgent,
    CaseDiscoveryReport,
)
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent, DiagnosisResult
from evalvitals.eval_agent.stages.experiment_writer import (
    ExperimentWriter,
    ExperimentWriterConfig,
    ExperimentWriterResult,
    SolutionNode,
    build_model_context,
)
from evalvitals.eval_agent.stages.fix_agent import (
    FixAgent,
    FixCandidate,
    FixOutcome,
    FixValidation,
)
from evalvitals.eval_agent.stages.fix_internals import (
    INTERNALS_PRIMITIVES,
    FinetuneSpec,
    InternalsPrimitive,
)
from evalvitals.eval_agent.stages.fix_tiers import FixTier, parse_tier, route_min_tier
from evalvitals.eval_agent.stages.fix_tools import PipelineSpec
from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester, HypothesisTestResult
from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe
from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
from evalvitals.eval_agent.stages.probe_candidate_generator import VLMProbeCandidateGenerator
from evalvitals.eval_agent.stages.probe_generator import GeneratedProbe, ProbeGenerator
from evalvitals.eval_agent.stages.probe_search_agent import ProbeSearchAgent
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol, ProbingSchema
from evalvitals.eval_agent.stages.surgery import InterventionResult, SurgeryAgent
from evalvitals.eval_agent.stages.whitebox_probe_generator import (
    GeneratedWhiteboxProbe,
    WhiteboxProbeGenerator,
)
from evalvitals.eval_agent.store import InMemoryStore, JsonlStore, Store

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
    # M4
    "SurgeryAgent",
    "InterventionResult",
    # Loop
    "AutoDiagnoseLoop",
    "AutoDiagnoseReport",
    "SelfEvolveLoop",
    "VLDiagnoseLoop",
    "VLDiagnoseReport",
    # Agentic loop (judge-decided M1-M5, alternative to VLDiagnoseLoop's fixed cycle)
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
    # M5 hypothesis tester
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
