"""CLI-agent runtime: sandboxing, code generation, providers, judge models, and skills.

Shared by ``evalrx.analysis`` and ``evalrx.eval_agent`` — this package
imports nothing from either, so it is safe for both to depend on it.

Submodules:
  cli_types      CliAgentConfig / CliAgentResult and BINARY_DEFAULTS
  cli_runtime    SubprocessRunner, ProcessRun, collect_py_files
  cli_transcript CLI stdout post-processing (e.g. Claude stream-json rendering)
  sandbox        ExperimentSandbox — safe subprocess execution of generated code
  factory        create_sandbox() — subprocess/Docker sandbox backend selection
  codegen        CodegenRunner — the single boundary stages use to invoke a CLI
                 coding agent and harvest generated files
  providers      CLI coding-provider adapters (claude_code, codex, antigravity, ...)
  judges         CLI-backed judge model wrappers (ClaudeModel, AgyModel) +
                 liveness-probe autodetection (resolve_cli_judge)
  skills         Agent Skills resolution, sandbox installation, and prompt policy
"""

from __future__ import annotations

from evalrx.agent_runtime.cli_types import (
    BINARY_DEFAULTS,
    CliAgentConfig,
    CliAgentResult,
)
from evalrx.agent_runtime.codegen import CodegenCodeResult, CodegenRunner
from evalrx.agent_runtime.factory import (
    SandboxConfig,
    SandboxFactoryConfig,
    create_sandbox,
)
from evalrx.agent_runtime.json_shape import validate_json_shape
from evalrx.agent_runtime.judges import (
    AgyModel,
    ClaudeModel,
    CodexModel,
    ResolvedJudge,
    resolve_cli_judge,
)
from evalrx.agent_runtime.providers import create_cli_agent
from evalrx.agent_runtime.sandbox import (
    ExperimentSandbox,
    SandboxProtocol,
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)
from evalrx.agent_runtime.skills.prompt_policy import fences_hint, skills_hint
from evalrx.agent_runtime.skills.resolver import resolve_skill_paths

__all__ = [
    "BINARY_DEFAULTS",
    "CliAgentConfig",
    "CliAgentResult",
    "CodegenCodeResult",
    "CodegenRunner",
    "SandboxConfig",
    "SandboxFactoryConfig",
    "create_sandbox",
    "AgyModel",
    "ClaudeModel",
    "CodexModel",
    "ResolvedJudge",
    "resolve_cli_judge",
    "create_cli_agent",
    "ExperimentSandbox",
    "SandboxProtocol",
    "SandboxResult",
    "parse_metrics",
    "validate_entry_point",
    "validate_entry_point_resolved",
    "resolve_skill_paths",
    "skills_hint",
    "fences_hint",
    "validate_json_shape",
]
