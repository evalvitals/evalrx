"""Compatibility facade for CLI coding agents and CLI-backed judge models.

The implementation lives in ``evalrx.agent_runtime``:

- ``evalrx.agent_runtime.providers`` contains CLI coding-provider adapters.
- ``evalrx.agent_runtime.judges`` contains judge model wrappers.
- ``evalrx.agent_runtime.cli_types`` contains public config/result types.

This module keeps the historical ``evalrx.eval_agent.cli_agent`` import
path stable.
"""

from __future__ import annotations

from evalrx.agent_runtime.cli_types import BINARY_DEFAULTS, CliAgentConfig, CliAgentResult
from evalrx.agent_runtime.judges.agy import (
    AgyModel,
)
from evalrx.agent_runtime.judges.agy import (
    safe_unlink as _safe_unlink,
)
from evalrx.agent_runtime.judges.agy import (
    scan_agy_log as _scan_agy_log,
)
from evalrx.agent_runtime.judges.claude import ClaudeModel
from evalrx.agent_runtime.judges.codex import CodexModel
from evalrx.agent_runtime.providers.antigravity import AntigravityAgent
from evalrx.agent_runtime.providers.base import CliAgentBase as _CliAgentBase
from evalrx.agent_runtime.providers.claude_code import ClaudeCodeAgent
from evalrx.agent_runtime.providers.codex import CodexAgent
from evalrx.agent_runtime.providers.gemini_cli import GeminiCliAgent
from evalrx.agent_runtime.providers.kimi_cli import KimiCliAgent
from evalrx.agent_runtime.providers.opencode import OpenCodeAgent
from evalrx.agent_runtime.providers.registry import (
    PROVIDER_CLASSES as _PROVIDER_CLASSES,
)
from evalrx.agent_runtime.providers.registry import (
    create_cli_agent,
)

__all__ = [
    "AgyModel",
    "AntigravityAgent",
    "BINARY_DEFAULTS",
    "ClaudeCodeAgent",
    "ClaudeModel",
    "CodexModel",
    "CliAgentConfig",
    "CliAgentResult",
    "CodexAgent",
    "GeminiCliAgent",
    "KimiCliAgent",
    "OpenCodeAgent",
    "_CliAgentBase",
    "_PROVIDER_CLASSES",
    "_safe_unlink",
    "_scan_agy_log",
    "create_cli_agent",
]
