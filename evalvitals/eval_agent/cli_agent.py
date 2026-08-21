"""Compatibility facade for CLI coding agents and CLI-backed judge models.

The implementation lives in ``evalvitals.agent_runtime``:

- ``evalvitals.agent_runtime.providers`` contains CLI coding-provider adapters.
- ``evalvitals.agent_runtime.judges`` contains judge model wrappers.
- ``evalvitals.agent_runtime.cli_types`` contains public config/result types.

This module keeps the historical ``evalvitals.eval_agent.cli_agent`` import
path stable.
"""

from __future__ import annotations

from evalvitals.agent_runtime.cli_types import BINARY_DEFAULTS, CliAgentConfig, CliAgentResult
from evalvitals.agent_runtime.judges.agy import (
    AgyModel,
)
from evalvitals.agent_runtime.judges.agy import (
    safe_unlink as _safe_unlink,
)
from evalvitals.agent_runtime.judges.agy import (
    scan_agy_log as _scan_agy_log,
)
from evalvitals.agent_runtime.judges.claude import ClaudeModel
from evalvitals.agent_runtime.judges.codex import CodexModel
from evalvitals.agent_runtime.providers.antigravity import AntigravityAgent
from evalvitals.agent_runtime.providers.base import CliAgentBase as _CliAgentBase
from evalvitals.agent_runtime.providers.claude_code import ClaudeCodeAgent
from evalvitals.agent_runtime.providers.codex import CodexAgent
from evalvitals.agent_runtime.providers.gemini_cli import GeminiCliAgent
from evalvitals.agent_runtime.providers.kimi_cli import KimiCliAgent
from evalvitals.agent_runtime.providers.opencode import OpenCodeAgent
from evalvitals.agent_runtime.providers.registry import (
    PROVIDER_CLASSES as _PROVIDER_CLASSES,
)
from evalvitals.agent_runtime.providers.registry import (
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
