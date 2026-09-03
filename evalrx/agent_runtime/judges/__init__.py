"""CLI-backed judge model wrappers."""

from evalrx.agent_runtime.judges.agy import AgyModel, scan_agy_log
from evalrx.agent_runtime.judges.autodetect import (
    DEFAULT_AGY_CANDIDATES,
    DEFAULT_CLAUDE_CANDIDATES,
    ResolvedJudge,
    pick_agy_model,
    pick_claude_model,
    pick_live_model,
    resolve_cli_judge,
)
from evalrx.agent_runtime.judges.claude import ClaudeModel
from evalrx.agent_runtime.judges.codex import CodexModel

__all__ = [
    "AgyModel",
    "ClaudeModel",
    "CodexModel",
    "scan_agy_log",
    "DEFAULT_AGY_CANDIDATES",
    "DEFAULT_CLAUDE_CANDIDATES",
    "ResolvedJudge",
    "pick_agy_model",
    "pick_claude_model",
    "pick_live_model",
    "resolve_cli_judge",
]
