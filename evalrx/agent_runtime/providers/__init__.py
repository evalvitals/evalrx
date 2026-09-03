"""CLI coding-provider adapters."""

from evalrx.agent_runtime.providers.antigravity import AntigravityAgent
from evalrx.agent_runtime.providers.base import CliAgentBase
from evalrx.agent_runtime.providers.claude_code import ClaudeCodeAgent
from evalrx.agent_runtime.providers.codex import CodexAgent
from evalrx.agent_runtime.providers.gemini_cli import GeminiCliAgent
from evalrx.agent_runtime.providers.kimi_cli import KimiCliAgent
from evalrx.agent_runtime.providers.opencode import OpenCodeAgent
from evalrx.agent_runtime.providers.registry import create_cli_agent

__all__ = [
    "AntigravityAgent",
    "CliAgentBase",
    "ClaudeCodeAgent",
    "CodexAgent",
    "GeminiCliAgent",
    "KimiCliAgent",
    "OpenCodeAgent",
    "create_cli_agent",
]
