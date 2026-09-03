"""Provider registry and factory for CLI coding agents."""

from __future__ import annotations

import shutil

from evalrx.agent_runtime.cli_types import BINARY_DEFAULTS, CliAgentConfig
from evalrx.agent_runtime.providers.antigravity import AntigravityAgent
from evalrx.agent_runtime.providers.base import CliAgentBase
from evalrx.agent_runtime.providers.claude_code import ClaudeCodeAgent
from evalrx.agent_runtime.providers.codex import CodexAgent
from evalrx.agent_runtime.providers.gemini_cli import GeminiCliAgent
from evalrx.agent_runtime.providers.kimi_cli import KimiCliAgent
from evalrx.agent_runtime.providers.opencode import OpenCodeAgent

PROVIDER_CLASSES: dict[str, type[CliAgentBase]] = {
    "claude_code": ClaudeCodeAgent,
    "codex": CodexAgent,
    "opencode": OpenCodeAgent,
    "gemini_cli": GeminiCliAgent,
    "kimi_cli": KimiCliAgent,
    "antigravity": AntigravityAgent,
}


def create_cli_agent(config: CliAgentConfig) -> CliAgentBase:
    """Instantiate the appropriate CLI coding provider for *config*."""
    provider = config.provider

    if provider == "llm":
        raise ValueError(
            "'llm' is not a CLI provider. Use ExperimentWriter directly "
            "(leave cli_agent=None or CliAgentConfig(provider='llm'))."
        )

    cls = PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown CLI provider: {provider!r}. Valid: {sorted(PROVIDER_CLASSES)}"
        )

    binary = config.binary_path or shutil.which(BINARY_DEFAULTS[provider]) or ""
    if not binary:
        raise RuntimeError(
            f"CLI agent binary for {provider!r} not found in PATH. "
            f"Install '{BINARY_DEFAULTS[provider]}' or pass "
            f"CliAgentConfig(binary_path='/path/to/{BINARY_DEFAULTS[provider]}')."
        )

    return cls(
        binary_path=binary,
        model=config.model,
        max_budget_usd=config.max_budget_usd,
        timeout_sec=config.timeout_sec,
        extra_args=list(config.extra_args),
        skills=list(config.skills),
        allow_skills=config.allow_skills,
    )
