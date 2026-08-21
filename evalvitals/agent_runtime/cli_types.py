"""Public types and provider metadata for CLI coding agents."""

from __future__ import annotations

from dataclasses import dataclass

VALID_PROVIDERS = frozenset(
    {"llm", "claude_code", "codex", "opencode", "gemini_cli", "kimi_cli", "antigravity"}
)

BINARY_DEFAULTS: dict[str, str] = {
    "claude_code": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "gemini_cli": "gemini",
    "kimi_cli": "kimi",
    "antigravity": "agy",
}


@dataclass(frozen=True)
class CliAgentConfig:
    """Configuration for a CLI-based coding-agent backend.

    Args:
        provider: Which CLI agent to use. ``"llm"`` means no CLI agent.
        binary_path: Explicit path to the CLI binary. Auto-detected when empty.
        model: Model override flag forwarded to the binary.
        max_budget_usd: Spend cap forwarded to providers that support it.
        timeout_sec: Hard wall-clock limit for the agent subprocess.
        extra_args: Additional flags appended verbatim to the CLI command.
        skills: Paths to Agent-Skill directories containing ``SKILL.md``.
        allow_skills: Enable provider-native skill invocation where supported.
    """

    provider: str = "llm"
    binary_path: str = ""
    model: str = ""
    max_budget_usd: float = 5.0
    timeout_sec: int = 600
    extra_args: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    allow_skills: bool = False

    @property
    def skills_enabled(self) -> bool:
        return self.allow_skills or bool(self.skills)


@dataclass
class CliAgentResult:
    """Output of one CLI agent invocation."""

    files: dict[str, str]
    provider_name: str
    elapsed_sec: float
    raw_output: str = ""
    usage: dict | None = None
    error: str | None = None
    audit: dict | None = None
    # Path (inside the invocation workdir) of the UNTRUNCATED raw agent
    # stream — ``raw_output`` above is a capped rendering for UIs; the full
    # stream (every tool call and result) is persisted here for debugging.
    raw_stream_path: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.files)
