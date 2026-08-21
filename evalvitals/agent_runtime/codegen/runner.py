"""Shared runner for CLI coding-provider invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from evalvitals.agent_runtime.cli_types import CliAgentConfig, CliAgentResult


@dataclass
class CodegenCodeResult:
    """Selected source file plus the underlying CLI run metadata."""

    code: str
    raw_output: str
    usage: dict | None = None
    files: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    elapsed_sec: float = 0.0
    audit: dict | None = None
    # Path (relative to the invocation workdir) of the UNTRUNCATED raw agent
    # stream persisted by the provider base class.
    raw_stream_path: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.code) and self.error is None


def _select_py_file(
    files: dict[str, str],
    *,
    preferred: tuple[str, ...] = (),
) -> str:
    py_files = {name: body for name, body in files.items() if name.endswith(".py")}
    for name in preferred:
        if name in py_files:
            return py_files[name]
    if not py_files:
        return ""
    return max(py_files.values(), key=len)


class CodegenRunner:
    """Run a configured CLI coding provider and select generated code files."""

    def __init__(self, cli_config: CliAgentConfig) -> None:
        self._cli_config = cli_config

    @property
    def provider(self) -> str:
        return self._cli_config.provider

    def run(
        self,
        prompt: str,
        *,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CliAgentResult:
        # Lazy import: this is the sole call site, so importing here avoids
        # pulling every provider module in at CodegenRunner import time.
        from evalvitals.agent_runtime.providers.registry import create_cli_agent

        agent = create_cli_agent(self._cli_config)
        return agent.run(prompt, workdir=workdir, timeout_sec=timeout_sec)

    def write_code(
        self,
        prompt: str,
        *,
        workdir: Path,
        timeout_sec: int | None = None,
        preferred_filenames: tuple[str, ...] = (),
        include_error_in_raw: bool = False,
    ) -> CodegenCodeResult:
        result = self.run(prompt, workdir=workdir, timeout_sec=timeout_sec)
        raw_output = result.raw_output or (
            (result.error or "") if include_error_in_raw else ""
        )
        code = _select_py_file(result.files, preferred=preferred_filenames)
        if not result.ok:
            code = ""
        return CodegenCodeResult(
            code=code,
            raw_output=raw_output,
            usage=result.usage,
            files=result.files,
            error=result.error,
            elapsed_sec=result.elapsed_sec,
            audit=result.audit,
            raw_stream_path=result.raw_stream_path,
        )
