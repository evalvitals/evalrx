"""Base class for CLI coding-provider adapters."""

from __future__ import annotations

import logging
from pathlib import Path

from evalvitals.agent_runtime.cli_runtime import ProcessRun, SubprocessRunner, collect_py_files
from evalvitals.agent_runtime.cli_transcript import RAW_OUTPUT_CAP
from evalvitals.agent_runtime.cli_types import CliAgentResult
from evalvitals.agent_runtime.skill_audit import build_agent_audit
from evalvitals.agent_runtime.skills.installer import SkillInstaller

logger = logging.getLogger(__name__)


class CliAgentBase:
    """Subprocess runner shared by all CLI coding providers."""

    _provider_name: str = "unknown"

    # Every provider's ``_build_cmd`` places *prompt* directly in argv (``-p
    # <prompt>`` / ``exec <prompt>`` / ``--message <prompt>``). This
    # framework's prompts (a full M1 evidence dump, an L1/L2 fix-candidate
    # brief with per-case tables, ...) can exceed the kernel's argv+envp size
    # limit -- observed in practice as ``OSError: [Errno 7] Argument list
    # too long`` on exec. Above this conservative threshold, ``run()`` spills
    # the prompt to a file in *workdir* instead of passing it inline.
    _LARGE_PROMPT_BYTES = 60_000

    def __init__(
        self,
        binary_path: str,
        model: str = "",
        max_budget_usd: float = 5.0,
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
        skills: list[str] | None = None,
        allow_skills: bool = False,
    ) -> None:
        self._binary = binary_path
        self._model = model
        self._max_budget_usd = max_budget_usd
        self._timeout_sec = timeout_sec
        self._extra_args: list[str] = extra_args or []
        self._skills: list[str] = list(skills or [])
        self._allow_skills: bool = bool(allow_skills or self._skills)
        self._runner = SubprocessRunner()
        self._skill_installer = self._make_skill_installer(self._skills)

    def _make_skill_installer(self, skills: list[str]) -> SkillInstaller:
        return SkillInstaller(skills)

    def _install_skills(self, workdir: Path) -> None:
        """Install configured skills into the provider-visible workspace."""
        self._skill_installer.install(workdir)

    def run(
        self,
        prompt: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CliAgentResult:
        """Invoke the CLI provider with *prompt* and return collected files."""
        timeout = timeout_sec if timeout_sec is not None else self._timeout_sec
        workdir.mkdir(parents=True, exist_ok=True)
        self._install_skills(workdir)
        cli_prompt = prompt
        if len(prompt.encode("utf-8", errors="replace")) > self._LARGE_PROMPT_BYTES:
            (workdir / "prompt.txt").write_text(prompt, encoding="utf-8")
            cli_prompt = (
                "Your full task instructions are in the file `prompt.txt` in "
                "this workspace (too large to pass inline). Read it "
                "completely and follow it exactly -- do not summarize, "
                "truncate, or skip any part of it."
            )
        cmd = self._build_cmd(cli_prompt, workdir)
        logger.debug("%s: running %s", self._provider_name, cmd[0])

        run = self._run_subprocess(cmd, workdir, timeout)
        return self._build_result(workdir, run)

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _postprocess_output(self, stdout: str) -> tuple[str, dict | None]:
        return stdout[:RAW_OUTPUT_CAP], None

    def _run_subprocess(self, cmd: list[str], workdir: Path, timeout_sec: int) -> ProcessRun:
        return self._runner.run(cmd, workdir, timeout_sec)

    def _build_result(self, workdir: Path, run: ProcessRun) -> CliAgentResult:
        files = collect_py_files(workdir)
        error: str | None = None
        if run.timed_out:
            error = f"[TIMEOUT] agent killed after {run.elapsed_sec:.0f}s"
        elif run.returncode != 0 and not files:
            error = f"Exited {run.returncode}: {run.stderr[:500]}"

        raw_output, usage = self._postprocess_output(run.stdout)
        audit = build_agent_audit(
            provider=self._provider_name,
            skills=self._skills,
            workdir=workdir,
            stdout=run.stdout,
            stderr=run.stderr,
            returncode=run.returncode,
            timed_out=run.timed_out,
            elapsed_sec=run.elapsed_sec,
            files=files,
            error=error,
        )
        logger.debug(
            "%s: rc=%d files=%s elapsed=%.1fs timed_out=%s",
            self._provider_name,
            run.returncode,
            list(files),
            run.elapsed_sec,
            run.timed_out,
        )
        return CliAgentResult(
            files=files,
            provider_name=self._provider_name,
            elapsed_sec=run.elapsed_sec,
            raw_output=raw_output,
            usage=usage,
            error=error,
            audit=audit,
        )
