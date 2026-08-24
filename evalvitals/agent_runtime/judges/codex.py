"""OpenAI Codex CLI wrapped as a text-generation judge model.

The judge receives only the prompt placed in an isolated temporary directory.
It uses ``codex exec`` in read-only mode; coding-agent uses elsewhere in the
pipeline are configured independently through :class:`CliAgentConfig`.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path


class CodexModel:
    """Codex CLI wrapped behind EvalVitals' ``generate`` model protocol."""

    key = "codex"

    def __init__(
        self,
        binary_path: str = "",
        timeout_sec: int = 300,
        model: str = "gpt-5.6-terra",
        effort: str = "",
    ) -> None:
        from evalvitals.core.capability import Capability

        binary = binary_path or shutil.which("codex") or ""
        if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                "CodexModel: 'codex' binary not found or not executable. "
                "Set CODEX_PATH=$(which codex) and re-run, or pass binary_path= explicitly."
            )
        self._binary = binary
        self._timeout_sec = timeout_sec
        self._model = model
        self._effort = effort
        self.capabilities = frozenset({Capability.GENERATE})
        self.modalities = frozenset({"text"})

    def generate(
        self,
        inputs: object,
        *,
        images: "list[Path] | None" = None,
        **kwargs: object,
    ) -> str:
        """Run an isolated, non-interactive Codex turn and return its answer.

        The prompt is passed over stdin, never as an argv entry, so an M2
        evidence table cannot hit Linux's per-argument size limit.  Images are
        copied into the isolated workdir and named in the prompt; this mirrors
        the existing CLI judge contract without exposing the repository.
        """
        workspace = tempfile.mkdtemp(prefix="codex_judge_")
        output_path = Path(workspace) / "answer.txt"
        try:
            prompt_text = str(inputs)
            valid = [path for path in images or [] if isinstance(path, pathlib.Path) and path.exists()]
            if valid:
                names: list[str] = []
                for path in valid:
                    destination = Path(workspace) / path.name
                    shutil.copy2(path, destination)
                    names.append(destination.name)
                prompt_text = f"Images available in workspace: {', '.join(names)}\n\n{prompt_text}"

            cmd = [
                self._binary,
                "exec",
                "-",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--output-last-message", str(output_path),
                "-C", workspace,
            ]
            if self._model:
                cmd += ["--model", self._model]
            if self._effort:
                cmd += ["-c", f'model_reasoning_effort="{self._effort}"']
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt_text,
                    capture_output=True,
                    timeout=self._timeout_sec,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ},
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"CodexModel: codex timed out after {self._timeout_sec}s"
                ) from exc

            output = output_path.read_text(encoding="utf-8", errors="replace").strip() if output_path.is_file() else ""
            if proc.returncode != 0 and not output:
                reason = (proc.stderr or proc.stdout or "").strip()[-500:]
                raise RuntimeError(f"CodexModel: codex exited {proc.returncode}: {reason}")
            if not output:
                warnings.warn(
                    "CodexModel: codex returned an empty response; the caller will fall back to a non-LLM path.",
                    stacklevel=2,
                )
            return output
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def __repr__(self) -> str:
        return (
            f"CodexModel(binary={self._binary!r}, model={self._model!r}, "
            f"effort={self._effort!r})"
        )
