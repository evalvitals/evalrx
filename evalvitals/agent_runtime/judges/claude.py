"""Claude Code CLI wrapped as a text-generation judge model."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path


class ClaudeModel:
    """Claude Code CLI wrapped as a judge model."""

    key = "claude"

    def __init__(
        self,
        binary_path: str = "",
        timeout_sec: int = 240,
        model: str = "",
        effort: str = "",
    ) -> None:
        from evalvitals.core.capability import Capability

        binary = binary_path or shutil.which("claude") or ""
        if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                "ClaudeModel: 'claude' binary not found or not executable. "
                "Set CLAUDE_PATH=$(which claude) and re-run, or pass "
                "binary_path= explicitly."
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
        """Run ``claude -p`` with the prompt on STDIN and return the response.

        The prompt must not be an argv entry. Linux caps a SINGLE argument at
        ``MAX_ARG_STRLEN`` = 32 pages = 131,072 bytes — a limit ``ulimit -s`` and
        ``ARG_MAX`` do not describe and no configuration raises — and execve
        fails outright with ``OSError: [Errno 7] Argument list too long``.

        M2's prompt is the analyzers' findings JSON, so its size grows with how
        many analyzers M1 selected. Measured on qwen3.5-2b / bbh_word_sorting:
        9 analyzers produced 127,856 bytes and worked, 12 analyzers produced
        ~137,600 and did not. The size that fails is therefore a property of one
        judge's analyzer selection, which is exactly the thing that varies run to
        run — the same batch, the same code, and one run crosses the cliff.

        Stdin has no equivalent limit; 144,050 bytes round-trips fine.
        """
        img_dir: str | None = None
        try:
            prompt_text = str(inputs)
            if images:
                valid = [p for p in images if isinstance(p, pathlib.Path) and p.exists()]
                if valid:
                    img_dir = tempfile.mkdtemp(prefix="claude_imgs_")
                    for path in valid:
                        shutil.copy2(path, pathlib.Path(img_dir) / path.name)
                    names = ", ".join(path.name for path in valid)
                    prompt_text = f"Images available in workspace: {names}\n\n{prompt_text}"

            # A judge generates text; it does not act. Asking for the blanket
            # permission bypass was more than that needs, and it made the judge
            # unrunnable anywhere nothing wraps it: the CLI refuses
            # `--dangerously-skip-permissions` under root unless IS_SANDBOX=1,
            # which our compose files set (see _common/compose/base.yml) and a
            # bare root box — Colab, a plain container, CI — does not. There
            # every judge call exited 1 before reaching the model.
            #
            # `--tools ""` is the accurate request for a text completion: no
            # tools, therefore nothing to permission. An image-bearing call is
            # the one exception — the figures are staged in a temp dir and the
            # model has to Read them — so that call allows exactly Read, scoped
            # to the directory just added, and nothing else.
            cmd = [
                self._binary,
                "-p",
                "--output-format",
                "text",
            ]
            if img_dir:
                cmd += ["--add-dir", img_dir, "--allowed-tools", "Read"]
            else:
                cmd += ["--tools", ""]
            if self._model:
                cmd += ["--model", self._model]
            if self._effort:
                cmd += ["--effort", self._effort]

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
                    f"ClaudeModel: claude timed out after {self._timeout_sec}s"
                ) from exc

            output = (proc.stdout or "").strip()
            if proc.returncode != 0 and not output:
                reason = (proc.stderr or "").strip()[:240]
                raise RuntimeError(
                    f"ClaudeModel: claude exited {proc.returncode}: {reason}"
                )
            if not output:
                warnings.warn(
                    "ClaudeModel: claude returned an empty response -- likely "
                    "rate-limited or out of quota; the caller will fall back "
                    "to a non-LLM path.",
                    stacklevel=2,
                )
            return output
        finally:
            if img_dir:
                shutil.rmtree(img_dir, ignore_errors=True)

    def __repr__(self) -> str:
        return (
            f"ClaudeModel(binary={self._binary!r}, model={self._model!r}, "
            f"effort={self._effort!r})"
        )
