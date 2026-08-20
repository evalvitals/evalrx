"""Antigravity CLI wrapped as a text-generation judge model."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

AGY_ERROR_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "code 429",
    "quota",
    "ineligible",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "exhausted",
)


def safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def scan_agy_log(path: str) -> str:
    """Return the last agy-log error line, or ``""`` if none."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-300:]
    except OSError:
        return ""
    for line in reversed(lines):
        if any(marker.lower() in line.lower() for marker in AGY_ERROR_MARKERS):
            msg = line.strip()
            idx = msg.rfind("] ")
            return (msg[idx + 2:] if idx != -1 else msg)[:240]
    return ""


class AgyModel:
    """agy CLI wrapped as a judge model."""

    key = "agy"

    def __init__(
        self,
        binary_path: str = "",
        timeout_sec: int = 120,
        model: str = "",
    ) -> None:
        from evalvitals.core.capability import Capability

        binary = binary_path or shutil.which("agy") or ""
        if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                "AgyModel: 'agy' binary not found or not executable. "
                "Set AGY_PATH=$(which agy) and re-run, or pass binary_path= explicitly."
            )
        self._binary = binary
        self._timeout_sec = timeout_sec
        self._model = model
        self.capabilities = frozenset({Capability.GENERATE})
        self.modalities = frozenset({"text"})

    def generate(
        self,
        inputs: object,
        *,
        images: "list[Path] | None" = None,
        **kwargs: object,
    ) -> str:
        """Run ``agy -p <inputs>`` and return the text response."""
        fd, log_path = tempfile.mkstemp(prefix="agy_", suffix=".log")
        os.close(fd)
        img_dir: str | None = None
        try:
            prompt_text = str(inputs)
            if images:
                valid = [p for p in images if isinstance(p, pathlib.Path) and p.exists()]
                if valid:
                    img_dir = tempfile.mkdtemp(prefix="agy_imgs_")
                    for path in valid:
                        shutil.copy2(path, pathlib.Path(img_dir) / path.name)
                    names = ", ".join(path.name for path in valid)
                    prompt_text = f"Images available in workspace: {names}\n\n{prompt_text}"

            cmd = [
                self._binary,
                "-p",
                prompt_text,
                # A judge receives the evidence explicitly in its prompt. It
                # must not inspect repository files, benchmark manifests, or
                # held-out labels through terminal tools.
                "--sandbox",
                "--dangerously-skip-permissions",
                "--log-file",
                log_path,
            ]
            if img_dir:
                cmd += ["--add-dir", img_dir]
            if self._model:
                cmd += ["--model", self._model]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self._timeout_sec,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ},
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"AgyModel: agy timed out after {self._timeout_sec}s"
                ) from exc

            output = proc.stdout.strip()
            if proc.returncode != 0 and not output:
                reason = scan_agy_log(log_path) or (proc.stderr or "").strip()[:240]
                raise RuntimeError(f"AgyModel: agy exited {proc.returncode}: {reason}")

            output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
            if not output:
                reason = scan_agy_log(log_path)
                warnings.warn(
                    "AgyModel: agy returned an empty response"
                    + (f" -- {reason}" if reason else "")
                    + ". agy is likely rate-limited or quota-exhausted; the caller "
                    "will fall back to a non-LLM path.",
                    stacklevel=2,
                )
            return output
        finally:
            safe_unlink(log_path)
            if img_dir:
                shutil.rmtree(img_dir, ignore_errors=True)

    def __repr__(self) -> str:
        return f"AgyModel(binary={self._binary!r})"
