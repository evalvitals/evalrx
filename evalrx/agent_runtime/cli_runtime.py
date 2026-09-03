"""Runtime primitives for CLI coding-agent adapters."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessRun:
    """Completed subprocess execution."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool = False


def to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def collect_py_files(workdir: Path) -> dict[str, str]:
    """Read top-level Python files produced by a coding agent."""
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith(("_codex_", "_agent_")):
            continue
        try:
            files[pyfile.name] = pyfile.read_text(encoding="utf-8")
        except OSError:
            pass
    return files


class SubprocessRunner:
    """Run CLI agents in a workdir with timeout and environment normalization."""

    def run(self, cmd: list[str], workdir: Path, timeout_sec: int) -> ProcessRun:
        workdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        timed_out = False

        env = {**os.environ}
        bindir = os.path.dirname(sys.executable)
        if bindir:
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=env,
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                stdout_bytes, stderr_bytes = b"", b""
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001
                    pass

        elapsed = time.monotonic() - start
        rc = proc.returncode if proc.returncode is not None else -1
        return ProcessRun(
            returncode=rc,
            stdout=to_text(stdout_bytes),
            stderr=to_text(stderr_bytes),
            elapsed_sec=elapsed,
            timed_out=timed_out,
        )
