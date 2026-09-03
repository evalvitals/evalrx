"""Experiment sandbox — safe subprocess execution + ARC-compatible metric parsing.

Mirrors researchclaw/experiment/sandbox.py.

Metric output formats understood by ``parse_metrics``:

    Plain:      ``metric: 0.85``
    Condition:  ``condition=arithmetic metric: 0.42``
    Summary:    ``SUMMARY condition=X metric=Y mean=M std=S``

The diagnostic script printed by ``ExperimentWriter`` uses these same formats,
so ``SandboxResult.metrics`` is immediately ready for the surgery agent to
interpret.

Enhanced over the original with:
- ``SandboxProtocol`` for pluggable backends
- ``run_project()`` for multi-file experiment projects
- Path traversal protection (validate_entry_point / resolved)
- Numbered project dirs (``_project_{counter}``) with thread-safe counter
- Immutable harness injection (``experiment_harness.py``)
- Cleanup policy: delete scripts only on success
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry-point validation (path traversal protection)
# ---------------------------------------------------------------------------


def validate_entry_point(entry_point: str) -> str | None:
    """Validate *entry_point* syntax without filesystem access.

    Returns an error message if invalid, ``None`` if valid.
    Call this **before** copying files to fail fast on obviously bad input.
    """
    if not entry_point or not entry_point.strip():
        return "Entry point is empty"
    ep = Path(entry_point)
    posix_ep = PurePosixPath(entry_point)
    windows_ep = PureWindowsPath(entry_point)
    if ep.is_absolute() or posix_ep.is_absolute() or windows_ep.is_absolute():
        return f"Entry point must be a relative path, got: {entry_point}"
    if ".." in ep.parts:
        return f"Entry point must not contain '..': {entry_point}"
    return None


def validate_entry_point_resolved(staging: Path, entry_point: str) -> str | None:
    """Validate that *entry_point* resolves inside *staging*.

    Returns an error message if invalid, ``None`` if valid.
    Call this **after** copying files so that symlinks are resolved against
    the real staging contents.
    """
    resolved = (staging / entry_point).resolve()
    staging_resolved = staging.resolve()
    if not resolved.is_relative_to(staging_resolved):
        return f"Entry point escapes staging directory: {entry_point}"
    return None


# ---------------------------------------------------------------------------
# Metric parsing  (ARC-compatible formats)
# ---------------------------------------------------------------------------

_FLOAT_RE = r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?"

# Plain:      "metric: 0.85"
_PLAIN_RE = re.compile(rf"^(\w[\w./]*)\s*:\s*({_FLOAT_RE})\s*$")

# Condition:  "condition=X metric: 0.42"
_CONDITION_RE = re.compile(rf"^condition=(\S+)\s+(\w[\w./]*)\s*:\s*({_FLOAT_RE})\s*$")

# Summary:    "SUMMARY condition=X metric=Y mean=M std=S"
_SUMMARY_RE = re.compile(
    rf"^SUMMARY\s+condition=(\S+)\s+metric=(\S+)\s+mean=({_FLOAT_RE})\s+std=({_FLOAT_RE})"
)


def parse_metrics(stdout: str) -> dict[str, float]:
    """Extract ``name: float`` pairs from *stdout*.

    Matching priority: SUMMARY lines first, then condition-prefixed, then plain.
    Non-finite values are silently dropped.
    """
    metrics: dict[str, float] = {}

    for line in stdout.splitlines():
        stripped = line.strip()

        # SUMMARY condition=X metric=Y mean=M std=S
        m = _SUMMARY_RE.match(stripped)
        if m:
            cond, metric, mean_s, std_s = m.groups()
            try:
                mean_v, std_v = float(mean_s), float(std_s)
            except ValueError:
                continue
            if math.isfinite(mean_v):
                metrics[f"{cond}/{metric}"]      = mean_v
                metrics[f"{cond}/{metric}_mean"] = mean_v
                metrics[f"{cond}/{metric}_std"]  = std_v
                metrics[metric]                  = mean_v
            continue

        # condition=X metric: value
        m = _CONDITION_RE.match(stripped)
        if m:
            cond, name, val_s = m.groups()
            try:
                val = float(val_s)
            except ValueError:
                continue
            if math.isfinite(val):
                metrics[f"{cond}/{name}"] = val
                metrics[name]             = val
            continue

        # metric: value
        m = _PLAIN_RE.match(stripped)
        if m:
            name, val_s = m.groups()
            try:
                val = float(val_s)
            except ValueError:
                continue
            if math.isfinite(val):
                metrics[name] = val

    return metrics


# ---------------------------------------------------------------------------
# Result + Protocol
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Output of one :class:`ExperimentSandbox` run.

    Attributes:
        returncode:  Process exit code (0 = success, −1 = internal/timeout).
        stdout:      Captured standard output.
        stderr:      Captured standard error.
        elapsed_sec: Wall-clock seconds.
        metrics:     Parsed ``{name: float}`` from *stdout*.
        timed_out:   ``True`` when the process was killed for exceeding the budget.
    """

    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    metrics: dict[str, float] = field(default_factory=dict)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class SandboxProtocol(Protocol):
    """Structural type for sandbox backends (subprocess, Docker, SSH, …)."""

    def run(self, code: str, *, timeout_sec: int = 60) -> SandboxResult: ...

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 60,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult: ...


# ---------------------------------------------------------------------------
# ExperimentSandbox
# ---------------------------------------------------------------------------


def _make_resource_preexec(
    max_memory_bytes: int | None,
    max_cpu_seconds: int | None,
) -> "Callable[[], None] | None":
    """Return a ``preexec_fn`` that sets resource limits on Linux, or ``None``.

    Uses ``resource.setrlimit`` which is only available on Unix.  The function
    is constructed at call time and captures the limit values in its closure so
    each subprocess gets its own independent limits.

    Memory is capped via ``RLIMIT_AS`` (virtual address space).  CPU time is
    capped via ``RLIMIT_CPU`` (user+system seconds).  Hitting the CPU limit
    sends SIGXCPU to the process; the subprocess.run *timeout* handles wall-clock
    kills independently, so both limits are in effect simultaneously.
    """
    import sys
    if sys.platform == "win32":
        return None
    try:
        import resource as _resource
    except ImportError:
        return None

    def _preexec() -> None:
        if max_memory_bytes is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_AS,
                    (max_memory_bytes, max_memory_bytes),
                )
            except (ValueError, _resource.error):
                pass
        if max_cpu_seconds is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_CPU,
                    (max_cpu_seconds, max_cpu_seconds),
                )
            except (ValueError, _resource.error):
                pass

    return _preexec


class ExperimentSandbox:
    """Run Python code in a subprocess and parse metrics from stdout.

    Mirrors ``researchclaw.experiment.sandbox.ExperimentSandbox``.

    Args:
        workdir:          Directory where scripts are written.  A temporary
                          directory is created when ``None``.
        python_path:      Python executable to use.  Defaults to ``sys.executable``.
        max_memory_bytes: Virtual-address-space cap applied via ``RLIMIT_AS``
                          before the child process starts (Linux/macOS only).
                          ``None`` means no memory limit.
        max_cpu_seconds:  CPU-time cap applied via ``RLIMIT_CPU`` (Linux/macOS).
                          ``None`` means no CPU limit.  This complements the
                          wall-clock *timeout_sec* argument to ``run()``.
        cleanup:          When ``True`` (default), a successful run's script/
                          project is deleted (see ``_should_cleanup``) — the
                          original behaviour for an ephemeral *workdir*. Pass
                          ``False`` when *workdir* is a durable, self-contained
                          location (e.g. one fix/M4 trial's ``workspace/``)
                          whose code should stay on disk even on success.
    """

    def __init__(
        self,
        workdir: Path | str | None = None,
        *,
        python_path: str | None = None,
        max_memory_bytes: int | None = None,
        max_cpu_seconds: int | None = None,
        cleanup: bool = True,
    ) -> None:
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="evalrx_sandbox_"))
        # Resolved to absolute: _run_script passes this same path as both the
        # subprocess cwd and (via script_path) part of the command argv — a
        # relative workdir makes the child resolve the script path a second
        # time relative to its new cwd, doubling it and raising
        # FileNotFoundError instead of running the script.
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._python_path: str = python_path or sys.executable
        self._run_counter: int = 0
        self._counter_lock = threading.Lock()
        self._preexec_fn = _make_resource_preexec(max_memory_bytes, max_cpu_seconds)
        self._cleanup = cleanup

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 60) -> SandboxResult:
        """Write *code* to a numbered script file and execute it.

        Args:
            code:        Python source to execute.
            timeout_sec: Hard wall-clock limit; the process is killed on breach.

        Returns:
            :class:`SandboxResult` with parsed metrics.
        """
        with self._counter_lock:
            self._run_counter += 1
            counter = self._run_counter
        script_path = self.workdir / f"exp_{counter:04d}.py"
        script_path.write_text(code, encoding="utf-8")

        result = self._run_script(script_path, timeout_sec=timeout_sec)

        if self._should_cleanup(result):
            self._cleanup_path(script_path)

        return result

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 60,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Copy a multi-file project into the sandbox and execute *entry_point*.

        Security:
            - *entry_point* is validated for path traversal before and after copy.
            - The project cannot overwrite the injected ``experiment_harness.py``.

        Args:
            project_dir:   Source directory containing project files.
            entry_point:   Relative path to the script to execute (default ``main.py``).
            timeout_sec:   Hard wall-clock limit.
            args:          Extra command-line arguments passed to the script.
            env_overrides: Additional environment variables for the subprocess.

        Returns:
            :class:`SandboxResult` with parsed metrics.
        """
        # Pre-copy syntax validation — fail fast on obviously bad input
        err = validate_entry_point(entry_point)
        if err:
            return SandboxResult(returncode=-1, stdout="", stderr=err,
                                 elapsed_sec=0.0, metrics={})

        with self._counter_lock:
            self._run_counter += 1
            counter = self._run_counter
        sandbox_project = self.workdir / f"_project_{counter}"
        if sandbox_project.exists():
            shutil.rmtree(sandbox_project)
        sandbox_project.mkdir(parents=True, exist_ok=True)

        # Inject immutable experiment harness first
        self._inject_harness(sandbox_project)

        # Copy project files (harness cannot be overwritten)
        for src in Path(project_dir).iterdir():
            if src.is_file():
                dest = sandbox_project / src.name
                if dest.name == "experiment_harness.py":
                    logger.warning(
                        "Project contains experiment_harness.py — skipping (immutable)"
                    )
                    continue
                dest.write_bytes(src.read_bytes())
            elif src.is_dir() and not src.name.startswith("."):
                shutil.copytree(src, sandbox_project / src.name, dirs_exist_ok=True)

        # Post-copy resolve check — catches symlink-based escapes
        err = validate_entry_point_resolved(sandbox_project, entry_point)
        if err:
            return SandboxResult(returncode=-1, stdout="", stderr=err,
                                 elapsed_sec=0.0, metrics={})

        entry = sandbox_project / entry_point
        if not entry.exists():
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Entry point '{entry_point}' not found in project",
                elapsed_sec=0.0, metrics={},
            )

        result = self._run_script(
            entry,
            timeout_sec=timeout_sec,
            cwd=sandbox_project,
            args=args,
            env_overrides=env_overrides,
        )

        if self._should_cleanup(result):
            self._cleanup_path(sandbox_project)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_script(
        self,
        script_path: Path,
        *,
        timeout_sec: int,
        cwd: Path | None = None,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if env_overrides:
            env.update(env_overrides)
        command = self._build_command(script_path, args=args)
        run_cwd = str(cwd or self.workdir)
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                cwd=run_cwd,
                env=env,
                check=False,
                preexec_fn=self._preexec_fn,
            )
            return self._result_from_completed(completed, elapsed_sec=time.monotonic() - start)
        except subprocess.TimeoutExpired as exc:
            return self._result_from_timeout(
                exc, timeout_sec=timeout_sec, elapsed_sec=time.monotonic() - start
            )
        except Exception as exc:  # noqa: BLE001
            return self._result_from_exception(exc, elapsed_sec=time.monotonic() - start)

    def _build_command(
        self, script_path: Path, *, args: list[str] | None = None
    ) -> list[str]:
        python = self._python_path
        python_path = Path(python)
        if not python_path.is_absolute() and python != "python":
            python_path = Path.cwd() / python_path
        command = [str(python_path), "-u", str(script_path)]
        if args:
            command.extend(args)
        return command

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        """Copy the immutable experiment harness into *target_dir*."""
        harness_src = Path(__file__).parent / "experiment_harness.py"
        if harness_src.exists():
            dest = target_dir / "experiment_harness.py"
            dest.write_text(harness_src.read_text(encoding="utf-8"), encoding="utf-8")
            logger.debug("Injected experiment harness into %s", target_dir)
        else:
            logger.warning("Harness template not found at %s", harness_src)

    @staticmethod
    def _result_from_completed(
        completed: subprocess.CompletedProcess[str], *, elapsed_sec: float
    ) -> SandboxResult:
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        return SandboxResult(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed_sec,
            metrics=parse_metrics(stdout),
        )

    @staticmethod
    def _result_from_timeout(
        exc: subprocess.TimeoutExpired,
        *,
        timeout_sec: int,
        elapsed_sec: float,
    ) -> SandboxResult:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return SandboxResult(
            returncode=-1,
            stdout=stdout,
            stderr=f"[TIMEOUT after {timeout_sec}s]\n{stderr}",
            elapsed_sec=elapsed_sec,
            metrics=parse_metrics(stdout),
            timed_out=True,
        )

    @staticmethod
    def _result_from_exception(exc: Exception, *, elapsed_sec: float) -> SandboxResult:
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=f"[SANDBOX ERROR] {exc}",
            elapsed_sec=elapsed_sec,
            metrics={},
        )

    def _should_cleanup(self, result: SandboxResult) -> bool:
        return self._cleanup and result.returncode == 0 and not result.timed_out

    @staticmethod
    def _cleanup_path(path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to clean up sandbox path: %s", path)
