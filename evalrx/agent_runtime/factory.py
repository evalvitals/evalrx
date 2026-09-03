"""Sandbox factory — create the appropriate sandbox backend from config.

Mirrors ``researchclaw/experiment/factory.py``.

Supported modes:
    ``"subprocess"`` (default) — :class:`ExperimentSandbox` running in a local
        subprocess.  No extra dependencies required.
    ``"docker"`` — Docker-isolated sandbox.  Falls back to the subprocess
        backend when Docker is not available or the DockerSandbox import fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from evalrx.agent_runtime.sandbox import ExperimentSandbox, SandboxProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for the subprocess sandbox backend.

    Attributes:
        python_path: Python executable to use.  Empty string → ``sys.executable``.
        timeout_sec: Default timeout for sandbox runs (seconds).
    """

    python_path: str = ""
    timeout_sec: int = 60


@dataclass(frozen=True)
class SandboxFactoryConfig:
    """Top-level factory configuration.

    Attributes:
        mode:    Backend to use: ``"subprocess"`` (default) or ``"docker"``.
        sandbox: Settings for the subprocess backend.
    """

    mode: str = "subprocess"
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_sandbox(config: SandboxFactoryConfig, workdir: Path) -> SandboxProtocol:
    """Instantiate the sandbox backend specified by *config.mode*.

    Args:
        config:  Factory configuration.
        workdir: Directory where the sandbox writes scripts and project copies.

    Returns:
        A :class:`~evalrx.agent_runtime.sandbox.SandboxProtocol` implementation.
        Falls back to :class:`ExperimentSandbox` if the requested backend is
        unavailable.
    """
    if config.mode == "docker":
        sandbox = _try_create_docker_sandbox(workdir)
        if sandbox is not None:
            return sandbox
        logger.warning(
            "Docker sandbox unavailable — falling back to subprocess sandbox"
        )

    python_path = config.sandbox.python_path or None
    return ExperimentSandbox(workdir=workdir, python_path=python_path)


def _try_create_docker_sandbox(workdir: Path) -> SandboxProtocol | None:
    """Attempt to create a DockerSandbox; return None if not available."""
    try:
        import subprocess as _sp
        result = _sp.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("Docker not running (docker info returned %s)", result.returncode)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Docker check failed: %s", exc)
        return None

    # DockerSandbox is not yet implemented in evalrx — stub falls through.
    logger.debug("Docker is available but DockerSandbox is not yet implemented")
    return None
