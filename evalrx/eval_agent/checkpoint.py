"""Checkpoint / heartbeat file I/O for resumable diagnosis runs.

Free functions extracted from ``AutoDiagnoseLoop`` (the only current caller,
in ``legacy.py``) so the atomic-write/read logic is reusable by any future
loop that wants run_dir-based resume, without depending on the loop class
itself.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_checkpoint(
    checkpoint_path: Path, *, cycle: int, run_id: str, hypothesis_statuses: list[str],
) -> None:
    """Atomic checkpoint write (temp-file + rename)."""
    data = {
        "last_completed_cycle": cycle,
        "run_id": run_id,
        "hypothesis_statuses": hypothesis_statuses,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    fd, tmp_path = tempfile.mkstemp(
        dir=checkpoint_path.parent, suffix=".tmp", prefix="checkpoint_"
    )
    os.close(fd)
    try:
        Path(tmp_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        Path(tmp_path).replace(checkpoint_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def write_heartbeat(heartbeat_path: Path, *, cycle: int, run_id: str) -> None:
    data = {
        "pid": os.getpid(),
        "last_cycle": cycle,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    heartbeat_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_checkpoint(checkpoint_path: "Path | None") -> "dict[str, Any] | None":
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
