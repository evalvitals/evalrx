"""Hold a run directory's label-bearing artifacts in memory while the fix
stage runs, so nothing the repair coder or its sandbox can read from disk
reveals gold answers or per-case pass/fail.

Why this exists
---------------
The fix stage already withholds labels at the *prompt* level: the proposer sees
EXPLORE cases only, the coded-pipeline payload carries ``id`` / ``prompt`` /
``baseline_output`` and nothing else, and every FixAgent prompt says so. But the
coder CLI runs with ``Bash Edit Write Read`` and the pipeline sandbox is a plain
subprocess, both rooted *inside* the run directory — where, by the time the fix
stage starts, M1..M5 have written the same labels many times over:

* ``baseline.json`` and ``logs/report/discovery_cases.json`` — every case with
  ``expected`` and ``label``, CONFIRM cases included;
* ``logs/run_log.jsonl`` — ``case_record`` events with ``expected``, ``label``
  and ``metadata.gold``;
* ``logs/artifacts/*.result.json``, ``logs/contract/c*.m1.json``,
  ``explore/records.json`` — per-case analyzer signals, among them
  ``gold_yes``, which *is* the gold answer on a yes/no task;
* ``logs/workspace/post_m4/cases.json`` — the M4 experiment's cases with gold.

``cat ../../../baseline.json`` from a fix workspace is two directory levels
away. A candidate that joins its cases to any of these by id can hard-code the
held-out answers; the frozen-model control catches the crude lookup-table form
but not one that also manufactures consensus support through the bridge.

What it does
------------
:func:`quarantine_run_dir` is a context manager. On entry it reads every
regular file under the run directory into memory and unlinks it (directory
names stay; they carry nothing). Files handed in as ``append_logs`` — the
structured event log, which a :class:`logging.FileHandler` holds open in
append mode — are truncated in place instead, so events the fix stage writes
still land in the same file. On exit (also on an exception) everything is put
back byte-for-byte; an append log becomes ``old + what the fix stage wrote``;
a path the fix stage re-created with new content keeps the new content and
the pre-fix bytes are saved beside it as ``<name>.pre_fix`` — nothing
is ever lost. A ``fix_quarantine.json`` manifest is written on exit so the run
records exactly which files were hidden.

What it cannot do
-----------------
It only governs what *this run wrote*. A dataset manifest on a bind mount
(``/app/work/data/<dataset>/manifest.json`` in the benchmark containers)
carries ``gold`` too and cannot be hidden from a root process in the same
container — that needs uid separation or host-side label delivery, which is
out of scope here. The gap is deliberate and documented, not forgotten.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "fix_quarantine.json"


@dataclass
class QuarantineReport:
    """What was hidden and how it was restored (also written as JSON)."""

    root: Path
    hidden: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    n_bytes: int = 0

    def to_json(self) -> dict:
        return {
            "root": str(self.root),
            "n_hidden": len(self.hidden),
            "n_bytes": self.n_bytes,
            "hidden": sorted(self.hidden),
            "truncated_in_place": sorted(self.truncated),
            "kept_visible": sorted(self.kept),
            "merged_append_logs": sorted(self.merged),
            "conflicts_saved_as_pre_fix": sorted(self.conflicts),
        }


def _under(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in prefixes)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


@contextmanager
def quarantine_run_dir(
    run_dir: "str | os.PathLike[str]",
    *,
    append_logs: Iterable["str | os.PathLike[str]"] = (),
    keep: Iterable[str] = (),
    write_manifest: bool = True,
) -> Iterator[QuarantineReport]:
    """Hide every file under *run_dir* for the duration of the block.

    Args:
        run_dir:      The run directory (``baseline.json``, ``logs/``, ...).
        append_logs:  Files that a live handler holds open in append mode
                      (``RunContext.log_path``). They are truncated to zero
                      length instead of unlinked and restored as
                      ``old + new``.
        keep:         Relative path prefixes (POSIX style) to leave visible,
                      e.g. ``("logs/fixes",)`` when resuming a fix search.
        write_manifest: Write ``fix_quarantine.json`` into *run_dir* on exit.

    Yields the :class:`QuarantineReport`, filled in as the block runs.
    """
    root = Path(run_dir).resolve()
    keep_prefixes = tuple(k.strip("/") for k in keep if k.strip("/"))
    append_rel = set()
    for p in append_logs:
        p = Path(p).resolve()
        try:
            append_rel.add(_rel(p, root))
        except ValueError:
            logger.warning("label quarantine: append log %s is outside %s; ignored", p, root)

    report = QuarantineReport(root=root)
    stash: dict[str, bytes] = {}

    if root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink():
                    continue
                rel = _rel(path, root)
                if rel == MANIFEST_NAME or _under(rel, keep_prefixes):
                    report.kept.append(rel)
                    continue
                data = path.read_bytes()
                stash[rel] = data
                report.n_bytes += len(data)
                if rel in append_rel:
                    # A logging.FileHandler appends through an O_APPEND fd:
                    # after truncation its next write lands at offset 0.
                    with path.open("r+b") as handle:
                        handle.truncate(0)
                    report.truncated.append(rel)
                else:
                    path.unlink()
                report.hidden.append(rel)
    logger.info(
        "label quarantine: hid %d file(s), %d bytes, under %s for the fix stage",
        len(report.hidden), report.n_bytes, root,
    )

    try:
        yield report
    finally:
        for rel, old in stash.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if rel in append_rel:
                new = path.read_bytes() if path.exists() else b""
                path.write_bytes(old + new)
                report.merged.append(rel)
            elif path.exists() and path.stat().st_size > 0:
                # Appended AFTER the extension so a `*.json` scan (the contract
                # index) never mistakes the copy for a payload.
                side = path.with_name(path.name + ".pre_fix")
                side.write_bytes(old)
                report.conflicts.append(rel)
                logger.warning(
                    "label quarantine: %s was rewritten during the fix stage; "
                    "pre-fix content saved as %s", rel, side.name,
                )
            else:
                path.write_bytes(old)
            report.restored.append(rel)
        if write_manifest and root.is_dir():
            payload = report.to_json()
            payload["restored_at"] = datetime.now(timezone.utc).isoformat()
            (root / MANIFEST_NAME).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )


__all__ = ["MANIFEST_NAME", "QuarantineReport", "quarantine_run_dir"]
