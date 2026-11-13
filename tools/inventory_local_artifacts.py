"""Inventory retained, Git-ignored experiment files without moving or deleting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash", action="store_true", help="Read each file and record SHA-256")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent file reads (default: 8)")
    parser.add_argument("--output", type=Path, default=REPO / ".local-artifacts/inventory.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths = subprocess.check_output(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", "examples"],
        cwd=REPO,
    ).decode().split("\0")
    paths = sorted(p for p in paths if p and "__pycache__" not in Path(p).parts
                   and Path(p).suffix not in {".pyc", ".pyo"})

    def inspect(relative: str) -> tuple[dict | None, bool]:
        path = REPO / relative
        try:
            before = path.stat()
            row = {"path": relative, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
            if args.hash:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                row["sha256"] = digest.hexdigest()
            after = path.stat()
            changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            return row, changed
        except FileNotFoundError:
            return None, True

    print(f"Inventorying {len(paths)} files with {args.workers} workers", flush=True)
    rows = []
    unstable = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (relative, (row, changed)) in enumerate(zip(paths, pool.map(inspect, paths)), 1):
            if row is not None:
                rows.append(row)
            if changed:
                unstable.append(relative)
            if i % 5000 == 0:
                print(f"Inventoried {i}/{len(paths)} files", flush=True)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "scope": "Git-ignored files under examples/, excluding Python bytecode caches",
        "hash_algorithm": "sha256" if args.hash else None,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "changed_during_inventory": unstable,
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} files, {report['total_bytes']} bytes: {args.output}")
    if unstable:
        parser.exit(1, f"{len(unstable)} files changed during the inventory; rerun when idle.\n")


if __name__ == "__main__":
    main()
