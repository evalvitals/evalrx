"""Restore the original standalone MMAU report without rerunning an experiment."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVISION = "770947992e32e3dbd2b12ed4c26047941bc5b175"
RELATIVE_PATH = "examples/m1_m5/mmau_qwen2_audio/demo_page/index.html"
SHA256 = "005cfe088dcecdd3159fd318d9cd6473ccdc299464771cc3634011393ab1958b"
SIZE = 95617127
URL = f"https://raw.githubusercontent.com/evalvitals/evalrx/{REVISION}/{RELATIVE_PATH}"


def verify(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise ValueError(f"{path}: size or SHA-256 differs from the historical report")


def restore(target: Path, source: str) -> None:
    if target.exists():
        verify(target)
        print(f"Already present and verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mmau-demo-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            local = None
            if source != "http" and shutil.which("git"):
                local = subprocess.run(
                    ["git", "cat-file", "blob", f"{REVISION}:{RELATIVE_PATH}"],
                    cwd=REPO, stdout=stream, stderr=subprocess.PIPE,
                )
            if local is None or local.returncode:
                if source == "git":
                    raise RuntimeError("Pinned report is not in local Git history; use --source http")
                stream.seek(0)
                stream.truncate()
                with urllib.request.urlopen(URL, timeout=60) as response:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
        verify(temporary)
        # Publish without replacing a file another process may have created.
        os.link(temporary, target)
        print(f"Restored and verified: {target} ({SIZE} bytes)")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / RELATIVE_PATH)
    parser.add_argument("--source", choices=("auto", "git", "http"), default="auto")
    args = parser.parse_args()
    try:
        restore(args.output.absolute(), args.source)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\nExisting files were not replaced.\n")


if __name__ == "__main__":
    main()
