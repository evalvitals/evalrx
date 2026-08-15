#!/usr/bin/env python3
"""Download the five benchmark papers without placing their PDFs in git.

The manifest stores canonical source URLs; ``data/papers/`` is intentionally
ignored. Re-running the command is idempotent unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "papers.json"
DEFAULT_DATA_DIR = ROOT / "data" / "papers"


def load_papers(manifest: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the checked-in paper manifest."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    papers = payload.get("papers")
    if not isinstance(papers, list) or len(papers) < 5:
        raise ValueError("manifest must contain at least five papers")
    for paper in papers:
        if not isinstance(paper, dict) or not paper.get("id") or not paper.get("pdf_url"):
            raise ValueError("every paper needs non-empty 'id' and 'pdf_url' fields")
    return papers


def download_paper(paper: dict[str, Any], destination: Path, *, force: bool = False) -> Path:
    """Download one PDF atomically and return its local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"skip: {paper['id']} ({destination})")
        return destination

    request = Request(
        str(paper["pdf_url"]),
        headers={"User-Agent": "EvalVitals-paper-diagnosis-example/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"application/pdf", "application/octet-stream"}:
            raise ValueError(f"{paper['id']}: expected a PDF, received {content_type!r}")
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    print(f"downloaded: {paper['id']} -> {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--paper", action="append", default=[], help="Paper id to download (repeatable)."
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing PDFs.")
    args = parser.parse_args()

    selected = set(args.paper)
    papers = load_papers(args.manifest)
    available = {str(paper["id"]) for paper in papers}
    unknown = selected - available
    if unknown:
        parser.error(f"unknown paper id(s): {', '.join(sorted(unknown))}")

    for paper in papers:
        if selected and paper["id"] not in selected:
            continue
        download_paper(paper, args.data_dir / f"{paper['id']}.pdf", force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
