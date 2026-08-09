#!/usr/bin/env python3
"""Convert downloaded PDFs into page-level records consumable by EvalVitals.

This is the benchmark's data interface: each record preserves paper identity,
page number, diagnosis axis, and the verbatim extracted page text. Keeping the
source text in ignored ``data/`` lets a coding-agent analysis cite evidence
without committing copyrighted PDFs or derived corpora to this repository.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

from download_papers import DEFAULT_DATA_DIR, DEFAULT_MANIFEST, load_papers

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS_PATH = ROOT / "data" / "paper_records.jsonl"


def _reader(pdf_path: Path) -> Any:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "pypdf is required to extract PDFs. Install it with: "
            'pip install "evalvitals[dashboard]"'
        ) from exc
    return PdfReader(str(pdf_path))


def page_records(paper: dict[str, Any], pdf_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one normalized, traceable evidence record per non-empty PDF page."""
    for page_number, page in enumerate(_reader(pdf_path).pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            yield {
                "record_type": "paper_evidence_page",
                "paper_id": paper["id"],
                "paper_title": paper["title"],
                "paper_year": paper["year"],
                "source_url": paper["source_url"],
                "page": page_number,
                "diagnosis_axis": paper["diagnosis_axis"],
                "expected_stress_test": paper["expected_stress_test"],
                "text": text,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_RECORDS_PATH)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for paper in load_papers(args.manifest):
        pdf_path = args.data_dir / f"{paper['id']}.pdf"
        if not pdf_path.exists():
            missing.append(paper["id"])
            continue
        records.extend(page_records(paper, pdf_path))
    if missing:
        raise SystemExit(
            "missing PDFs for: " + ", ".join(missing) + "; run download_papers.py first"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} page records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
