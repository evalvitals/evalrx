#!/usr/bin/env python3
"""Download frozen local slices for the five intervention-paper pilot.

The raw Hugging Face cache and the exported JSONL files live under ``data/``;
the example-level .gitignore prevents both from being committed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "intervention_papers.json"
DEFAULT_DATA_DIR = ROOT / "data" / "intervention_pilot"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    papers = json.loads(path.read_text(encoding="utf-8")).get("papers", [])
    if len(papers) != 5:
        raise ValueError("intervention manifest must contain exactly five papers")
    return papers


def _gsm8k_rows(dataset: Any, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for index in indices[:limit]:
        row = dataset[index]
        answer = str(row["answer"])
        match = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
        if match is None:
            continue
        rows.append({
            "id": f"gsm8k-{index}",
            "question": str(row["question"]),
            "answer": match.group(1).replace(",", ""),
            "dataset": "openai/gsm8k",
        })
    return rows


def _truthfulqa_rows(dataset: Any, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        target = row["mc1_targets"]
        choices = list(target["choices"])
        gold = next((i for i, label in enumerate(target["labels"]) if label == 1), None)
        if gold is None or len(choices) < 2:
            continue
        order = list(range(len(choices)))
        rng.shuffle(order)
        rendered = "\n".join(
            f"{chr(65 + new_index)}. {choices[old_index]}"
            for new_index, old_index in enumerate(order)
        )
        prompt = (
            f"{row['question']}\n\n"
            "Choose the single best answer. Reply with only its letter.\n"
            f"{rendered}"
        )
        rows.append({
            "id": f"truthfulqa-{index}",
            "question": prompt,
            "answer": chr(65 + order.index(gold)),
            "dataset": "truthful_qa",
        })
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gsm8k-limit", type=int, default=120)
    parser.add_argument("--truthfulqa-limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    _load_manifest(args.manifest)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required; install it with: pip install 'evalrx[data]'"
        ) from exc

    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache = args.data_dir / "hf_cache"
    gsm8k = load_dataset("openai/gsm8k", "main", split="test", cache_dir=str(cache))
    truthfulqa = load_dataset(
        "truthful_qa", "multiple_choice", split="validation", cache_dir=str(cache)
    )
    _write_jsonl(
        _gsm8k_rows(gsm8k, args.gsm8k_limit, args.seed), args.data_dir / "gsm8k.jsonl"
    )
    _write_jsonl(
        _truthfulqa_rows(truthfulqa, args.truthfulqa_limit, args.seed),
        args.data_dir / "truthfulqa_mc1.jsonl",
    )
    print(f"wrote frozen benchmark slices to {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
