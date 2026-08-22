"""Freeze a deterministic sample of the human-authored ChartQA test split."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "HuggingFaceM4/ChartQA"
TEST_PARQUET = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--exclude-json",
        action="append",
        default=[],
        help="JSON list from an earlier run; exclude its ids from this fresh sample.",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    parquet = hf_hub_download(REPO, TEST_PARQUET, repo_type="dataset")
    table = pq.read_table(parquet)
    human_indices = [
        i for i, value in enumerate(table.column("human_or_machine").to_pylist())
        if value == 0  # ClassLabel names are ["human", "machine"]
    ]
    excluded: set[str] = set()
    for path in args.exclude_json:
        previous = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = previous if isinstance(previous, list) else previous.get("cases", [])
        excluded.update(
            str(row.get("id") or row.get("sample_id"))
            for row in rows
            if isinstance(row, dict) and (row.get("id") or row.get("sample_id"))
        )
    human_indices = [
        index for index in human_indices if f"chartqa-human-{index}" not in excluded
    ]
    rng = random.Random(args.seed)
    selected = rng.sample(human_indices, min(args.limit, len(human_indices)))
    rows = []
    for source_index in selected:
        item = table.slice(source_index, 1).to_pylist()[0]
        filename = f"chartqa-{source_index:05d}.png"
        destination = images / filename
        if not destination.exists():
            Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB").save(destination)
        rows.append({
            "id": f"chartqa-human-{source_index}",
            "dataset": REPO,
            "subset": "test_human",
            "source_index": source_index,
            "sample_seed": args.seed,
            "image": f"images/{filename}",
            "prompt": item["query"] + " Answer with only the short answer.",
            "answers": [str(answer) for answer in item["label"]],
            "numeric_tolerance": 0.05,
            "metadata": {"human_or_machine": "human"},
        })
    (output / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(rows)} frozen human-authored test questions and images to {output} "
        f"(excluded {len(excluded)} prior ids)"
    )


if __name__ == "__main__":
    main()
