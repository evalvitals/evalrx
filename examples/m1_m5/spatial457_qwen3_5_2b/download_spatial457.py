"""Freeze a deterministic Spatial457 L5 sample without dataset loader scripts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "RyanWW/Spatial457"
SUBSET = "L5_6d_spatial"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=457)
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
    question_path = hf_hub_download(
        REPO, f"questions/{SUBSET}.json", repo_type="dataset"
    )
    payload = json.loads(Path(question_path).read_text(encoding="utf-8"))
    questions = payload["questions"]
    excluded: set[str] = set()
    for path in args.exclude_json:
        previous = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = previous if isinstance(previous, list) else previous.get("cases", [])
        excluded.update(
            str(row.get("id") or row.get("sample_id"))
            for row in rows
            if isinstance(row, dict) and (row.get("id") or row.get("sample_id"))
        )
    eligible = [
        index
        for index, item in enumerate(questions)
        if f"spatial457-{item.get('question_index', index)}" not in excluded
    ]
    rng = random.Random(args.seed)
    indices = rng.sample(eligible, min(args.limit, len(eligible)))
    rows = []
    for source_index in indices:
        item = questions[source_index]
        filename = item["image_filename"]
        source = Path(hf_hub_download(REPO, f"images/{filename}", repo_type="dataset"))
        destination = images / filename
        if not destination.exists():
            shutil.copy2(source, destination)
        rows.append({
            "id": f"spatial457-{item.get('question_index', source_index)}",
            "dataset": "RyanWW/Spatial457",
            "subset": SUBSET,
            "source_index": source_index,
            "sample_seed": args.seed,
            "image": f"images/{filename}",
            "prompt": item["question"] + " Answer with only the short answer.",
            "answers": [str(item["answer"])],
            "numeric_tolerance": 0.0,
            "metadata": {
                "split": item.get("split"),
                "program": item.get("program"),
                "question_index": item.get("question_index"),
            },
        })
    (output / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(rows)} frozen questions and "
        f"{len(set(r['image'] for r in rows))} images to {output} "
        f"(excluded {len(excluded)} prior ids)"
    )


if __name__ == "__main__":
    main()
