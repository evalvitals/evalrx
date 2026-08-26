"""ChartQA, human-authored test split (Masry et al. 2022) — frozen deterministic sample.

Port of examples/m1_m4/chartqa_qwen3_5_2b/download_chartqa.py: same parquet, same
human-only filter, same seeded sample, same prompt suffix and relaxed 5 % numeric
tolerance; the manifest now follows the benchmark protocol (image/audio slots).
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from .base import Task, _protocol, write_manifest

REPO = "HuggingFaceM4/ChartQA"
TEST_PARQUET = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"


def download(out_dir: Path, limit: int = 256, seed: int = 5022, exclude_ids: set | None = None) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(hf_hub_download(REPO, TEST_PARQUET, repo_type="dataset"))
    human = [i for i, v in enumerate(table.column("human_or_machine").to_pylist()) if v == 0]
    excluded = set(exclude_ids or ())
    human = [i for i in human if f"chartqa-human-{i}" not in excluded]
    selected = random.Random(seed).sample(human, min(limit, len(human)))
    rows = []
    for source_index in selected:
        item = table.slice(source_index, 1).to_pylist()[0]
        filename = f"chartqa-{source_index:05d}.png"
        destination = images / filename
        if not destination.exists():
            Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB").save(destination)
        rows.append({
            "id": f"chartqa-human-{source_index}",
            "dataset": REPO, "subset": "test_human", "source_index": source_index,
            "sample_seed": seed,
            "image": f"images/{filename}", "audio": None,
            "prompt": item["query"] + " Answer with only the short answer.",
            "answers": [str(a) for a in item["label"]],
            "task": "exact_or_numeric", "numeric_tolerance": 0.05,
            "metadata": {"human_or_machine": "human"},
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "excluded": len(excluded), "manifest": str(out_dir / "manifest.json")}


def protocol(model_label: str):
    return _protocol(
        description=(
            f"Evaluate a vision-language model ({model_label}) on the human-authored ChartQA "
            "test split. Questions require reading chart marks and labels, comparing "
            "quantities, and performing arithmetic or logical reasoning. Diagnose whether "
            "errors reflect answer extraction, unstable visual evidence, or failure to "
            "verify a candidate answer."
        ),
        task_domain="chart visual question answering",
        success_criteria=(
            "The short answer must match an official ChartQA label after normalization; "
            "numeric answers use ChartQA's relaxed five-percent tolerance."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


TASK = Task(
    name="chartqa", modality="vlm", kind="exact_or_numeric", title="ChartQA/test_human",
    download=download, protocol=protocol,
    pinned_m1=("answer_extraction_audit", "selfcheck_consistency", "coverage_verification_gap"),
    default_limit=256, default_seed=5022, max_new_tokens=64,
    source="HuggingFaceM4/ChartQA (test, human-authored)",
)
