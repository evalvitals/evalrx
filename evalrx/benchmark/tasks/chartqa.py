"""ChartQA, human-authored test split (Masry et al. 2022) — frozen deterministic sample.

Port of examples/m1_m5/chartqa_qwen3_5_2b/download_chartqa.py: same parquet, same
human-only filter, same seeded sample, same prompt suffix and relaxed 5 % numeric
tolerance; the manifest now follows the benchmark protocol (image/audio slots).

:func:`validation_source` freezes a validation-only manifest: every human-authored
row a discovery pool left (the rest of the test split, then the whole val split),
for ``--triage-validation-source``::

    python -m evalrx.benchmark.tasks.chartqa <out_dir> --exclude <pool>/manifest.json
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from .base import Task, _protocol, load_rows, write_manifest

REPO = "HuggingFaceM4/ChartQA"
REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
TEST_PARQUET = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
VAL_PARQUET = "data/val-00000-of-00001-0f11003c77497969.parquet"
#: per parquet: the subset label, the case id prefix and the image file prefix of its rows
SPLITS = {TEST_PARQUET: ("test_human", "chartqa-human-", "chartqa-"),
          VAL_PARQUET: ("val_human", "chartqa-val-human-", "chartqa-val-")}


def _table(parquet: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(REPO, parquet, repo_type="dataset", revision=REVISION))


def _human(table) -> list[int]:
    return [i for i, v in enumerate(table.column("human_or_machine").to_pylist()) if v == 0]


def _rows(table, indices: list[int], images: Path, *, parquet: str, sample_seed: int | None) -> list[dict]:
    from PIL import Image

    subset, id_prefix, file_prefix = SPLITS[parquet]
    rows = []
    for source_index in indices:
        item = table.slice(source_index, 1).to_pylist()[0]
        filename = f"{file_prefix}{source_index:05d}.png"
        destination = images / filename
        if not destination.exists():
            Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB").save(destination)
        rows.append({
            "id": f"{id_prefix}{source_index}",
            "dataset": REPO, "revision": REVISION, "subset": subset, "source_index": source_index,
            "sample_seed": sample_seed,
            "image": f"images/{filename}", "audio": None,
            "prompt": item["query"] + " Answer with only the short answer.",
            "answers": [str(a) for a in item["label"]],
            "task": "exact_or_numeric", "numeric_tolerance": 0.05,
            "metadata": {"human_or_machine": "human"},
        })
    return rows


def download(out_dir: Path, limit: int = 256, seed: int = 5022, exclude_ids: set | None = None,
             val_limit: int = 0) -> dict:
    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    table = _table(TEST_PARQUET)
    excluded = set(exclude_ids or ())
    human = [i for i in _human(table) if f"chartqa-human-{i}" not in excluded]
    selected = random.Random(seed).sample(human, min(limit, len(human)))
    write_manifest(out_dir / "manifest.json", _rows(table, selected, images, parquet=TEST_PARQUET, sample_seed=seed))
    summary = {"kept": len(selected), "excluded": len(excluded),
               "manifest": str(out_dir / "manifest.json")}
    if val_limit and val_limit > 0:
        # Held-out validation manifest: a FRESH draw (seed + 1) from the human
        # rows the main sample did not take — disjoint by construction.
        leftover = [i for i in human if i not in set(selected)]
        val_selected = random.Random(seed + 1).sample(leftover, min(val_limit, len(leftover)))
        val_rows = _rows(table, val_selected, images, parquet=TEST_PARQUET, sample_seed=seed + 1)
        write_manifest(out_dir / "manifest_val.json", val_rows)
        summary.update(kept_val=len(val_rows),
                       manifest_val=str(out_dir / "manifest_val.json"))
    return summary


def validation_source(out_dir: Path, exclude: "list[str | Path] | tuple[str | Path, ...]" = ()) -> dict:
    """Freeze ``<out_dir>/manifest.json`` with every human-authored row whose id is in none of the
    *exclude* manifests: the rest of the test split, then the whole val split, in source order (no
    sampling: this is the data the pool left). Validation only (``--triage-validation-source``); the
    partition, not this, leaves out what shares a chart with DISCOVERY."""
    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    excluded = {str(r["id"]) for path in exclude for r in load_rows(Path(path))}
    rows: list[dict] = []
    by_subset: dict[str, int] = {}
    for parquet in (TEST_PARQUET, VAL_PARQUET):
        table = _table(parquet)
        subset, id_prefix, _ = SPLITS[parquet]
        indices = [i for i in _human(table) if f"{id_prefix}{i}" not in excluded]
        rows.extend(_rows(table, indices, images, parquet=parquet, sample_seed=None))
        by_subset[subset] = len(indices)
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "by_subset": by_subset, "excluded": len(excluded),
            "manifest": str(out_dir / "manifest.json")}


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
    default_limit=256, val_limit=128, default_seed=5022, max_new_tokens=64,
    source="HuggingFaceM4/ChartQA (test, human-authored)",
)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="freeze ChartQA's validation-only source (see validation_source)")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--exclude", nargs="*", default=[], help="manifests whose rows are left out (the discovery pool)")
    ns = ap.parse_args()
    print(json.dumps(validation_source(ns.out_dir, ns.exclude)))
