"""Spatial457 L5 6D-spatial questions (Wang et al. 2025) — frozen deterministic sample.

Port of examples/m1_m5/spatial457_qwen3_5_2b/download_spatial457.py (no dataset
loader scripts: questions JSON + per-image hub downloads).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .base import Task, _protocol, write_manifest

REPO = "RyanWW/Spatial457"
SUBSET = "L5_6d_spatial"


def download(out_dir: Path, limit: int = 256, seed: int = 457, exclude_ids: set | None = None,
             val_limit: int = 0) -> dict:
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    question_path = hf_hub_download(REPO, f"questions/{SUBSET}.json", repo_type="dataset")
    questions = json.loads(Path(question_path).read_text(encoding="utf-8"))["questions"]
    excluded = set(exclude_ids or ())
    eligible = [
        i for i, item in enumerate(questions)
        if f"spatial457-{item.get('question_index', i)}" not in excluded
    ]
    indices = random.Random(seed).sample(eligible, min(limit, len(eligible)))

    def rows_for(selected: list[int], sample_seed: int) -> list[dict]:
        rows = []
        for source_index in selected:
            item = questions[source_index]
            filename = item["image_filename"]
            destination = images / filename
            if not destination.exists():
                source = Path(hf_hub_download(REPO, f"images/{filename}", repo_type="dataset"))
                shutil.copy2(source, destination)
            rows.append({
                "id": f"spatial457-{item.get('question_index', source_index)}",
                "dataset": REPO, "subset": SUBSET, "source_index": source_index,
                "sample_seed": sample_seed,
                "image": f"images/{filename}", "audio": None,
                "prompt": item["question"] + " Answer with only the short answer.",
                "answers": [str(item["answer"])],
                "task": "exact_or_numeric", "numeric_tolerance": 0.0,
                "metadata": {
                    "split": item.get("split"), "program": item.get("program"),
                    "question_index": item.get("question_index"),
                },
            })
        return rows

    write_manifest(out_dir / "manifest.json", rows_for(indices, seed))
    summary = {"kept": len(indices), "excluded": len(excluded),
               "manifest": str(out_dir / "manifest.json")}
    if val_limit and val_limit > 0:
        # Held-out validation manifest: a FRESH draw (seed + 1) from the
        # questions the main sample did not take — disjoint by construction.
        leftover = [i for i in eligible if i not in set(indices)]
        val_indices = random.Random(seed + 1).sample(leftover, min(val_limit, len(leftover)))
        val_rows = rows_for(val_indices, seed + 1)
        write_manifest(out_dir / "manifest_val.json", val_rows)
        summary.update(kept_val=len(val_rows),
                       manifest_val=str(out_dir / "manifest_val.json"))
    return summary


def protocol(model_label: str):
    return _protocol(
        description=(
            f"Evaluate a vision-language model ({model_label}) on Spatial457 level-5 6D "
            "spatial questions. Each synthetic scene requires grounding multiple objects "
            "and reasoning about location, orientation, depth, and object attributes. "
            "Diagnose whether errors come from answer extraction, unstable visual "
            "grounding, or a verification gap."
        ),
        task_domain="6D visual spatial reasoning",
        success_criteria=(
            "The short answer must exactly match the official Spatial457 answer after "
            "case, punctuation, article, and yes/no normalization."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


TASK = Task(
    name="spatial457", modality="vlm", kind="exact_or_numeric", title="Spatial457/L5_6d_spatial",
    download=download, protocol=protocol,
    pinned_m1=("answer_extraction_audit", "selfcheck_consistency", "coverage_verification_gap"),
    default_limit=256, val_limit=128, default_seed=457, max_new_tokens=64,
    source="RyanWW/Spatial457 (L5_6d_spatial)",
)
