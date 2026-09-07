"""CHAIR object-hallucination captioning (Rohrbach et al., EMNLP 2018) —
OPERA's MSCOCO val2014 recipe at a seeded 450-image sample: prompt
``Please describe this image in detail.``, hallucinated COCO objects counted
with the official ``chair.py`` word rules.

CHAIR is a metric, not a dataset; the frozen slice is built locally from
``$CHAIR_DIR`` (default eval_experiments/chair — see chair_data.md there):
``coco/val2014`` + ``coco/annotations`` (train AND val instance/caption files,
``chair.py`` needs both) and ``chair.py`` (OPERA's copy), whose ``CHAIR``
evaluator gives every image its gold object set (segmentation categories +
the 5 ground-truth captions). Sampling follows chair_data.md exactly:
``random.seed(seed); random.sample(sorted(ids_with_instances), limit)`` — the
list is also written to ``$CHAIR_DIR/sampled_<limit>.json``.

Per-case label (kind ``chair_caption``): PASS iff the caption names NO object
outside the image's gold set (per-image CHAIR_S == 0) AND names at least one
gold object (recall > 0 — an empty or evasive caption is not a repair). The
population CHAIR_S/CHAIR_I/Recall/Len come from ``chair.py`` on the exported
captions (``chair/captions/*.jsonl``); the harness only needs the binary.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

from .base import Task, _protocol, write_manifest

PROMPT = "Please describe this image in detail."
DEFAULT_CHAIR_DIR = "/tealab-data/jiaqiliu/evalsmith/eval_experiments/chair"


def _chair_dir() -> Path:
    return Path(os.environ.get("CHAIR_DIR", DEFAULT_CHAIR_DIR))


def _evaluator(chair_dir: Path):
    """OPERA's ``CHAIR`` object (gold objects per image), pickled once as chair/chair.pkl."""
    os.environ.setdefault("CHAIR_NLTK_DATA", str(chair_dir / "nltk_data"))
    from . import chair_words

    chair_words._nltk()                     # registers $CHAIR_NLTK_DATA before chair.py tokenises
    cache = chair_dir / "chair.pkl"
    if str(chair_dir) not in sys.path:
        sys.path.insert(0, str(chair_dir))
    import chair as official  # noqa: F401  (chair/chair.py; pickle needs the module)

    if cache.is_file():
        return pickle.load(open(cache, "rb"))
    evaluator = official.CHAIR(str(chair_dir / "coco" / "annotations"))
    pickle.dump(evaluator, open(cache, "wb"))
    return evaluator


def download(out_dir: Path, limit: int = 450, seed: int = 0) -> dict:
    chair_dir = _chair_dir()
    coco = chair_dir / "coco"
    val_dir, ann = coco / "val2014", coco / "annotations" / "instances_val2014.json"
    if not ann.is_file() or not val_dir.is_dir():
        raise SystemExit(f"CHAIR needs {ann} and {val_dir} (see eval_experiments/chair_data.md)")
    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    instances = json.load(open(ann))
    ids = sorted({a["image_id"] for a in instances["annotations"]})     # only images with object annotations
    random.seed(seed)
    sampled = random.sample(ids, limit) if limit and limit > 0 else ids
    sample_file = chair_dir / f"sampled_{len(sampled)}.json"
    if not sample_file.is_file():
        json.dump(sampled, open(sample_file, "w"))
    evaluator = _evaluator(chair_dir)
    rows, copied = [], 0
    for image_id in sampled:
        file_name = f"COCO_val2014_{image_id:012d}.jpg"
        dest = images / file_name
        if not dest.is_file():
            src = val_dir / file_name
            if not src.is_file():
                raise FileNotFoundError(src)
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
            copied += 1
        gt = sorted(evaluator.imid_to_objects.get(image_id, ()))
        rows.append({
            "id": f"chair-{image_id}", "dataset": "MSCOCO val2014 (CHAIR/OPERA 500)", "subset": "val2014",
            "source_index": int(image_id), "sample_seed": seed,
            "image": f"images/{file_name}", "audio": None,
            "prompt": PROMPT,
            "answers": [", ".join(gt)], "gold": gt,
            "task": "chair_caption", "numeric_tolerance": 0.0,
            "metadata": {"image_id": int(image_id), "file_name": file_name, "gt_objects": gt,
                         "n_gt_objects": len(gt)},
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "images_linked": copied, "sample_file": str(sample_file),
            "candidates": len(ids), "manifest": str(out_dir / "manifest.json")}


def grade(output: str, gold) -> bool:
    from . import chair_words

    gt = gold if isinstance(gold, (list, tuple, set)) else [g.strip() for g in str(gold).split(",") if g.strip()]
    case = chair_words.chair_case(output, gt)
    return not case["hallucinated"] and bool(case["recalled"])


def protocol(model_label: str):
    return _protocol(
        description=(
            f"We evaluate a vision-language model ({model_label}) on CHAIR object hallucination "
            "in open-ended image captioning (OPERA's MSCOCO val2014 recipe, a seeded 450-image "
            "sample): the "
            "model is asked 'Please describe this image in detail.' and its description is "
            "scanned for the 80 COCO object categories (with the official CHAIR synonym table). "
            "A case FAILS when the description names an object that is not in the image's gold "
            "object set (segmentation annotations + the five reference captions), or names no "
            "gold object at all. We want to know what distinguishes the images it describes "
            "faithfully from the ones where it hallucinates objects."
        ),
        task_domain="open-ended image captioning, object hallucination (CHAIR, MSCOCO)",
        success_criteria=(
            "The description mentions at least one gold object and no COCO object absent from "
            "the image (per-image CHAIR_S = 0); mentions are matched after lemmatisation via the "
            "CHAIR synonym table, so 'puppy' counts as 'dog' and 'people' as 'person'."
        ),
        failure_patterns=(
            "descriptions that add plausible-but-absent objects (a 'fork' beside a plate, a "
            "'person' behind a parked bicycle, a 'clock' on a wall) — co-occurrence priors and "
            "long-tail elaboration past the visual evidence; a 'fix' that simply shortens the "
            "description or refuses to name objects lowers hallucinations by lowering recall and "
            "is not an improvement — repairs must keep the description detailed and grounded"
        ),
        target_modalities=frozenset({"text", "image"}),
    )


TASK = Task(
    name="chair", modality="vlm", kind="chair_caption", title="CHAIR/MSCOCO-val2014-450",
    download=download, protocol=protocol,
    pinned_m1=("termination_audit", "selfcheck_consistency", "self_consistency", "perturbation_battery"),
    default_limit=450, default_seed=0, max_new_tokens=512, short_answer=False,
    source="MSCOCO val2014 + OPERA chair.py (Rohrbach et al. 2018; Huang et al. 2024 recipe, 450 images)",
)
