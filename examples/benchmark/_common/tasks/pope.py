"""POPE object-presence probes (Li et al., EMNLP 2023) — one yes + one no per image.

Three vlm datasets, one per negative-sampling split (``random`` / ``popular`` /
``adversarial``) of the official COCO probe files, pinned to the same POPE
commit as ``examples/m1_m5/deco_pope``. Each split file asks 6 questions per
image (3 present, 3 absent) over the same 500 COCO val2014 images; the frozen
slice keeps the FIRST yes and the FIRST no question of every image in file
order -> 500 x 2 = 1000 rows, class-balanced by construction. The three splits
share their yes questions (POPE reuses the present-object probes; only the
absent-object sampling differs), so the cross-split difficulty contrast lives
entirely in the gold=No half — and so do the images, which is why ``download``
hardlinks them from a sibling ``pope_*`` dataset before hitting the network.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import urllib.request
from functools import partial
from pathlib import Path

from .base import Task, _protocol, write_manifest

POPE_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
POPE_URL_FMT = ("https://raw.githubusercontent.com/AoiDragon/POPE/"
                + POPE_COMMIT + "/output/coco/coco_pope_{split}.json")
COCO_URL_FMT = "http://images.cocodataset.org/val2014/{file_name}"
SPLITS = ("random", "popular", "adversarial")
PROMPT_SUFFIX = " Answer with only the single word Yes or No."
_OBJECT_RE = re.compile(r"Is there an? (.+?) in the image\?")


def _probe_lines(split: str, cache: Path) -> list[dict]:
    """The split's official probe list (JSON-lines), downloaded once per dataset."""
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POPE_URL_FMT.format(split=split), cache)
    return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fetch_image(file_name: str, dest: Path) -> None:
    urllib.request.urlretrieve(COCO_URL_FMT.format(file_name=file_name), dest)


def _materialise_image(images_dir: Path, file_name: str) -> str:
    """``cached`` | ``reused`` (hardlink/copy from a sibling pope_* dataset) | ``downloaded``."""
    dest = images_dir / file_name
    if dest.is_file():
        return "cached"
    data_dir = images_dir.parent.parent
    for sibling in sorted(data_dir.glob(f"pope_*/images/{file_name}")):
        if sibling.is_file() and sibling.resolve() != dest.resolve():
            try:
                os.link(sibling, dest)
            except OSError:
                shutil.copy2(sibling, dest)
            return "reused"
    _fetch_image(file_name, dest)
    return "downloaded"


def download(out_dir: Path, limit: int = 1000, seed: int = 2305, split: str = "adversarial") -> dict:
    """Freeze ``limit // 2`` images x (first yes + first no) from one POPE split.

    ``limit`` counts ROWS like every other task, but the slice always keeps
    whole (yes, no) pairs; 0 freezes every image. When fewer images than
    available are requested the subset is seed-sampled, then kept in POPE file
    order.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown POPE split {split!r}; expected one of {SPLITS}")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    lines = _probe_lines(split, out_dir / f"coco_pope_{split}.json")
    per_image: dict[str, dict[str, list[dict]]] = {}
    order: list[str] = []
    for line in lines:
        image = str(line["image"])
        if image not in per_image:
            per_image[image] = {"yes": [], "no": []}
            order.append(image)
        per_image[image][str(line["label"]).strip().lower()].append(line)
    usable = [img for img in order if per_image[img]["yes"] and per_image[img]["no"]]
    n_images = min(limit // 2, len(usable)) if limit and limit > 0 else len(usable)
    if n_images < len(usable):
        keep = set(random.Random(seed).sample(usable, n_images))
        chosen = [img for img in usable if img in keep]
    else:
        chosen = usable
    rows, sourcing = [], {"cached": 0, "reused": 0, "downloaded": 0}
    for i, image in enumerate(chosen):
        sourcing[_materialise_image(images_dir, image)] += 1
        for line in (per_image[image]["yes"][0], per_image[image]["no"][0]):
            text = str(line["text"]).strip()
            match = _OBJECT_RE.match(text)
            label = str(line["label"]).strip().lower()
            rows.append({
                "id": f"pope-{split}-{line['question_id']}",
                "dataset": "AoiDragon/POPE", "subset": f"coco_pope_{split}",
                "source_index": int(line["question_id"]), "sample_seed": seed,
                "image": f"images/{image}", "audio": None,
                "prompt": text + PROMPT_SUFFIX,
                "answers": [label.capitalize()],
                "task": "yes_no", "numeric_tolerance": 0.0,
                "metadata": {"object": match.group(1) if match else "", "pope_label": label,
                             "split": split, "file_name": image},
            })
        if (i + 1) % 50 == 0:
            print(f"  [pope_{split}] images {i + 1}/{len(chosen)} ({sourcing})")
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "images": len(chosen), "skipped_unpaired": len(order) - len(usable),
            **sourcing, "manifest": str(out_dir / "manifest.json")}


_SPLIT_BLURB = {
    "random": ("the absent object is drawn uniformly from the COCO vocabulary minus the image's "
               "present objects — the easiest negatives, probing whether the model asserts objects "
               "with no contextual pull at all"),
    "popular": ("the absent object is one of the globally most frequent COCO objects that is not in "
                "the image — negatives with a language-prior pull, since frequent words invite Yes"),
    "adversarial": ("the absent object is the one most frequently co-occurring with the image's "
                    "present objects while itself absent — negatives built so co-occurrence priors "
                    "(keyboard+monitor -> 'mouse') fight the visual evidence"),
}


def protocol(model_label: str, split: str = "adversarial"):
    return _protocol(
        description=(
            f"We evaluate a vision-language model ({model_label}) on the {split} split of POPE, "
            "a discriminative object-hallucination benchmark over COCO val2014 images: the model "
            "answers a binary question of the form 'Is there a X in the image?'. Each image "
            "contributes one present-object question (gold=Yes) and one absent-object question "
            f"(gold=No); {_SPLIT_BLURB[split]}. We want to know what distinguishes the questions "
            "it gets right from the ones it gets wrong."
        ),
        task_domain=f"visual object-hallucination detection (POPE coco {split})",
        success_criteria=(
            "The model's Yes/No answer must match whether the named object actually appears in "
            "the image according to the COCO instance annotations."
        ),
        failure_patterns=(
            "wrong answers concentrated on absent-object (gold=No) questions, where the model "
            "answers Yes anyway -- object hallucination driven by language or co-occurrence "
            "priors rather than visual evidence shows exactly this asymmetric pattern; a fix "
            "that trades away present-object (gold=Yes) recall to gain absent-object accuracy "
            "is a different error, not an improvement"
        ),
        target_modalities=frozenset({"text", "image"}),
    )


def _make(split: str) -> Task:
    return Task(
        name=f"pope_{split}", modality="vlm", kind="yes_no", title=f"POPE/coco_{split}",
        download=partial(download, split=split), protocol=partial(protocol, split=split),
        pinned_m1=(
            "answer_extraction_audit", "termination_audit", "selfcheck_consistency",
            "self_consistency", "calibration", "logprob_entropy", "perturbation_battery",
        ),
        default_limit=1000, default_seed=2305, max_new_tokens=16,
        source=f"AoiDragon/POPE coco_pope_{split} @{POPE_COMMIT[:8]} + COCO val2014",
    )


TASKS = tuple(_make(split) for split in SPLITS)
