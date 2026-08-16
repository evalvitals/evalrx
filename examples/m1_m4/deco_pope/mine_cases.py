"""Offline miner — build the frozen fail/success manifest for one model size.

Run ONCE per model key (failure sets don't transfer across 2B/4B/8B):

    python mine_cases.py --model qwen3-vl-8b-instruct --n-images 100

Steps:
    1. download the POPE COCO adversarial/random split JSONs (pinned URLs)
    2. group probes into per-image triplets (adversarial-absent / present / random-absent)
    3. download the COCO val2014 images into data/images/
    4. greedy-generate the model's answer per probe (do_sample=False)
    5. label PASS/FAIL via pope.parse_yes_no, attach yes/no token-id sets
    6. split 60/40 explore/validate BY IMAGE, freeze to data/cases/{model_key}.json

See DESIGN.md §4 for the rationale of every choice here.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
IMG_DIR = DATA / "images"
RAW_DIR = DATA / "raw"
COCO_URL = "http://images.cocodataset.org/val2014/{file_name}"
# POPE official probe lists (Li et al. 2023), pinned to a commit hash.
POPE_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
POPE_URLS = {
    "adversarial": f"https://raw.githubusercontent.com/AoiDragon/POPE/{POPE_COMMIT}/output/coco/coco_pope_adversarial.json",
    "random": f"https://raw.githubusercontent.com/AoiDragon/POPE/{POPE_COMMIT}/output/coco/coco_pope_random.json",
}
POPE_TMPL = "Is there a {obj} in the image? Please answer Yes or No."
QUESTION_RE = re.compile(r"Is there an? (.+?) in the image\?")


def _pope_lines(split: str) -> list[dict]:
    """Download (once) and parse one POPE split — the files are JSON-lines."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"coco_pope_{split}.json"
    if not dest.exists():
        urllib.request.urlretrieve(POPE_URLS[split], dest)
    return [json.loads(line) for line in dest.read_text().splitlines() if line.strip()]


def fetch_pope_probes(n_images: int) -> list[dict]:
    """Download POPE splits and assemble per-image triplets (DESIGN.md §4.2).

    Returns entries: {image_id, file_name, object, pope_label, probe_type}.
    Selection is deterministic — first usable probe of each type in file order
    (POPE question order), images in first-appearance order of the adversarial
    split. The random-absent object must differ from the adversarial-absent one
    so the co-occurrence gradient contrast is preserved.
    """
    adv, rnd = _pope_lines("adversarial"), _pope_lines("random")

    per_image: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []
    for line in adv:
        obj = QUESTION_RE.match(line["text"]).group(1)
        slot = per_image.setdefault(line["image"], {"present": [], "adversarial": [], "random": []})
        if line["image"] not in order:
            order.append(line["image"])
        slot["present" if line["label"] == "yes" else "adversarial"].append(obj)
    for line in rnd:
        if line["label"] == "no" and line["image"] in per_image:
            per_image[line["image"]]["random"].append(QUESTION_RE.match(line["text"]).group(1))

    probes, skipped = [], 0
    for file_name in order:
        slot = per_image[file_name]
        adv_objs = slot["adversarial"]
        rnd_objs = [o for o in slot["random"] if o not in adv_objs]
        if not (slot["present"] and adv_objs and rnd_objs):
            skipped += 1  # incomplete triplet — drop the whole image (TODO.md)
            continue
        image_id = int(file_name.split("_")[-1].split(".")[0])
        for probe_type, obj, label in (
            ("adversarial", adv_objs[0], "no"),
            ("present", slot["present"][0], "yes"),
            ("random", rnd_objs[0], "no"),
        ):
            probes.append({
                "image_id": image_id, "file_name": file_name,
                "object": obj, "pope_label": label, "probe_type": probe_type,
            })
        if len(probes) >= n_images * 3:
            break
    print(f"assembled {len(probes)} probes from {len(probes) // 3} images "
          f"(skipped {skipped} incomplete)")
    return probes


def download_images(probes: list[dict]) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    todo = sorted({p["file_name"] for p in probes})
    for i, fn in enumerate(todo):
        dest = IMG_DIR / fn
        if not dest.exists():
            urllib.request.urlretrieve(COCO_URL.format(file_name=fn), dest)
        if (i + 1) % 25 == 0:
            print(f"  images {i + 1}/{len(todo)}")


def answer_token_sets(tokenizer) -> dict[str, list[int]]:
    """Token-id equivalence sets for yes/no (DESIGN.md §3.5)."""
    def ids(variants):
        out = []
        for v in variants:
            t = tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1:          # only single-token variants are usable at pos -1
                out.append(t[0])
        return sorted(set(out))
    return {"yes": ids(["Yes", " Yes", "yes", " yes"]),
            "no": ids(["No", " No", "no", " no"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explore-frac", type=float, default=0.6)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch
    import transformers
    from PIL import Image

    from evalvitals import compose
    from evalvitals.analyzers.hallucination.pope import parse_yes_no
    from evalvitals.core.capability import Capability
    from evalvitals.core.case import Inputs

    probes = fetch_pope_probes(args.n_images)
    download_images(probes)

    model = compose(args.model, "hf_local", want={Capability.GENERATE})
    # Public tokenizer accessor: load from the spec's repo (the backend keeps its
    # processor private; the plain tokenizer is all we need for answer-token ids).
    tokenizer = transformers.AutoTokenizer.from_pretrained(model.spec.hf_repo)
    tok_sets = answer_token_sets(tokenizer)

    # By-image split BEFORE labeling, so yields per split are honest.
    image_ids = sorted({p["image_id"] for p in probes})
    random.shuffle(image_ids)
    n_explore = int(len(image_ids) * args.explore_frac)
    split_of = {iid: ("explore" if i < n_explore else "validate")
                for i, iid in enumerate(image_ids)}

    records, yields = [], {"explore": {"fail": 0, "pass": 0}, "validate": {"fail": 0, "pass": 0}}
    for i, p in enumerate(probes):
        image = Image.open(IMG_DIR / p["file_name"]).convert("RGB")
        ans = model.generate(
            Inputs(prompt=POPE_TMPL.format(obj=p["object"]), image=image),
            max_new_tokens=8, do_sample=False,
        )
        pred = parse_yes_no(ans)
        label = "pass" if pred == p["pope_label"] else "fail"
        split = split_of[p["image_id"]]
        yields[split][label] += 1
        records.append({
            **p, "observed": ans, "pred": pred, "label": label, "split": split,
            "gt_token_ids": tok_sets[p["pope_label"]],
            "out_token_ids": tok_sets[pred] if pred else [],
        })
        if (i + 1) % 30 == 0:
            print(f"  probes {i + 1}/{len(probes)}  yields={yields}")

    out = DATA / "cases" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "seed": args.seed, "prompt_template": POPE_TMPL,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        "pope_commit": POPE_COMMIT,
        "triplet_rule": ("first probe per type in POPE file order; random-absent "
                         "object forced != adversarial-absent; incomplete images dropped"),
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "dtype": "bfloat16"},
        "yields": yields, "cases": records,
    }, indent=2, ensure_ascii=False))
    print(f"frozen -> {out}  yields={yields}")
    # DESIGN.md §8: if explore fail count < ~15, raise --n-images or merge AMBER.


if __name__ == "__main__":
    main()
