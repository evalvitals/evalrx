"""Offline miner (deco_chair) — captions -> CHAIR-matched mentions -> frozen manifest.

    python mine_cases.py --model qwen3-vl-8b-instruct --n-images 50

Steps (DESIGN.md §2):
    1. pick ~50 cluttered COCO val2014 images (selection criteria coded, not handpicked)
    2. extract per-image GT object lists from instances_val2014.json (cached, not committed)
    3. greedy caption each image ("Please help me describe the image in detail.")
    4. CHAIR-match mentions vs GT (synonym table) -> hallucinated | grounded
    5. locate each mention's first-token index in the full tokenized sequence,
       then VERIFY it: re-feed ids[:token_index], greedy argmax must equal the
       mention's first token (recorded per mention — the probe step relies on it)
    6. freeze to data/cases/{model_key}.json (60/40 explore/validate by image,
       stratified by hallucinated)
"""

from __future__ import annotations

import argparse
import json
import random
import string
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
RAW_DIR = DATA / "raw"
IMG_DIR = DATA / "images"
CAPTION_PROMPT = "Please help me describe the image in detail."
COCO_IMG_URL = "http://images.cocodataset.org/val2014/{file_name}"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
# CHAIR standard synonym table (Rohrbach et al. 2018), pinned to a commit hash.
SYNONYMS_COMMIT = "6e4d33c4bedf6dfcd24c61bf527f40714717be47"
SYNONYMS_URL = (f"https://raw.githubusercontent.com/LisaAnne/Hallucination/"
                f"{SYNONYMS_COMMIT}/data/synonyms.txt")
# Cluttered indoor scenes have the strongest co-occurrence priors (DESIGN.md §2).
INDOOR_SUPERCATS = {"kitchen", "furniture", "electronic", "appliance"}
# Punctuation -> space, LENGTH-PRESERVING, so char offsets survive normalization.
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _instances() -> dict:
    path = RAW_DIR / "annotations" / "instances_val2014.json"
    if not path.exists():
        raise SystemExit(
            f"{path} missing — download + unzip instances_val2014.json from "
            f"{ANNOTATIONS_URL} (250MB zip; cached, never committed)")
    return json.loads(path.read_text())


def select_images(n: int, inst: dict) -> list[dict]:
    """Cluttered-scene selection: rank val2014 images by (#distinct indoor-supercat
    categories, #distinct categories), take top n. Criteria recorded in
    data/image_list.json — coded, not handpicked."""
    cats = {c["id"]: (c["name"].lower(), c["supercategory"]) for c in inst["categories"]}
    file_of = {i["id"]: i["file_name"] for i in inst["images"]}
    per_img: dict[int, set[int]] = {}
    for a in inst["annotations"]:
        per_img.setdefault(a["image_id"], set()).add(a["category_id"])

    rows = []
    for iid, cset in per_img.items():
        n_indoor = sum(1 for c in cset if cats[c][1] in INDOOR_SUPERCATS)
        rows.append({"image_id": iid, "file_name": file_of[iid],
                     "n_categories": len(cset), "n_indoor_categories": n_indoor})
    rows.sort(key=lambda r: (-r["n_indoor_categories"], -r["n_categories"], r["image_id"]))
    chosen = rows[:n]

    (DATA / "image_list.json").write_text(json.dumps({
        "criteria": ("val2014 images ranked by (-n_indoor_categories, -n_categories, "
                     f"image_id); indoor supercategories = {sorted(INDOOR_SUPERCATS)} "
                     "('indoor' supercat itself excluded: book/clock/vase/... are not "
                     "scene-defining); top n taken"),
        "n": n, "images": chosen,
    }, indent=2))
    return chosen


def extract_gt_objects(images: list[dict], inst: dict) -> dict[str, list[str]]:
    """instances_val2014.json -> {str(image_id): [coco category names]} for chosen
    images. Written to data/gt_objects.json (small, committed)."""
    cats = {c["id"]: c["name"].lower() for c in inst["categories"]}
    want = {img["image_id"] for img in images}
    gt: dict[str, set[str]] = {str(i): set() for i in want}
    for a in inst["annotations"]:
        if a["image_id"] in want:
            gt[str(a["image_id"])].add(cats[a["category_id"]])
    out = {k: sorted(v) for k, v in gt.items()}
    (DATA / "gt_objects.json").write_text(json.dumps(out, indent=2))
    return out


def build_synonyms() -> dict[str, list[str]]:
    """CHAIR standard table -> {coco_category: [surface forms]} (first entry of each
    line is the canonical COCO name; verified to match instances categories 1:1).
    Frozen to data/chair_synonyms.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    src = RAW_DIR / "synonyms.txt"
    if not src.exists():
        urllib.request.urlretrieve(SYNONYMS_URL, src)
    table: dict[str, list[str]] = {}
    for line in src.read_text().splitlines():
        surfaces = [s.strip().lower() for s in line.split(",") if s.strip()]
        table[surfaces[0]] = sorted(set(surfaces))
    (DATA / "chair_synonyms.json").write_text(json.dumps({
        "_meta": {"source": SYNONYMS_URL, "commit": SYNONYMS_COMMIT},
        "synonyms": table,
    }, indent=2))
    return table


def chair_match(caption: str, gt: list[str], synonyms: dict[str, list[str]]) -> list[dict]:
    """Return mentions: {surface, coco_category, mention_kind, char_start}.

    Matching reuses the convention of evalvitals.analyzers.hallucination.chair.
    extract_objects (space-padded word, optional plural 's') on a PUNCTUATION-
    NORMALIZED copy of the caption (punct -> space, length-preserving) — the
    package matcher alone misses every mention followed by ',' or '.', which
    would contaminate the PASS group. One mention per category: its EARLIEST
    surface occurrence (where the model first commits to the object).
    """
    norm = caption.lower().translate(PUNCT_TABLE)
    padded = f" {norm} "
    gt_set = {g.lower() for g in gt}
    mentions = []
    for cat, surfaces in synonyms.items():
        best: tuple[int, str] | None = None
        for surf in surfaces:
            for form in (surf, surf + "s"):
                idx = padded.find(f" {form} ")
                # padded = " " + norm, so the word starts at norm[idx]
                if idx != -1 and (best is None or idx < best[0]):
                    best = (idx, surf)
        if best is not None:
            mentions.append({
                "surface": best[1], "coco_category": cat, "char_start": best[0],
                "mention_kind": "grounded" if cat in gt_set else "hallucinated",
            })
    return sorted(mentions, key=lambda m: m["char_start"])


def token_index_of(offsets: list[tuple[int, int]], char_start: int) -> int | None:
    """Index (within the CAPTION encoding) of the token whose span covers
    char_start. BPE space-prefixed tokens include the leading space in their
    span, so coverage (start <= char_start < end) is the right condition."""
    for j, (s, e) in enumerate(offsets):
        if s <= char_start < e:
            return j
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explore-frac", type=float, default=0.6)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch
    import transformers
    from PIL import Image

    from evalvitals import compose
    from evalvitals.core.capability import Capability
    from evalvitals.core.case import Inputs

    inst = _instances()
    images = select_images(args.n_images, inst)
    gt_objects = extract_gt_objects(images, inst)
    synonyms = build_synonyms()
    del inst

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for img in images:
        dest = IMG_DIR / img["file_name"]
        if not dest.exists():
            urllib.request.urlretrieve(COCO_IMG_URL.format(file_name=img["file_name"]), dest)

    model = compose(args.model, "hf_local", want={Capability.GENERATE})
    # Public processor/tokenizer (the backend keeps its own private); the chat
    # template + image expansion must match the backend's encoding, so mirror
    # _encode_vlm: same content layout, add_generation_prompt, template kwargs.
    processor = transformers.AutoProcessor.from_pretrained(model.spec.hf_repo)
    tokenizer = processor.tokenizer

    records = []
    for i, img in enumerate(images):
        pil = Image.open(IMG_DIR / img["file_name"]).convert("RGB")
        caption = model.generate(
            Inputs(prompt=CAPTION_PROMPT, image=pil),
            max_new_tokens=512, do_sample=False,
        )
        mentions = chair_match(caption, gt_objects[str(img["image_id"])], synonyms)

        # --- token alignment: full sequence = templated prompt (with image
        # expansion) + re-encoded caption. Greedy decode->re-encode roundtrips
        # are not guaranteed identical, so every mention is VERIFIED below.
        content = [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **model.spec.chat_template_kwargs,
        )
        enc = processor(text=[text], images=[pil], return_tensors="pt")
        prompt_len = enc["input_ids"].shape[1]
        cap_enc = tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
        cap_ids, cap_offsets = cap_enc["input_ids"], cap_enc["offset_mapping"]

        hf_model = model._loaded[0]  # read-only access to the underlying HF module
        device = next(hf_model.parameters()).device
        full_ids = torch.cat([enc["input_ids"], torch.tensor([cap_ids])], dim=1).to(device)
        pixel_kwargs = {k: v.to(device) for k, v in enc.items()
                        if k not in ("input_ids", "attention_mask")}

        n_verified = 0
        for m in mentions:
            j = token_index_of(cap_offsets, m["char_start"])
            if j is None:
                m["token_index"], m["first_token_id"], m["verified"] = None, None, False
                continue
            k = prompt_len + j
            m["token_index"], m["first_token_id"] = k, cap_ids[j]
            with torch.no_grad():
                out = hf_model(input_ids=full_ids[:, :k],
                               attention_mask=torch.ones_like(full_ids[:, :k]),
                               **pixel_kwargs)
            m["verified"] = int(out.logits[0, -1].argmax()) == cap_ids[j]
            n_verified += m["verified"]

        # Mid-sentence first-token ids of the image's GT objects (probe targets;
        # the Eq.2 top-p candidate filter is applied at probe time, not here).
        gt_first_token_ids = {
            cat: tokenizer.encode(" " + cat, add_special_tokens=False)[0]
            for cat in gt_objects[str(img["image_id"])]
        }

        records.append({
            "image_id": img["image_id"], "file_name": img["file_name"],
            "caption": caption, "prompt_len": prompt_len,
            "caption_token_ids": cap_ids,
            "mentions": mentions,
            "hallucinated": any(m["mention_kind"] == "hallucinated" for m in mentions),
            "gt_first_token_ids": gt_first_token_ids,
        })
        n_hal = sum(m["mention_kind"] == "hallucinated" for m in mentions)
        print(f"  [{i + 1}/{len(images)}] {img['file_name']}: {len(mentions)} mentions "
              f"({n_hal} hallucinated), {n_verified}/{len(mentions)} verified")

    # Image-level 60/40 explore/validate split, stratified by `hallucinated`.
    for group in (True, False):
        idxs = [i for i, r in enumerate(records) if r["hallucinated"] is group]
        random.shuffle(idxs)
        n_explore = round(len(idxs) * args.explore_frac)
        for rank, i in enumerate(idxs):
            records[i]["split"] = "explore" if rank < n_explore else "validate"

    yields = {s: {"hallucinated": 0, "clean": 0} for s in ("explore", "validate")}
    n_mentions = {"hallucinated": 0, "grounded": 0, "verified": 0}
    for r in records:
        yields[r["split"]]["hallucinated" if r["hallucinated"] else "clean"] += 1
        for m in r["mentions"]:
            n_mentions[m["mention_kind"]] += 1
            n_mentions["verified"] += bool(m.get("verified"))

    out = DATA / "cases" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "prompt": CAPTION_PROMPT, "seed": args.seed,
        "decoding": {"do_sample": False, "max_new_tokens": 512},
        "versions": {"transformers": transformers.__version__,
                     "torch": torch.__version__, "dtype": "bfloat16"},
        "matcher": ("package chair.extract_objects convention (space-padded word, "
                    "plural 's') on punctuation-normalized text (punct->space, "
                    "length-preserving); one mention per category at its earliest "
                    "surface occurrence; CHAIR synonym table pinned in "
                    "chair_synonyms.json"),
        "synonyms_commit": SYNONYMS_COMMIT,
        "yields": yields, "mention_counts": n_mentions,
        "images": records,
    }, indent=2, ensure_ascii=False))
    print(f"frozen -> {out}")
    print(f"yields={yields}")
    print(f"mentions={n_mentions}")
    # TODO.md acceptance: hallucinated (FAIL) images >= 10; token_index
    # verification must be 100% on re-fed mentions (see per-mention 'verified').


if __name__ == "__main__":
    main()
