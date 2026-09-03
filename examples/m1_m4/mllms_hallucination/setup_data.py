"""Download POPE adversarial annotations + required COCO val2014 images.

Run this once before starting the example:

    python examples/m1_m4/mllms_hallucination/setup_data.py
    python examples/m1_m4/mllms_hallucination/setup_data.py --max-images 100

The script:
  1. Downloads coco_pope_adversarial.json from the POPE GitHub repo.
  2. Parses the JSONL to find which COCO val2014 images are referenced.
  3. Downloads those images from the public COCO mirror
     (http://images.cocodataset.org/val2014/).

Data is written to /data/rjin02/evalrx/pope_coco/:
    coco_pope_adversarial.json   POPE annotations
    images/                      COCO val2014 JPEGs
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATA_ROOT = Path("/data/rjin02/evalrx/pope_coco")
IMAGE_DIR = DATA_ROOT / "images"

POPE_URL = (
    "https://raw.githubusercontent.com/AoiDragon/POPE/master/"
    "output/coco/coco_pope_adversarial.json"
)
COCO_IMAGE_BASE = "http://images.cocodataset.org/val2014/"


def _download(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    label = desc or dest.name
    print(f"  downloading {label} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done ({size_kb} KB)")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        print(f"FAILED: {exc}")
        raise


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download POPE + COCO val2014 subset")
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Download at most this many unique images (default: all ~500).",
    )
    ap.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help=f"Destination directory (default: {DATA_ROOT})",
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: POPE annotations
    print("\n[1/2] POPE adversarial annotations")
    pope_dest = root / "coco_pope_adversarial.json"
    _download(POPE_URL, pope_dest, "coco_pope_adversarial.json")

    # Step 2: parse unique image filenames
    content = pope_dest.read_text(encoding="utf-8").strip()
    if content.startswith("["):
        records = json.loads(content)
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]

    unique_images: list[str] = []
    seen: set[str] = set()
    for rec in records:
        fname = rec["image"]
        if fname not in seen:
            seen.add(fname)
            unique_images.append(fname)

    if args.max_images:
        unique_images = unique_images[: args.max_images]

    print(f"\n[2/2] COCO val2014 images ({len(unique_images)} unique files)")
    failed: list[str] = []
    for i, fname in enumerate(unique_images, 1):
        dest = img_dir / fname
        if dest.exists():
            if (i % 50) == 0:
                print(f"  {i}/{len(unique_images)} already present ...")
            continue
        url = COCO_IMAGE_BASE + fname
        try:
            urllib.request.urlretrieve(url, dest)
            if (i % 20) == 0 or i == len(unique_images):
                print(f"  {i}/{len(unique_images)} downloaded")
        except Exception as exc:
            dest.unlink(missing_ok=True)
            failed.append(fname)
            if len(failed) <= 5:
                print(f"  WARN: could not download {fname}: {exc}")

    present = sum(1 for f in unique_images if (img_dir / f).exists())
    print(f"\nDone.  {present}/{len(unique_images)} images available in {img_dir}")
    if failed:
        print(f"  {len(failed)} image(s) failed to download — they will be skipped during run.py")

    total_records = len(records)
    loadable = sum(1 for r in records if (img_dir / r["image"]).exists())
    print(f"  {loadable}/{total_records} POPE cases have images ready for run.py")


if __name__ == "__main__":
    main()
