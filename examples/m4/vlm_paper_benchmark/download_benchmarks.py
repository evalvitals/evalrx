#!/usr/bin/env python3
"""Fetch deterministic, image-bearing slices from five public VLM benchmarks.

Only this adapter and :mod:`papers.json` are versioned. Source data, decoded
images, Hugging Face cache and sampled record JSONL stay below ``data/`` and
are ignored by git.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import itertools
import json
import random
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "papers.json"
DEFAULT_DATA = ROOT / "data"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    papers = json.loads(path.read_text(encoding="utf-8")).get("papers", [])
    if len(papers) < 5:
        raise ValueError("papers.json must contain at least five VLM papers")
    return papers


def first_value(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, "", [], {}):
            return value
    return None


def decoded_images(row: dict[str, Any], fields: Iterable[str]) -> list[Any]:
    """Decode all source images, accepting common Hugging Face schemas."""
    values = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, (list, tuple)):
            values.extend(item for item in value if item is not None)
        elif value is not None:
            values.append(value)
    images = []
    try:
        from PIL import Image

        for value in values:
            if isinstance(value, dict):
                value = value.get("image") or value.get("bytes") or value.get("path")
            if value is None:
                continue
            if isinstance(value, Image.Image):
                images.append(value.convert("RGB"))
                continue
            try:
                images.append(Image.open(value).convert("RGB"))
            except (OSError, TypeError, ValueError):
                # some sources (e.g. HR-Bench) ship images as base64 strings
                payload = io.BytesIO(base64.b64decode(value, validate=True))
                images.append(Image.open(payload).convert("RGB"))
    except Exception:
        return []
    return images


def compose_images(images: list[Any]):
    """Keep a multi-image question intact as a labelled contact sheet."""
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    from PIL import Image, ImageDraw

    cell_w, cell_h = 768, 512
    columns = 2
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), color="white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((cell_w - 20, cell_h - 36))
        x = (index % columns) * cell_w + (cell_w - thumb.width) // 2
        y = (index // columns) * cell_h + 28 + (cell_h - 36 - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text(
            (index % columns * cell_w + 8, index // columns * cell_h + 6),
            f"Image {index + 1}",
            fill="black",
        )
    return canvas


def normalize_question(value: Any) -> str:
    return re.sub(r"^\s*<image(?:\s+\d+)?>\s*", "", str(value or "")).strip()


def normalize_expected(spec: dict[str, Any], value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if spec["task"] == "vqa_consensus":
        answers = [str(item).strip() for item in value if str(item).strip()]
        return answers or None
    return str(value).strip() or None


def normalize_options(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    return []


def normalized_bbox(spec: dict[str, Any], value: Any, image: Any) -> list[float] | None:
    """Convert a paper-provided bbox to normalized xyxy coordinates.

    This field is deliberately stored under a ``paper_oracle_`` key later: it
    supports causal/oracle controls, but it is never exposed to auto-fix
    candidate generation.
    """
    if value is None or not spec.get("bbox_field"):
        return None
    try:
        x, y, a, b = (float(item) for item in value)
        width, height = image.size
    except (TypeError, ValueError, AttributeError):
        return None
    if width <= 0 or height <= 0:
        return None
    if spec.get("bbox_format") == "xywh_abs":
        x2, y2 = x + a, y + b
    elif spec.get("bbox_format") == "xyxy_abs":
        x2, y2 = a, b
    else:
        return None
    left, right = sorted((max(0.0, x), min(float(width), x2)))
    top, bottom = sorted((max(0.0, y), min(float(height), y2)))
    if right <= left or bottom <= top:
        return None
    return [left / width, top / height, right / width, bottom / height]


def reservoir_rows(
    dataset: Iterable[dict[str, Any]], n: int, scan: int, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    """Sample rows while preserving their upstream streaming positions.

    ``source_index`` must not be the final reservoir position: that position
    changes with sample size and seed and cannot support overlap exclusion.
    """
    rng = random.Random(seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    for source_index, row in enumerate(itertools.islice(dataset, scan)):
        seen = source_index + 1
        if len(reservoir) < n:
            reservoir.append((source_index, row))
        else:
            index = rng.randrange(seen)
            if index < n:
                reservoir[index] = (source_index, row)
    return reservoir


def _vstar_question_and_options(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    options = []
    question = []
    for line in lines:
        match = re.match(r"^\(([A-D])\)\s*(.+)$", line)
        if match:
            options.append(match.group(2).strip())
        elif not options and "answer with the option" not in line.lower():
            question.append(line)
    return " ".join(question), options


def records_for_vstar(
    spec: dict[str, Any], *, per_paper: int, scan_rows: int, seed: int, data_dir: Path
) -> list[dict[str, Any]]:
    """Fetch V*Bench directly from its image/JSON repository layout.

    The dataset viewer exposes image paths as strings, so the generic
    ``datasets`` streaming adapter cannot resolve them. Download only the
    selected images and annotation sidecars; both remain under ignored data/.
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image

    cache_dir = data_dir / "hf_cache"
    questions_path = hf_hub_download(
        spec["dataset"], "test_questions.jsonl", repo_type="dataset", cache_dir=str(cache_dir)
    )
    source_rows = [
        json.loads(line) for line in Path(questions_path).read_text().splitlines() if line
    ]
    candidates = reservoir_rows(source_rows, per_paper, min(scan_rows, len(source_rows)), seed)
    image_dir = data_dir / "images" / spec["id"]
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_index, row in candidates:
        relative_image = str(row["image"])
        source_image = hf_hub_download(
            spec["dataset"], relative_image, repo_type="dataset", cache_dir=str(cache_dir)
        )
        # V*Bench contains high-resolution source images.  The runner can
        # consume their native JPEG/PNG/WebP files, so do not decode and
        # re-encode each one merely to create a PNG cache.  Reading ``size``
        # only is enough to normalize the paper's bbox oracle below.
        with Image.open(source_image) as opened:
            image_size = opened.size
        sidecar_path = hf_hub_download(
            spec["dataset"],
            str(Path(relative_image).with_suffix(".json")),
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        sidecar = json.loads(Path(sidecar_path).read_text())
        question, options = _vstar_question_and_options(str(row["text"]))
        if (
            not question
            or not 2 <= len(options) <= 4
            or str(row.get("label", "")).upper() not in "ABCD"
        ):
            continue
        bbox = normalized_bbox(
            {"bbox_field": "bbox", "bbox_format": "xywh_abs"},
            (sidecar.get("bbox") or [None])[0],
            SimpleNamespace(size=image_size),
        )
        image_name = f"{len(records):05d}{Path(relative_image).suffix.lower()}"
        shutil.copyfile(source_image, image_dir / image_name)
        records.append(
            {
                "id": f"{spec['id']}-{row.get('question_id', source_index)}",
                "paper_id": spec["id"],
                "task": "multiple_choice",
                "question": question,
                "expected": str(row["label"]).upper(),
                "image": str(Path("images") / spec["id"] / image_name),
                "options": options,
                "metadata": {
                    "source_index": source_index,
                    "source_question_id": str(row.get("question_id", "")),
                    "vstar_category": row.get("category"),
                    "failure_axis": spec["failure_axis"],
                    **({"paper_oracle_bbox_xyxy_norm": bbox} if bbox is not None else {}),
                },
            }
        )
    if len(records) < per_paper:
        raise RuntimeError(
            f"{spec['id']}: obtained {len(records)}/{per_paper} usable V*Bench examples"
        )
    return records


def records_for_paper(
    spec: dict[str, Any], *, per_paper: int, scan_rows: int, seed: int, data_dir: Path
) -> list[dict[str, Any]]:
    if spec["dataset"] == "craigwu/vstar_bench":
        return records_for_vstar(
            spec, per_paper=per_paper, scan_rows=scan_rows, seed=seed, data_dir=data_dir
        )
    from datasets import load_dataset

    cache_dir = data_dir / "hf_cache"
    dataset = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        streaming=True,
        cache_dir=str(cache_dir),
    )
    candidates = reservoir_rows(
        dataset,
        per_paper * int(spec.get("oversample_factor", 3)),
        scan_rows,
        seed,
    )
    image_dir = data_dir / "images" / spec["id"]
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_index, row in candidates:
        question = normalize_question(first_value(row, spec["question_fields"]))
        expected = normalize_expected(spec, first_value(row, spec["answer_fields"]))
        images = decoded_images(row, spec["image_fields"])
        image = compose_images(images)
        if not question or expected is None or image is None:
            continue
        oracle_bbox = normalized_bbox(spec, row.get(spec.get("bbox_field", "")), image)
        if spec.get("max_bbox_area_frac") is not None:
            if oracle_bbox is None:
                continue
            left, top, right, bottom = oracle_bbox
            if (right - left) * (bottom - top) >= float(spec["max_bbox_area_frac"]):
                continue
        options = normalize_options(first_value(row, ("options", "choices")))
        if not options and spec.get("option_letter_fields"):
            letters = [str(row.get(field) or "").strip() for field in spec["option_letter_fields"]]
            if not all(letters):
                continue  # a missing lettered option would shift the answer mapping
            options = letters
        image_name = f"{len(records):05d}.png"
        image.save(image_dir / image_name, format="PNG")
        records.append(
            {
                "id": f"{spec['id']}-{source_index}",
                "paper_id": spec["id"],
                "task": spec["task"],
                "question": question,
                "expected": expected,
                "image": str(Path("images") / spec["id"] / image_name),
                "options": options,
                "metadata": {
                    "source_index": source_index,
                    "failure_axis": spec["failure_axis"],
                    "source_image_count": len(images),
                    **(
                        {"paper_oracle_bbox_xyxy_norm": oracle_bbox}
                        if oracle_bbox is not None
                        else {}
                    ),
                },
            }
        )
        if len(records) == per_paper:
            break
    if len(records) < per_paper:
        raise RuntimeError(
            f"{spec['id']}: obtained {len(records)}/{per_paper} usable image examples; "
            "increase --scan-rows or check the upstream schema"
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--per-paper", type=int, default=96)
    parser.add_argument("--scan-rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--paper", action="append", default=[], help="repeat to fetch only named ids"
    )
    args = parser.parse_args()
    if args.per_paper < 12:
        raise SystemExit("--per-paper must be >=12 for diagnosis/selection/confirmation")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        spec
        for spec in load_manifest(args.manifest)
        if not args.paper or spec["id"] in set(args.paper)
    ]
    if not selected:
        raise SystemExit("no paper id matched --paper")
    for offset, spec in enumerate(selected):
        records = records_for_paper(
            spec,
            per_paper=args.per_paper,
            scan_rows=args.scan_rows,
            seed=args.seed + offset,
            data_dir=args.data_dir,
        )
        path = args.data_dir / f"{spec['id']}.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        print(f"{spec['id']}: wrote {len(records)} records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
