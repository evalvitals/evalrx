#!/usr/bin/env python3
"""White-box ViCrop control for ``MLLMs Know Where to Look``.

This is intentionally an *example-level* paper executor, not a canned
``FixAgent`` primitive.  It makes the paper-specific assumptions explicit:
the model must expose attention, the original image and crop are supplied
together, and only a held-out paired result can admit the intervention.

The paper's released LLaVA implementation uses a relative-attention ratio,
then selects a crop with a multiscale sliding window whose peak has the
sharpest local contrast.  This executor preserves those decisions for
LLaVA-1.5; Qwen2.5-VL remains an architecture adaptation and is labelled as
such in its reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from run_autofix import DATA, OUT, diagnostic_split, evaluate, shuffled_rows, task_prompt

from evalvitals.analyzers.attention.relative_attn import attention_heatmap
from evalvitals.core.case import FailureCase, Inputs
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.specs import get_spec
from evalvitals.stats import compare

GENERAL_PROMPT = "Describe the image generally. Do not focus on any particular question or detail."


def relative_attention_map(
    model: HFLocalModel, row: dict[str, Any], layer: int | float
) -> np.ndarray:
    """Return paper-style task-relative image-patch attention for one row."""
    specific = attention_heatmap(
        model,
        FailureCase(id=str(row["id"]), inputs=Inputs(task_prompt(row), row["image"])),
        layer=layer,
    )
    general = attention_heatmap(
        model,
        FailureCase(id=f"{row['id']}-general", inputs=Inputs(GENERAL_PROMPT, row["image"])),
        layer=layer,
    )
    if specific is None or general is None or specific.shape != general.shape:
        raise RuntimeError("could not obtain compatible image attention maps")
    eps = np.finfo(np.float64).eps
    # This is the paper's relative-attention rule (specific / general), not a
    # subtraction of globally normalized maps.  The ratio boosts a region
    # only when the task prompt makes it more salient than the general prompt.
    return specific / np.maximum(general, eps)


def parse_layer(value: str) -> int | float:
    """Accept an absolute layer (``14``) or a fractional depth (``0.78``)."""
    return float(value) if any(mark in value for mark in (".", "e", "E")) else int(value)


def default_layer(model_key: str) -> int | float:
    """Officially released relative-attention layer where one is available."""
    return 14 if model_key == "llava-1.5-7b-hf" else 22


def load_records_from(data_dir: str, paper: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load an isolated local sample without changing the versioned runner.

    A new sample directory makes it possible to validate a frozen candidate on
    images that were not used while developing it.  Both source data and image
    files remain gitignored.
    """
    root = Path(data_dir)
    path = root / f"{paper}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing; run download_benchmarks.py --data-dir {root}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit is not None and len(rows) < limit:
        raise SystemExit(f"{path} has {len(rows)} rows, fewer than --limit {limit}")
    selected = rows if limit is None else rows[:limit]
    for row in selected:
        row["image"] = str(root / row["image"])
    return selected


def record_fingerprint(row: dict[str, Any]) -> str:
    """Stable identity for a local benchmark item, independent of sample order."""
    payload = {
        "task": row.get("task"),
        "question": row.get("question"),
        "expected": row.get("expected"),
        "options": row.get("options"),
        "bbox": row.get("metadata", {}).get("paper_oracle_bbox_xyxy_norm"),
        "image_sha256": hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def exclude_seen_rows(
    rows: list[dict[str, Any]], seen_data_dirs: list[str]
) -> tuple[list[dict[str, Any]], int]:
    """Drop exact previously observed benchmark items from a new local sample."""
    if not seen_data_dirs:
        return rows, 0
    seen_fingerprints = {
        record_fingerprint(row)
        for seen_data_dir in seen_data_dirs
        for row in load_records_from(seen_data_dir, "mllms_know_textvqa_small")
    }
    unseen = [row for row in rows if record_fingerprint(row) not in seen_fingerprints]
    return unseen, len(rows) - len(unseen)


def sliding_window_box(
    relative_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    bbox_size: int,
) -> tuple[float, float, float, float]:
    """Reproduce ViCrop's adaptive window rule, returned as normalized xyxy.

    The released implementation tries crop scales 1 through 2, finds the
    largest attention sum at each scale, and selects the scale whose maximum
    differs most from its four adjacent windows.  Labels and answers never
    enter this decision.
    """
    if relative_map.ndim != 2 or min(relative_map.shape) < 2:
        raise ValueError("relative attention must be a 2-D patch grid")
    h, w = relative_map.shape
    width, height = image_size
    block_width, block_height = width / w, height / h
    candidates: list[tuple[float, tuple[int, int], tuple[int, int], float]] = []
    for ratio in (1, 1.2, 1.4, 1.6, 1.8, 2):
        box_w = min(int(bbox_size * ratio / block_width), w)
        box_h = min(int(bbox_size * ratio / block_height), h)
        if w - box_w < 1 and h - box_h < 1:
            if ratio == 1:
                return 0.0, 0.0, 1.0, 1.0
            continue
        window_scores = np.empty((h - box_h + 1, w - box_w + 1), dtype=float)
        for top in range(window_scores.shape[0]):
            for left in range(window_scores.shape[1]):
                window_scores[top, left] = relative_map[
                    top : top + box_h, left : left + box_w
                ].sum()
        top, left = np.unravel_index(np.argmax(window_scores), window_scores.shape)
        adjacent: list[float] = []
        for neighbor_top, neighbor_left in (
            (top, left - 1),
            (top, left + 1),
            (top - 1, left),
            (top + 1, left),
        ):
            if (
                0 <= neighbor_top < window_scores.shape[0]
                and 0 <= neighbor_left < window_scores.shape[1]
            ):
                adjacent.append(float(window_scores[neighbor_top, neighbor_left]))
        contrast = (float(window_scores[top, left]) - float(np.mean(adjacent))) / (box_w * box_h)
        candidates.append((contrast, (left, top), (box_w, box_h), bbox_size * ratio))
    _, (left, top), (box_w, box_h), selected_bbox_size = max(candidates, key=lambda item: item[0])
    center_x = int(left * block_width + block_width * box_w / 2)
    center_y = int(top * block_height + block_height * box_h / 2)
    half = selected_bbox_size // 2
    center_x = min(max(center_x, half), width - half)
    center_y = min(max(center_y, half), height - half)
    x1, y1 = max(0, center_x - half), max(0, center_y - half)
    x2, y2 = min(width, center_x + half), min(height, center_y + half)
    return x1 / width, y1 / height, x2 / width, y2 / height


def crop_from_box(image_path: str, box: tuple[float, float, float, float]):
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, math.floor(left * width)))
    y1 = max(0, min(height - 1, math.floor(top * height)))
    x2 = max(x1 + 1, min(width, math.ceil(right * width)))
    y2 = max(y1 + 1, min(height, math.ceil(bottom * height)))
    return image.crop((x1, y1, x2, y2))


def vicrop_answer(
    model: HFLocalModel, row: dict[str, Any], layer: int | float
) -> tuple[str, dict[str, Any]]:
    rel = relative_attention_map(model, row, layer)
    from PIL import Image

    with Image.open(row["image"]) as image:
        image_size = image.size
    base_size = 336 if model.spec.key == "llava-1.5-7b-hf" else 224
    box = sliding_window_box(rel, image_size, bbox_size=base_size)
    crop = crop_from_box(row["image"], box)
    prompt = (
        "The first image is the full scene and the second is a task-relative visual crop. "
        "Use both views, prioritizing visible detail in the crop, then answer the question.\n\n"
        + task_prompt(row)
    )
    return model.generate(Inputs(prompt, [row["image"], crop])), {"relative_attention_box": box}


def summary(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    a = [bool(case["correct"]) for case in baseline["cases"]]
    b = [bool(case["correct"]) for case in candidate["cases"]]
    result = compare(a, b)
    fixed = [
        base["id"]
        for base, cand in zip(baseline["cases"], candidate["cases"])
        if not base["correct"] and cand["correct"]
    ]
    broken = [
        base["id"]
        for base, cand in zip(baseline["cases"], candidate["cases"])
        if base["correct"] and not cand["correct"]
    ]
    return {
        "n_pairs": len(a),
        "baseline_accuracy": baseline["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "fixed_cases": fixed,
        "broken_cases": broken,
        "n_fixed": len(fixed),
        "n_broken": len(broken),
        "effect": result.effect,
        "e_value": result.e_value,
        "reject": result.reject,
        "summary": result.summary(),
    }


def merge_evaluations(*evaluations: dict[str, Any]) -> dict[str, Any]:
    """Combine paired arms for an auditable sequential-evidence summary."""
    cases = [case for evaluation in evaluations for case in evaluation["cases"]]
    correct = sum(bool(case["correct"]) for case in cases)
    return {"n": len(cases), "correct": correct, "accuracy": correct / len(cases), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--diagnosis-cases", type=int, default=12)
    parser.add_argument("--selection-cases", type=int, default=24)
    parser.add_argument("--layer", type=parse_layer)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--data-dir", default=str(DATA), help="local benchmark sample directory")
    parser.add_argument(
        "--exclude-data-dir",
        action="append",
        default=[],
        help="repeat for every prior local sample to exclude by content fingerprint",
    )
    parser.add_argument(
        "--confirm-from",
        type=Path,
        help="completed report whose positive selection freezes this candidate for pure confirmation",
    )
    parser.add_argument("--output-name", default="mllms_know_hf_vicrop")
    args = parser.parse_args()
    split = args.diagnosis_cases + args.selection_cases
    if args.confirm_from is None and (
        args.diagnosis_cases < 1 or args.selection_cases < 8 or split >= args.limit
    ):
        raise SystemExit("need diagnosis >=1, selection >=8, and a held-out confirmation split")

    rows, n_excluded = exclude_seen_rows(
        load_records_from(args.data_dir, "mllms_know_textvqa_small"), args.exclude_data_dir
    )
    if len(rows) < args.limit:
        raise SystemExit(
            f"only {len(rows)} unseen records remain after exclusion, fewer than --limit {args.limit}"
        )
    rows = shuffled_rows(rows[: args.limit], args.seed)
    probe_rows, confirmation_rows = rows[:split], rows[split:]
    model = HFLocalModel(
        get_spec(args.model),
        RuntimeConfig(
            device=args.device,
            dtype="bfloat16",
            attn_impl="eager",
            max_new_tokens=args.max_tokens,
        ),
    )
    model.load()
    layer = args.layer if args.layer is not None else default_layer(args.model)
    if args.confirm_from is not None:
        prior = json.loads(args.confirm_from.read_text(encoding="utf-8"))
        prior_selection = prior.get("selection", {})
        if not (prior_selection.get("reject") and prior_selection.get("effect", 0) > 0):
            raise SystemExit("--confirm-from must contain a positive, validated selection result")
        if prior.get("model") != args.model or prior.get("layer") != layer:
            raise SystemExit("--confirm-from model/layer must match the frozen executor")
        baseline_confirmation = evaluate(
            rows, lambda row: model.generate(Inputs(task_prompt(row), row["image"]))
        )
        candidate_confirmation = evaluate(rows, lambda row: vicrop_answer(model, row, layer))
        confirmation = summary(baseline_confirmation, candidate_confirmation)
        report = {
            "paper": "mllms_know",
            "method": "relative-attention ViCrop (official ratio and adaptive window; hf_local executor)",
            "fidelity": (
                "LLaVA attention/crop selector faithfully ported; framework prompt adapter"
                if args.model == "llava-1.5-7b-hf"
                else "architecture-adapted attention/crop selector; framework prompt adapter"
            ),
            "model": args.model,
            "layer": layer,
            "data_dir": args.data_dir,
            "exclude_data_dirs": args.exclude_data_dir,
            "n_excluded_as_previously_seen": n_excluded,
            "mode": "frozen_candidate_confirmation",
            "frozen_from": str(args.confirm_from),
            "selection": {"skipped": "candidate frozen by prior validated selection"},
            "confirmation": confirmation,
            "confirmation_cases": {
                "baseline": baseline_confirmation["cases"],
                "candidate": candidate_confirmation["cases"],
            },
        }
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{args.output_name}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(path), "confirmation": confirmation}, indent=2))
        return 0
    baseline_probe = evaluate(
        probe_rows, lambda row: model.generate(Inputs(task_prompt(row), row["image"]))
    )
    diagnosis_rows, selection_rows = diagnostic_split(
        probe_rows, baseline_probe, args.diagnosis_cases
    )
    baseline_selection = {
        "n": len(selection_rows),
        "cases": [
            case
            for case in baseline_probe["cases"]
            if case["id"] in {row["id"] for row in selection_rows}
        ],
    }
    baseline_selection["correct"] = sum(case["correct"] for case in baseline_selection["cases"])
    baseline_selection["accuracy"] = baseline_selection["correct"] / len(selection_rows)
    selected = evaluate(selection_rows, lambda row: vicrop_answer(model, row, layer))
    selection = summary(baseline_selection, selected)
    confirmation: dict[str, Any] = {"skipped": "selection did not validate ViCrop"}
    sequential_evidence: dict[str, Any] | None = None
    if selection["reject"] and selection["effect"] > 0:
        baseline_confirmation = evaluate(
            confirmation_rows, lambda row: model.generate(Inputs(task_prompt(row), row["image"]))
        )
        candidate_confirmation = evaluate(
            confirmation_rows, lambda row: vicrop_answer(model, row, layer)
        )
        confirmation = summary(baseline_confirmation, candidate_confirmation)
        sequential_evidence = {
            "pooled": summary(
                merge_evaluations(baseline_selection, baseline_confirmation),
                merge_evaluations(selected, candidate_confirmation),
            ),
            "interpretation": (
                "The ViCrop configuration was fixed before this fresh sample; the pooled e-value is "
                "valid under sequential monitoring. This is not a substitute for a separately powered "
                "confirmation split."
            ),
        }
    report = {
        "paper": "mllms_know",
        "method": "relative-attention ViCrop (official ratio and adaptive window; hf_local executor)",
        "fidelity": (
            "LLaVA attention/crop selector faithfully ported; framework prompt adapter"
            if args.model == "llava-1.5-7b-hf"
            else "architecture-adapted attention/crop selector; framework prompt adapter"
        ),
        "model": args.model,
        "layer": layer,
        "data_dir": args.data_dir,
        "exclude_data_dirs": args.exclude_data_dir,
        "n_excluded_as_previously_seen": n_excluded,
        "splits": {
            "diagnosis": len(diagnosis_rows),
            "selection": len(selection_rows),
            "confirmation": len(confirmation_rows),
            "shuffle_seed": args.seed,
        },
        "selection": selection,
        "confirmation": confirmation,
        "sequential_evidence": sequential_evidence,
        # Keep the exact local split membership and paired answers so a future
        # run can exclude observed IDs.  Outputs are gitignored with the data.
        "selection_cases": {
            "baseline": baseline_selection["cases"],
            "candidate": selected["cases"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.output_name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(path), "selection": selection, "confirmation": confirmation}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
