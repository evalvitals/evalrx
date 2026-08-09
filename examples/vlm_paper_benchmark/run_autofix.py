#!/usr/bin/env python3
"""Run visual auto-fix with image-level diagnosis, selection and confirmation.

The input must be created by ``download_benchmarks.py``. This is deliberately
an evaluation runner rather than a paper-reproduction claim: the paper's data
and task define the visual failure surface; EvalVitals selects exactly one
repair candidate on held-out images and confirms it on images it never saw.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import mimetypes
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model, TokenLogprob, Trace
from evalvitals.eval_agent.hypothesis import Hypothesis, hypothesis_to_dict
from evalvitals.eval_agent.stages.fix_agent import FixAgent

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODEL_ID = os.environ.get("AUTOFIX_MODEL_ID", "gpt-qwen3-vl-8b")
BASE_URL = os.environ.get("AUTOFIX_BASE_URL", "http://127.0.0.1:8010/v1")
PAPER_IDS = tuple(item["id"] for item in json.loads((ROOT / "papers.json").read_text())["papers"])
PAPER_SPECS = {
    item["id"]: item for item in json.loads((ROOT / "papers.json").read_text())["papers"]
}


def image_data_url(image: Any) -> str:
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image
    if isinstance(image, str) and Path(image).is_file():
        # Preserve local JPEG/WebP/PNG bytes when possible. This changes only
        # transport cost, not pixels or model input; repeated PIL PNG
        # re-encoding made high-resolution V*Bench runs unnecessarily slow.
        mime, _ = mimetypes.guess_type(image)
        if mime in {"image/jpeg", "image/png", "image/webp"}:
            return (
                "data:"
                + mime
                + ";base64,"
                + base64.b64encode(Path(image).read_bytes()).decode("ascii")
            )
    from PIL import Image

    img = image if isinstance(image, Image.Image) else Image.open(image)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class VLMEndpoint(Model):
    """OpenAI-compatible VLM adapter that retains completion telemetry."""

    capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
    modalities = frozenset({"text", "image"})

    def __init__(self, client: OpenAI, model_id: str, *, max_tokens: int = 256) -> None:
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._grounding_dino: tuple[Any, Any] | None = None

    def generate_with_metadata(self, inputs: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        image = getattr(inputs, "image", None)
        content: Any = prompt
        if image is not None:
            images = image if isinstance(image, (list, tuple)) else [image]
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": image_data_url(item)}} for item in images
            )
        max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
        temperature = float(kwargs.pop("temperature", 0.0))
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = response.choices[0]
        return choice.message.content or "", {
            "finish_reason": choice.finish_reason or "unknown",
            "generation_config": {"max_tokens": max_tokens, "temperature": temperature},
        }

    def generate(self, inputs: Any, **kwargs: Any) -> str:
        return self.generate_with_metadata(inputs, **kwargs)[0]

    def logprobs(self, inputs: Any, *, top_k: int = 20, **kwargs: Any) -> list[TokenLogprob]:
        """Retrieve first-token alternatives from an OpenAI-compatible vLLM server."""
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        image = getattr(inputs, "image", None)
        content: Any = prompt
        if image is not None:
            images = image if isinstance(image, (list, tuple)) else [image]
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": image_data_url(item)}} for item in images
            )
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=int(kwargs.pop("max_tokens", 1)),
            temperature=0.0,
            logprobs=True,
            top_logprobs=min(20, max(2, int(top_k))),
            **kwargs,
        )
        items = response.choices[0].logprobs.content if response.choices[0].logprobs else []
        return [
            TokenLogprob(
                token=item.token,
                logprob=float(item.logprob),
                top={entry.token: float(entry.logprob) for entry in item.top_logprobs},
            )
            for item in items
        ]

    @staticmethod
    def _diffusion_noise(image: Any, noise_step: int) -> Any:
        """Deterministic image-space counterpart to VCD's diffusion corruption."""
        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            original = image.convert("RGB")
        else:
            with Image.open(image) as opened:
                original = opened.convert("RGB")
        pixels = np.asarray(original, dtype=np.float32) / 255.0
        step = max(0, min(999, int(noise_step)))
        # Match VCD's released ``add_diffusion_noise`` schedule (a sigmoid
        # beta schedule from 1e-5 to 5e-3), rather than substituting the
        # common DDPM linear schedule.  The noise source is deterministically
        # seeded per image so paired evaluation is reproducible.
        betas = (1.0 / (1.0 + np.exp(-np.linspace(-6.0, 6.0, 1000, dtype=np.float32)))) * (
            0.5e-2 - 1e-5
        ) + 1e-5
        alpha_bar = float(np.cumprod(1.0 - betas)[step])
        digest = hashlib.sha256(pixels.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        noisy = alpha_bar**0.5 * pixels + (1.0 - alpha_bar) ** 0.5 * rng.standard_normal(
            pixels.shape
        )
        return Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype("uint8"), mode="RGB")

    @staticmethod
    def _answer_logprob(tokens: list[TokenLogprob], answer: str) -> float:
        answer = answer.lower()
        values = [
            value
            for item in tokens[:1]
            for token, value in {item.token: item.logprob, **item.top}.items()
            if token.strip().lower() == answer
        ]
        return max(values) if values else -100.0

    def generate_vcd(
        self, inputs: Any, *, alpha: float = 0.5, beta: float = 0.1, noise_step: int = 500
    ) -> str:
        """VCD for a binary answer: contrast original/noised first-token logits.

        The paper's decoding formula is exact for a one-token Yes/No decision;
        we intentionally reject open generation rather than approximate every
        later token with an unrelated API request.
        """
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("VCD requires one image and a binary answer task")
        clean = self.logprobs(inputs, max_tokens=1)
        distorted = self.logprobs(
            Inputs(prompt=inputs.prompt, image=self._diffusion_noise(image, noise_step)),
            max_tokens=1,
        )
        clean_scores = {answer: self._answer_logprob(clean, answer) for answer in ("yes", "no")}
        distorted_scores = {
            answer: self._answer_logprob(distorted, answer) for answer in ("yes", "no")
        }
        threshold = max(clean_scores.values()) + math.log(float(beta))
        scores = {
            answer: (1.0 + float(alpha)) * clean_scores[answer]
            - float(alpha) * distorted_scores[answer]
            for answer in clean_scores
            if clean_scores[answer] >= threshold
        }
        if not scores:
            scores = clean_scores
        return max(scores, key=scores.get).title()

    @staticmethod
    def _crop_from_box(image: Any, cx: float, cy: float, side: float) -> Any:
        """Crop a normalized square while preserving a valid image boundary."""
        from PIL import Image

        if isinstance(image, Image.Image):
            original = image.convert("RGB")
        else:
            with Image.open(image) as opened:
                original = opened.convert("RGB")
        width, height = original.size
        crop_side = max(1, int(min(width, height) * side))
        x = int(cx * width - crop_side / 2)
        y = int(cy * height - crop_side / 2)
        x = max(0, min(width - crop_side, x))
        y = max(0, min(height - crop_side, y))
        return original.crop((x, y, x + crop_side, y + crop_side))

    @staticmethod
    def _crop_from_xyxy(image: Any, box: Any, padding: float = 0.15) -> Any:
        """Crop an absolute detector box with proportional context padding."""
        from PIL import Image

        if isinstance(image, Image.Image):
            original = image.convert("RGB")
        else:
            with Image.open(image) as opened:
                original = opened.convert("RGB")
        width, height = original.size
        x1, y1, x2, y2 = (float(value) for value in box)
        pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
        return original.crop(
            (
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(width, int(x2 + pad_x)),
                min(height, int(y2 + pad_y)),
            )
        )

    def _grounding_dino_detector(self) -> tuple[Any, Any]:
        """Lazy local open-vocabulary detector used by the V* control."""
        if self._grounding_dino is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            repo = "IDEA-Research/grounding-dino-tiny"
            processor = AutoProcessor.from_pretrained(repo)
            detector = AutoModelForZeroShotObjectDetection.from_pretrained(repo).to("cuda").eval()
            self._grounding_dino = (detector, processor)
        return self._grounding_dino

    @staticmethod
    def _parse_search_targets(text: str) -> list[str]:
        """Extract 1--3 concrete detector prompts from a controller response."""
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group()).get("targets", [])
                if isinstance(value, list):
                    cleaned = [str(item).strip(" .\t\n") for item in value]
                    return [item for item in cleaned if 1 <= len(item) <= 80][:3]
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                pass
        candidates = [part.strip(" -•.\t") for part in re.split(r"[,\n]", text)]
        return [item for item in candidates if 1 <= len(item) <= 80][:3]

    def _detect_target_crop(self, image: Any, target: str) -> tuple[Any, float]:
        import torch
        from PIL import Image

        detector, processor = self._grounding_dino_detector()
        if isinstance(image, Image.Image):
            original = image.convert("RGB")
        else:
            with Image.open(image) as opened:
                original = opened.convert("RGB")
        encoded = processor(images=original, text=target.strip() + " .", return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            output = detector(**encoded)
        result = processor.post_process_grounded_object_detection(
            output,
            encoded.input_ids,
            threshold=0.18,
            text_threshold=0.18,
            target_sizes=[original.size[::-1]],
        )[0]
        if len(result["boxes"]) == 0:
            raise ValueError(f"detector found no box for {target!r}")
        index = int(result["scores"].argmax())
        return self._crop_from_xyxy(original, result["boxes"][index]), float(
            result["scores"][index]
        )

    @staticmethod
    def _parse_visual_search_box(
        text: str,
        *,
        min_side: float,
        max_side: float,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[float, float, float] | None:
        """Parse normalized or pixel tight boxes (or legacy centre/side)."""
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            box = json.loads(match.group())
            if {"x1", "y1", "x2", "y2"} <= set(box):
                x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
                if max(x1, y1, x2, y2) > 1.0:
                    if image_size is None:
                        return None
                    width, height = image_size
                    if width <= 0 or height <= 0:
                        return None
                    x1, x2 = x1 / width, x2 / width
                    y1, y2 = y1 / height, y2 / height
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    return None
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = max(x2 - x1, y2 - y1)
            else:
                cx, cy, side = (float(box[key]) for key in ("cx", "cy", "side"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            return None
        return cx, cy, max(float(min_side), min(float(max_side), side))

    def generate_visual_search(
        self,
        inputs: Any,
        *,
        min_side: float = 0.25,
        max_side: float = 0.70,
        scales: list[float] | None = None,
        decision: str | None = None,
        baseline_answer: str | None = None,
    ) -> str:
        """Black-box V* control: question-guided locate -> crop -> answer.

        This is a label-free endpoint-compatible control in the V* method
        family. It is not represented as the paper's trained SEAL module.
        """
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("guided visual search requires exactly one image")
        from PIL import Image

        with Image.open(image) as opened:
            image_size = opened.size
        controller_prompt = (
            "You control a zoom lens for visual question answering. Inspect the image and "
            "find the specific object or objects named in the task that determine the answer. "
            "Return exactly one JSON object, with no prose, using the keys x1, y1, x2, y2. "
            "They are normalized top-left and bottom-right coordinates in [0, 1]. The box "
            "must tightly contain all answer-relevant objects; do not default to the image "
            "centre and do not answer the task.\n\nTask:\n" + inputs.prompt
        )
        controller = self.generate(Inputs(prompt=controller_prompt, image=image), max_tokens=96)
        box = self._parse_visual_search_box(
            controller,
            min_side=float(min_side),
            max_side=float(max_side),
            image_size=image_size,
        )
        if box is None:
            raise ValueError("visual-search controller did not return a valid crop")
        cx, cy, _ = box
        requested_scales = scales or [box[2]]
        crops = [
            self._crop_from_box(
                image, cx, cy, max(float(min_side), min(float(max_side), float(scale)))
            )
            for scale in requested_scales
        ]
        if decision == "unanimous_crop_override" and baseline_answer is not None:
            # Conservative self-consistency gate: visual search may change a
            # stable baseline only when every independent crop view returns
            # the same answer.  It uses no labels and leaves ambiguity alone.
            crop_answers = [
                self.generate(Inputs(prompt=inputs.prompt, image=[image, crop])) for crop in crops
            ]
            parsed = [parsed_choice(answer) for answer in crop_answers]
            if len(parsed) == len(crops) and len(set(parsed)) == 1 and parsed[0]:
                return parsed[0]
            return baseline_answer
        answer_prompt = (
            "The first image is the full scene. The remaining images are progressively "
            "sized, question-guided crops around the same target. Use all views and prefer "
            "visible evidence over priors.\n\n" + inputs.prompt
        )
        return self.generate(Inputs(prompt=answer_prompt, image=[image, *crops]))

    def generate_detector_visual_search(
        self,
        inputs: Inputs,
        *,
        baseline_answer: str | None = None,
        decision: str = "unanimous_crop_override",
    ) -> str:
        """V*-style target extraction -> open-vocabulary detection -> query.

        This is a label-free method-family control.  The paper's trained visual
        search model is not substituted or claimed here.
        """
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("detector visual search requires exactly one image")
        controller_prompt = (
            "For the visual question below, name one to three concrete visible objects or "
            "regions that must be inspected to answer it. Do not answer the question and do "
            'not give coordinates. Return exactly JSON: {"targets": ["object phrase"]}.\n\n'
            "Question:\n" + inputs.prompt
        )
        controller = self.generate(Inputs(controller_prompt, image), max_tokens=80)
        targets = self._parse_search_targets(controller)
        if not targets:
            raise ValueError("visual-search controller returned no detector targets")
        crops: list[Any] = []
        for target in targets:
            try:
                crop, _score = self._detect_target_crop(image, target)
                crops.append(crop)
            except ValueError:
                continue
        if not crops:
            raise ValueError("detector returned no usable target crop")
        prompt = (
            "The first image is the full scene. The later image(s) are localized visual "
            "evidence for objects needed by the question. Use all views and return only the "
            "requested answer.\n\n" + inputs.prompt
        )
        candidate = self.generate(Inputs(prompt, [image, *crops]))
        if decision == "unanimous_crop_override" and baseline_answer is not None:
            # One detector crop can be a false positive. Make an independent
            # crop-only query and retain the baseline unless both agree.
            crop_only = self.generate(Inputs(inputs.prompt, crops[0]))
            first, second = parsed_choice(candidate), parsed_choice(crop_only)
            if first and first == second:
                return first
            return baseline_answer
        return candidate

    def forward(self, inputs: Any, capture: set[Capability], spec: Any = None) -> Trace:
        raise NotImplementedError("the VLM paper runner is generation-only")


def load_records_from(
    data_dir: str | Path, paper: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load a local sample without assuming it is the canonical data directory."""
    root = Path(data_dir)
    path = root / f"{paper}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} missing; run download_benchmarks.py --paper {paper} --data-dir {root}"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit is not None and len(rows) < limit:
        raise SystemExit(f"{path} has {len(rows)} rows, fewer than --limit {limit}")
    selected = rows if limit is None else rows[:limit]
    for row in selected:
        row["image"] = str(root / row["image"])
    return selected


def record_fingerprint(row: dict[str, Any]) -> str:
    """Content identity independent of reservoir order or an adapter-local ID."""
    payload = {
        "task": row.get("task"),
        "question": row.get("question"),
        "expected": row.get("expected"),
        "options": row.get("options"),
        "metadata": {
            key: value
            for key, value in (row.get("metadata") or {}).items()
            if key not in {"source_index"}
        },
        "image_sha256": hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def exclude_seen_rows(
    rows: list[dict[str, Any]], paper: str, seen_data_dirs: list[str]
) -> tuple[list[dict[str, Any]], int]:
    """Remove exact items already observed in any prior local sample."""
    if not seen_data_dirs:
        return rows, 0
    seen = {
        record_fingerprint(row)
        for data_dir in seen_data_dirs
        for row in load_records_from(data_dir, paper)
    }
    unseen = [row for row in rows if record_fingerprint(row) not in seen]
    return unseen, len(rows) - len(unseen)


def load_records(paper: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Back-compatible canonical-sample loader."""
    return load_records_from(DATA, paper, limit)


def normalized(text: Any) -> str:
    value = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. -]", "", str(text).lower())).strip()
    # TextVQA answers frequently differ only in typography around a unit
    # (e.g. ``345ml`` vs ``345 ml``); that is not a visual recognition error.
    return re.sub(r"(?<=\d)\s+(?=[a-z])", "", value)


def parsed_yes_no(output: str) -> str:
    match = re.search(r"\b(yes|no)\b", output.lower())
    return match.group(1) if match else ""


def parsed_choice(output: str) -> str:
    marked = re.findall(r"(?:answer|choice|final)\s*[:=-]?\s*([A-D])\b", output.upper())
    letters = re.findall(r"\b([A-D])\b", output.upper())
    return marked[-1] if marked else (letters[-1] if letters else "")


def score(row: dict[str, Any], output: str) -> bool:
    task = row["task"]
    if task == "yes_no":
        return parsed_yes_no(output) == normalized(row["expected"])
    if task == "multiple_choice":
        return parsed_choice(output) == str(row["expected"]).strip().upper()
    if task == "vqa_consensus":
        expected = [normalized(answer) for answer in row["expected"]]
        return expected.count(normalized(output)) >= 3
    expected = normalized(row["expected"])
    observed = normalized(output)
    try:
        return abs(float(expected) - float(observed)) < 1e-4
    except ValueError:
        return observed == expected


def task_prompt(row: dict[str, Any]) -> str:
    """The common, task-valid prompt used by every experimental arm.

    This is deliberately not counted as an auto-fix: benchmark answer-format
    instructions belong in the baseline, otherwise an L1 prompt candidate can
    appear to repair serialization rather than visual reasoning.
    """
    task = row["task"]
    if task == "yes_no":
        suffix = "Inspect the image and reply with exactly Yes or No."
    elif task == "multiple_choice":
        suffix = "Inspect the image and reply with only the option letter shown in the choices."
    else:
        suffix = (
            "Inspect the image and reply with only the requested final answer, without explanation."
        )
    options = row.get("options") or []
    option_text = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options[:4])
    )
    return f"{row['question']}\n\n{option_text}\n\n{suffix}".strip()


def evaluate(
    rows: list[dict[str, Any]], strategy: Callable[[dict[str, Any]], Any]
) -> dict[str, Any]:
    cases = []
    for row in rows:
        result = strategy(row)
        output, telemetry = result if isinstance(result, tuple) else (result, {})
        cases.append(
            {"id": row["id"], "output": output, "correct": score(row, output), **telemetry}
        )
    correct = sum(case["correct"] for case in cases)
    return {"n": len(cases), "correct": correct, "accuracy": correct / len(cases), "cases": cases}


def subset_evaluation(baseline: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Project one probe baseline onto a split without re-sampling the VLM."""
    cases_by_id = {case["id"]: case for case in baseline["cases"]}
    cases = [cases_by_id[row["id"]] for row in rows]
    correct = sum(case["correct"] for case in cases)
    return {"n": len(cases), "correct": correct, "accuracy": correct / len(cases), "cases": cases}


def diagnostic_split(
    rows: list[dict[str, Any]], baseline: dict[str, Any], diagnosis_cases: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make diagnosis informative while retaining a separate selection split.

    The probe includes only diagnosis+selection rows. Its labels never reach a
    prompt or candidate; they only ensure that the diagnosis partition contains
    observed failures when the pool has them. Confirmation rows are untouched.
    """
    correct_by_id = {case["id"]: case["correct"] for case in baseline["cases"]}
    failures = [row for row in rows if not correct_by_id[row["id"]]]
    passes = [row for row in rows if correct_by_id[row["id"]]]
    # Preserve at least one failure for selection whenever the probe has two.
    n_fail = min(len(failures), max(1, diagnosis_cases // 2))
    if len(failures) > 1:
        n_fail = min(n_fail, len(failures) - 1)
    diagnosis = failures[:n_fail]
    diagnosis.extend(passes[: diagnosis_cases - len(diagnosis)])
    if len(diagnosis) < diagnosis_cases:
        diagnosis.extend(failures[n_fail:diagnosis_cases])
    diagnosis_ids = {row["id"] for row in diagnosis}
    selection = [row for row in rows if row["id"] not in diagnosis_ids]
    return diagnosis, selection


def shuffled_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return a deterministic permutation before any diagnosis/selection split.

    Several public datasets are grouped by source, question type, or
    adversarial construction.  Taking contiguous slices can otherwise make a
    diagnosis partition absorb nearly all observed failures and leave the
    held-out selection partition under-powered.  The permutation is fixed and
    recorded in the report; labels are never used to construct it.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def reference_prompt(row: dict[str, Any]) -> str:
    axis = row["metadata"]["failure_axis"]
    return (
        f"Inspect the image carefully for {axis}. Work from visible evidence, then give only "
        f"the final answer requested.\n\n{task_prompt(row)}"
    )


def oracle_human_crop_pair(row: dict[str, Any]) -> list[Any]:
    """Original image plus a paper-supplied answer-region crop (oracle only)."""
    bbox = row["metadata"]["paper_oracle_bbox_xyxy_norm"]
    from PIL import Image

    with Image.open(row["image"]) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    left, top, right, bottom = bbox
    x1, y1 = int(left * width), int(top * height)
    x2, y2 = int(right * width), int(bottom * height)
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, min(width - side, cx - side // 2))
    y1 = max(0, min(height - side, cy - side // 2))
    return [row["image"], image.crop((x1, y1, x1 + side, y1 + side))]


def oracle_human_crop_prompt(row: dict[str, Any]) -> str:
    return (
        "The first image is the original scene; the second is an oracle crop around "
        "the answer-bearing visual detail. Use both views, then answer the question.\n\n"
        + task_prompt(row)
    )


def make_cases(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    prompt_fn: Callable[[dict[str, Any]], str] = task_prompt,
) -> CaseBatch:
    """Create auto-fix cases using the exact prompt used for the baseline.

    Most benchmark runs use :func:`task_prompt`.  Paper reproductions can
    instead supply the source paper's prompt contract; keeping that choice at
    the call site prevents a candidate from being evaluated against a
    different prompt than the recorded baseline.
    """
    baseline_by_id = {case["id"]: case for case in baseline["cases"]}
    return CaseBatch(
        FailureCase(
            id=row["id"],
            inputs=Inputs(prompt=prompt_fn(row), image=row["image"]),
            expected=row["expected"],
            observed=baseline_by_id[row["id"]]["output"],
            label=Label.PASS if score(row, baseline_by_id[row["id"]]["output"]) else Label.FAIL,
            metadata={
                **{
                    key: value
                    for key, value in row["metadata"].items()
                    if not key.startswith("paper_oracle_")
                },
                "task": row["task"],
                "options": row.get("options", []),
                "finish_reason": baseline_by_id[row["id"]].get("finish_reason"),
                "generation_config": baseline_by_id[row["id"]].get("generation_config", {}),
            },
        )
        for row in rows
    )


def score_case(case: FailureCase, output: str) -> bool:
    return score({"task": case.metadata["task"], "expected": case.expected}, output)


def diagnose(
    model: VLMEndpoint, baseline: dict[str, Any], rows: list[dict[str, Any]]
) -> Hypothesis:
    failure_rows = [
        {"question": row["question"][:280], "output": case["output"][:280]}
        for row, case in zip(rows, baseline["cases"])
        if not case["correct"]
    ][:8]
    failed_examples = [
        (row, case) for row, case in zip(rows, baseline["cases"]) if not case["correct"]
    ]
    if not failure_rows:
        text = "No failures in the diagnosis split."
    else:
        representative, _ = failed_examples[0]
        text = model.generate(
            Inputs(
                "Infer one narrow visual failure mechanism from these incorrect image-question "
                "responses. The attached image is the first failure: inspect it to ground the "
                "diagnosis. Do not propose a fix or mention unavailable hidden model state.\n\n"
                + json.dumps(failure_rows, ensure_ascii=False),
                representative["image"],
            )
        )
    return Hypothesis(
        statement=text[:1200],
        target_model=MODEL_ID,
        predicted_failure_mode="image-grounded VLM failure",
        metadata={"fix_tier": "L2"},
    )


def method_comparison(
    paper: str, rows: list[dict[str, Any]], selection: Any, oracle: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not rows or "paper_oracle_bbox_xyxy_norm" not in rows[0]["metadata"]:
        return None
    spec = PAPER_SPECS[paper]
    paper_method = str(spec.get("paper_method", "paper-provided crop control"))
    if paper == "mllms_know_textvqa_small":
        tier, matched_name, limitation = (
            "L3a",
            None,
            "The OpenAI-compatible endpoint exposes no attention/gradient maps, so this "
            "runner cannot execute or claim the paper's internal ViCrop method.",
        )
    elif paper == "vstar_bench":
        tier, matched_name, limitation = (
            "L2",
            "guided_visual_search",
            "The paper's trained SEAL components are unavailable through this endpoint; "
            "guided_visual_search is a label-free black-box method-family control only.",
        )
    else:
        tier, matched_name, limitation = "L2", None, "No exact paper-method adapter is registered."
    candidate = selection.best.candidate if selection.best is not None else None
    if candidate is None:
        alignment = "no auto-fix was statistically validated; do not claim paper-method agreement"
    elif candidate.name == matched_name:
        alignment = "method-family match; exact reproduction remains subject to access limits"
    else:
        alignment = "selected repair differs from the paper method"
    return {
        "paper_method": paper_method,
        "required_intervention_tier": tier,
        "oracle_control": oracle,
        "auto_fix_candidate": candidate.name if candidate else None,
        "alignment": alignment,
        "black_box_limitation": limitation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", choices=PAPER_IDS)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--diagnosis-cases", type=int, default=24)
    parser.add_argument("--selection-cases", type=int, default=36)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--auto-candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        help="local prior report whose baseline case ids must be excluded before splitting",
    )
    parser.add_argument(
        "--output-name",
        help="output stem under outputs/ (defaults to the paper id)",
    )
    parser.add_argument(
        "--max-tier",
        default="L2",
        choices=("L0", "L1", "L2", "L3a", "L3b", "L4"),
        help="bound auto-fix intervention space; use L0 to isolate decoding methods such as VCD",
    )
    parser.add_argument(
        "--paper-methods-only",
        action="store_true",
        help="test only paper-aligned candidates declared by the active backend",
    )
    args = parser.parse_args()
    split = args.diagnosis_cases + args.selection_cases
    if args.diagnosis_cases < 1 or args.selection_cases < 8 or split >= args.limit:
        raise SystemExit("need diagnosis >=1, selection >=8 and a non-empty confirmation split")

    available_rows = load_records(args.paper)
    if args.exclude_report:
        prior = json.loads(args.exclude_report.read_text(encoding="utf-8"))
        seen_ids = {
            case["id"]
            for split in prior.get("baseline", {}).values()
            if isinstance(split, dict)
            for case in split.get("cases", [])
        }
        available_rows = [row for row in available_rows if row["id"] not in seen_ids]
    if len(available_rows) < args.limit:
        raise SystemExit(
            f"only {len(available_rows)} unseen rows remain, fewer than --limit {args.limit}"
        )
    rows = shuffled_rows(available_rows, args.seed)[: args.limit]
    probe_rows, confirm_rows = rows[:split], rows[split:]
    model = VLMEndpoint(
        OpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=900),
        MODEL_ID,
        max_tokens=args.max_tokens,
    )
    baseline_probe = evaluate(
        probe_rows,
        lambda row: model.generate_with_metadata(Inputs(task_prompt(row), row["image"])),
    )
    diagnosis_rows, selection_rows = diagnostic_split(
        probe_rows, baseline_probe, args.diagnosis_cases
    )
    baseline_diagnosis = subset_evaluation(baseline_probe, diagnosis_rows)
    hypothesis = diagnose(model, baseline_diagnosis, diagnosis_rows)
    baseline_selection = subset_evaluation(baseline_probe, selection_rows)

    agent = FixAgent(
        judge=model,
        max_tier=args.max_tier,
        score_fn=score_case,
        max_validation_cases=0,
        max_judge_candidates=args.auto_candidates,
        allow_codegen=False,
        paper_methods_only=args.paper_methods_only,
    )
    selection = agent.propose_and_validate(
        model, make_cases(selection_rows, baseline_selection), [hypothesis]
    )
    confirmation: dict[str, Any] = {"skipped": "no selection candidate"}
    baseline_confirm = None
    reference = None
    oracle = None
    if selection.best is not None:
        # Confirmation and diagnostic controls must be paid only after a
        # candidate has passed the independent selection gate.  Running them
        # earlier both wastes expensive VLM calls and tempts callers to peek
        # at held-out evidence while still iterating on a candidate.
        baseline_confirm = evaluate(
            confirm_rows,
            lambda row: model.generate_with_metadata(Inputs(task_prompt(row), row["image"])),
        )
        reference = evaluate(
            confirm_rows, lambda row: model.generate(Inputs(reference_prompt(row), row["image"]))
        )
        if confirm_rows and "paper_oracle_bbox_xyxy_norm" in confirm_rows[0]["metadata"]:
            oracle = evaluate(
                confirm_rows,
                lambda row: model.generate(
                    Inputs(oracle_human_crop_prompt(row), oracle_human_crop_pair(row))
                ),
            )
        validated = FixAgent(score_fn=score_case).validate_candidate(
            model, make_cases(confirm_rows, baseline_confirm), selection.best.candidate
        )
        paired_baseline_accuracy = (
            validated.n_baseline_correct / validated.n_pairs if validated.n_pairs else None
        )
        candidate_accuracy = (
            validated.n_candidate_correct / validated.n_pairs if validated.n_pairs else None
        )
        confirmation = {
            "candidate": selection.best.candidate.name,
            "tier": selection.best.candidate.tier.label,
            "payload": selection.best.candidate.payload,
            "fixed": validated.fixed,
            "n_fixed": validated.n_fixed,
            "n_broken": validated.n_broken,
            "effect": validated.effect,
            "e_value": validated.e_value,
            "paired_baseline_accuracy": paired_baseline_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "accuracy_delta": (
                candidate_accuracy - paired_baseline_accuracy
                if candidate_accuracy is not None and paired_baseline_accuracy is not None
                else None
            ),
            "summary": validated.summary,
        }
    report = {
        "paper": args.paper,
        "model": MODEL_ID,
        "splits": {
            "diagnosis": len(diagnosis_rows),
            "selection": len(selection_rows),
            "confirmation": len(confirm_rows),
            "shuffle_seed": args.seed,
        },
        "baseline": {
            "diagnosis": baseline_diagnosis,
            "selection": baseline_selection,
            "confirmation": baseline_confirm,
        },
        "reference_prompt": reference,
        "paper_oracle_control": oracle,
        "hypothesis": hypothesis_to_dict(hypothesis),
        "auto_fix": {"selection": selection.to_dict(), "confirmation": confirmation},
        "paper_method_comparison": method_comparison(args.paper, confirm_rows, selection, oracle),
        "paper_methods_only": args.paper_methods_only,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.output_name or args.paper}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "paper": args.paper,
                "baseline": baseline_confirm["accuracy"] if baseline_confirm is not None else None,
                "auto_fix": confirmation.get("candidate_accuracy"),
                "fixed": confirmation.get("fixed", False),
                "output": str(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
