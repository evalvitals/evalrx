"""LLaVA relative-attention ViCrop executor from *MLLMs Know Where to Look*.

The implementation is intentionally limited to the paper's LLaVA-style
attention selector: task/general attention ratio, adaptive local-contrast
window, then answer from original image plus crop.  It is used by the local
backend and can therefore be proposed as a read-only L3a auto-fix candidate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from evalrx.analyzers.attention.relative_attn import attention_heatmap
from evalrx.core.case import FailureCase, Inputs

if TYPE_CHECKING:
    from evalrx.models.backends.hf_local import HFLocalModel


GENERAL_PROMPT = "Describe the image generally. Do not focus on any particular question or detail."


def relative_attention_map(model: "HFLocalModel", inputs: Inputs, layer: int | float) -> np.ndarray:
    """Paper-style ratio of task-specific to general image-patch attention."""
    image = getattr(inputs, "image", None)
    if image is None or isinstance(image, (list, tuple)):
        raise ValueError("ViCrop requires exactly one source image")
    prompt = str(getattr(inputs, "prompt", ""))
    specific = attention_heatmap(
        model, FailureCase(id="vicrop-specific", inputs=Inputs(prompt, image)), layer=layer
    )
    general = attention_heatmap(
        model, FailureCase(id="vicrop-general", inputs=Inputs(GENERAL_PROMPT, image)), layer=layer
    )
    if specific is None or general is None or specific.shape != general.shape:
        raise RuntimeError("could not obtain compatible task/general image attention maps")
    return specific / np.maximum(general, np.finfo(np.float64).eps)


def sliding_window_box(
    relative_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    bbox_size: int,
) -> tuple[float, float, float, float]:
    """Released ViCrop adaptive-window rule, returned as normalized xyxy."""
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
        scores = np.empty((h - box_h + 1, w - box_w + 1), dtype=float)
        for top in range(scores.shape[0]):
            for left in range(scores.shape[1]):
                scores[top, left] = relative_map[top : top + box_h, left : left + box_w].sum()
        top, left = np.unravel_index(np.argmax(scores), scores.shape)
        neighbors = [
            float(scores[neighbor_top, neighbor_left])
            for neighbor_top, neighbor_left in (
                (top, left - 1), (top, left + 1), (top - 1, left), (top + 1, left)
            )
            if 0 <= neighbor_top < scores.shape[0] and 0 <= neighbor_left < scores.shape[1]
        ]
        contrast = (float(scores[top, left]) - float(np.mean(neighbors))) / (box_w * box_h)
        candidates.append((contrast, (left, top), (box_w, box_h), bbox_size * ratio))
    _, (left, top), (box_w, box_h), selected_size = max(candidates, key=lambda item: item[0])
    center_x = int(left * block_width + block_width * box_w / 2)
    center_y = int(top * block_height + block_height * box_h / 2)
    half = selected_size // 2
    center_x = min(max(center_x, half), width - half)
    center_y = min(max(center_y, half), height - half)
    return (
        max(0, center_x - half) / width,
        max(0, center_y - half) / height,
        min(width, center_x + half) / width,
        min(height, center_y + half) / height,
    )


def crop_from_box(image: Any, box: tuple[float, float, float, float]):
    """Load one source image and return its selected crop."""
    from PIL import Image

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
    width, height = rgb.size
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, math.floor(left * width)))
    y1 = max(0, min(height - 1, math.floor(top * height)))
    x2 = max(x1 + 1, min(width, math.ceil(right * width)))
    y2 = max(y1 + 1, min(height, math.ceil(bottom * height)))
    return rgb.crop((x1, y1, x2, y2))


def prepare_views(
    model: "HFLocalModel", inputs: Inputs, *, layer: int | float = 14
) -> tuple[Any, Any, str]:
    """Return source image, selected crop, and ViCrop's original+crop prompt."""
    image = getattr(inputs, "image", None)
    if image is None or isinstance(image, (list, tuple)):
        raise ValueError("ViCrop requires exactly one source image")
    relative = relative_attention_map(model, inputs, layer)
    from PIL import Image

    if isinstance(image, Image.Image):
        image_size = image.size
    else:
        with Image.open(image) as opened:
            image_size = opened.size
    box = sliding_window_box(relative, image_size, bbox_size=336)
    crop = crop_from_box(image, box)
    prompt = (
        "The first image is the full scene and the second is a task-relative visual crop. "
        "Use both views, prioritizing visible detail in the crop, then answer the question.\n\n"
        + str(getattr(inputs, "prompt", ""))
    )
    return image, crop, prompt


def generate(model: "HFLocalModel", inputs: Inputs, *, layer: int | float = 14) -> str:
    """Answer from original image and a task-relative ViCrop crop."""
    image, crop, prompt = prepare_views(model, inputs, layer=layer)
    return str(model.generate(Inputs(prompt, [image, crop])))
