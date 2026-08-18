"""L2 scaffold tool catalog + pipeline executor for the fix module.

An L2 candidate fix wraps the *unchanged* model in a small pipeline: image
preprocessing tools applied to the case image, an optional prompt template,
and optional multi-sample aggregation.  Tools are a registered catalog (the
judge selects and parameterises them — same select-from-catalog pattern as
M1's analyzers); pipeline specs are plain dicts so they serialise into run
logs and can be re-executed.

PIL is imported lazily — the core package does not depend on pillow; any
environment that loads images for a VLM already has it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from evalvitals.core.case import FailureCase
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_to_bool(value: Any) -> "Optional[bool]":
    """Normalize scorer outputs to the fix-module success contract.

    The fix module accepts user-provided scorers.  Some examples reuse
    ``CaseDiscoveryAgent`` scorers that return ``Label.PASS`` / ``Label.FAIL``
    instead of bare booleans.  Enum instances are truthy in Python, including
    ``Label.FAIL``, so every fix executor must normalize before doing boolean
    algebra or passing vectors into statistical tests.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    text = str(getattr(value, "value", value)).strip().lower()
    if text in {"pass", "passed", "true", "correct", "ok", "1", "yes"}:
        return True
    if text in {"fail", "failed", "false", "incorrect", "wrong", "0", "no"}:
        return False
    if text in {"unknown", "none", "unscored", ""}:
        return None

    if isinstance(value, (int, float)):
        return bool(value)
    return None


# ---------------------------------------------------------------------------
# Image tool catalog
# ---------------------------------------------------------------------------

def _as_pil(image: Any):
    """Decode *image* (PIL.Image | path/URL) to a PIL image, or ``None``."""
    if image is None:
        return None
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    try:
        return Image.open(str(image))
    except Exception as exc:
        logger.debug("fix_tools: cannot decode image %r: %s", image, exc)
        return None


def zoom_center(img, factor: float = 1.5):
    """Crop the central 1/factor region and resize back to the original size."""
    factor = max(1.0, float(factor))
    w, h = img.size
    cw, ch = int(w / factor), int(h / factor)
    left, top = (w - cw) // 2, (h - ch) // 2
    from PIL import Image

    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def enhance_contrast(img, factor: float = 1.5):
    from PIL import ImageEnhance

    return ImageEnhance.Contrast(img).enhance(float(factor))


def sharpen(img, factor: float = 2.0):
    from PIL import ImageEnhance

    return ImageEnhance.Sharpness(img).enhance(float(factor))


def equalize(img):
    """Histogram equalization (high dynamic-range scans, e.g. radiology)."""
    from PIL import ImageOps

    return ImageOps.equalize(img.convert("RGB"))


def upscale(img, factor: float = 2.0):
    factor = max(1.0, float(factor))
    from PIL import Image

    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)


def crop_region(img, box=(0.25, 0.25, 0.75, 0.75)):
    """Crop a normalized (left, top, right, bottom) box, resize to original size.

    Like :func:`zoom_center` but for an arbitrary region — the building block
    for attention-guided cropping (L3a), also usable by coded pipelines.
    """
    from PIL import Image

    w, h = img.size
    left, top, right, bottom = (float(v) for v in box)
    left, top = max(0.0, min(left, 0.95)), max(0.0, min(top, 0.95))
    right, bottom = min(1.0, max(right, left + 0.05)), min(1.0, max(bottom, top + 0.05))
    px = (int(left * w), int(top * h), max(int(right * w), int(left * w) + 1),
          max(int(bottom * h), int(top * h) + 1))
    return img.crop(px).resize((w, h), Image.LANCZOS)


def crop_case_bbox(
    img,
    case: "FailureCase | None" = None,
    bbox_key: str = "answer_bbox_xyxy_norm",
    padding: float = 0.15,
    min_size_frac: float = 0.08,
    sharpen_factor: float = 1.0,
    contrast_factor: float = 1.0,
):
    """Crop a per-case normalized bbox from metadata, then resize back.

    TextVQA-style size-sensitivity experiments often include answer bboxes.
    This L2 scaffold implements the paper's human-CROP intervention: use a
    dataset-provided visual localization annotation to magnify the small answer
    region.  It does not read labels or expected answers, and is a no-op for
    cases without a usable bbox.
    """
    meta = getattr(case, "metadata", {}) or {}
    raw = meta.get(bbox_key) or meta.get("answer_bbox_norm") or meta.get("bbox_xyxy_norm")
    if raw is None:
        return img
    if isinstance(raw, dict):
        try:
            box = [raw["left"], raw["top"], raw["right"], raw["bottom"]]
        except KeyError:
            try:
                box = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]]
            except KeyError:
                return img
    else:
        box = raw
    try:
        left, top, right, bottom = (float(v) for v in box)
    except Exception:
        return img
    if right <= left or bottom <= top:
        return img

    left, top = max(0.0, min(1.0, left)), max(0.0, min(1.0, top))
    right, bottom = max(0.0, min(1.0, right)), max(0.0, min(1.0, bottom))
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    side = max(right - left, bottom - top, float(min_size_frac))
    side = min(1.0, side * (1.0 + 2.0 * max(0.0, float(padding))))
    new_left = cx - side / 2.0
    new_top = cy - side / 2.0
    new_right = cx + side / 2.0
    new_bottom = cy + side / 2.0
    if new_left < 0.0:
        new_right -= new_left
        new_left = 0.0
    if new_top < 0.0:
        new_bottom -= new_top
        new_top = 0.0
    if new_right > 1.0:
        new_left -= new_right - 1.0
        new_right = 1.0
    if new_bottom > 1.0:
        new_top -= new_bottom - 1.0
        new_bottom = 1.0
    out = crop_region(
        img,
        box=(
            max(0.0, new_left),
            max(0.0, new_top),
            min(1.0, new_right),
            min(1.0, new_bottom),
        ),
    )
    try:
        from PIL import ImageEnhance

        if float(sharpen_factor) != 1.0:
            out = ImageEnhance.Sharpness(out).enhance(float(sharpen_factor))
        if float(contrast_factor) != 1.0:
            out = ImageEnhance.Contrast(out).enhance(float(contrast_factor))
    except Exception:
        return out
    return out


def crop_salient_region(img, padding: float = 0.05, min_delta: float = 18.0):
    """Crop non-background content, then resize back to the original size.

    The background is estimated from image-border pixels.  This is a generic L2
    preprocessing tool for small colored marks, narrow bands, dots, and other
    objects that are visually distinct from a mostly uniform canvas.  It does
    not inspect labels or answer text; it only magnifies the detected content.
    """
    import numpy as np
    from PIL import Image

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    border = max(1, min(h, w) // 32)
    samples = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    border_dist = np.linalg.norm(samples - bg, axis=1)
    border_median = float(np.median(border_dist))
    border_mad = float(np.median(np.abs(border_dist - border_median)))
    thresh = max(float(min_delta), border_median + 6.0 * border_mad + float(min_delta))
    mask = dist > thresh
    if not bool(mask.any()):
        return img

    ys, xs = np.where(mask)
    pad_px = max(1, int(round(float(padding) * max(h, w))))
    left = max(0, int(xs.min()) - pad_px)
    right = min(w, int(xs.max()) + pad_px + 1)
    top = max(0, int(ys.min()) - pad_px)
    bottom = min(h, int(ys.max()) + pad_px + 1)
    if right <= left or bottom <= top:
        return img
    return rgb.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)


def separate_horizontal_bands(
    img,
    min_delta: float = 18.0,
    color_delta: float = 35.0,
    min_width_frac: float = 0.35,
):
    """Render detected horizontal color runs as separated, thick bands.

    This is a deterministic visibility transform for sub-patch horizontal
    structures: it detects rows whose color differs from the border-estimated
    background, splits adjacent rows when their median color changes, and
    redraws each run with gray gaps.  It preserves the number/order/color of
    detected bands while making individual bands resolvable to a VLM.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    border = max(1, min(h, w) // 32)
    samples = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > float(min_delta)
    if not bool(mask.any()):
        return img

    ys, xs = np.where(mask)
    if (int(xs.max()) - int(xs.min()) + 1) / max(1, w) < float(min_width_frac):
        return img

    row_salient = mask.mean(axis=1) > 0.05
    row_colors: list[tuple[int, np.ndarray]] = []
    for y in np.where(row_salient)[0].tolist():
        row_mask = mask[y]
        if not bool(row_mask.any()):
            continue
        row_colors.append((y, np.median(arr[y, row_mask, :], axis=0)))
    if not row_colors:
        return img

    segments: list[tuple[int, int, np.ndarray]] = []
    start_y, prev_y, running = row_colors[0][0], row_colors[0][0], [row_colors[0][1]]
    prev_color = row_colors[0][1]
    for y, color in row_colors[1:]:
        new_band = (y != prev_y + 1) or (
            float(np.linalg.norm(color - prev_color)) > float(color_delta)
        )
        if new_band:
            segments.append((start_y, prev_y, np.median(np.stack(running), axis=0)))
            start_y, running = y, [color]
        else:
            running.append(color)
        prev_y, prev_color = y, color
    segments.append((start_y, prev_y, np.median(np.stack(running), axis=0)))
    if len(segments) <= 1:
        return img

    bg_color = tuple(int(max(0, min(255, round(v)))) for v in bg)
    out = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(out)
    margin_y = max(8, int(round(0.06 * h)))
    gap = max(2, min(8, int(round(h / (len(segments) * 12)))))
    band_h = max(3, int((h - 2 * margin_y - gap * (len(segments) - 1)) / len(segments)))
    total_h = len(segments) * band_h + (len(segments) - 1) * gap
    y = max(0, (h - total_h) // 2)
    x0 = max(0, int(xs.min()) - max(2, int(round(0.02 * w))))
    x1 = min(w - 1, int(xs.max()) + max(2, int(round(0.02 * w))))
    for _, _, color in segments:
        fill = tuple(int(max(0, min(255, round(v)))) for v in color)
        draw.rectangle([x0, y, x1, min(h - 1, y + band_h - 1)], fill=fill)
        y += band_h + gap
    return out


def _horizontal_band_count(
    img,
    min_delta: float = 18.0,
    color_delta: float = 35.0,
    min_width_frac: float = 0.35,
) -> int:
    """Count horizontal color runs detected against the border background."""
    import numpy as np

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    border = max(1, min(h, w) // 32)
    samples = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > float(min_delta)
    if not bool(mask.any()):
        return 0
    _, xs = np.where(mask)
    if (int(xs.max()) - int(xs.min()) + 1) / max(1, w) < float(min_width_frac):
        return 0

    row_salient = mask.mean(axis=1) > 0.05
    row_colors: list[tuple[int, np.ndarray]] = []
    for y in np.where(row_salient)[0].tolist():
        row_mask = mask[y]
        if bool(row_mask.any()):
            row_colors.append((y, np.median(arr[y, row_mask, :], axis=0)))
    if not row_colors:
        return 0

    count = 1
    prev_y, prev_color = row_colors[0]
    for y, color in row_colors[1:]:
        if (y != prev_y + 1) or float(np.linalg.norm(color - prev_color)) > float(color_delta):
            count += 1
        prev_y, prev_color = y, color
    return count


def annotate_horizontal_band_count(
    img,
    min_delta: float = 18.0,
    color_delta: float = 35.0,
    min_width_frac: float = 0.35,
    min_count: int = 1,
):
    """Overlay a deterministic horizontal-band count on the image.

    The count is computed from the image pixels using the same horizontal run
    detector as :func:`separate_horizontal_bands`.  This is an L2 tool-assisted
    scaffold: it gives the unchanged VLM an explicit visual measurement without
    exposing labels or expected answers.
    """
    from PIL import ImageDraw, ImageFont

    count = _horizontal_band_count(
        img,
        min_delta=min_delta,
        color_delta=color_delta,
        min_width_frac=min_width_frac,
    )
    if count < int(min_count):
        return img

    out = separate_horizontal_bands(
        img,
        min_delta=min_delta,
        color_delta=color_delta,
        min_width_frac=min_width_frac,
    ).convert("RGB")
    w, h = out.size
    draw = ImageDraw.Draw(out)
    banner_h = max(44, int(round(0.16 * h)))
    draw.rectangle([0, 0, w - 1, banner_h], fill=(255, 255, 255), outline=(0, 0, 0))
    text = f"COUNT: {count}"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(28, int(round(banner_h * 0.55))))
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 120, 20
    draw.text(((w - tw) // 2, max(4, (banner_h - th) // 2)), text, fill=(0, 0, 0), font=font)
    return out


#: name -> (function, parameter hint for the judge prompt, description)
IMAGE_TOOLS: "dict[str, tuple[Callable, str, str]]" = {
    "zoom_center": (zoom_center, "factor: float >= 1 (default 1.5)",
                    "crop the central 1/factor region, resize back — magnifies small findings"),
    "enhance_contrast": (enhance_contrast, "factor: float (default 1.5)",
                         "global contrast boost — low-contrast findings"),
    "sharpen": (sharpen, "factor: float (default 2.0)",
                "edge sharpening — blurred or subtle boundaries"),
    "equalize": (equalize, "(no params)",
                 "histogram equalization — compressed dynamic range"),
    "upscale": (upscale, "factor: float >= 1 (default 2.0)",
                "resize up before encoding — more vision tokens per region"),
    "crop_region": (crop_region, "box: [left, top, right, bottom] normalized 0..1",
                    "crop an arbitrary region and resize back — magnify a known area"),
    "crop_case_bbox": (
        crop_case_bbox,
        "bbox_key: str metadata key (default answer_bbox_xyxy_norm), padding: float "
        "(default 0.15), min_size_frac: float (default 0.08), "
        "sharpen_factor: float (default 1), contrast_factor: float (default 1)",
        "crop the per-case bbox stored in metadata and resize back — paper-style "
        "human-CROP for TextVQA answer regions, optionally sharpened/enhanced",
    ),
    "crop_salient_region": (crop_salient_region,
                            "padding: float 0-0.5 (default 0.05), min_delta: float "
                            "(default 18)",
                            "detect content that differs from a uniform border/background, "
                            "crop it with padding, and resize back — magnifies small objects"),
    "separate_horizontal_bands": (
        separate_horizontal_bands,
        "min_delta: float (default 18), color_delta: float (default 35), "
        "min_width_frac: float 0-1 (default 0.35)",
        "detect adjacent horizontal color bands and redraw them as separated, "
        "thick bands — makes sub-patch stripe structure countable",
    ),
    "annotate_horizontal_band_count": (
        annotate_horizontal_band_count,
        "min_delta: float (default 18), color_delta: float (default 35), "
        "min_width_frac: float 0-1 (default 0.35), min_count: int (default 1)",
        "detect adjacent horizontal color bands and overlay a visual COUNT: N "
        "measurement derived from the image pixels",
    ),
}


def catalog_text() -> str:
    """Render the tool catalog for inclusion in a judge prompt."""
    return "\n".join(
        f"- {name}: {desc}  [params: {params}]"
        for name, (_, params, desc) in IMAGE_TOOLS.items()
    )


def apply_image_ops(
    image: Any,
    ops: "list[dict[str, Any]]",
    case: "FailureCase | None" = None,
) -> Any:
    """Apply ``[{"tool": name, "params": {...}}, ...]`` to *image*.

    Unknown tools and per-op failures are skipped (logged); returns the
    original object when nothing could be applied (e.g. no image).
    """
    img = _as_pil(image)
    if img is None:
        return image
    for op in ops:
        name = str(op.get("tool", ""))
        entry = IMAGE_TOOLS.get(name)
        if entry is None:
            logger.warning("fix_tools: unknown tool %r skipped", name)
            continue
        fn = entry[0]
        try:
            if name == "crop_case_bbox":
                img = fn(img, case=case, **dict(op.get("params") or {}))
            else:
                img = fn(img, **dict(op.get("params") or {}))
        except Exception as exc:
            logger.warning("fix_tools: tool %r failed (%s); skipped", name, exc)
    return img


# ---------------------------------------------------------------------------
# Pipeline spec + executor
# ---------------------------------------------------------------------------

@dataclass
class PipelineSpec:
    """A serialisable L2 scaffold around the unchanged model.

    Attributes:
        name:            Short identifier for logs.
        image_ops:       Tool applications, in order (see :data:`IMAGE_TOOLS`).
        prompt_template: Must contain ``{prompt}``; identity by default.
        n_samples:       Independent passes per case (majority vote on the
                         extracted final answer when > 1). Applies to every
                         strategy — a multi-call strategy is repeated end-to-end.
        generation_kwargs: Safe, backend-neutral decoding overrides. This is
                           deliberately part of the serialized candidate: a
                           length stop is an execution-health failure, not a
                           prompt failure.
        strategy:        A reviewed multi-call scaffold. ``direct`` is the
                         normal one-call path; the other choices encode common
                         general-purpose reasoning patterns without arbitrary
                         code execution.
        output_key_pattern: Optional safe regex (one capture group) used to
                         aggregate structured final answers across samples.
    """

    name: str
    image_ops: "list[dict[str, Any]]" = field(default_factory=list)
    prompt_template: str = "{prompt}"
    n_samples: int = 1
    generation_kwargs: "dict[str, Any]" = field(default_factory=dict)
    strategy: str = "direct"
    output_key_pattern: str = ""

    @classmethod
    def from_dict(cls, d: "dict[str, Any]") -> "PipelineSpec | None":
        """Validate a judge-proposed spec dict; ``None`` when unusable."""
        name = str(d.get("name", "")).strip()
        template = str(d.get("prompt_template") or "{prompt}")
        if not name or "{prompt}" not in template:
            return None
        ops = [
            {"tool": str(op["tool"]), "params": dict(op.get("params") or {})}
            for op in d.get("image_ops") or []
            if isinstance(op, dict) and str(op.get("tool", "")) in IMAGE_TOOLS
        ]
        try:
            n_samples = max(1, int(d.get("n_samples", 1)))
        except (TypeError, ValueError):
            n_samples = 1
        generation_kwargs = _safe_generation_kwargs(d.get("generation_kwargs"))
        strategy = str(d.get("strategy", "direct")).strip().lower()
        if strategy not in _SCAFFOLD_STRATEGIES:
            strategy = "direct"
        pattern = _safe_output_key_pattern(d.get("output_key_pattern"))
        return cls(name=name, image_ops=ops, prompt_template=template,
                   n_samples=min(n_samples, 5), generation_kwargs=generation_kwargs,
                   strategy=strategy, output_key_pattern=pattern)

    def to_dict(self) -> "dict[str, Any]":
        return {"name": self.name, "image_ops": self.image_ops,
                "prompt_template": self.prompt_template, "n_samples": self.n_samples,
                "generation_kwargs": self.generation_kwargs, "strategy": self.strategy,
                "output_key_pattern": self.output_key_pattern}


_SCAFFOLD_STRATEGIES = frozenset({
    "direct", "least_to_most", "self_refine", "chain_of_verification",
})

#: The ordered call chain each multi-call strategy issues, as
#: ``(step label, instruction prepended for that call)``. An empty instruction
#: means the call sends the task prompt with nothing added.
#:
#: Hoisted out of ``run_pipeline`` (which now formats from these) so that the
#: dashboard can SHOW the text a strategy actually sends instead of
#: paraphrasing it. ``analysis/case_studio.py`` keeps a mirror of this table —
#: it may not import this package at runtime (analysis has to stay standalone)
#: — and ``test_strategy_flow_matches_what_the_pipeline_sends`` asserts the two
#: are equal, so re-wording a prompt here without updating the mirror fails the
#: suite rather than silently leaving the UI describing a pipeline that is gone.
STRATEGY_CALLS: "dict[str, tuple[tuple[str, str], ...]]" = {
    "self_refine": (
        ("answer", ""),
        ("critique that answer",
         "Check the attempted answer for factual, reasoning, arithmetic, and "
         "instruction-following errors. Give concise correction advice only."),
        ("revise it",
         "Produce a corrected final answer to the original task using the "
         "feedback. Do not discuss the revision process."),
    ),
    "least_to_most": (
        ("break into subproblems",
         "Break the following task into the smallest useful subproblems. "
         "Do not answer the task yet."),
        ("solve using the decomposition",
         "Solve the original task using the decomposition. Give only the final "
         "answer required by the task."),
    ),
    "chain_of_verification": (
        ("answer", ""),
        ("list verification checks",
         "List short, independent checks needed to verify this attempted answer. "
         "Do not answer the original task yet."),
        ("answer after the checks",
         "Answer the original task after applying the independent verification "
         "checks. Give only the final answer required by the task."),
    ),
}


def _safe_output_key_pattern(value: Any) -> str:
    """Accept one bounded capture regex for evaluator-declared answer formats."""
    pattern = str(value or "").strip()
    if not pattern or len(pattern) > 256:
        return ""
    try:
        compiled = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    except re.error:
        return ""
    return pattern if compiled.groups >= 1 else ""


#: Upper bound on a candidate's ``max_tokens``. This is an audit ceiling, not a
#: budget: it only stops a judge from smuggling an absurd number in. It sits
#: well above every baseline budget this repo runs (20480 for the long-form
#: llm_benchmark datasets) — the old 8192 was BELOW those, so a candidate could
#: never even match the baseline on them. The floor (a candidate may not decode
#: with LESS than the baseline budget) is enforced by the fix agent, which
#: knows the baseline; see ``FixAgent._enforce_generation_floor``.
MAX_TOKENS_CAP = 32768


def _safe_generation_kwargs(value: Any) -> "dict[str, Any]":
    """Keep only portable, bounded decoding controls from a candidate spec.

    The model adapter owns its provider-specific API.  Repair candidates may
    only tune controls that are safe to replay and audit; this avoids a judge
    smuggling arbitrary provider arguments or callbacks into ``generate``.
    """
    raw = value if isinstance(value, dict) else {}
    out: "dict[str, Any]" = {}
    try:
        max_tokens = int(raw.get("max_tokens"))
        if 1 <= max_tokens <= MAX_TOKENS_CAP:
            out["max_tokens"] = max_tokens
    except (TypeError, ValueError):
        pass
    try:
        temperature = float(raw.get("temperature"))
        if 0.0 <= temperature <= 2.0:
            out["temperature"] = temperature
    except (TypeError, ValueError):
        pass
    try:
        top_p = float(raw.get("top_p"))
        if 0.0 < top_p <= 1.0:
            out["top_p"] = top_p
    except (TypeError, ValueError):
        pass
    stop = raw.get("stop")
    if isinstance(stop, str) and len(stop) <= 128:
        out["stop"] = stop
    elif isinstance(stop, list) and len(stop) <= 4 and all(
        isinstance(item, str) and len(item) <= 128 for item in stop
    ):
        out["stop"] = list(stop)
    return out


def spec_changes_input(spec: PipelineSpec, case: "FailureCase") -> bool:
    """True when *spec* actually alters this case's effective input.

    A spec that leaves both the prompt and the image untouched is a *no-op* on
    the case — e.g. ``crop_case_bbox`` on a case that carries no answer bbox, or
    an enhancement with unit factors.  Such a case is outside the candidate's
    applicability: the fix can neither repair nor break it, so it must be scoped
    out of the safety/coverage accounting rather than counted as an unchanged
    "control".  This is the structural half of an applicability predicate; an
    explicit :attr:`FixCandidate.predicate` overrides it.
    """
    if (
        spec.prompt_template.strip() != "{prompt}"
        or spec.generation_kwargs
        or spec.strategy != "direct"
        or spec.n_samples > 1
    ):
        return True
    if not spec.image_ops:
        return False
    inp = getattr(case, "inputs", None)
    image = getattr(inp, "image", None) if inp is not None else None
    before = _as_pil(image)
    if before is None:
        return False
    after = apply_image_ops(before, spec.image_ops, case=case)
    if after is before:
        return False
    try:
        import numpy as np

        a, b = np.asarray(before), np.asarray(after)
        return a.shape != b.shape or not np.array_equal(a, b)
    except Exception:
        return True  # cannot prove it is a no-op — treat as applicable


def _run_strategy_once(
    generate: "Callable[[str], str]", strategy: str, base_prompt: str
) -> str:
    """One pass of a reviewed multi-call strategy; returns its final output."""
    if strategy == "least_to_most":
        calls = STRATEGY_CALLS["least_to_most"]
        decomposition = generate(calls[0][1] + "\n\n" + base_prompt)
        return generate(
            f"{calls[1][1]}\n\nOriginal task:\n"
            f"{base_prompt}\n\nDecomposition:\n{decomposition}"
        )
    if strategy == "self_refine":
        calls = STRATEGY_CALLS["self_refine"]
        draft = generate(base_prompt)
        feedback = generate(
            f"{calls[1][1]}\n\n"
            f"Original task:\n{base_prompt}\n\nAttempt:\n{draft}"
        )
        return generate(
            f"{calls[2][1]}\n\n"
            f"Original task:\n{base_prompt}\n\nAttempt:\n{draft}\n\nFeedback:\n{feedback}"
        )
    if strategy == "chain_of_verification":
        calls = STRATEGY_CALLS["chain_of_verification"]
        draft = generate(base_prompt)
        checks = generate(
            f"{calls[1][1]}\n\n"
            f"Original task:\n{base_prompt}\n\nAttempt:\n{draft}"
        )
        return generate(
            f"{calls[2][1]}\n\n"
            f"Original task:\n{base_prompt}\n\nAttempt:\n{draft}\n\nChecks:\n{checks}"
        )
    return generate(base_prompt)


def answer_key(output: str, pattern: str = "") -> str:
    """Label-free key two samples must share to count as the same answer.

    *pattern* (evaluator-declared, one capture group) wins when it matches.
    Otherwise the key is the FINAL ANSWER span — last ``Answer:`` tag /
    ``\\boxed{}`` / last non-empty line, normalised — via the same
    ``extract_answer`` + ``normalize_answer`` the reasoning analyzers use.

    Why not the whole normalised text (the previous fallback): with a
    chain-of-thought model no two samples are ever byte-identical, so every
    sample sat in its own group and ``n_samples=5`` silently degenerated to
    "return the first sample" — five calls, no vote. Measured on the
    qwen3.5-2b/bbh_tracking7 candidates: the judge's ``n_samples=5`` proposals
    voted on nothing. Neither path reads the gold answer.
    """
    text = str(output or "")
    if pattern:
        try:
            key_re = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        except re.error:
            key_re = None
        if key_re is not None:
            matches = list(key_re.finditer(text))
            if matches and matches[-1].lastindex:
                return matches[-1].group(1).strip().lower()
    try:
        from evalvitals.analyzers.reasoning._text import extract_answer, normalize_answer

        key = normalize_answer(extract_answer(text))
        if key:
            return key
    except Exception:  # analyzers are optional at this layer; degrade gracefully
        pass
    return re.sub(r"\s+", " ", text.strip().lower())


def run_pipeline(
    model: "Model",
    case: "FailureCase",
    spec: PipelineSpec,
    score_fn: "Callable[[FailureCase, str], Optional[bool]]",
    capture: "dict[str, Any] | None" = None,
) -> "Optional[bool]":
    """Execute *spec* on one case; aggregate outputs before host-side scoring.

    ``n_samples`` applies to EVERY strategy: a multi-call strategy is run
    ``n_samples`` times end-to-end and its final outputs vote (previously only
    ``direct`` honoured ``n_samples`` — a judge's ``least_to_most, n_samples=3``
    silently ran once). Samples vote on an evaluator-declared final-answer key
    when one is available, otherwise on the extracted final answer
    (:func:`answer_key`). The scoring rubric is applied *after* aggregation, so
    a pipeline cannot use the hidden gold answer to select its preferred
    sample. Reviewed multi-call strategies are prompt-only and keep the
    underlying model unchanged. Returns ``None`` when the case cannot be scored
    (no rubric / all calls failed).

    *capture*, when a dict, receives ``{"prompt", "outputs", "winner",
    "n_calls"}`` so the caller can persist WHAT the candidate produced (the
    only way to tell a truncated answer from a wrong one after the fact).
    """
    import dataclasses

    from evalvitals.core.case import Inputs

    inp = getattr(case, "inputs", None)
    prompt = str(getattr(inp, "prompt", "")) if inp is not None else ""
    image = getattr(inp, "image", None) if inp is not None else None
    if spec.image_ops:
        image = apply_image_ops(image, spec.image_ops, case=case)
    base_prompt = spec.prompt_template.format(prompt=prompt)
    n_calls = 0

    def generate(text: str) -> str:
        nonlocal n_calls
        n_calls += 1
        try:
            # dataclasses.replace(inp, ...), not a bare Inputs(prompt=...,
            # image=...): the bare form silently dropped .video/.audio, so
            # every strategy (least_to_most/self_refine/chain_of_verification)
            # was unconditionally inapplicable on any non-image FailureCase --
            # generate() raising on the missing required modality field,
            # caught below, every call returning "" (all outputs empty ->
            # run_pipeline returns None -> unscoreable for every case). Only
            # `image` is deliberately overridden (it may have been rewritten
            # by spec.image_ops just above); no fallback needed for inp is
            # None since FailureCase.inputs is never optional in practice.
            new_inputs = (
                dataclasses.replace(inp, prompt=text, image=image)
                if inp is not None
                else Inputs(prompt=text, image=image)
            )
            return str(model.generate(new_inputs, **spec.generation_kwargs))
        except Exception as exc:
            logger.debug("run_pipeline: generate failed on %s: %s", case.id, exc)
            return ""

    outputs = [
        _run_strategy_once(generate, spec.strategy, base_prompt)
        for _ in range(max(1, spec.n_samples))
    ]
    outputs = [output for output in outputs if output]
    if capture is not None:
        capture.update({"prompt": base_prompt, "outputs": list(outputs),
                        "winner": None, "n_calls": n_calls})
    if not outputs:
        return None
    # Keep the first original response for the winning normalized key; stable
    # tie-breaking avoids injecting a score-dependent preference.
    pattern = spec.output_key_pattern or str(
        (getattr(case, "metadata", {}) or {}).get("output_key_pattern", "")
    )
    grouped: "dict[str, list[str]]" = {}
    for output in outputs:
        grouped.setdefault(answer_key(output, pattern), []).append(output)
    winner = max(grouped.values(), key=len)[0]
    if capture is not None:
        capture["winner"] = winner
    return score_to_bool(score_fn(case, winner))
