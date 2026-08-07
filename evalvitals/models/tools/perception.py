"""Structured-perception tools — the T1 tier: ``image_ocr`` and ``image_detect``.

Both follow the house tool constraints: deterministic engines (versioned
weights, no sampling), normalized ``[0, 1]`` coordinates in results, engine
failures surfaced through the standard executor error envelope, and
:class:`~evalvitals.core.tool.ToolResult` outputs whose ``meta`` records what
actually ran.

Engines are **injectable** (any callable with the documented signature) so the
tools stay dependency-light and unit-testable; the default engines are built
lazily on first call:

* OCR   — ``easyocr`` (reuses the already-installed torch stack),
* detect — open-vocabulary Grounding DINO via ``transformers``
  (``IDEA-Research/grounding-dino-tiny``).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from evalvitals.core.tool import Tool, ToolResult
from evalvitals.models.tools.visual import _sanitize_bbox, _to_pil

# ----------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------
OCR_DESCRIPTION = (
    "Read the text in the CURRENT image (optical character recognition). "
    "Optionally pass bbox=[x0, y0, x1, y1] (fractions of width/height in "
    "[0, 1]) to read only that region. Returns the recognized text lines with "
    "their locations. Use this for signs, labels, documents, or any question "
    "about what text says."
)

OCR_PARAMETERS = {
    "type": "object",
    "properties": {
        "bbox": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "Optional region [x0, y0, x1, y1], fractions in [0, 1].",
        },
    },
    "required": [],
}


def default_ocr_engine(languages: "list[str] | None" = None, device: str = "cpu") -> Callable:
    """Build the default OCR engine on ``easyocr`` (deterministic, local weights).

    Returns ``engine(pil_image) -> list[(bbox_px, text, confidence)]`` where
    ``bbox_px`` is a 4-point polygon in pixels (easyocr's native shape).
    """
    try:
        import easyocr
    except ImportError as exc:
        raise ImportError(
            "image_ocr's default engine needs easyocr (pip install easyocr), "
            "or inject your own engine via ocr_tool(..., engine=...)."
        ) from exc
    import numpy as np

    reader = easyocr.Reader(languages or ["en"], gpu=device != "cpu", verbose=False)

    def _engine(pil_image):
        return reader.readtext(np.array(pil_image))

    return _engine


def ocr_tool(
    image: Any,
    *,
    engine: Optional[Callable] = None,
    max_lines: int = 40,
) -> Tool:
    """Build the ``image_ocr`` tool bound to *image*.

    *engine* is ``callable(pil_image) -> list[(polygon_px, text, confidence)]``;
    defaults to :func:`default_ocr_engine` built lazily on first call.
    """
    base = _to_pil(image)
    width, height = base.size
    state = {"engine": engine}

    def _get_engine() -> Callable:
        if state["engine"] is None:
            state["engine"] = default_ocr_engine()
        return state["engine"]

    def _ocr(bbox=None) -> ToolResult:
        region = base
        offset = (0, 0)
        meta: dict = {"engine": "injected" if engine else "easyocr", "region": "full"}
        if bbox is not None:
            frac, mode = _sanitize_bbox(bbox, width, height)
            px = (
                int(round(frac[0] * width)),
                int(round(frac[1] * height)),
                int(round(frac[2] * width)),
                int(round(frac[3] * height)),
            )
            region = base.crop(px)
            offset = (px[0], px[1])
            meta.update({"region": "bbox", "bbox_frac": [round(v, 4) for v in frac], "coord_mode": mode})

        detections = _get_engine()(region)
        lines: list[str] = []
        for det in detections[:max_lines]:
            poly, text, conf = det[0], det[1], (det[2] if len(det) > 2 else None)
            xs = [p[0] + offset[0] for p in poly]
            ys = [p[1] + offset[1] for p in poly]
            loc = (
                f"[{min(xs) / width:.2f}, {min(ys) / height:.2f}, "
                f"{max(xs) / width:.2f}, {max(ys) / height:.2f}]"
            )
            conf_s = f" (conf {conf:.2f})" if isinstance(conf, (int, float)) else ""
            lines.append(f"- {text!r} at {loc}{conf_s}")
        n_total = len(detections)
        meta["n_detections"] = n_total
        if not lines:
            text_out = "No text detected in the requested region."
        else:
            clipped = f" (showing {max_lines} of {n_total})" if n_total > max_lines else ""
            text_out = (
                f"Detected {n_total} text line(s){clipped}, locations as "
                f"[x0, y0, x1, y1] fractions of the full image:\n" + "\n".join(lines)
            )
        return ToolResult(text=text_out, meta=meta)

    return Tool(name="image_ocr", description=OCR_DESCRIPTION, parameters=OCR_PARAMETERS, fn=_ocr)


# ----------------------------------------------------------------------
# Open-vocabulary detection
# ----------------------------------------------------------------------
DETECT_DESCRIPTION = (
    "Detect objects in the CURRENT image by name (open-vocabulary detection). "
    "Pass query as the object name(s) to look for, separated by ' . ' for "
    "multiple (e.g. 'dog . frisbee'). Returns each detected instance with a "
    "confidence score and its box as [x0, y0, x1, y1] fractions of the image, "
    "plus an annotated copy of the image with the boxes drawn. Use this for "
    "counting, presence checks, and locating objects."
)

DETECT_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Object name(s) to detect, ' . '-separated for multiple.",
        },
    },
    "required": ["query"],
}

DEFAULT_DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"


def default_detect_engine(
    model_id: str = DEFAULT_DETECTOR_ID,
    device: str = "cpu",
    threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Callable:
    """Build the default detector on Grounding DINO via transformers.

    Returns ``engine(pil_image, query) -> list[{"label", "score", "box_px"}]``
    (deterministic: pure forward pass, fixed thresholds, versioned weights).
    """
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "image_detect's default engine needs torch+transformers "
            "(pip install 'evalvitals[local]'), or inject an engine via "
            "detect_tool(..., engine=...)."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def _engine(pil_image, query: str):
        text = query.strip().rstrip(".") + "."  # grounding-dino wants lowercase, dot-terminated
        inputs = processor(images=pil_image, text=text.lower(), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        out = []
        for label, score, box in zip(
            results["text_labels" if "text_labels" in results else "labels"],
            results["scores"],
            results["boxes"],
        ):
            out.append(
                {
                    "label": str(label),
                    "score": float(score),
                    "box_px": [float(v) for v in box.tolist()],
                }
            )
        return out

    return _engine


def _annotate(image, detections: list) -> Any:
    """Draw detection boxes + labels on a copy of *image*."""
    from PIL import ImageDraw

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det["box_px"]
        draw.rectangle([x0, y0, x1, y1], outline=(208, 59, 59), width=3)
        draw.text((x0 + 3, max(0, y0 - 12)), f"{i}: {det['label']} {det['score']:.2f}", fill=(208, 59, 59))
    return annotated


def detect_tool(
    image: Any,
    *,
    engine: Optional[Callable] = None,
    annotate: bool = True,
    max_detections: int = 20,
) -> Tool:
    """Build the ``image_detect`` tool bound to *image*.

    *engine* is ``callable(pil_image, query) -> list[{"label", "score",
    "box_px"}]``; defaults to :func:`default_detect_engine` built lazily on
    first call (weights load once per tool instance).
    """
    base = _to_pil(image)
    width, height = base.size
    state = {"engine": engine}

    def _get_engine() -> Callable:
        if state["engine"] is None:
            state["engine"] = default_detect_engine()
        return state["engine"]

    def _detect(query: str) -> ToolResult:
        if not str(query).strip():
            raise ValueError("query must name at least one object to detect")
        detections = _get_engine()(base, str(query))
        detections = sorted(detections, key=lambda d: -d["score"])[:max_detections]
        meta: dict = {
            "engine": "injected" if engine else DEFAULT_DETECTOR_ID,
            "query": str(query),
            "n_detections": len(detections),
            "detections": [
                {
                    "label": d["label"],
                    "score": round(d["score"], 3),
                    "box_frac": [
                        round(d["box_px"][0] / width, 4),
                        round(d["box_px"][1] / height, 4),
                        round(d["box_px"][2] / width, 4),
                        round(d["box_px"][3] / height, 4),
                    ],
                }
                for d in detections
            ],
        }
        if not detections:
            return ToolResult(
                text=f"No instances of {query!r} detected (score threshold not reached).",
                meta=meta,
            )
        lines = [
            f"- #{i} {d['label']} (score {d['score']:.2f}) at "
            f"[{d['box_frac'][0]:.2f}, {d['box_frac'][1]:.2f}, {d['box_frac'][2]:.2f}, {d['box_frac'][3]:.2f}]"
            for i, d in enumerate(meta["detections"])
        ]
        images = []
        text_out = (
            f"Detected {len(detections)} instance(s) of {query!r}, boxes as "
            f"[x0, y0, x1, y1] fractions of the image:\n" + "\n".join(lines)
        )
        if annotate:
            images.append(_annotate(base, detections))
            text_out += "\nAn annotated copy of the image with numbered boxes is attached."
        return ToolResult(text=text_out, images=images, meta=meta)

    return Tool(
        name="image_detect",
        description=DETECT_DESCRIPTION,
        parameters=DETECT_PARAMETERS,
        fn=_detect,
    )
