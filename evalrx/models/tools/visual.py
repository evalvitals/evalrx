"""Visual perception tools — the T0 tier of the default visual-reasoning kit.

``zoom_in`` is the single highest-value visual tool (the o3 / DeepEyes /
Qwen3-VL-demo mechanism): crop a region, upscale it, and hand it back as a new
image so the model can inspect fine detail.  Torch-free; PIL imported lazily.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

from evalrx.core.tool import Tool, ToolResult

_MIN_SIDE_FRAC = 0.05        # never zoom into a sliver thinner than 5% of a side
_TARGET_SHORT_SIDE = 672     # upscale the crop so detail is actually visible
_MAX_UPSCALE = 4.0           # ... but never blow pixels up more than 4x

ZOOM_IN_DESCRIPTION = (
    "Zoom into a rectangular region of the CURRENT image to inspect fine detail. "
    "Pass bbox=[x0, y0, x1, y1] as FRACTIONS of the image width/height, each in "
    "[0, 1], with (x0, y0) the top-left and (x1, y1) the bottom-right corner "
    "(e.g. the right half is [0.5, 0.0, 1.0, 1.0]). The cropped, enlarged view "
    "comes back as a new image. Zoom before answering questions about small "
    "objects, text, or fine attributes."
)

ZOOM_IN_PARAMETERS = {
    "type": "object",
    "properties": {
        "bbox": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x0, y0, x1, y1], fractions of width/height in [0, 1].",
        },
        "target": {
            "type": "string",
            "description": "What you are trying to see in this region (short phrase).",
        },
    },
    "required": ["bbox"],
}


def _to_pil(image: Any):
    """Return *image* as a PIL image (paths are opened and converted to RGB)."""
    if hasattr(image, "size") and hasattr(image, "mode"):
        return image
    from PIL import Image

    return Image.open(image).convert("RGB")


def _sanitize_bbox(bbox: Any, width: int, height: int) -> "tuple[list[float], str]":
    """Normalize *bbox* to fractional [x0, y0, x1, y1] and report the coord mode.

    Models that ground in pixel coordinates are auto-detected (any value > 1.5)
    and converted — the mode is recorded so the trajectory keeps evidence of
    which convention the model actually used.
    """
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        raise ValueError(f"bbox must be four numbers [x0, y0, x1, y1], got {bbox!r}")
    if len(vals) != 4:
        raise ValueError(f"bbox must have exactly 4 values [x0, y0, x1, y1], got {len(vals)}")

    mode = "normalized"
    if max(vals) > 1.5:  # not fractions: pixel coordinates or a Qwen-style 0-1000 grid
        fits_pixels = (
            max(vals[0], vals[2]) <= width * 1.05
            and max(vals[1], vals[3]) <= height * 1.05
        )
        if fits_pixels:
            mode = "pixel"
            scale = (width, height, width, height)
        else:  # out of pixel bounds -> 0-1000 normalized grid
            mode = "grid_1000"
            scale = (1000.0, 1000.0, 1000.0, 1000.0)
        vals = [v / s for v, s in zip(vals, scale)]

    x0, y0, x1, y1 = vals
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)

    # expand degenerate/sliver boxes around their center to a usable minimum
    for lo, hi in ((0, 2), (1, 3)):
        box = [x0, y0, x1, y1]
        if box[hi] - box[lo] < _MIN_SIDE_FRAC:
            center = (box[lo] + box[hi]) / 2
            lo_v = min(max(0.0, center - _MIN_SIDE_FRAC / 2), 1.0 - _MIN_SIDE_FRAC)
            if lo == 0:
                x0, x1 = lo_v, lo_v + _MIN_SIDE_FRAC
            else:
                y0, y1 = lo_v, lo_v + _MIN_SIDE_FRAC
    return [x0, y0, x1, y1], mode


def zoom_in_tool(
    image: Any,
    *,
    save_dir: "str | None" = None,
    target_short_side: int = _TARGET_SHORT_SIDE,
    max_upscale: float = _MAX_UPSCALE,
) -> Tool:
    """Build the ``image_zoom_in`` tool bound to *image* (PIL or path).

    The returned crop is upscaled (LANCZOS) so its short side reaches
    *target_short_side* (capped at *max_upscale*), and re-injected into the
    conversation by the agent loop.  When *save_dir* is given every crop is
    also written there (``zoom_00.png``, ...) and the path recorded in
    ``ToolResult.meta`` — trajectories reference media rather than embed it.
    """
    base = _to_pil(image)
    width, height = base.size
    counter = itertools.count()

    def _zoom(bbox, target: str = "") -> ToolResult:
        frac, mode = _sanitize_bbox(bbox, width, height)
        px = (
            int(round(frac[0] * width)),
            int(round(frac[1] * height)),
            int(round(frac[2] * width)),
            int(round(frac[3] * height)),
        )
        crop = base.crop(px)
        short = min(crop.size)
        if short > 0 and short < target_short_side:
            factor = min(target_short_side / short, max_upscale)
            if factor > 1.0:
                from PIL import Image

                crop = crop.resize(
                    (int(round(crop.size[0] * factor)), int(round(crop.size[1] * factor))),
                    Image.LANCZOS,
                )

        meta: dict = {
            "bbox_frac": [round(v, 4) for v in frac],
            "crop_px": list(px),
            "coord_mode": mode,
        }
        if target:
            meta["target"] = target
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"zoom_{next(counter):02d}.png")
            crop.save(path)
            meta["saved_path"] = path

        note = "" if mode == "normalized" else f" (coordinates interpreted as {mode})"
        return ToolResult(
            text=(
                f"Zoomed into region x:[{frac[0]:.2f}, {frac[2]:.2f}] "
                f"y:[{frac[1]:.2f}, {frac[3]:.2f}] of the {width}x{height} image{note}. "
                f"The enlarged {crop.size[0]}x{crop.size[1]} view is attached as a new image."
            ),
            images=[crop],
            meta=meta,
        )

    return Tool(
        name="image_zoom_in",
        description=ZOOM_IN_DESCRIPTION,
        parameters=ZOOM_IN_PARAMETERS,
        fn=_zoom,
    )
