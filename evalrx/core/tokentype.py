"""TokenTypeMap — the shared VLM primitive that locates image tokens + the patch grid.

Every white-box VLM analyzer (Grad-CAM, relative-attention, MM-SHAP,
image-token logit-lens, "what's in the image") needs to know which sequence
positions are image-patch tokens and how they map to a spatial grid.  Building
that is genuinely model-specific (dynamic resolution + spatial-merge), so it is
centralised here and driven by a :class:`~evalrx.core.spec.VisionSpec`.

Design notes that this encodes:
  * prefer the processor's ``mm_token_type_ids`` when emitted (cleanest), else
    match ``input_ids == config.<image_token_id_attr>`` — the id is read from the
    live config (GLM-4.5V 151363 vs GLM-4.1V 151343), never baked in;
  * the grid is read from processor metadata per ``grid_source``:
      grid_thw  -> image_grid_thw (PRE-merge patch units) -> post = (h//m, w//m)   [Qwen/GLM]
      grid_hw   -> image_grid_hws (pre-merge) -> //m                                [Kimi]
      fixed     -> a fixed token budget per tile                                    [Gemma/Step]
  * merge size is read from config (``vision_config.spatial_merge_size``), not literal.

This module is torch-tolerant: it accepts torch tensors OR plain lists, importing
torch nowhere — so it unit-tests on the light install with synthetic inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


def _flat_ids(x: Any) -> list[int]:
    """Flatten input_ids (torch tensor / np / nested list / 1-D list) to a 1-D int list."""
    if x is None:
        return []
    if hasattr(x, "flatten") and hasattr(x, "tolist"):  # torch / numpy tensor
        return [int(v) for v in x.flatten().tolist()]
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)):  # [[...]] -> first row
            return [int(v) for v in x[0]]
        return [int(v) for v in x]
    return [int(v) for v in list(x)]


def _rows2(x: Any) -> list[list[int]]:
    """Coerce an (n, k) grid spec (tensor / nested list) to a list of int rows."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    rows = []
    for row in x:
        rows.append([int(v) for v in (row if isinstance(row, (list, tuple)) else [row])])
    return rows


def _resolve(obj: Any, dotted: Optional[str]) -> Any:
    """Resolve a dotted attribute path against a config object; None on miss."""
    if obj is None or not dotted:
        return None
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _enc_get(enc: Any, key: str) -> Any:
    """Read a key from a processor BatchFeature / dict."""
    if enc is None:
        return None
    if hasattr(enc, "get"):
        try:
            return enc.get(key)
        except Exception:
            pass
    return getattr(enc, key, None)


@dataclass
class TokenTypeMap:
    """Where the image tokens are, and how they map to a spatial grid.

    Attributes:
        seq_len:    total sequence length.
        image_pos:  sequence positions that are image-patch tokens (flat, in order).
        text_pos:   the remaining (text/control) positions.
        grids:      per image, the POST-merge grid ``(t, h, w)``.
        patches:    aligned with ``image_pos`` — ``(image_idx, t, row, col)`` for each.
        image_token_id: the placeholder id used (if matched by id).
    """

    seq_len: int
    image_pos: list[int] = field(default_factory=list)
    text_pos: list[int] = field(default_factory=list)
    grids: list[tuple[int, int, int]] = field(default_factory=list)
    patches: list[tuple[int, int, int, int]] = field(default_factory=list)
    image_token_id: Optional[int] = None

    @property
    def n_images(self) -> int:
        return len(self.grids)

    @property
    def has_image(self) -> bool:
        return len(self.image_pos) > 0


def build_token_type_map(input_ids: Any, enc: Any, config: Any, vision_spec: Any) -> TokenTypeMap:
    """Construct a :class:`TokenTypeMap` from a forward's encoding + model config + VisionSpec."""
    ids = _flat_ids(input_ids)
    seq = len(ids)

    # 1. locate image positions
    img_id = _resolve(config, vision_spec.image_token_id_attr) if config is not None else None
    mm = _enc_get(enc, "mm_token_type_ids")
    if getattr(vision_spec, "prefer_mm_token_type_ids", True) and mm is not None:
        mm_flat = _flat_ids(mm)
        image_pos = [i for i, v in enumerate(mm_flat) if v == 1]
    elif img_id is not None:
        image_pos = [i for i, t in enumerate(ids) if t == img_id]
    else:
        image_pos = []
    img_set = set(image_pos)
    text_pos = [i for i in range(seq) if i not in img_set]

    # 2. per-image POST-merge grid
    merge = _resolve(config, getattr(vision_spec, "merge_size_attr", None)) or 1
    merge = int(merge) if merge else 1
    grids: list[tuple[int, int, int]] = []
    source = getattr(vision_spec, "grid_source", "grid_thw")
    if source == "grid_thw":
        for row in _rows2(_enc_get(enc, "image_grid_thw")):
            t, h, w = (row + [1, 1, 1])[:3]
            grids.append((int(t), max(int(h) // merge, 1), max(int(w) // merge, 1)))
    elif source == "grid_hw":
        for row in _rows2(_enc_get(enc, "image_grid_hws")):
            h, w = (row + [1, 1])[:2]
            grids.append((1, max(int(h) // merge, 1), max(int(w) // merge, 1)))
    elif source == "fixed":
        per = getattr(vision_spec, "fixed_tokens_per_tile", None)
        if per:
            n_tiles = max(len(image_pos) // per, 1) if image_pos else 0
            side = int(math.isqrt(per))
            for _ in range(n_tiles):
                grids.append((1, side, side) if side * side == per else (1, 1, per))

    # 3. map each image position to (image_idx, t, row, col), row-major
    patches: list[tuple[int, int, int, int]] = []
    k = 0
    for img_idx, (t, h, w) in enumerate(grids):
        for ti in range(t):
            for r in range(h):
                for c in range(w):
                    if k < len(image_pos):
                        patches.append((img_idx, ti, r, c))
                        k += 1

    return TokenTypeMap(
        seq_len=seq,
        image_pos=image_pos,
        text_pos=text_pos,
        grids=grids,
        patches=patches,
        image_token_id=int(img_id) if isinstance(img_id, int) else None,
    )
