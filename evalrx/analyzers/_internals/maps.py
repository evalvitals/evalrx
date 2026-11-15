"""Per-image-token scores → spatial maps, shape features, and an overlayable Result.

Grad-CAM and attention relevance both end in one score per image token. This
module turns those into what ``relative_attention`` already produces, so the
run logger, the tensor-level omnibus and the multimodal judge consume them the
same way:

* ``artifacts["spatial_map"]`` (first case), ``per_case_maps`` (float16 per
  case), ``fail_mean_map`` / ``pass_mean_map`` / ``diff_map_fail_minus_pass``;
* ``findings["<key>_case_id"]`` naming the case whose image each map overlays;
* :class:`SpatialMapResult.image_overlays` — the duck-typed hook RunLoggerV2
  calls to save heatmap-on-photo PNGs.

numpy and the overlay helpers are imported inside functions: this module is
imported by analyzers that load on the light install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Trace

OVERLAYABLE_KEYS = ("spatial_map", "fail_mean_map", "pass_mean_map", "diff_map_fail_minus_pass")


def image_token_scores(trace: "Trace", values: Any) -> "tuple[Any, Any]":
    """``(flat, spatial)`` for the FIRST image of *trace*.

    *values* is a 1-D array aligned with ``trace.token_type_map.image_pos``.
    ``spatial`` is the post-merge ``(h, w)`` grid (frames of a video summed),
    or the flat vector when the grid cannot be rebuilt.
    """
    import numpy as np

    flat = np.asarray(values, dtype=np.float64).ravel()
    ttm = getattr(trace, "token_type_map", None)
    if ttm is None or not ttm.grids or len(ttm.patches) != flat.size:
        shape = (getattr(trace, "extras", None) or {}).get("image_spatial_shape")
        if shape is not None and int(shape[0]) * int(shape[1]) == flat.size:
            return flat, flat.reshape(int(shape[0]), int(shape[1]))
        return flat, flat
    _, h, w = ttm.grids[0]
    grid = np.zeros((int(h), int(w)), dtype=np.float64)
    for value, (img, _t, row, col) in zip(flat, ttm.patches):
        if img == 0:
            grid[row, col] += value
    return flat, grid


def map_features(flat: Any, spatial: Any, prefix: str) -> "dict[str, float]":
    """Shape of a non-negative per-token map, as ``{prefix}_{feature}`` scalars.

    Same features ``relative_attention`` reports, so M2 can compare readings:
    normalized entropy (1 = diffuse), top-1 share, centre-of-mass offset from
    the image centre, and mass on the border ring.
    """
    import numpy as np

    from evalrx.analyzers.attention.relative_attn import _attention_features

    w = np.clip(np.asarray(flat, dtype=np.float64), 0.0, None)
    if not w.size or float(w.sum()) <= 0.0:
        return {}
    spatial = np.clip(np.asarray(spatial, dtype=np.float64), 0.0, None)
    feats = _attention_features(w, spatial)
    k = min(5, w.size)
    feats["top5_share"] = round(float(np.sort(w)[::-1][:k].sum() / w.sum()), 4)
    return {f"{prefix}_{name.replace('attention_', '')}": value for name, value in feats.items()}


def group_maps(per_case_maps: "dict[str, Any]", cases: "CaseBatch") -> "tuple[dict, dict]":
    """``(artifacts, findings)`` for the FAIL/PASS mean maps and their difference."""
    from evalrx.analyzers.attention.relative_attn import _group_mean_map

    label_of = {c.id: getattr(getattr(c, "label", None), "value", None) for c in cases}
    fail = {cid: m for cid, m in per_case_maps.items() if label_of.get(cid) == "fail" and m.ndim == 2}
    passed = {cid: m for cid, m in per_case_maps.items() if label_of.get(cid) == "pass" and m.ndim == 2}
    artifacts: dict = {}
    findings: dict = {}
    fail_mean = _group_mean_map([m.astype("float64") for m in fail.values()])
    pass_mean = _group_mean_map([m.astype("float64") for m in passed.values()])
    if fail_mean is not None:
        artifacts["fail_mean_map"] = fail_mean
        findings["fail_mean_map_case_id"] = next(iter(fail))
    if pass_mean is not None:
        artifacts["pass_mean_map"] = pass_mean
        findings["pass_mean_map_case_id"] = next(iter(passed))
    if fail_mean is not None and pass_mean is not None and fail_mean.shape == pass_mean.shape:
        artifacts["diff_map_fail_minus_pass"] = fail_mean - pass_mean
        findings["diff_map_fail_minus_pass_case_id"] = next(iter(fail))
    return artifacts, findings


@dataclass
class SpatialMapResult(Result):
    """A Result carrying image-token maps that can be blended onto the case photo."""

    def _representative_image(self, key: str) -> Any:
        from evalrx.analyzers.attention.relative_attn import _resolve_image

        case_id = (self.findings or {}).get(f"{key}_case_id")
        if case_id is None or self.cases is None:
            return None
        for c in self.cases:
            if c.id == case_id:
                return _resolve_image(getattr(c.inputs, "image", None))
        return None

    def overlay(self, key: str = "spatial_map", *, alpha: float = 0.6, cmap: str = "jet") -> Any:
        """``artifacts[key]`` alpha-blended on its representative image, or ``None``."""
        from evalrx.analyzers.attention.relative_attn import _blend_heatmap_on_image

        mat = self.artifacts.get(key)
        image = self._representative_image(key)
        if mat is None or getattr(mat, "ndim", 0) != 2 or not hasattr(image, "convert"):
            return None
        try:
            return _blend_heatmap_on_image(image, mat, alpha=alpha, cmap=cmap, diverging="diff" in key)
        except ImportError:
            return None

    def image_overlays(self, fig_dir: Any, stem_prefix: str) -> "list[Any]":
        """Save every available overlay under ``fig_dir`` (RunLoggerV2 hook)."""
        from pathlib import Path

        out = []
        for key in OVERLAYABLE_KEYS:
            if key not in self.artifacts:
                continue
            path = Path(fig_dir) / f"{stem_prefix}_{key}_overlay.png"
            try:
                img = self.overlay(key)
                if img is not None:
                    img.save(path)
                    out.append(path)
            except Exception:  # noqa: BLE001 - best-effort visualisation
                pass
        return out
