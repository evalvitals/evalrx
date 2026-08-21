"""Relative attention to image tokens — "MLLMs Know Where to Look" (arXiv 2502.17422).

For a task-specific prompt, divides its image-patch attention weights by those from
a generic baseline prompt to reveal which patches are *uniquely* important for the
question being asked.  VLM-only; requires a white-box model whose ``forward()``
populates ``trace.extras["image_token_mask"]`` and optionally
``trace.extras["image_spatial_shape"]``.

References:
- Paper: "MLLMs Know Where to Look" — https://arxiv.org/abs/2502.17422
- Code:  https://github.com/saccharomycetes/mllms_know
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import Inputs
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)


def _blend_heatmap_on_image(
    image: "Any", heatmap: np.ndarray, *, alpha: float = 0.6, cmap: str = "jet",
    diverging: bool = False,
) -> "Any":
    """Alpha-blend a 2-D heatmap onto *image*, weighted by per-pixel intensity.

    Background (near-zero) regions stay close to the original photo; the
    colour-mapped heatmap shows through more strongly where the signal is
    high (CAM-style overlay) — this is what lets a human or a multimodal
    judge sanity-check whether attention lands on something that makes sense,
    rather than reading an abstract colour grid with no spatial reference.

    *diverging*: when True (signed maps like a FAIL-minus-PASS difference),
    normalises symmetrically around zero and uses ``cmap``'s full range for
    sign; the alpha weight is driven by ``|heatmap|`` magnitude either way.
    """
    from matplotlib import colormaps
    from PIL import Image

    base = image.convert("RGB")
    h = heatmap.astype(np.float64)
    if diverging:
        bound = float(max(abs(h.min()), abs(h.max()))) or 1.0
        norm = (h + bound) / (2 * bound)       # -bound..bound -> 0..1
        intensity = np.abs(h) / bound          # 0..1 magnitude
    else:
        lo, hi = float(h.min()), float(h.max())
        norm = (h - lo) / (hi - lo + 1e-8)
        intensity = norm

    def _upsample(mat: np.ndarray) -> np.ndarray:
        img = Image.fromarray((np.clip(mat, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        return np.asarray(img.resize(base.size, Image.BILINEAR), dtype=np.float64) / 255.0

    norm_up = _upsample(norm)
    intensity_up = _upsample(intensity)

    cmap_fn = colormaps[cmap]
    heat_rgb = (cmap_fn(norm_up)[:, :, :3] * 255).astype(np.uint8)

    base_arr = np.asarray(base, dtype=np.float64)
    a = (intensity_up * alpha)[:, :, None]
    out = base_arr * (1 - a) + heat_rgb.astype(np.float64) * a
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def _resolve_image(image: "Any") -> "Any | None":
    """Decode ``Inputs.image`` to a PIL Image when it's a path/URL the backend
    resolves rather than an already-decoded object (see ``Inputs`` docstring
    in :mod:`evalvitals.core.case` — datasets like ``TextVQASizeDataset``
    store a lazy file path, only opened when the model actually forwards).

    Delegates to ``transformers.image_utils.load_image`` — the exact helper
    ``hf_local``'s processor call resolves images through — so the overlay
    sees the same image the model's forward pass did.  Returns ``None``
    (never raises) when *image* is already unusable or can't be resolved.
    """
    if image is None or hasattr(image, "convert"):
        return image
    if not isinstance(image, str):
        return None
    try:
        from transformers.image_utils import load_image
        return load_image(image)
    except Exception:
        return None


def resolve_attention_layer(layer: "int | float", n_layers: int) -> int:
    """Map a layer spec to an absolute decoder-layer index.

    - ``int`` (incl. negative): used as an absolute index unchanged.
    - ``float`` in (0, 1): a *fractional depth*, ``round(layer * (n_layers - 1))``.

    The last decoder layer is dominated by attention sinks (BOS/edge tokens) and
    is **not** spatially grounded; spatial localization over image patches lives
    in the late-middle layers.  "MLLMs Know Where to Look" reads layer 22 of 28
    for Qwen2.5-VL-7B (≈0.78 depth).  A fractional default keeps this choice
    model-agnostic — it scales with the network's depth instead of hard-coding
    one layer index that only happens to suit a single model.
    """
    if isinstance(layer, float):
        if not 0.0 < layer < 1.0:
            raise ValueError(f"fractional layer must be in (0, 1), got {layer}")
        return int(round(layer * (n_layers - 1)))
    return int(layer)


def image_token_attention(attn_layer, mask):
    """Head-averaged attention from the last query position to image patches.

    The single reduction every image-attention consumer shares: average one
    decoder layer's attention over heads, take the last query row, and keep only
    the image-token columns.

    Args:
        attn_layer: one layer's attention, shape ``(heads, seq, seq)``.
        mask:       boolean image-token mask over the ``seq`` axis.

    Returns:
        1-D tensor of length ``n_image_tokens``.
    """
    return attn_layer.float().mean(dim=0)[-1][mask]


def attention_heatmap(
    model: "Model", case: "FailureCase", layer: "int | float" = 0.75
) -> "np.ndarray | None":
    """One ATTENTION forward → (H, W) image-patch heatmap, or ``None``.

    Host-side capture used both by analyzers and by the fix module's read-only
    ``model_attend()`` bridge (:mod:`evalvitals.eval_agent.stages.fix_pipeline`):
    run one attention forward, reduce it with :func:`image_token_attention`, and
    reshape to the backend's ``image_spatial_shape`` (near-square fallback).
    ``layer`` is resolved by :func:`resolve_attention_layer` — a float is a
    fractional depth (default 0.75, a spatially-grounded late-middle layer; the
    last layer is sink-dominated and localizes poorly).
    """
    from evalvitals.core.capability import Capability

    if Capability.ATTENTION not in getattr(model, "capabilities", frozenset()):
        return None
    try:
        trace = model.forward(case.inputs, capture={Capability.ATTENTION})
        attns = trace.require(Capability.ATTENTION)
        mask = trace.extras.get("image_token_mask")
        if mask is None:
            return None
        layer_idx = resolve_attention_layer(layer, len(attns))
        heat = image_token_attention(attns[layer_idx], mask).cpu().numpy().astype(np.float64)
        if heat.size == 0:
            return None
        shape = trace.extras.get("image_spatial_shape")
        if shape is not None and int(shape[0]) * int(shape[1]) == heat.size:
            h, w = int(shape[0]), int(shape[1])
        else:  # near-square fallback
            h = max(1, int(np.sqrt(heat.size)))
            while heat.size % h:
                h -= 1
            w = heat.size // h
        return heat.reshape(h, w)
    except Exception as exc:
        logger.debug("attention_heatmap failed for %s: %s", getattr(case, "id", "?"), exc)
        return None


#: Spatial-map artifact keys that can be overlaid on a representative case
#: image; each has a matching ``findings["<key>_case_id"]`` recorded by
#: ``RelativeAttentionAnalyzer._run`` naming which case to anchor it on.
_OVERLAYABLE_KEYS = ("spatial_map", "fail_mean_map", "pass_mean_map",
                    "diff_map_fail_minus_pass")


@dataclass
class RelativeAttentionResult(Result):
    """Result of relative attention analysis.

    Artifacts:
        attn_map         — 1-D float32 array of relative weights over image-patch tokens.
        spatial_map       — 2-D array (H, W) reshaped from attn_map; same as attn_map
                            when the grid shape cannot be inferred.
        fail_mean_map     — mean spatial_map over FAIL-labelled cases (when present).
        pass_mean_map     — mean spatial_map over PASS-labelled cases (when present).
        diff_map_fail_minus_pass — fail_mean_map - pass_mean_map (when both present).
    """

    @property
    def attn_map(self) -> np.ndarray:
        return self.artifacts["attn_map"]

    @property
    def spatial_map(self) -> np.ndarray:
        return self.artifacts.get("spatial_map", self.artifacts["attn_map"])

    def _representative_image(self, key: str) -> "Any | None":
        """The PIL image of the case backing ``artifacts[key]``, or ``None``.

        Looked up via ``findings[f"{key}_case_id"]`` — the case whose forward
        pass produced ``spatial_map``, or one representative case from the
        FAIL/PASS group for the mean/diff maps (see :meth:`overlay`).
        """
        case_id = (self.findings or {}).get(f"{key}_case_id")
        if case_id is None or self.cases is None:
            return None
        for c in self.cases:
            if c.id == case_id:
                return _resolve_image(getattr(c.inputs, "image", None))
        return None

    def overlay(self, key: str = "spatial_map", *, alpha: float = 0.6, cmap: str = "jet"):
        """Alpha-blend ``artifacts[key]`` onto its representative case image.

        Lets a human (or a multimodal judge) sanity-check whether the
        highlighted patches correspond to something sensible in the actual
        photo, instead of reading an abstract heatmap with no spatial
        reference — see :func:`_blend_heatmap_on_image`.

        ``fail_mean_map`` / ``pass_mean_map`` / ``diff_map_fail_minus_pass``
        are averages that can span *different* source images; the overlay
        anchors them on ONE representative case from that group purely to
        give the heatmap a spatial anchor, not a literal per-pixel
        correspondence — the image is genuinely illustrative there, not the
        exact image each contributing forward pass saw.

        Returns ``None`` (does not raise) when the map, its representative
        image, or matplotlib is unavailable — overlays are best-effort.
        """
        mat = self.artifacts.get(key)
        image = self._representative_image(key)
        if mat is None or mat.ndim != 2 or not hasattr(image, "convert"):
            return None
        try:
            return _blend_heatmap_on_image(
                image, mat, alpha=alpha, cmap=cmap, diverging="diff" in key)
        except ImportError:
            return None

    def save_overlay(self, key: str, path: "Any", **kwargs) -> bool:
        """Render :meth:`overlay` and save it to *path*; ``True`` on success."""
        img = self.overlay(key, **kwargs)
        if img is None:
            return False
        img.save(path)
        return True

    def image_overlays(self, fig_dir: "Any", stem_prefix: str) -> "list[Any]":
        """Save every available overlay under ``fig_dir``; return the paths.

        Hook called by :class:`~evalvitals.eval_agent.run_logger.RunLogger`
        (duck-typed — any :class:`Result` subclass may define this) so the
        overlay PNGs land in ``figures/`` alongside the bare heatmaps and get
        forwarded to a multimodal judge the same way.
        """
        from pathlib import Path

        out = []
        for key in _OVERLAYABLE_KEYS:
            if key not in self.artifacts:
                continue
            path = Path(fig_dir) / f"{stem_prefix}_{key}_overlay.png"
            try:
                if self.save_overlay(key, path):
                    out.append(path)
            except Exception:  # noqa: BLE001 - best-effort visualisation
                pass
        return out

    def plot(self, ax=None, figsize=(6, 6), cmap="hot"):
        """Heatmap of the relative attention spatial map (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Visualisation requires matplotlib: pip install evalvitals[viz]"
            ) from None

        mat = self.spatial_map
        if mat.ndim == 1:
            raise ValueError(
                "Spatial shape is unknown (image_spatial_shape missing from trace.extras). "
                "The 1-D attn_map is stored in self.artifacts['attn_map']."
            )
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title("Relative Attention Map")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if created:
            plt.tight_layout()
            return fig
        return ax


@register_analyzer("relative_attention")
class RelativeAttentionAnalyzer(Analyzer):
    """Relative attention to image-patch tokens.

    Runs two forward passes through the same VLM — one with the task-specific
    prompt, one with a generic ``general_prompt`` — then computes the element-wise
    ratio of their last-token attention weights over the image-patch positions.
    Regions with a high ratio are uniquely relevant to the task question.

    Reference:
        "MLLMs Know Where to Look" (arXiv 2502.17422)
        https://arxiv.org/abs/2502.17422 · https://github.com/saccharomycetes/mllms_know

    Runs over the **whole batch** (one specific/general forward pair per case)
    and emits per-case focus metrics so the M2 stats layer can correlate
    attention behaviour with PASS/FAIL labels — e.g. "missed-finding cases show
    significantly lower task-specific attention".  When cases carry labels, the
    artifacts additionally include the mean spatial maps of the FAIL and PASS
    groups and their difference (``diff_map_fail_minus_pass``) for visual
    comparison.

    Hyper-parameters:
        general_prompt: Baseline prompt (same image, generic question).
        layer:          Decoder layer to read attention from.  An ``int`` is an
                        absolute index (negative allowed); a ``float`` in (0, 1)
                        is a fractional depth (default 0.75 — a late-middle layer
                        where image-patch attention is spatially grounded; the
                        last layer is sink-dominated, see
                        :func:`resolve_attention_layer`).
        top_k:          Number of highest-scoring patches to include in findings.
        max_cases:      Cap on analysed cases (2 attention-captured forwards each); 0 (the default) = every case.

    Example::

        from PIL import Image
        from evalvitals.analyzers.attention import RelativeAttentionAnalyzer
        from evalvitals.core.case import Inputs

        analyzer = RelativeAttentionAnalyzer(layer=22)
        case = Inputs(prompt="What color is the car?", image=Image.open("scene.jpg"))
        result = analyzer.run(vlm_model, case)
        result.plot()
    """

    name = "relative_attention"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"image", "video"})

    def __init__(
        self,
        general_prompt: str = "Describe the image.",
        layer: "int | float" = 0.75,
        top_k: int = 5,
        max_cases: int = 0,
    ) -> None:
        super().__init__(
            general_prompt=general_prompt, layer=layer, top_k=top_k, max_cases=max_cases
        )

    def _run(self, model: "Model", cases: "CaseBatch") -> RelativeAttentionResult:
        per_case: list[dict] = []
        errors: list[str] = []
        first_exc: Exception | None = None
        first: tuple | None = None  # (rel_np, spatial_map, n_img, n_layers)
        first_case_id: str | None = None
        maps_fail: list[np.ndarray] = []
        maps_pass: list[np.ndarray] = []
        fail_rep_id: str | None = None
        pass_rep_id: str | None = None
        # Per-case spatial maps (float16) — kept as an artifact (dropped from the
        # light result.json) so the WS4 attention_decoding omnibus can read the
        # full map per case, not just the scalar reductions.
        per_case_maps: dict[str, np.ndarray] = {}

        for case in cases.stratified_head(self.max_cases):
            try:
                rel_np, spatial_map, n_img, n_layers = self._relative_map(model, case)
            except Exception as exc:  # noqa: BLE001 - record and continue the batch
                errors.append(f"{case.id}: {exc}")
                if first_exc is None:
                    first_exc = exc
                continue
            if first is None:
                first = (rel_np, spatial_map, n_img, n_layers)
                first_case_id = case.id

            k = min(self.top_k, rel_np.size)
            topk_share = float(np.sort(rel_np)[::-1][:k].sum() / (rel_np.sum() + 1e-8))
            entry = {
                "id": case.id,
                "max_relative_weight": round(float(rel_np.max()), 4),
                "mean_relative_weight": round(float(rel_np.mean()), 4),
                "focus_share": round(topk_share, 4),
            }
            entry.update(_attention_features(rel_np, spatial_map))
            per_case.append(entry)
            per_case_maps[case.id] = spatial_map.astype(np.float16)

            label = getattr(getattr(case, "label", None), "value", None)
            if label == "fail":
                maps_fail.append(spatial_map)
                if fail_rep_id is None:
                    fail_rep_id = case.id
            elif label == "pass":
                maps_pass.append(spatial_map)
                if pass_rep_id is None:
                    pass_rep_id = case.id

        if first is None:
            # Whole batch failed — preserve the original single-case semantics
            # (probe_agent catches this and warns with the actionable message).
            raise first_exc if first_exc is not None else ValueError("empty case batch")

        rel_np, spatial_map, n_img, n_layers = first
        topk_idx = np.argsort(rel_np)[::-1][: self.top_k]
        top_patches = [
            {"patch_idx": int(i), "relative_weight": round(float(rel_np[i]), 4)}
            for i in topk_idx
        ]

        artifacts: dict = {"attn_map": rel_np, "spatial_map": spatial_map}
        if per_case_maps:
            artifacts["per_case_maps"] = per_case_maps
        fail_mean = _group_mean_map(maps_fail)
        pass_mean = _group_mean_map(maps_pass)
        if fail_mean is not None:
            artifacts["fail_mean_map"] = fail_mean
        if pass_mean is not None:
            artifacts["pass_mean_map"] = pass_mean
        if (
            fail_mean is not None and pass_mean is not None
            and fail_mean.shape == pass_mean.shape
        ):
            artifacts["diff_map_fail_minus_pass"] = fail_mean - pass_mean

        maxes = [e["max_relative_weight"] for e in per_case]
        findings: dict = {
            "n_image_tokens": n_img,
            "n_layers": n_layers,
            "layer_used": resolve_attention_layer(self.layer, n_layers),
            "map_shape": list(spatial_map.shape),
            "top_patches": top_patches,
            "n_cases_analyzed": len(per_case),
            "n_errors": len(errors),
            "mean_max_relative_weight": round(float(np.mean(maxes)), 4),
            "per_case": per_case,
            # Anchors RelativeAttentionResult.overlay(): which case's image each
            # spatial-map-shaped artifact should be alpha-blended onto.
            "spatial_map_case_id": first_case_id,
        }
        if "fail_mean_map" in artifacts:
            findings["fail_mean_map_case_id"] = fail_rep_id
        if "pass_mean_map" in artifacts:
            findings["pass_mean_map_case_id"] = pass_rep_id
        if "diff_map_fail_minus_pass" in artifacts:
            findings["diff_map_fail_minus_pass_case_id"] = fail_rep_id
        if maps_fail and maps_pass:
            label_by_id = {
                c.id: getattr(getattr(c, "label", None), "value", None) for c in cases
            }
            fail_maxes = [e["max_relative_weight"] for e in per_case
                          if label_by_id.get(e["id"]) == "fail"]
            pass_maxes = [e["max_relative_weight"] for e in per_case
                          if label_by_id.get(e["id"]) == "pass"]
            if fail_maxes and pass_maxes:
                findings["fail_mean_max_relative_weight"] = round(float(np.mean(fail_maxes)), 4)
                findings["pass_mean_max_relative_weight"] = round(float(np.mean(pass_maxes)), 4)

        result = RelativeAttentionResult(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            artifacts=artifacts,
        )
        result.findings = findings
        return result

    def _relative_map(self, model: "Model", case) -> "tuple[np.ndarray, np.ndarray, int, int]":
        """One specific/general forward pair → (rel_map, spatial_map, n_img, n_layers)."""
        import torch  # noqa: F401 - tensor ops below

        # Forward pass 1: task-specific prompt
        specific_trace = model.forward(case.inputs, capture={Capability.ATTENTION})

        # Forward pass 2: generic baseline — same image/video, generic question
        general_inputs = Inputs(
            prompt=self.general_prompt,
            image=case.inputs.image,
            video=getattr(case.inputs, "video", None),
        )
        general_trace = model.forward(general_inputs, capture={Capability.ATTENTION})

        # Image-token masks — each trace reports its own positions because the
        # two prompts have different sequence lengths.
        specific_mask = specific_trace.extras.get("image_token_mask")
        general_mask = general_trace.extras.get("image_token_mask")

        if specific_mask is None or general_mask is None:
            raise ValueError(
                "trace.extras['image_token_mask'] is missing. "
                "RelativeAttentionAnalyzer requires a VLM model whose forward() "
                "populates image token positions (hf_local with is_vlm=True)."
            )

        n_img = int(specific_mask.sum())
        if n_img == 0:
            raise ValueError("image_token_mask contains no True entries — no image tokens found.")
        if int(general_mask.sum()) != n_img:
            raise ValueError(
                f"Image token count differs between specific ({n_img}) and "
                f"general ({int(general_mask.sum())}) forward passes. "
                "Both prompts must use the same image."
            )

        n_layers = len(specific_trace.require(Capability.ATTENTION))
        layer_idx = resolve_attention_layer(self.layer, n_layers)

        def _img_attn(trace, mask):
            """Head-averaged attention from the last query position to image patches."""
            attns = trace.require(Capability.ATTENTION)
            return image_token_attention(attns[layer_idx], mask)  # (n_img_tokens,)

        specific_attn = _img_attn(specific_trace, specific_mask)
        general_attn = _img_attn(general_trace, general_mask)

        # Relative attention: ratio highlighting task-relevant patches
        rel = specific_attn / (general_attn + 1e-8)
        rel_np = rel.cpu().float().numpy()

        # Spatial reshape using the grid shape stored by the VLM backend
        spatial_shape = specific_trace.extras.get("image_spatial_shape")
        if spatial_shape is not None:
            h, w = spatial_shape
            spatial_map = rel_np.reshape(h, w) if h * w == n_img else rel_np
        else:
            side = int(n_img ** 0.5)
            spatial_map = rel_np.reshape(side, side) if side * side == n_img else rel_np

        return rel_np, spatial_map, n_img, n_layers


def _attention_features(rel_np: np.ndarray, spatial_map: np.ndarray) -> dict:
    """Richer per-case scalars derived from one case's relative-attention map.

    Beyond the raw max/mean weight, these capture the SHAPE of attention so M2
    can test the mechanisms M3 reasons about:
      - ``attention_entropy``  normalized Shannon entropy of the weight
        distribution (1 = diffuse/smeared, 0 = a single spike) → the
        "diffuse vs. concentrated" regime split.
      - ``top1_share``         fraction of total weight on the single peak patch
        (single-patch dominance).
      - ``center_offset``      distance of the attention centre-of-mass from the
        image centre, normalized to [0, 1] (peripheral vs. central attention).
      - ``edge_mass``          fraction of weight on the border ring of patches
        (a positional-sink proxy — sinks tend to sit at edges/corners).
    All are continuous in [0, 1] (or ≥0), so they enter M2 as genuine
    discriminators, never as a label-reconstructing flag.
    """
    w = rel_np.astype(np.float64).ravel()
    total = float(w.sum()) + 1e-12
    p = w / total
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    ent_norm = ent / (np.log(max(w.size, 2))) if w.size > 1 else 0.0
    feats: dict = {
        "attention_entropy": round(float(ent_norm), 4),
        "top1_share": round(float(w.max() / total), 4),
    }
    if spatial_map.ndim == 2 and spatial_map.size > 1:
        m = spatial_map.astype(np.float64)
        msum = float(m.sum()) + 1e-12
        h, w2 = m.shape
        rows = np.arange(h)[:, None]
        cols = np.arange(w2)[None, :]
        com_r = float((m * rows).sum() / msum)
        com_c = float((m * cols).sum() / msum)
        cr, cc = (h - 1) / 2.0, (w2 - 1) / 2.0
        off = np.hypot((com_r - cr) / (cr + 1e-9), (com_c - cc) / (cc + 1e-9)) / np.sqrt(2.0)
        feats["center_offset"] = round(float(min(off, 1.0)), 4)
        border = np.zeros((h, w2), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        feats["edge_mass"] = round(float(m[border].sum() / msum), 4)
    return feats


def _group_mean_map(maps: "list[np.ndarray]") -> "np.ndarray | None":
    """Mean of the spatial maps sharing the most common shape (None when empty).

    Different source images can produce different patch grids; averaging is only
    meaningful within one grid shape, so minority shapes are dropped.
    """
    if not maps:
        return None
    from collections import Counter

    shape = Counter(m.shape for m in maps).most_common(1)[0][0]
    same = [m for m in maps if m.shape == shape]
    return np.mean(np.stack(same), axis=0)