"""EvalRX summaries alongside lossless native VLMProbe results and plots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from evalrx.core.result import Result


def json_safe(value):
    """Bounded provenance descriptors, never embedding model/tensor contents."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"callable": getattr(value, "__qualname__", type(value).__name__)}
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    return {"type": type(value).__name__}


def summarize_probe(result, sample_id):
    summary = {
        "sample_id": sample_id,
        "method": result.method,
        "tensor_shapes": {key: list(value.shape) for key, value in result.tensors.items()},
    }
    # These are scalar task scores/effects, not layer x token x vocabulary
    # readouts or edge tables. Keep the complete tensors in artifacts.
    for key in ("baseline_score", "intervention_score", "effect", "estimated_effect", "donor_score"):
        value = result.tensors.get(key)
        if value is not None and value.numel() <= 128:
            summary[key] = json_safe(value.detach().cpu().reshape(-1).tolist())
    if "decoded_text" in result.metadata:
        summary["decoded_text"] = json_safe(result.metadata["decoded_text"])
    for key in ("effect_sign", "score_sign", "effect_measured", "readout", "sweep", "alignment"):
        if key in result.metadata:
            summary[key] = json_safe(result.metadata[key])
    return summary


@dataclass
class VLMProbeResult(Result):
    """Native tensors stay available for signed, coordinate-preserving figures."""

    @property
    def probe_results(self):
        return self.artifacts["probe_results"]

    def save_probes(self, directory):
        """Save tensors plus metadata by case index, with a JSON case manifest."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (row, native) in enumerate(zip(self.findings["per_case"], self.probe_results)):
            name = f"case-{index:04d}"
            native.save(root / name)
            rows.append({"sample_id": row["sample_id"], "directory": name, "method": native.method})
            if index in self.artifacts.get("evaluations", {}):
                self.artifacts["evaluations"][index].save(root / name / "faithfulness")
        manifest = {**self.to_dict(), "probe_files": rows}
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        return root

    def visualize(self, index=0, *, plot="lens_heatmap", **kwargs):
        """Delegate to built-in plots, retaining native results and geometry.

        ``attention_overlay`` additionally needs ``image=...``; coordinates and
        patch-grid geometry remain explicit caller inputs. Other plots accepting
        prepared series/edges use ``data=...``. Nothing infers visual grids from
        tensor shape or silently re-normalizes signed scores.
        """
        from .bridge import import_vlmprobe

        viz = import_vlmprobe().visualization
        allowed = {
            "lens_heatmap", "attention_overlay", "causal_heatmap", "intervention_curves",
            "connections", "head_paths", "receiver_paths",
        }
        if plot not in allowed:
            raise ValueError(f"Unknown plot {plot!r}; choose {sorted(allowed)}")
        data = kwargs.pop("data", self.probe_results[index])
        function = getattr(viz, f"plot_{plot}")
        if plot == "attention_overlay":
            if "image" not in kwargs:
                raise ValueError("attention_overlay requires image=... and the native visual-grid geometry")
            return function(kwargs.pop("image"), data, **kwargs)
        return function(data, **kwargs)
