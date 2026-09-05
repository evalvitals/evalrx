"""Route B: full-coverage attention extraction for all three Qwen3-VL
checkpoints over ALL their cases (no max_cases cap).

For each model file in ../../diagnosis_loops/deco_hallu/data/cases/, runs the
RelativeAttentionAnalyzer (2 attention-captured forwards per case) and writes
the 7 per-case attention-geometry scalars back into an enriched copy of the
cases JSON under data_attn_full/, ready to hand to `evalrx explore`.
Per-case spatial maps are kept as float16 .npz alongside for future
tensor-level analyses.

    python extract_attention_all.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LOOP_DIR = HERE.parent.parent / "m1_m5" / "deco_hallu"
sys.path.insert(0, str(LOOP_DIR))  # reuse run.py's manifest loader (images incl.)

OUT_DATA = HERE / "data_attn_full"
OUT_MAPS = OUT_DATA / "maps"

MODELS = ["qwen3-vl-2b-instruct", "qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct"]
ATTN_FIELDS = ["attention_entropy", "focus_share", "center_offset", "edge_mass",
               "top1_share", "max_relative_weight", "mean_relative_weight"]

COCO_URL = "http://images.cocodataset.org/val2014/{}"


def ensure_images(model_keys: list[str]) -> Path:
    """Build data_images_all/: symlinks to deco_pope's shared images plus any
    missing COCO val2014 files downloaded from the official server. The shared
    dir only holds the 2B batch's images (and may be read-only), so the 4B/8B
    batches need this merged, locally-owned image dir."""
    import urllib.request

    shared = LOOP_DIR.parent / "deco_pope" / "data" / "images"
    merged = HERE / "data_images_all"
    merged.mkdir(exist_ok=True)
    for src in shared.glob("*.jpg"):
        dst = merged / src.name
        if not dst.exists():
            dst.symlink_to(src)
    need: set[str] = set()
    for mk in model_keys:
        raw = json.load(open(LOOP_DIR / "data" / "cases" / f"{mk}.json"))
        need |= {r["file_name"] for r in raw["cases"]}
    missing = sorted(f for f in need if not (merged / f).exists())
    if missing:
        print(f"downloading {len(missing)} missing COCO val2014 image(s)...", flush=True)
        for name in missing:
            urllib.request.urlretrieve(COCO_URL.format(name), merged / name)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    import numpy as np

    import run as loop_run  # examples/m1_m5/deco_hallu/run.py

    loop_run.IMAGES = ensure_images(args.models)

    from evalrx import compose
    from evalrx.analyzers.attention.relative_attn import RelativeAttentionAnalyzer
    from evalrx.core.capability import Capability
    from evalrx.models.backends.base import RuntimeConfig

    OUT_DATA.mkdir(exist_ok=True)
    OUT_MAPS.mkdir(exist_ok=True)

    for model_key in args.models:
        t0 = time.time()
        print(f"\n=== {model_key} ===", flush=True)
        cases, raw = loop_run.load_manifest(model_key)
        n = len(list(cases))
        print(f"cases={n}", flush=True)

        model = compose(model_key, "hf_local",
                        runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                        want={Capability.GENERATE, Capability.HIDDEN_STATES,
                              Capability.ATTENTION})

        analyzer = RelativeAttentionAnalyzer(max_cases=n)  # no cap: every case
        result = analyzer.run(model, cases)
        per_case = (result.findings or {}).get("per_case") or []
        maps = (result.artifacts or {}).get("per_case_maps") or {}
        print(f"analyzed={len(per_case)}/{n} in {time.time()-t0:.0f}s "
              f"(errors={len((result.findings or {}).get('errors') or [])})", flush=True)

        # join scalars back to raw rows via case.id -> (image_id, object)
        key_by_case = {c.id: (c.metadata.get("image_id"), c.metadata.get("object"))
                       for c in cases}
        attn_by_key = {key_by_case[r["id"]]: {f: r.get(f) for f in ATTN_FIELDS}
                       for r in per_case if r.get("id") in key_by_case}
        n_hit = 0
        for row in raw["cases"]:
            k = (row.get("image_id"), row.get("object"))
            attn = attn_by_key.get(k)
            if attn:
                row.update(attn)
                n_hit += 1
            else:
                row.update({f: None for f in ATTN_FIELDS})
        raw["attention_extraction"] = {
            "analyzer": "relative_attention (MLLMs Know Where to Look)",
            "coverage": f"{n_hit}/{len(raw['cases'])}",
            "note": "full-batch extraction, no max_cases cap",
        }
        out = OUT_DATA / f"{model_key}.json"
        json.dump(raw, open(out, "w"), indent=1)
        print(f"wrote {out} (attention on {n_hit}/{len(raw['cases'])} rows)", flush=True)

        if maps:
            np.savez_compressed(OUT_MAPS / f"{model_key}_maps.npz",
                                **{str(k): v for k, v in maps.items()})
            print(f"wrote {OUT_MAPS / (model_key + '_maps.npz')} ({len(maps)} maps)",
                  flush=True)

        # free VRAM before the next checkpoint
        del model, analyzer, result
        try:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
