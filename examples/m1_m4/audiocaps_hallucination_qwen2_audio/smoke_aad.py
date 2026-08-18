"""One-off, real-weights smoke test for generate_aad -- cheap pre-flight
check before spending a full mining+loop GPU run. Deletable after use.

    python smoke_aad.py --n 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evalvitals.core.case import Inputs
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.specs import get_spec

DATA = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    records = [json.loads(l) for l in (DATA / "audiohallucination.jsonl").read_text().splitlines() if l]
    picked = records[: args.n]
    print(f"picked {len(picked)} cases")

    print("loading model...")
    t0 = time.monotonic()
    model = HFLocalModel(get_spec("qwen2-audio-7b-instruct"), RuntimeConfig(device="cuda", dtype="bfloat16", max_new_tokens=16))
    model.load()
    print(f"loaded in {time.monotonic() - t0:.1f}s")
    print(f"paper_method_fidelity('aad') = {model.paper_method_fidelity('aad')!r}")

    for row in picked:
        inputs = Inputs(prompt=row["instruction"], audio=str(DATA / row["audio_path"]))
        print(f"\n=== {row['id']}: {row['instruction']!r} (gold={row['expected']!r}) ===")

        t0 = time.monotonic()
        try:
            base = model.generate(inputs)
            print(f"  generate()   -> {base!r}  ({time.monotonic()-t0:.1f}s)")
        except Exception as exc:
            print(f"  generate() FAILED: {exc!r}")
            import traceback; traceback.print_exc()

        t0 = time.monotonic()
        try:
            aad_out = model.generate_aad(inputs)
            print(f"  generate_aad -> {aad_out!r}  ({time.monotonic()-t0:.1f}s)")
        except Exception as exc:
            print(f"  generate_aad FAILED: {exc!r}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
