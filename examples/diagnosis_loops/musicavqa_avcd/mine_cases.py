"""Offline miner (musicavqa_avcd) — Music-AVQA questions -> frozen manifest.

Mirrors examples/diagnosis_loops/deco_chair/mine_cases.py's shape: run the
model once, offline, over a sampled question pool; label each case
PASS/FAIL by comparing the model's answer to the dataset's ground truth;
freeze to data/cases/{model_key}.json so run.py's loop never needs to know
how labeling worked.

    python mine_cases.py --mock --n 40                    # CPU smoke test, no real weights
    python mine_cases.py --model videollama2.1-7b-av --n 200 --device cuda

Data paths default to the Drive-mounted dataset/model locations used in this
Colab session; override with --data-dir / --model-path if running elsewhere.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from avqa_data import answers_match, load_records

DATA = Path(__file__).parent / "data"
DEFAULT_DATASET_DIR = "/content/drive/MyDrive/av_llm_project/datasets/Music-AVQA"
DEFAULT_MODEL_PATH = "/content/drive/MyDrive/av_llm_project/models/VideoLLaMA2.1-7B-AV"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="videollama2.1-7b-av",
                     help="label used for the frozen manifest filename (not a registry key)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--modalities", nargs="+", default=["Audio-Visual", "Audio", "Visual"])
    ap.add_argument("--n", type=int, default=40, help="number of questions to mine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explore-frac", type=float, default=0.6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--mock", action="store_true",
                     help="use MockAVModel instead of loading the real 17GB checkpoint "
                          "(for wiring smoke tests on RAM-constrained hosts)")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)

    json_path = Path(args.data_dir) / f"avqa-{args.split}.json"
    videos_dir = Path(args.data_dir) / "videos"
    pool = load_records(json_path, videos_dir, modalities=set(args.modalities))
    if not pool:
        raise SystemExit(f"no usable records found under {args.data_dir} (split={args.split})")
    random.shuffle(pool)
    pool = pool[: args.n]
    print(f"mining {len(pool)} questions (pool had {len(pool)} after modality filter)")

    if args.mock:
        from videollama2_model import MockAVModel
        model = MockAVModel()
        print("model: MockAVModel (no real weights loaded)")
    else:
        from videollama2_model import VideoLLaMA2AVModel
        model = VideoLLaMA2AVModel(
            args.model_path, device=args.device,
            max_new_tokens=args.max_new_tokens, load_4bit=args.load_4bit,
            want_attention=False,  # mining only calls generate(); no forward() capture needed
        )
        print(f"model: VideoLLaMA2AVModel path={args.model_path} device={args.device}")

    from evalvitals.core.case import Inputs

    records = []
    n_fail = 0
    for i, rec in enumerate(pool):
        observed = model.generate(
            Inputs(prompt=rec["question"] + " Answer concisely.", video=rec["video_path"]),
            max_new_tokens=args.max_new_tokens, do_sample=False,
        )
        ok = answers_match(observed, rec["answer"])
        n_fail += not ok
        records.append({**rec, "observed": observed, "label": "pass" if ok else "fail"})
        print(f"  [{i + 1}/{len(pool)}] {rec['video_id']} q{rec['question_id']} "
              f"({rec['modality']}/{rec['qtype']}) gold={rec['answer']!r} "
              f"got={observed[:60]!r} -> {'PASS' if ok else 'FAIL'}")

    # Stratified explore/validate split by label (M2/M3 only ever see "explore";
    # "validate" is held out for M5-style confirmation the same way deco_chair does).
    for group in ("fail", "pass"):
        idxs = [i for i, r in enumerate(records) if r["label"] == group]
        random.shuffle(idxs)
        n_explore = round(len(idxs) * args.explore_frac)
        for rank, i in enumerate(idxs):
            records[i]["split"] = "explore" if rank < n_explore else "validate"

    out = DATA / "cases" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "mock": args.mock, "source_split": args.split,
        "modalities": args.modalities, "seed": args.seed, "n": len(records),
        "n_fail": n_fail, "n_pass": len(records) - n_fail,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "cases": records,
    }, indent=2, ensure_ascii=False))
    print(f"frozen -> {out}")
    print(f"fail={n_fail} pass={len(records) - n_fail} (fail-rate={n_fail / len(records):.2f})")


if __name__ == "__main__":
    main()
