"""Stage 0 — run the model once per item and freeze a LABELLED CaseBatch.

This is the only GPU-bound step before M4. It is separated from the pipeline so
that M2/M3/M5 can be re-run, re-prompted, and debugged against a FROZEN batch
without paying for generation again, and so two judges see literally the same
PASS/FAIL labels.

    python build_cases.py --model qwen3.5-9b --dataset supergpqa_law --n 120

Writes outputs/<model>/<dataset>/cases.json.

Why it refuses to write outside [0.15, 0.85]: M2 contrasts PASS against FAIL, so
a batch with almost none of either carries no signal, and every mechanism number
downstream would be computed on a pool that cannot support it. The band was
measured on Qwen3.5-9B; on 2B/4B the SAME slice can be floored or saturated,
which is a property of the pair, not a bug.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_band_probe"))
#: The checkout root (holds pyproject.toml). Found by walking up rather than
#: counting directories: this example moved from examples/ to
#: examples/dataset_selection/ in one reorg, and a hardcoded ``parent.parent``
#: silently started pointing at examples/ instead of the package root.
PKG_ROOT = next(p for p in HERE.resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(PKG_ROOT))

import band_locate as B  # noqa: E402
import datasets as CATALOG  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text())

#: Outside this range the batch has too little of one class to diagnose.
MIN_ACC, MAX_ACC = 0.15, 0.85


def build(model_id: str, base_url: str, dataset: str, n: int,
          concurrency: int, max_tokens: int, sampling: dict) -> dict:
    entry = CATALOG.get(dataset)
    spec = entry.spec
    B.MODEL_ID = model_id
    B.BASE_URL = base_url

    # n <= 0 means "every item in the slice". Capping is pointless below the
    # slice size and impossible above it, so a census is the honest default.
    requested = n
    if n <= 0:
        n = entry.items

    rows = B.fetch_rows(spec, n)
    adapter = spec.adapter or B._adapter_plain(spec.question_field, spec.answer_field)
    items = []
    for row in rows:
        pair = adapter(row)
        if pair:
            items.append(pair)
        if len(items) >= n:
            break
    if not items:
        raise SystemExit(f"{dataset}: adapter produced no gradable items")
    # A slice cannot yield more than it holds. Asking for 240 from a 125-item set
    # silently returns 125, and a batch half the requested size changes every
    # interval downstream -- so say so rather than letting it pass as n=240.
    if len(items) < n:
        print(f"  NOTE asked for {n} but the slice yields {len(items)} "
              f"({entry.items} recorded) — using every item there is")

    grade = spec.grader or B.answer_equal

    def _one(item):
        question, gold = item
        prompt = (f"{question}\n\n{spec.instruction}" if spec.append_instruction
                  else question)
        output, finish = B.generate(prompt, max_tokens, sampling["temperature"],
                                    sampling={k: v for k, v in sampling.items()
                                              if k != "temperature"})
        graded = output if spec.grades_raw_output else B.extract_answer(output)
        return {
            "prompt": prompt,
            "gold": gold if not isinstance(gold, (list, tuple)) else list(gold),
            "output": output,
            "label": "PASS" if bool(grade(graded, gold)) else "FAIL",
            "finish_reason": finish,
            "truncated": finish == "length",
        }

    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        cases = list(pool.map(_one, items))

    n_pass = sum(1 for c in cases if c["label"] == "PASS")
    acc = n_pass / len(cases)
    truncated = sum(1 for c in cases if c["truncated"]) / len(cases)
    errors = sum(1 for c in cases if str(c["finish_reason"]).startswith("error:"))
    return {
        "model": model_id,
        "dataset": dataset,
        "n": len(cases),
        "n_requested": requested,
        "slice_items": entry.items,
        #: the batch is the whole slice -- no sampling variability left for it,
        #: so the interval speaks to the task, not to which items were drawn
        "is_census": len(cases) >= entry.items,
        "accuracy": round(acc, 4),
        "n_pass": n_pass,
        "n_fail": len(cases) - n_pass,
        "truncated_rate": round(truncated, 4),
        "error_rate": round(errors / len(cases), 4),
        "max_tokens": max_tokens,
        "sampling": sampling,
        "seconds": round(time.time() - started, 1),
        "reference_9b_accuracy": entry.accuracy_9b,
        "cases": cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--base-url", default=CFG["base_url"])
    ap.add_argument("--dataset", default=CFG["dataset"])
    ap.add_argument("--n", type=int, default=CFG["n_cases"],
                    help="0 (default) = every item in the slice; >0 caps the sample")
    ap.add_argument("--concurrency", type=int, default=CFG["concurrency"])
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 (default) = the dataset's own measured budget if it "
                         "declares one, else config max_tokens")
    ap.add_argument("--force", action="store_true",
                    help="write the batch even if it is outside the usable band")
    args = ap.parse_args()

    sampling = {"temperature": float(CFG["temperature"]),
                "top_p": float(CFG["top_p"]), "top_k": int(CFG["top_k"])}

    # A dataset whose reference accuracy was measured at a larger budget must be
    # run at that budget or its FAIL labels are token-cap artefacts, not
    # capability. minervamath reads 0.500 at 65k and 0.320/budget_limited at 40k.
    entry = CATALOG.get(args.dataset)
    max_tokens = args.max_tokens or entry.max_tokens or int(CFG["max_tokens"])
    if not args.max_tokens and entry.max_tokens:
        print(f"  NOTE {args.dataset} declares max_tokens={entry.max_tokens} "
              f"(config default is {CFG['max_tokens']}) — using the declared one")

    n_label = "ALL" if args.n <= 0 else str(args.n)
    print(f"[build_cases] {args.model} x {args.dataset} n={n_label} "
          f"max_tokens={max_tokens}", flush=True)
    report = build(args.model, args.base_url, args.dataset, args.n,
                   args.concurrency, max_tokens, sampling)

    acc, trunc = report["accuracy"], report["truncated_rate"]
    census = " (CENSUS of the slice)" if report["is_census"] else ""
    print(f"  accuracy {report['n_pass']}/{report['n']} = {acc:.3f}{census} "
          f"(9B reference {report['reference_9b_accuracy']:.3f})")
    print(f"  truncated {trunc:.0%}   errors {report['error_rate']:.0%}   "
          f"{report['seconds']:.0f}s")

    if trunc > 0.10:
        print(f"  WARNING truncation {trunc:.0%} > 10%: some FAIL labels are budget "
              f"artefacts, not capability. Raise --max-tokens before trusting M2.")
    if not (MIN_ACC <= acc <= MAX_ACC) and not args.force:
        raise SystemExit(
            f"  REFUSING to write: accuracy {acc:.3f} outside [{MIN_ACC}, {MAX_ACC}] "
            f"— one class is too thin for M2 to contrast. Pick another dataset for "
            f"this model size, or pass --force if you know why you want it."
        )

    out = HERE / "outputs" / args.model / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cases.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
