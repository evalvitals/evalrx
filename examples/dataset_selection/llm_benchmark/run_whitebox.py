"""Stage W runner — attention readouts over a label-balanced subset of the batch.

    WHITEBOX_PYTHON=/path/to/vllm-venv/bin/python \
      $WHITEBOX_PYTHON run_whitebox.py --model qwen3.5-9b --dataset supergpqa_law --n 24

Runs AFTER the endpoint chain, on the same frozen cases.json, and writes
outputs/<model>/<dataset>/whitebox.json next to summary.json.

**Why it aggregates itself instead of handing the batch to the analyzers.**
``attention_sink``, ``attention_rollout`` and ``attention`` all analyse
``cases[0]`` and say so in their own source ("Stage 1: single-case
ergonomics"). Passing a 24-case batch would return one case's number wearing a
batch's clothes. So each analyzer is run once per case here and the PASS/FAIL
contrast is computed in this file, where the n is visible.

That contrast is the whole point: a sink fraction of 0.43 means nothing on its
own, and means a great deal if FAIL cases sit at 0.43 while PASS cases sit at
0.21 on the same slice.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
#: The checkout root (holds pyproject.toml). Found by walking up rather than
#: counting directories: this example moved from examples/ to
#: examples/dataset_selection/ in one reorg, and a hardcoded ``parent.parent``
#: silently started pointing at examples/ instead of the package root.
PKG_ROOT = next(p for p in HERE.resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(PKG_ROOT))

import whitebox as W  # noqa: E402

#: Run these when --analyzers is left at "auto".
#:
#: ``attention_rollout`` is deliberately NOT here. Rollout multiplies the
#: per-layer matrices to trace influence back to the input, which assumes every
#: layer contributes one. Qwen3.5 is a hybrid stack — 24 of its 32 layers are
#: linear attention with no matrix at all — so the product runs over 8 of 32
#: layers and the result is a partial path wearing the name of a full one. Ask
#: for it explicitly with --analyzers if you want it, and read it as such.
DEFAULT_ANALYZERS = ("attention_sink",)

#: Composing analyzers whose meaning changes on a HYBRID_SPARSE stack.
DEPTH_COMPOSING = {"attention_rollout"}


def numeric_findings(findings: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (findings or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = float(v)
    return out


def contrast(rows: list, key: str) -> dict:
    """PASS vs FAIL on one scalar, with the n that produced it."""
    p = [r[key] for r in rows if r["label"] == "PASS" and key in r]
    f = [r[key] for r in rows if r["label"] == "FAIL" and key in r]
    out = {"n_pass": len(p), "n_fail": len(f)}
    if p:
        out["pass_mean"] = round(statistics.mean(p), 4)
    if f:
        out["fail_mean"] = round(statistics.mean(f), 4)
    if len(p) >= 2 and len(f) >= 2:
        out["gap"] = round(statistics.mean(f) - statistics.mean(p), 4)
        sp, sf = statistics.stdev(p), statistics.stdev(f)
        pooled = ((sp ** 2 + sf ** 2) / 2) ** 0.5
        # Cohen's d, reported as the honest "how big relative to the noise".
        # Deliberately NOT a p-value: n is ~12 per side by design here, and a
        # p-value at that n invites a claim the sample cannot carry.
        out["cohens_d"] = round(out["gap"] / pooled, 3) if pooled > 0 else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=24, help="label-BALANCED total (n/2 each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyzers", default="auto")
    ap.add_argument("--attn-budget-gb", type=float, default=8.0)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="drop longer cases; attention is O(seq^2) so a few long "
                         "prompts dominate the whole run's memory")
    ap.add_argument("--layers", default="",
                    help="comma-separated positions INTO THE CAPTURED LIST, not "
                         "model layer numbers — on a hybrid stack those differ "
                         "(0..7 here, mapping to model layers 3,7,...,31)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    path = HERE / "outputs" / args.model / args.dataset / "cases.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run build_cases.py first")
    report = json.loads(path.read_text())

    layers = [int(x) for x in args.layers.split(",") if x.strip()] or None
    model = W.build(args.model, device=args.device, layers=layers,
                    budget_bytes=int(args.attn_budget_gb * 1024 ** 3))
    _, processor = model._loaded
    tok = getattr(processor, "tokenizer", processor)

    selected = W.select_cases(report, args.n, seed=args.seed,
                              max_prompt_tokens=args.max_prompt_tokens, tokenizer=tok)
    n_lab = {l: sum(1 for c in selected if c["label"] == l) for l in ("PASS", "FAIL")}
    print(f"[stage W] {args.model} x {args.dataset}: {len(selected)} cases {n_lab} "
          f"(from {report['n']}, {report['accuracy']:.3f} accuracy)")
    if min(n_lab.values()) == 0:
        raise SystemExit("one label is empty after filtering — nothing to contrast")

    from evalrx.core.case import CaseBatch
    from evalrx.core.registry import registry

    names = (list(DEFAULT_ANALYZERS) if args.analyzers == "auto"
             else [s.strip() for s in args.analyzers.split(",") if s.strip()])
    available = set(registry.analyzers.names_compatible_with(model))
    missing = [n for n in names if n not in available]
    if missing:
        print(f"  skipping {missing} — not compatible with this model "
              f"(available: {sorted(available)})")
    names = [n for n in names if n in available]
    if not names:
        raise SystemExit("no compatible analyzer selected")

    attn_layers = model.attention_layers()
    semantics = str(getattr(model.spec.attn_semantics, "value", ""))
    print(f"  attn_semantics={semantics}: {len(attn_layers)} capturable layers "
          f"at model indices {attn_layers}")
    risky = [n for n in names if n in DEPTH_COMPOSING]
    if risky and semantics == "hybrid_sparse":
        print(f"  WARNING {risky} compose across the stack, but this model only "
              f"exposes {len(attn_layers)} of its layers — the output is a "
              f"partial path, not a full-depth one. Reported anyway; label it.")

    rows, failures = [], []
    for i, case in enumerate(selected):
        batch = CaseBatch(W.to_batch([case]))
        row = {"label": case["label"], "prompt_tokens": len(tok(case["prompt"])["input_ids"])}
        for name in names:
            try:
                analyzer = registry.analyzers.get(name)()
                result = analyzer.run(model, batch)
                row.update(numeric_findings(result.findings, prefix=f"{name}."))
            except MemoryError as exc:
                failures.append({"i": i, "analyzer": name, "error": str(exc).splitlines()[0]})
            except Exception as exc:
                failures.append({"i": i, "analyzer": name,
                                 "error": f"{type(exc).__name__}: {exc}"})
            finally:
                # AttentionResult keeps the full attention stack in .artifacts.
                # Holding 24 of those is tens of GB; drop it before the next case.
                result = None
        rows.append(row)
        print(f"  [{i+1}/{len(selected)}] {case['label']} "
              f"seq={row['prompt_tokens']}", flush=True)

    keys = sorted({k for r in rows for k in r if k not in ("label",)})
    contrasts = {k: contrast(rows, k) for k in keys}

    print("\n  PASS vs FAIL")
    for k, c in contrasts.items():
        if "gap" in c:
            print(f"    {k:38s} pass={c['pass_mean']:.4f} fail={c['fail_mean']:.4f} "
                  f"gap={c['gap']:+.4f} d={c['cohens_d']}")

    # Length is the confound that eats this whole stage if it goes unchecked.
    # Sink mass is attention on token 0 averaged over all query positions, so it
    # falls as the sequence grows for purely mechanical reasons. If PASS and FAIL
    # differ in prompt length, EVERY seq-sensitive readout inherits that
    # difference and reads as a mechanism. Observed on the first real run:
    # sink gap -0.0014 with d=-1.15, alongside a prompt_tokens gap of +8.8
    # tokens with d=1.32 -- the "effect" was mostly the length.
    length = contrasts.get("prompt_tokens", {})
    confounded = abs(length.get("cohens_d") or 0.0) >= 0.5
    if confounded:
        print(f"\n  ⚠ prompt length itself separates the labels "
              f"(pass={length.get('pass_mean')} fail={length.get('fail_mean')} "
              f"d={length.get('cohens_d')}).")
        print("    Attention readouts are sequence-length sensitive, so treat every "
              "gap above as length-confounded until you re-run on a "
              "length-matched subset. A large d on a tiny gap is the usual tell.")

    out = path.parent / "whitebox.json"
    out.write_text(json.dumps({
        "model": args.model, "dataset": args.dataset,
        "transformers": W.require_transformers(),
        "n_selected": len(selected), "labels": n_lab,
        "batch_n": report["n"], "batch_accuracy": report["accuracy"],
        "seed": args.seed, "layers": layers,
        "attn_semantics": semantics,
        "capturable_layer_indices": attn_layers,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_seq_seen": model.max_seq_seen, "n_forwards": model.n_forwards,
        "analyzers": names, "contrasts": contrasts, "per_case": rows,
        "length_confounded": confounded,
        "failures": failures,
    }, indent=2, ensure_ascii=False))
    print(f"\n  wrote {out}")
    if failures:
        print(f"  {len(failures)} analyzer call(s) failed — see 'failures' in the json")


if __name__ == "__main__":
    main()
