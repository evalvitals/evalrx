"""Can a loop-vs-productive contrast find the published edit on its own?

The p50c edit of "Sparks of Weight-Space Programming" was written by an AI
system. This script asks whether evalrx can PROPOSE a comparable edit from the
base model's own behaviour, with no access to that edit:

  1. Derive: on one set of the unedited model's looping GSM8K generations,
     contrast the MLP writes at loop positions with those at productive
     positions (``continuation_contrast_edit``). That yields, per layer, a
     direction and a ranking of neurons.
  2. Compare with the published edit: where its two layers rank among all
     layers, the cosine between the derived and the published direction at
     those layers, and how many of the published 64 neurons the derived ranking
     recovers (against the number expected by chance).
  3. Audit on a DISJOINT set of generations (``weight_edit_continuation_audit``):
     the published edit and the derived one, each at several strengths alpha.

Nothing here is assumed to succeed. The contrast finds the direction along
which loops differ from ordinary text; the published edit works by overshooting
(alpha = 6, where alpha = 1 does almost nothing), and its direction need not be
that difference.

    python tools/recover_astra_edit_qwen35.py --edit-lens ~/Github/edit-lens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_astra_edit_qwen35 as base  # noqa: E402


def with_alpha(edit, alpha: float, name: str):
    """The same rank-one edit (direction, neurons) at another strength."""
    from evalrx.wsp import TensorEdit, WeightEdit

    return WeightEdit([TensorEdit.rank_one(t.name, t.u, alpha, cols=t.cols)
                       for t in edit.tensors], name=name, metadata=dict(edit.metadata))


def audit(model, cases, edit, args, *, controls=(), steering=False, n_seeds=1) -> dict:
    from evalrx.analyzers.weights import WeightEditContinuationAudit

    findings = WeightEditContinuationAudit(
        edit=edit, groups="repetition", repeat_ngram=args.repeat_ngram, controls=controls,
        n_seeds=n_seeds, steering=steering, max_tokens=args.max_tokens,
    ).run(model, cases).findings
    findings.pop("structure", None)
    return findings


def summary_row(label: str, f: dict) -> str:
    def g(key: str, spec: str = "+.4f") -> str:
        return "n/a" if f.get(key) is None else format(f[key], spec)

    return (f"{label:<34}{g('mean_dnll_edit_loop'):>10}{g('mean_dnll_edit_productive'):>10}"
            f"{g('mean_kl_edit_loop', '.5f'):>10}{g('mean_kl_edit_productive', '.5f'):>10}"
            f"{g('frac_flips_edit_loop', '.4f'):>10}{g('frac_flips_edit_productive', '.4f'):>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edit-lens", required=True, help="path to a clone of edit-lens")
    parser.add_argument("--edit-dir", default="")
    parser.add_argument("--model", default=base.BASE_REPO)
    parser.add_argument("--revision", default=base.BASE_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-derive", type=int, default=12,
                        help="generations the direction and neurons are derived from")
    parser.add_argument("--n-audit", type=int, default=12,
                        help="other generations the edits are audited on")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repeat-ngram", type=int, default=32)
    parser.add_argument("--min-loop-frac", type=float, default=0.03)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 6.0, 10.0])
    parser.add_argument("--no-verify-base", action="store_true")
    parser.add_argument("--out", default="astra_edit_recovery.json")
    args = parser.parse_args()

    import torch

    from evalrx.wsp import (
        WeightEdit,
        applied,
        collect_continuation_stats,
        continuation_contrast_edit,
        edit_agreement,
        fit_rank_one,
        rank_columns,
        resolve_parameters,
        write_sites,
    )

    edit_dir = args.edit_dir or os.path.join(args.edit_lens, "astra_logs", "exp8_math", "runs",
                                             "p50c", "edit")
    published = WeightEdit.from_astra(edit_dir)
    module, tokenizer = base.load_model(args.model, args.revision or None, args.device)
    print(f"loaded {args.model} as {type(module).__name__} on {args.device}")
    if not args.no_verify_base:
        try:
            with applied(module, published, verify_base=True):
                print("base hashes match")
        except (ValueError, KeyError) as exc:
            print(f"REFUSED: {exc}")
            return 1
    model = base.as_evalrx_model(module, tokenizer)

    common = dict(max_tokens=args.max_tokens, repeat_ngram=args.repeat_ngram,
                  min_loop_frac=args.min_loop_frac)
    print("\nderive set:")
    derive = base.looping_cases(args.edit_lens, tokenizer, n_texts=args.n_derive, **common)
    print("audit set:")
    held_out = base.looping_cases(args.edit_lens, tokenizer, n_texts=args.n_audit,
                                  offset=len(derive), **common)
    assert not ({c.id for c in derive} & {c.id for c in held_out})

    # The published edit as an explicit rank-one rule, so it can be compared and re-scaled.
    reference = fit_rank_one(module, published, name="published")
    sites = {s.name: s for s in write_sites(model)}
    ref_layers = [sites[key].layer for key, _ in resolve_parameters(module, reference)]
    n_cols = max(int(len(t.cols)) for t in reference.tensors)
    print(f"\npublished edit as a rank-one rule: layers {ref_layers}, "
          + ", ".join(f"alpha={r['alpha']:.3f} residual={r['rank_one_residual']:.4f}"
                      for r in reference.metadata["fit"]))

    # ---- 1. Derive ----------------------------------------------------------------
    started = time.time()
    stats = collect_continuation_stats(model, derive, groups="repetition",
                                       repeat_ngram=args.repeat_ngram, max_tokens=args.max_tokens)
    print(f"\n[1] contrast from {stats.n_cases} generations: {stats.n_positions('loop'):,} loop "
          f"vs {stats.n_positions('productive'):,} productive positions "
          f"({time.time() - started:.0f}s)")
    free = continuation_contrast_edit(model, derive, stats=stats, n_layers=len(ref_layers),
                                      n_columns=n_cols, alpha=6.0, name="derived_free_layers")
    forced = continuation_contrast_edit(model, derive, stats=stats, layers=ref_layers,
                                        n_columns=n_cols, alpha=6.0, name="derived_same_layers")
    separation = {int(k): v for k, v in free.metadata["layer_separation"].items()}
    order = sorted(separation, key=lambda i: -separation[i])
    print("  layers by loop-vs-productive separation of the MLP write (largest first):")
    print("   ", "  ".join(f"L{i}:{separation[i]:.3f}" for i in order[:8]))
    for layer in ref_layers:
        print(f"  published layer L{layer}: rank {order.index(layer) + 1} of {len(order)} "
              f"(separation {separation[layer]:.3f})")
    free_layers = sorted(int(k) for k in free.metadata["layers"])
    print(f"  layers the contrast would pick on its own: {free_layers}")

    # ---- 2. Compare with the published edit ------------------------------------------
    print("\n[2] agreement with the published edit at its own layers")
    agreement = edit_agreement(module, forced, reference)
    recovery: list[dict] = []
    for tensor, row in zip(reference.tensors, agreement["tensors"]):
        layer = sites[row["name"]].layer
        weight = sites[row["name"]].module.weight
        ranked = rank_columns(stats.groups["loop"][layer], stats.groups["productive"][layer],
                              weight)
        published_cols = set(tensor.cols.tolist())
        k = len(published_cols)
        u_pub = torch.from_numpy(tensor.u).double()
        signals = {
            "push (derived direction)": ranked["push"],
            "|projection| (derived direction)": ranked["projection"].abs(),
            "activation difference": ranked["activation"],
            "push (published direction)":
                ranked["activation"] * (u_pub @ weight.detach().double().cpu()),
            "|projection| (published direction)": (u_pub @ weight.detach().double().cpu()).abs(),
        }
        overlap = {name: len(set(torch.topk(score, k).indices.tolist()) & published_cols)
                   for name, score in signals.items()}
        chance = k * k / weight.shape[1]
        print(f"  L{layer}: |cos(derived u, published u)| = {row['direction_cosine']:.4f}")
        print(f"       neurons recovered in the top {k} (chance: {chance:.1f}):")
        for name, hits in overlap.items():
            print(f"         {name:<36} {hits:>3} of {k}")
        recovery.append({"layer": layer, "direction_cosine": row["direction_cosine"],
                         "top_k": k, "chance": chance, "overlap": overlap})

    # ---- 3. Audit on held-out generations --------------------------------------------
    print(f"\n[3] audit on {len(held_out)} held-out generations "
          "(dNLL of the recorded next token; KL from the unedited model; top-1 changed)")
    print(f"{'edit':<34}{'dNLL loop':>10}{'dNLL prod':>10}{'KL loop':>10}{'KL prod':>10}"
          f"{'flip loop':>10}{'flip prod':>10}")
    audits: dict[str, dict] = {}
    started = time.time()
    for alpha in args.alphas:
        label = f"published rule, alpha={alpha:g}"
        audits[label] = audit(model, held_out, with_alpha(reference, alpha, label), args)
        print(summary_row(label, audits[label]))
    for alpha in args.alphas:
        label = f"derived, same layers, alpha={alpha:g}"
        audits[label] = audit(model, held_out, with_alpha(forced, alpha, label), args)
        print(summary_row(label, audits[label]))
    label = "derived, own layers, alpha=6"
    audits[label] = audit(model, held_out, free, args)
    print(summary_row(label, audits[label]))
    label = "published edit as stored"
    audits[label] = audit(model, held_out, published, args)
    print(summary_row(label, audits[label]))

    print("\n  derived edit (same layers, alpha=6) against its own controls:")
    controlled = audit(model, held_out, forced, args,
                       controls=("random_direction", "random_columns"), steering=True, n_seeds=2)
    audits["derived, same layers, alpha=6, with controls"] = controlled
    for group in ("loop", "productive"):
        print(f"    [{group}] specificity vs random direction "
              f"{controlled.get(f'specificity_random_direction_{group}')}, vs other neurons "
              f"{controlled.get(f'specificity_random_columns_{group}')}, steering recovery "
              f"{controlled.get(f'steer_recovery_{group}')}")
    print(f"  ({time.time() - started:.0f}s)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "model": args.model, "revision": args.revision, "settings": vars(args),
            "derive_ids": [c.id for c in derive], "audit_ids": [c.id for c in held_out],
            "published_fit": reference.metadata["fit"], "published_layers": ref_layers,
            "layer_separation": separation, "layers_picked_by_contrast": free_layers,
            "recovery": recovery, "agreement": agreement, "audits": audits,
        }, fh, indent=1, default=str)
    print(f"\nfull result written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
