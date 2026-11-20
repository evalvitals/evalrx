"""Check a published weight edit against the real Qwen3.5-0.8B base model.

Two steps, both read-only with respect to the files on disk:

  1. Load the edit (``edit.npz`` + ``manifest.json``) and apply it in memory with
     the base-hash check on. This passes only if every edited tensor of the
     loaded model is bit-identical to the tensor the edit was made on. While the
     edit is applied, the edited tensors are hashed and compared with the
     ``edited_sha256`` recorded in the manifest, which tests that the rebuild is
     bit-exact.
  2. Read the structure of the edit from the weights: changed columns, rank,
     and the strength ``alpha`` of the rule ``w' = w - alpha * u (u^T w)``.

For the p50c edit of "Sparks of Weight-Space Programming", EditLens reports 64
changed columns of ``mlp.down_proj`` in each of layers 15 and 19, a rank-one
change per layer, and alpha = 6. The script prints what it measures next to
those numbers; it does not assume them.

Requirements: the evalrx branch with ``evalrx.wsp`` installed, torch,
transformers (a release that knows Qwen3.5), and a clone of
https://github.com/assassin808/edit-lens for the edit files.

    python tools/check_astra_edit_qwen35.py \
        --edit-dir /path/to/edit-lens/astra_logs/exp8_math/runs/p50c/edit

No GPU is needed for these two steps; a 0.8B model fits in CPU memory.
"""

from __future__ import annotations

import argparse
import json
import sys

BASE_REPO = "Qwen/Qwen3.5-0.8B"
# The revision the edit was made on, as stated in the EditLens README.
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

# What EditLens reports for p50c. Used only to label the printed comparison.
EXPECTED = {"changed_columns": 64, "alpha": 6.0, "layers": (15, 19)}


def load_module(model: str, revision: "str | None", device: str):
    """Load the checkpoint in its stored dtype (bfloat16) without changing any value."""
    import torch
    import transformers

    kwargs = {"dtype": torch.bfloat16}
    if revision:
        kwargs["revision"] = revision
    errors = []
    # Only the parameters are used, so any class that loads the checkpoint will do.
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            module = cls.from_pretrained(model, **kwargs)
            print(f"loaded {model} with {cls_name} ({type(module).__name__})")
            return module.to(device).eval()
        except Exception as exc:  # noqa: BLE001 - try the next loader, report all at the end
            errors.append(f"{cls_name}: {type(exc).__name__}: {exc}")
    raise SystemExit("could not load the model:\n  " + "\n  ".join(errors))


def step1_verify(module, edit) -> "list[dict]":
    """Apply the edit with the base-hash check; compare the rebuilt tensors' hashes."""
    from evalrx.wsp import applied, resolve_parameters, tensor_sha256

    resolved = resolve_parameters(module, edit)
    rows = []
    for tensor_edit, (name, param) in zip(edit.tensors, resolved):
        rows.append({
            "edit_name": tensor_edit.name,
            "resolved_name": name,
            "dtype": str(param.dtype).replace("torch.", ""),
            "base_hash_matches": tensor_sha256(param) == tensor_edit.base_sha256,
        })
    if not all(r["base_hash_matches"] for r in rows):
        return rows  # applying would be refused; report which tensors differ

    with applied(module, edit, verify_base=True) as report:
        for row, tensor_edit, (_, param) in zip(rows, edit.tensors, resolved):
            row["edited_hash_matches"] = (
                tensor_sha256(param) == tensor_edit.edited_sha256
                if tensor_edit.edited_sha256 else None
            )
        for row, applied_row in zip(rows, report.rows):
            row["changed_entries"] = applied_row["changed_entries"]
            row["relative_change"] = applied_row["relative_change"]
    # Outside the block the original weights are back.
    for row, tensor_edit, (_, param) in zip(rows, edit.tensors, resolved):
        row["restored"] = tensor_sha256(param) == tensor_edit.base_sha256
    return rows


def step2_structure(module, edit) -> dict:
    from evalrx.wsp import edit_structure

    return edit_structure(module, edit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edit-dir", required=True,
                        help="directory with edit.npz and manifest.json (e.g. .../runs/p50c/edit)")
    parser.add_argument("--model", default=BASE_REPO,
                        help="Hugging Face repo id or local directory of the base model")
    parser.add_argument("--revision", default=BASE_REVISION,
                        help="model revision; pass an empty string for a local directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="astra_edit_check.json",
                        help="where to write the full result as JSON")
    args = parser.parse_args()

    from evalrx.wsp import WeightEdit

    edit = WeightEdit.from_astra(args.edit_dir)
    print(f"edit '{edit.name}': {len(edit)} tensor(s)")
    for tensor in edit.tensors:
        n_overwritten = 0 if tensor.index is None else len(tensor.index)
        print(f"  {tensor.name}  form={tensor.form}  overwritten={n_overwritten:,}")

    module = load_module(args.model, args.revision or None, args.device)

    # ---- Step 1 -------------------------------------------------------
    print("\n[1] base-hash check and bit-exact rebuild")
    try:
        verify = step1_verify(module, edit)
    except KeyError as exc:
        print(f"  FAILED to resolve a tensor name: {exc}")
        return 2
    for row in verify:
        print(f"  {row['resolved_name']}")
        print(f"    stored dtype           : {row['dtype']}")
        print(f"    base hash matches      : {row['base_hash_matches']}")
        if "edited_hash_matches" in row:
            print(f"    rebuilt hash matches   : {row['edited_hash_matches']}")
            print(f"    changed entries        : {row['changed_entries']:,}")
            print(f"    relative change        : {row['relative_change']:.4f}")
            print(f"    restored after exit    : {row['restored']}")
    base_ok = all(r["base_hash_matches"] for r in verify)
    rebuilt_ok = base_ok and all(r.get("edited_hash_matches") is not False for r in verify)
    if not base_ok:
        print("  -> the loaded weights are NOT the ones the edit was made on "
              "(wrong revision, a re-export, or a dtype other than bfloat16). "
              "Step 2 is skipped: the overwritten entries are absolute values and "
              "mean nothing on another base.")
        json.dump({"edit": edit.describe(), "verify": verify}, open(args.out, "w"), indent=1)
        return 1

    # ---- Step 2 -------------------------------------------------------
    print("\n[2] structure of the edit, read from the weights")
    structure = step2_structure(module, edit)
    for row in structure["tensors"]:
        print(f"  {row['name']}")
        print(f"    changed columns        : {row.get('changed_columns')}"
              f"   (EditLens: {EXPECTED['changed_columns']})")
        print(f"    changed rows           : {row.get('changed_rows')}")
        print(f"    effective rank         : {row.get('effective_rank', float('nan')):.4f}"
              "   (1 = rank one)")
        print(f"    rank-one residual      : {row.get('rank_one_residual', float('nan')):.4g}"
              "   (0 = exactly rank one; bfloat16 rounding keeps it above 0)")
        alpha = row.get("alpha_estimate")
        spread = row.get("alpha_spread")
        if alpha is not None:
            print(f"    alpha estimate         : {alpha:.4f} +/- {spread:.4f}"
                  f"   (EditLens: {EXPECTED['alpha']})")
        print(f"    projection selectivity : {row.get('projection_selectivity', float('nan')):.3f}"
              "   (>1: the edited columns write along u more than average)")
        print(f"    top singular values    : "
              f"{[round(v, 4) for v in row.get('singular_values', [])[:4]]}")
    for pair, cosine in structure["direction_cosines"].items():
        print(f"  |cos| between edit directions: {cosine:.4f}   ({pair})")

    result = {
        "model": args.model, "revision": args.revision,
        "edit": edit.describe(), "verify": verify, "structure": structure,
        "base_ok": base_ok, "rebuilt_ok": rebuilt_ok,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(f"\nfull result written to {args.out}")
    if not rebuilt_ok:
        print("NOTE: the base matched but a rebuilt tensor's hash did not; "
              "the loader's rebuild is not bit-exact for this edit.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
