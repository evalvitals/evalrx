"""Follow a published weight edit along the base model's own looping generations.

Runs ``weight_edit_continuation_audit`` with the p50c edit of "Sparks of
Weight-Space Programming" on Qwen3.5-0.8B, on the texts EditLens used: GSM8K
generations of the UNEDITED model that never left the thinking block (no
``</think>``) and contain repetition loops.

For every position of those generations the script measures, teacher-forced,
how much less likely the recorded next token becomes (dNLL) and how far the
next-token distribution moves (KL), under

  * the edit,
  * a random direction of the same size on the same neurons,
  * the same rule on other neurons,
  * a constant activation shift with the weights untouched,

separately for loop positions and productive positions.

EditLens's own teacher-forced table (results/02_loops/steering/
q4_simpler_intervention.md, 6 texts) has the edit raise the NLL of the repeated
continuation by +0.093 inside loops and by +0.053 at productive positions: a
larger effect in loops, with measurable collateral outside them. The script
prints what it measures; it does not assume that outcome.

Requirements: this branch of evalrx installed, a GPU with ~12 GB free (two
``[4096, vocab]`` logit tensors plus float32 work chunks), and a clone of
https://github.com/assassin808/edit-lens for the edit and the recorded texts.

    python tools/audit_astra_edit_qwen35.py --edit-lens ~/Github/edit-lens

Run ``tools/check_astra_edit_qwen35.py`` first: this script refuses to continue
if the loaded weights are not the ones the edit was made on.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

BASE_REPO = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# The suffix EditLens appends to each GSM8K question (scripts/q4_tf_interventions.py).
QUESTION_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def load_model(model: str, revision: "str | None", device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    module = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, **kwargs)
    return module.to(device).eval(), tokenizer


def as_evalrx_model(module, tokenizer):
    """An evalrx model handle that tokenises the prompt verbatim (no template added)."""
    import evalrx

    try:
        handle = evalrx.wrap(module, tokenizer)
        if getattr(handle.runtime, "apply_chat_template", False):
            raise RuntimeError("the handle would re-apply the chat template")
        print(f"model handle: evalrx.wrap -> {type(handle).__name__}")
        return handle
    except Exception as exc:  # noqa: BLE001 - fall back to a minimal handle, and say so
        print(f"evalrx.wrap not used ({type(exc).__name__}: {exc}); using a minimal handle")

    import torch

    from evalrx.core.capability import Capability
    from evalrx.core.model import Model, Trace

    class Handle(Model):
        def __init__(self) -> None:
            self.capabilities = frozenset({Capability.GENERATE, Capability.LOGITS})
            self.modalities = frozenset({"text"})
            self._hf = (module, tokenizer)

        @property
        def _loaded(self):
            return self._hf

        def generate(self, inputs, **kwargs) -> str:  # not used by the audit
            raise NotImplementedError

        def forward(self, inputs, capture, spec=None) -> Trace:
            ids = tokenizer(getattr(inputs, "prompt", str(inputs)))["input_ids"]
            device = next(module.parameters()).device
            with torch.no_grad():
                logits = module(input_ids=torch.tensor([ids], device=device)).logits[0]
            return Trace(tokens=[str(i) for i in ids], token_ids=list(ids),
                         provided={Capability.LOGITS}, logits=logits)

    return Handle()


def looping_cases(edit_lens: str, tokenizer, *, n_texts: int, max_tokens: int,
                  repeat_ngram: int, min_loop_frac: float, offset: int = 0):
    """Unfinished GSM8K generations of the unedited model with enough loop positions.

    *offset* skips that many qualifying generations first, so two calls with
    ``offset=0`` and ``offset=n`` return disjoint sets.
    """
    from evalrx.analyzers.weights.continuation_audit import repetition_mask
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label

    questions = {}
    with open(os.path.join(edit_lens, "data", "gsm8k_test.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions[row["id"]] = row["question"]
    pattern = os.path.join(edit_lens, "astra_logs", "exp8_math", "eval", "unedited", "parts",
                           "gsm8k_s*", "gsm8k_first_samples.jsonl.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no recorded generations found at {pattern}")
    cases, seen, unfinished = [], 0, 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                seen += 1
                if "</think>" in row["text"] or row["id"] not in questions:
                    continue
                unfinished += 1
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": questions[row["id"]] + QUESTION_SUFFIX}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
                prompt_ids = tokenizer(prompt)["input_ids"]
                text_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
                ids = (list(prompt_ids) + list(text_ids))[:max_tokens]
                mask = repetition_mask(ids, repeat_ngram)[len(prompt_ids):]
                if not mask or sum(mask) / len(mask) <= min_loop_frac:
                    continue
                cases.append(FailureCase(
                    inputs=Inputs(prompt=prompt), observed=row["text"], label=Label.FAIL,
                    id=str(row["id"]), metadata={"source": os.path.basename(os.path.dirname(path))}))
                if len(cases) >= offset + n_texts:
                    break
        if len(cases) >= offset + n_texts:
            break
    cases = cases[offset:]
    print(f"recorded generations read: {seen}; unfinished (no </think>): {unfinished}; "
          f"kept with loop share > {min_loop_frac}: {len(cases)} (after skipping {offset})")
    if not cases:
        raise SystemExit("no looping generation found; lower --min-loop-frac or --repeat-ngram")
    return CaseBatch(cases)


def fmt(value, spec: str = "+.4f") -> str:
    return "n/a" if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edit-lens", required=True, help="path to a clone of edit-lens")
    parser.add_argument("--edit-dir", default="",
                        help="edit directory (default: <edit-lens>/astra_logs/exp8_math/runs/p50c/edit)")
    parser.add_argument("--model", default=BASE_REPO)
    parser.add_argument("--revision", default=BASE_REVISION,
                        help="model revision; pass an empty string for a local directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-texts", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repeat-ngram", type=int, default=32)
    parser.add_argument("--min-loop-frac", type=float, default=0.03)
    parser.add_argument("--n-seeds", type=int, default=2, help="random draws per control")
    parser.add_argument("--no-verify-base", action="store_true",
                        help="skip the base-hash check (only for a dry run on a stand-in model)")
    parser.add_argument("--out", default="astra_edit_continuation_audit.json")
    args = parser.parse_args()

    from evalrx.analyzers.weights import WeightEditContinuationAudit
    from evalrx.wsp import WeightEdit, applied

    edit_dir = args.edit_dir or os.path.join(args.edit_lens, "astra_logs", "exp8_math", "runs",
                                             "p50c", "edit")
    edit = WeightEdit.from_astra(edit_dir)
    module, tokenizer = load_model(args.model, args.revision or None, args.device)
    print(f"loaded {args.model} as {type(module).__name__} on {args.device}")

    if not args.no_verify_base:
        try:
            with applied(module, edit, verify_base=True) as report:
                print("base hashes match; edit applies to:",
                      ", ".join(r["name"] for r in report.rows))
        except (ValueError, KeyError) as exc:
            print(f"REFUSED: {exc}")
            return 1

    model = as_evalrx_model(module, tokenizer)
    cases = looping_cases(args.edit_lens, tokenizer, n_texts=args.n_texts,
                          max_tokens=args.max_tokens, repeat_ngram=args.repeat_ngram,
                          min_loop_frac=args.min_loop_frac)

    started = time.time()
    findings = WeightEditContinuationAudit(
        edit=edit, groups="repetition", repeat_ngram=args.repeat_ngram,
        controls=("random_direction", "random_columns"), n_seeds=args.n_seeds,
        steering=True, max_tokens=args.max_tokens,
    ).run(model, cases).findings
    print(f"audit finished in {time.time() - started:.0f}s")

    f = findings
    print(f"\ncases: {f['n_cases_analyzed']} (skipped {f['n_skipped']}: {f['skip_reasons']})")
    print(f"positions: {f['n_positions']:,} = {f.get('n_positions_loop', 0):,} loop + "
          f"{f.get('n_positions_productive', 0):,} productive  (ngram {f['repeat_ngram']})")
    states = [s for s in ("edit", "random_direction", "random_columns", "steer")
              if f"mean_kl_{s}_productive" in f or f"mean_kl_{s}_loop" in f]
    header = f"{'':<18}" + "".join(f"{s:>18}" for s in states)
    for group in ("loop", "productive"):
        if f"mean_nll_clean_{group}" not in f:
            print(f"\n[{group}] no positions")
            continue
        print(f"\n[{group}] clean NLL of the recorded next token: "
              f"{f[f'mean_nll_clean_{group}']:.4f}")
        print(header)
        print(f"{'dNLL':<18}" + "".join(
            f"{fmt(f.get(f'mean_dnll_{s}_{group}')):>18}" for s in states))
        print(f"{'KL from clean':<18}" + "".join(
            f"{fmt(f.get(f'mean_kl_{s}_{group}'), '.3e'):>18}" for s in states))
        print(f"{'top-1 changed':<18}" + "".join(
            f"{fmt(f.get(f'frac_flips_{s}_{group}'), '.4f'):>18}" for s in states))
        print(f"  specificity vs random direction : "
              f"{fmt(f.get(f'specificity_random_direction_{group}'), '.2f')}")
        print(f"  specificity vs other neurons    : "
              f"{fmt(f.get(f'specificity_random_columns_{group}'), '.2f')}")
        print(f"  steering recovery               : "
              f"{fmt(f.get(f'steer_recovery_{group}'), '.3f')}")
    if "dnll_edit_loop_minus_productive" in f:
        print(f"\nedit dNLL, loop minus productive: {f['dnll_edit_loop_minus_productive']:+.4f}")
    for key in ("controls_not_run", "steering_not_run"):
        if key in f:
            print(f"{key}: {f[key]}")

    print("\nper case (edit dNLL: loop / productive, share of loop positions):")
    for row in f["per_case"]:
        print(f"  {row['sample_id']:>16}  "
              f"{fmt(row.get('wc_dnll_edit_loop')):>9} / {fmt(row.get('wc_dnll_edit_productive')):>9}"
              f"   loop share {row.get('wc_frac_loop', float('nan')):.3f}"
              f"   n={row['wc_n_positions']}")
    n_loop_up = sum(1 for r in f["per_case"]
                    if r.get("wc_dnll_edit_loop") is not None
                    and r.get("wc_dnll_edit_productive") is not None
                    and r["wc_dnll_edit_loop"] > r["wc_dnll_edit_productive"])
    print(f"cases where the edit raises NLL more in loops than outside: "
          f"{n_loop_up} of {len(f['per_case'])}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "revision": args.revision, "edit": edit.describe(),
                   "settings": vars(args), "findings": findings}, fh, indent=1, default=str)
    print(f"\nfull result written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
