"""Pipeline phase 0 — carve data_attn_full/ into held-in / held-out halves.

Each case already carries a ``split`` field (explore/validate, assigned when the
batches were built). This writes two derived input dirs:

  data_attn_explore/   split == "explore"   -> phase 1 (M2/M3 hypothesis proposal)
  data_attn_validate/  split == "validate"  -> phase 2 (held-out hypothesis testing)

The wrapper metadata (model, seed, prompt_template, ...) is preserved per file so
`evalvitals explore` sees the same shape as the full dataset. Derived dirs are
gitignored — rerun this script to regenerate.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "data_attn_full"
OUTS = {"explore": HERE / "data_attn_explore", "validate": HERE / "data_attn_validate"}


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"{SRC} missing — run extract_attention_all.py first")
    for out in OUTS.values():
        out.mkdir(exist_ok=True)
    totals = {k: 0 for k in OUTS}
    for src in sorted(SRC.glob("*.json")):
        raw = json.loads(src.read_text())
        for split, out_dir in OUTS.items():
            part = dict(raw)
            part["cases"] = [r for r in raw["cases"] if r.get("split") == split]
            part["split_partition"] = {
                "split": split,
                "n_cases": len(part["cases"]),
                "of_total": len(raw["cases"]),
                "source": str(src.relative_to(HERE)),
            }
            (out_dir / src.name).write_text(json.dumps(part, indent=1))
            totals[split] += len(part["cases"])
    for split, n in totals.items():
        print(f"{OUTS[split].name}: {n} cases")


if __name__ == "__main__":
    main()
