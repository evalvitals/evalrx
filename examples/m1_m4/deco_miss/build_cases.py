"""Offline slice: deco_pope's frozen manifest -> the MISS subset (present probes).

DeCo's fixable failure mode is the *miss* — the object IS in the image but the
model answers "No". It is the natural substrate for an internals-write fix: the
object is visually present (so the evidence exists mid-network) yet the output
says absent. deco_pope's manifest already mined every present-object probe with
its greedy answer + token-id sets, so we just re-slice (no GPU, no re-mining):

    FAIL = present probe answered "No"  (a missed detection)
    PASS = present probe answered "Yes" (correct)

    python build_cases.py            # writes data/cases/{model}.json for all sizes

Images are shared with deco_pope (../deco_pope/data/images), not copied.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
POPE = HERE.parent / "deco_pope"
OUT = HERE / "data" / "cases"
MODELS = ("qwen3-vl-2b-instruct", "qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        src = POPE / "data" / "cases" / f"{model}.json"
        if not src.exists():
            print(f"skip {model}: {src} missing (run deco_pope/mine_cases.py first)")
            continue
        raw = json.loads(src.read_text())
        present = [c for c in raw["cases"] if c["probe_type"] == "present"]
        n_miss = sum(1 for c in present if c["label"] == "fail")
        yields = {s: {"fail": 0, "pass": 0} for s in ("explore", "validate")}
        for c in present:
            yields[c["split"]][c["label"]] += 1
        out = {
            "model": model,
            "prompt_template": raw["prompt_template"],
            "seed": raw.get("seed"),
            "decoding": raw.get("decoding"),
            "pope_commit": raw.get("pope_commit"),
            "versions": raw.get("versions"),
            "source": f"deco_pope/data/cases/{model}.json (present-probe slice)",
            "subset": "miss = present-object probe answered 'No' (gold yes)",
            "yields": yields,
            "cases": present,
        }
        dst = OUT / f"{model}.json"
        dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"{model}: present={len(present)} miss(FAIL)={n_miss} -> {dst}")


if __name__ == "__main__":
    main()
