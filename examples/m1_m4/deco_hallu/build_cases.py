"""Offline slice: deco_pope's frozen manifest -> the HALLUCINATION subset.

The hard, complementary slice to deco_miss. The failure here is a *false Yes*:
the model answers "Yes" for an object that is NOT in the image. Unlike the miss
(where the object's evidence exists and can be amplified), a confident
hallucination has no obvious latent signal to recover, so this slice tests
whether the loop can find a mitigation that survives the no-free-lunch guard.

CRITICAL — the batch is built so a degenerate "always answer No" fix CANNOT win:
a fix is scored on every case against its OWN gold label, so it must keep the
present-object detections ("Yes") while flipping the absent-object
hallucinations ("No"). The PASS controls therefore include present-Yes cases.

    FAIL  = adversarial-absent probe answered "Yes"  (the hallucination)
    PASS  = adversarial-absent answered "No"         (correct rejection)
          + present-object answered "Yes"            (correct detection — the
                                                      recall a skeptical fix must
                                                      not break)

    python build_cases.py            # writes data/cases/{model}.json for all sizes

Images shared with deco_pope (../deco_pope/data/images). No GPU, no re-mining.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
POPE = HERE.parent / "deco_pope"
OUT = HERE / "data" / "cases"
MODELS = ("qwen3-vl-2b-instruct", "qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct")
# How many control cases to keep per type (all hallucinations are always kept).
# Roughly balanced against the hallucination count so the batch is not swamped.
N_REJECT = 80      # adversarial-absent answered "No" (correct rejection)
N_PRESENT = 80     # present-object answered "Yes" (correct detection / recall guard)
SEED = 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        src = POPE / "data" / "cases" / f"{model}.json"
        if not src.exists():
            print(f"skip {model}: {src} missing (run deco_pope/mine_cases.py first)")
            continue
        raw = json.loads(src.read_text())
        rng = random.Random(SEED)

        hallu, reject, present = [], [], []
        for c in raw["cases"]:
            if c["probe_type"] == "adversarial" and c["label"] == "fail":
                hallu.append(c)                       # false Yes (gold no)
            elif c["probe_type"] == "adversarial" and c["label"] == "pass":
                reject.append(c)                      # correct No
            elif c["probe_type"] == "present" and c["label"] == "pass":
                present.append(c)                     # correct Yes (recall guard)

        # split-stratified control sampling so explore/validate both stay mixed
        def sample(pool, n):
            out = []
            for split, frac in (("explore", 0.6), ("validate", 0.4)):
                s = [c for c in pool if c["split"] == split]
                rng.shuffle(s)
                out += s[: round(n * frac)]
            return out

        # Interleave the two control types so ANY prefix of the PASS cases holds
        # both correct-rejections AND present-detections. The fix validation
        # subset (FixAgent.stratified_head) takes PASS in document order, and the
        # no-free-lunch guard only works if present-detections are in that subset
        # (else a degenerate "always No" fix would look flawless).
        rej_s, pres_s = sample(reject, N_REJECT), sample(present, N_PRESENT)
        controls = []
        for i in range(max(len(rej_s), len(pres_s))):
            if i < len(rej_s):
                controls.append(rej_s[i])
            if i < len(pres_s):
                controls.append(pres_s[i])
        cases = hallu + controls
        yields = {s: {"fail": 0, "pass": 0} for s in ("explore", "validate")}
        for c in cases:
            yields[c["split"]][c["label"]] += 1

        out = {
            "model": model,
            "prompt_template": raw["prompt_template"],
            "seed": raw.get("seed"),
            "decoding": raw.get("decoding"),
            "pope_commit": raw.get("pope_commit"),
            "versions": raw.get("versions"),
            "source": f"deco_pope/data/cases/{model}.json (hallucination slice)",
            "subset": ("hallucination = adversarial-absent answered 'Yes'; controls "
                       "= correct rejections + present-object detections (no-free-lunch)"),
            "yields": yields,
            "n_hallucination": len(hallu),
            "n_correct_reject": sum(1 for c in cases if c["probe_type"] == "adversarial" and c["label"] == "pass"),
            "n_present_detect": sum(1 for c in cases if c["probe_type"] == "present"),
            "cases": cases,
        }
        dst = OUT / f"{model}.json"
        dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"{model}: hallu(FAIL)={len(hallu)} + reject={out['n_correct_reject']} "
              f"+ present={out['n_present_detect']} -> {len(cases)} cases -> {dst.name}")


if __name__ == "__main__":
    main()
