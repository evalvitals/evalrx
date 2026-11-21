"""Check the held-out comparison designs against data with a known answer.

Usage:
    python tools/check_holdout_calibration.py [--runs 400] [--n-boot 1000]

Two questions, both answered by simulation because neither can be answered by
a unit test on one dataset:

1. When nothing is going on, how often does a comparison come out
   "significant"? It should be about 5%. The script draws data in which a flag
   has no effect at all but rows cut from one source item share an outcome
   tendency, and counts rejections with and without the group column.
2. When something is going on, does the 95% interval contain the true effect
   about 95% of the time? The script draws data with a known effect for each
   design (two-group, stratified, difference in differences, dose-response)
   and counts how often the interval covers it.

A third part runs the keyword pipeline end to end: words are chosen on an
exploration half and re-tested on the rest, on text with one real keyword and
six filler words, and the script counts how often a filler is accepted.

It prints one table per part. No model and no network are needed.
"""

from __future__ import annotations

import argparse
import math
import random

from evalrx.analysis.holdout import holdout_confirm, outcome_values, split_records
from evalrx.analysis.holdout_designs import estimate_design
from evalrx.analysis.keywords import keyword_candidates
from evalrx.analysis.operationalize import SignalRecipe


def _bernoulli(rng: random.Random, p: float) -> float:
    return 1.0 if rng.random() < p else 0.0


def null_data(seed: int, flag_level: str, n_groups: int = 120):
    """No effect of ``f``. Rows of a group share a failure tendency."""
    rng = random.Random(seed)
    rows, y, groups = [], [], []
    for g in range(n_groups):
        tendency = 1.0 / (1.0 + math.exp(-(-0.5 + rng.gauss(0.0, 1.2))))
        group_flag = rng.random() < 0.4
        for _ in range(rng.randint(2, 10)):
            flag = group_flag if flag_level == "group" else rng.random() < 0.4
            rows.append({"f": int(flag)})
            y.append(_bernoulli(rng, tendency))
            groups.append(g)
    return rows, y, groups


def effect_data(seed: int, design: str, n_groups: int = 150):
    """Data with a known effect for *design*; returns rows, outcome, groups, truth, recipe."""
    rng = random.Random(seed)
    rows, y, groups = [], [], []
    for g in range(n_groups):
        shift = rng.gauss(0.0, 0.08)             # group-level shift in the failure rate
        stratum = "weak" if rng.random() < 0.5 else "strong"
        modifier = int(rng.random() < 0.5)
        for _ in range(rng.randint(2, 8)):
            if design == "two_group":
                flag = int(rng.random() < 0.4)
                p = 0.25 + 0.20 * flag
                row = {"f": flag}
            elif design == "stratified":
                # The flag is far more common in the stratum that fails more;
                # within a stratum it adds 0.10.
                flag = int(rng.random() < (0.75 if stratum == "weak" else 0.25))
                p = (0.50 if stratum == "weak" else 0.15) + 0.10 * flag
                row = {"f": flag, "s": stratum}
            elif design == "did":
                flag = int(rng.random() < 0.5)
                p = 0.20 + 0.05 * flag + 0.05 * modifier + 0.25 * flag * modifier
                row = {"f": flag, "m": modifier}
            else:  # dose_response: +0.12 per bin over three bins
                score = rng.random()
                step = (score > 1 / 3) + (score > 2 / 3)
                p = 0.20 + 0.12 * step
                row = {"score": score}
            rows.append(row)
            y.append(_bernoulli(rng, min(max(p + shift, 0.01), 0.99)))
            groups.append(g)
    recipes = {
        "two_group": (SignalRecipe(name="r", expr="f == 1"), 0.20),
        "stratified": (SignalRecipe(name="r", expr="f == 1", design="stratified",
                                    strata=["s"]), 0.10),
        "did": (SignalRecipe(name="r", expr="f == 1", design="did", modifier="m == 1"), 0.25),
        "dose_response": (SignalRecipe(name="r", expr="score", bins=[1 / 3, 2 / 3]), 0.12),
    }
    recipe, truth = recipes[design]
    return rows, y, groups, truth, recipe


def keyword_data(seed: int, n_groups: int = 120) -> list[dict]:
    """Clips of 3-7 rows. "timeout" marks failures; the other words mark nothing."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for g in range(n_groups):
        clip_fails = rng.random() < 0.5
        for _ in range(rng.randint(3, 7)):
            fail = clip_fails if rng.random() < 0.9 else not clip_fails
            filler = rng.choice(["alpha beta", "gamma delta", "epsilon zeta"])
            rows.append({
                "clip": f"clip{g}",
                "note": ("timeout " if fail and rng.random() < 0.6 else "") + filler,
                "label": "fail" if fail else "pass",
            })
    return rows


def keyword_runs(runs: int, n_boot: int) -> tuple[float, float, float]:
    """(share of runs accepting a filler, mean false share, share finding the real word)."""
    any_false, found, false_share = 0, 0, 0.0
    for seed in range(runs):
        rows = keyword_data(20_000 + seed)
        explore, held_out = split_records(rows, 0.4, seed=seed, outcome_col="label",
                                          group_col="clip")
        assert held_out is not None
        side = outcome_values(explore, "label")[0]
        report = {"candidate_signals": [c.to_dict() for c in
                                        keyword_candidates(explore, "note", side)],
                  "hypotheses": []}
        confirm = holdout_confirm(report, held_out, outcome_col="label",
                                  group_col="clip", n_boot=n_boot, seed=seed)
        held = [v["name"] for v in confirm["signal_verdicts"]
                if (v.get("claim") or {}).get("status") == "holds"]
        false = [name for name in held if name != "kw_timeout"]
        any_false += bool(false)
        found += "kw_timeout" in held
        false_share += len(false) / len(held) if held else 0.0
    return any_false / runs, false_share / runs, found / runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    runs, n_boot = args.runs, args.n_boot
    half_width = 1.96 * math.sqrt(0.05 * 0.95 / runs)

    print(f"1. False positives under no effect (target 0.050, runs={runs}, "
          f"simulation error about +/-{half_width:.3f})")
    print(f"   {'flag varies by':<16}{'rows independent':>18}{'group column':>14}")
    recipe = SignalRecipe(name="f", expr="f == 1")
    for level in ("group", "row"):
        rates = []
        for use_groups in (False, True):
            rejections = 0
            for seed in range(runs):
                rows, y, groups = null_data(seed, level)
                est = estimate_design(recipe, rows, y, n_boot=n_boot, seed=seed,
                                      clusters=groups if use_groups else None)
                rejections += (est.p_value or 1.0) <= 0.05
            rates.append(rejections / runs)
        print(f"   {level:<16}{rates[0]:>18.3f}{rates[1]:>14.3f}")

    print(f"\n2. Coverage of the 95% interval when the effect is known (target 0.950, "
          f"runs={runs})")
    print(f"   {'design':<16}{'true effect':>12}{'mean estimate':>15}{'coverage':>10}"
          f"{'unadjusted':>12}")
    for design in ("two_group", "stratified", "did", "dose_response"):
        covered, total, unadjusted = 0, 0.0, 0.0
        for seed in range(runs):
            rows, y, groups, truth, design_recipe = effect_data(10_000 + seed, design)
            est = estimate_design(design_recipe, rows, y, clusters=groups,
                                  n_boot=n_boot, seed=seed)
            assert est.ok and est.ci is not None and est.effect is not None, est.reason
            covered += est.ci[0] <= truth <= est.ci[1]
            total += est.effect
            unadjusted += est.details.get("unadjusted_difference", float("nan"))
        extra = f"{unadjusted / runs:>12.3f}" if design == "stratified" else f"{'':>12}"
        print(f"   {design:<16}{truth:>12.3f}{total / runs:>15.3f}{covered / runs:>10.3f}{extra}")

    keyword_n = max(50, runs // 2)
    any_false, false_share, found = keyword_runs(keyword_n, n_boot)
    print(f"\n3. Keyword recipes chosen on one half and re-tested on the other "
          f"(runs={keyword_n})")
    print(f"   real keyword accepted:                        {found:.3f}")
    print(f"   runs in which a filler word was accepted:     {any_false:.3f}")
    print(f"   mean share of accepted keywords that were filler: {false_share:.3f}")


if __name__ == "__main__":
    main()
