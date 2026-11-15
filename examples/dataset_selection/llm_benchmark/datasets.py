"""CLI over the LLM task catalog, plus the 2B probe records behind it.

The catalog itself (slice, provenance, reference accuracy) is
:mod:`evalrx.benchmark.tasks.llm.catalog`; the slice definitions (fetch,
prompt, grader) are :mod:`evalrx.benchmark.tasks.llm.specs`.

    python datasets.py                 # the catalog
    python datasets.py --acquisition   # exact HF ids per slice
    python datasets.py --probe         # confirm every slice still resolves
    python datasets.py --plan --n 240  # what an n_cases buys per slice
    python datasets.py --max-tokens minervamath   # used by run_all.sh
"""

from __future__ import annotations

from pathlib import Path

from evalrx.benchmark.tasks.llm.catalog import CATALOG, acquisition, get
from evalrx.benchmark.tasks.llm.fetch import FILTER_API, ROWS_API

#: Qwen3.5-**2B**, n=50 probe, measured 2026-08-16 with the FIXED extractor.
#:
#: The band is a property of the (model, dataset) PAIR, and this is the evidence
#: for it: five of seven candidates land IN band on a 2B, so a small model is not
#: the reason a slice is unusable — bbh_tracking7 was unusable because the grader
#: was inventing its position (0.988 once regraded, on the 9B).
#:
#: ``seconds`` is per 50 items at concurrency 16 and spans a factor of FORTY-FIVE,
#: which no accuracy column shows: cruxeval_output is a whole census in ~4 min
#: while supergpqa_economics is ~2.5 h. It is here so a dataset gets picked on
#: cost as well as band position.
#:
#: ``chars`` is the mean generation length, and it is the column that decides
#: whether M1 has anything to read. At 189 chars a 2B barely opens a chain on
#: cruxeval_output, so the reasoning analyzers measure almost nothing there —
#: same band position, very different diagnostic value.
#:
#: .. warning::
#:
#:    Each row ran at its own ``spec.max_tokens`` (band_locate's budget), which
#:    is NOT what build_cases uses — build_cases takes ``entry.max_tokens or
#:    CFG["max_tokens"]``.  The two agree only where the entry declares a budget
#:    (minervamath).  Everywhere else these accuracies are measured at a
#:    different budget than the census that follows them, and the gap is not
#:    small: bbh_causal_judgement reads 0.420 here at 8192 and 0.508 at 20480.
#:    Treat a row as "in band at THIS budget", never as a census prediction.
#:
#: Keys are ``band_locate`` SPEC names, NOT ``CATALOG`` entries — this table
#: records what was probed, and a probe that excludes a dataset is exactly the
#: result worth keeping. ``bbh_object_counting`` is here and deliberately not in
#: CATALOG, so do not look these names up in ``BY_NAME`` without a guard.
BAND_2B: dict = {
    #                        acc     band       seconds  chars
    "bbh_causal_judgement": (0.420, "USABLE",      90,     None),
    #: Probed 2026-08-17 to close a gap in the TextGrad dataset list. Excluded
    #: on the SMALL model, which settles it for the large one too: 0.800 on a 2B
    #: can only go up on a 9B. Its budget_bracket is the degenerate [0.80, 0.80]
    #: — budget_signal is 0 — so unlike a budget_limited row there is nothing a
    #: bigger cap could resolve. Counting objects in a list is arithmetic; the
    #: same 2B scores 0.508 on causal_judgement from the same repo.
    "bbh_object_counting":  (0.800, "marginal",    25,      351),
    "minervamath":          (0.380, "USABLE",     639,    12149),
    "supergpqa_law":        (0.360, "USABLE",     305,     6576),
    "supergpqa_economics":  (0.340, "USABLE",     507,     6338),
    "cruxeval_output":      (0.300, "USABLE",      14,      189),
    "bamboogle":            (0.160, "floor",      107,     None),
    "supergpqa_medicine_hard": (0.140, "floor",   701,     None),
    # bbh_tracking7 not probed: already saturated on the 9B once regraded.

    # ── 2026-08-17: the "just above 0.70 on the 9B" sweep ────────────────────
    # Rationale, and it held: a dataset the 9B has nearly saturated is where a
    # 2B lands mid-band. The drop is real but NOT predictable — same 0.720
    # starting point gave -28 (word_sorting) and -6 (gsm_symbolic_main) — so
    # this band is a hunting ground, never an estimate.
    #
    # The two rows that did NOT make it are the useful negative result: both
    # started from the saturated tier (0.94/0.98) and both fell exactly 20
    # points, landing 0.74/0.78 — still out. Entering the band from there needs
    # a cruxeval-sized 40-point fall, which happened once in five. Probe the
    # 0.70-0.80 tier; skip the saturated one.
    "mmlu_pro":             (0.480, "USABLE",     646,    16737),  # at 65536
    "bbh_word_sorting":     (0.440, "USABLE",      55,     2822),
    #: USABLE on the POINT ESTIMATE only — band_of tests acc in [0.30, 0.70] and
    #: this CI runs to 0.776, so the true value may be out of band. Raise n
    #: before using it.
    "gsm_symbolic_main":    (0.660, "USABLE",      52,     2126),
    "folio":                (0.740, "marginal",   221,    12300),
    "bbh_navigate":         (0.780, "marginal",    15,      874),
}

#: Full-census follow-ups, which are what the probe is FOR — checking that an
#: n=50 read survives the whole slice.  ``(accuracy, n_correct, n, budget)``.
#:
#: The two so far say different things, and the difference is the budget:
#:
#: * ``minervamath`` — probe 0.380 (CI 0.26-0.52) at 65536, census 0.419 at the
#:   SAME budget.  Inside the interval: the probe was representative.
#: * ``bbh_causal_judgement`` — probe 0.420 at the spec's 8192, census 0.508 at
#:   the config default 20480.  Outside the probe's point estimate by nearly
#:   nine points, and NOT a failure of the probe: the two ran at different
#:   budgets, because band_locate uses ``spec.max_tokens`` while build_cases
#:   uses ``entry.max_tokens or CFG["max_tokens"]`` and this entry declares
#:   none.  Band position is a property of (model, dataset, BUDGET).
#:
#: So compare a probe against a census only when both ran at the same budget —
#: otherwise the drift measures the budget, not the slice.
CENSUS_2B: dict = {
    "minervamath":          (0.419, 114, 272, 65536),
    "bbh_causal_judgement": (0.508,  95, 187, 20480),
}


def _plan(n_cases: int, confirm_split: float) -> None:
    """What a requested n actually buys, per dataset.

    `n_cases` is a SAMPLE SIZE drawn from the slice, so the slice size is a hard
    ceiling: asking 240 of a 125-item set gets 125. The number that decides
    whether M2 can conclude anything is neither n nor the half -- it is the
    SMALLER of PASS/FAIL within a half, since a paired contrast is limited by its
    thinner side.
    """
    label = "ALL (census)" if n_cases <= 0 else str(n_cases)
    print(f"n_cases={label}  confirm_split={confirm_split}  "
          f"(explore {1 - confirm_split:.0%} / confirm {confirm_split:.0%})")
    print()
    print(f"{'dataset':26s} {'slice':>6s} {'actual n':>9s} {'per half':>9s} "
          f"{'PASS/FAIL':>11s} {'thinner':>8s}")
    print("-" * 78)
    for e in CATALOG:
        actual = e.items if n_cases <= 0 else min(n_cases, e.items)
        half = int(actual * confirm_split) if confirm_split else actual
        n_pass = round(half * e.accuracy_9b)
        n_fail = half - n_pass
        thin = min(n_pass, n_fail)
        mark = ""
        if actual < n_cases:
            mark += " CAPPED"
        if thin < 25:
            mark += " THIN"
        print(f"{e.name:26s} {e.items:6d} {actual:9d} {half:9d} "
              f"{f'{n_pass}/{n_fail}':>11s} {thin:8d}{mark}")
    print()
    if n_cases > 0:
        print("CAPPED = the slice is smaller than the request; you get a CENSUS of it,")
        print("         so there is no sampling variability left for that slice.")
    else:
        print("Every row is a CENSUS: the batch IS the slice, so there is no sampling")
        print("variability left -- the interval speaks to the task, not to the draw.")
    print("THIN   = fewer than ~25 in the smaller class per half. A paired test")
    print("         there mostly reports 'not significant' from lack of power,")
    print("         which is not a finding.")
    print("PASS/FAIL uses the Qwen3.5-9B accuracy; a smaller model shifts it.")


def _probe(timeout: int = 60) -> int:
    """Fetch a couple of rows for each entry -- proves the ids actually resolve."""
    import requests

    bad = 0
    print(f"{'dataset':26s} {'rows in slice':>13s}  status")
    print("-" * 74)
    for e in CATALOG:
        acq = acquisition(e.name)
        url = FILTER_API if acq["where"] else ROWS_API
        params = {"dataset": acq["dataset"], "config": acq["config"],
                  "split": acq["split"], "offset": 0, "length": 2}
        if acq["where"]:
            params["where"] = acq["where"]
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            print(f"{e.name:26s} {'-':>13s}  UNREACHABLE {type(exc).__name__}")
            bad += 1
            continue
        if r.status_code != 200:
            print(f"{e.name:26s} {'-':>13s}  HTTP {r.status_code}")
            bad += 1
            continue
        total = r.json().get("num_rows_total")
        # the slice size is a property of the ids; a mismatch means the upstream
        # dataset moved under us and the recorded band no longer describes it
        flag = "OK" if total == e.items else f"SIZE CHANGED (recorded {e.items})"
        if total != e.items:
            bad += 1
        print(f"{e.name:26s} {total:13,d}  {flag}")
    return bad


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="the eight usable datasets")
    ap.add_argument("--probe", action="store_true",
                    help="hit datasets-server and confirm every slice resolves")
    ap.add_argument("--acquisition", action="store_true",
                    help="print the exact dataset/config/split/where for each")
    ap.add_argument("--plan", action="store_true",
                    help="what a given n_cases/confirm_split actually buys per dataset")
    ap.add_argument("--n", type=int, default=0, help="n_cases for --plan")
    ap.add_argument("--max-tokens", metavar="NAME",
                    help="print the generation budget NAME needs (0 = config "
                         "default is fine); run_all.sh uses this to size the "
                         "server's --max-model-len before it starts")
    ap.add_argument("--split", type=float, default=-1.0,
                    help="confirm_split for --plan")
    args = ap.parse_args()

    if args.max_tokens:
        print(get(args.max_tokens).max_tokens)
        raise SystemExit(0)

    if args.plan:
        import yaml
        cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
        _plan(args.n or int(cfg["n_cases"]),
              args.split if args.split >= 0 else float(cfg["confirm_split"]))
        raise SystemExit(0)

    if args.acquisition:
        for e in CATALOG:
            acq = acquisition(e.name)
            print(f"{e.name}")
            print(f"    dataset = {acq['dataset']}")
            print(f"    config  = {acq['config']}")
            print(f"    split   = {acq['split']}")
            if acq["where"]:
                print(f"    where   = {acq['where']}")
        raise SystemExit(0)

    if args.probe:
        raise SystemExit(1 if _probe() else 0)

    print(f"{'dataset':26s} {'items':>6s} {'9B acc':>7s} {'signal':>7s}  chapter")
    print("-" * 72)
    for e in CATALOG:
        print(f"{e.name:26s} {e.items:6d} {e.accuracy_9b:7.3f} "
              f"{e.budget_signal_9b:7.2f}  {e.chapter}")
    print("--acquisition: exact ids | --probe: verify they resolve | "
          "--plan: what an n_cases buys")
