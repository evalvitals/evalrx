"""The eight band-located datasets, with provenance and slicing criteria.

Each entry was MEASURED on Qwen3.5-9B into the 30-70% usable band (n=50, Qwen
thinking sampling). The band is a property of the (model, dataset) PAIR, not of
the dataset, so ``accuracy_9b`` is a starting point for 2B/4B, never a prediction
-- see README.md.

The dataset plumbing itself lives in ``../llm_band_probe/band_locate.py``; this
module only records WHICH slice and WHY, and re-exports the spec so the two can
never drift apart.

.. warning::

   **Every ``accuracy_9b`` below was measured with a BROKEN answer extractor**
   (fixed 2026-08-16 in ``evalvitals.analyzers.reasoning._text``).  Two defects,
   both of which could only ever push a number DOWN:

   * a bare ``(A)`` was discarded as a format placeholder, so multiple-choice
     answers were thrown away and extraction fell back into the reasoning prose;
   * ``\\boxed{}`` outranked a LATER ``Answer:`` line, so a chain that boxed its
     intermediate working outranked its own conclusion.

   Regrading the frozen full-census batches moved ``bbh_tracking7`` 0.592 ->
   0.988 and ``minervamath`` 0.360 -> 0.463, with ZERO PASS->FAIL either way.
   ``bbh_tracking7`` is therefore not a mid-band dataset at all — it is
   saturated, and its band placement was an artefact of the grader.

   Treat every ``accuracy_9b`` here as an unverified LOWER BOUND until it is
   re-measured.  The multiple-choice entries (the ``supergpqa_*`` trio) are the
   most exposed, because ``(X)`` is exactly the shape that was being discarded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_BAND_PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "llm_band_probe")
sys.path.insert(0, _BAND_PROBE)

import band_locate as B  # noqa: E402


@dataclass(frozen=True)
class Entry:
    """One usable dataset: where it comes from and why this slice."""

    name: str
    chapter: str
    items: int
    accuracy_9b: float
    ci95_9b: tuple
    budget_signal_9b: float
    source: str
    venue: str
    slicing: str
    grading: str
    caveat: str = ""
    #: Generation budget the reference accuracy was MEASURED at, when it differs
    #: from the config default. 0 = the default is fine. This is not a
    #: preference: minervamath reads 0.500 at 65k and 0.320/budget_limited at
    #: 40k, so running it at the default would diagnose the token cap and call
    #: it a capability. build_cases.py reads this; run_all.sh sizes the server's
    #: --max-model-len from it.
    max_tokens: int = 0

    @property
    def spec(self) -> "B.Spec":
        """The live band_locate spec — single source of truth for fetch/grade."""
        return next(s for s in B.SPECS if s.name == self.name)


CATALOG: tuple = (
    Entry(
        name="cruxeval_output",
        chapter="ch2-code",
        items=800,
        accuracy_9b=0.700,
        ci95_9b=(0.56, 0.81),
        budget_signal_9b=0.06,
        source="cruxeval-org/cruxeval · split=test",
        venue="CRUXEval, Gu et al., ICML 2024",
        slicing="Whole test split. The `output` direction only (predict what the "
                "function returns), not the `input` direction.",
        grading="Exact string match on the extracted answer. NO SANDBOX — the "
                "model predicts the return value, nothing is executed.",
        caveat="Sits exactly on the band's upper edge (0.700). If the true value "
               "is any higher it slides out into `saturated`.",
    ),
    Entry(
        name="bbh_causal_judgement",
        chapter="ch4-basic",
        items=187,
        accuracy_9b=0.600,
        ci95_9b=(0.46, 0.72),
        budget_signal_9b=0.10,
        source="lukaemon/bbh · config=causal_judgement · split=test",
        venue="BIG-Bench Hard, Suzgun et al., ACL Findings 2023",
        slicing="One named BBH task. Causal attribution questions answered Yes/No.",
        grading="Normalised exact match. Two-way answer space, no grader ambiguity.",
        caveat="MARGINAL PASS. Budget signal is exactly 0.10, sitting on the veto "
               "threshold, and the CI upper bound 0.724 crosses 0.70. It also ran "
               "at an 8k budget rather than 40k. Re-measure at n=100 / 24k before "
               "using it as a headline number. — THE BUDGET WORRY WAS JUSTIFIED: "
               "on qwen3.5-2b the same slice reads 0.420 at the spec's 8192 (n=50 "
               "probe) and 0.508 at 20480 (full census, 95/187, Wilson "
               "[0.44, 0.58]). Nearly nine points from budget alone, so band "
               "position here is a property of (model, dataset, BUDGET), not of "
               "the pair. accuracy_9b=0.600 above is still the 8k number and is "
               "therefore NOT comparable to a census run at the config default.",
    ),
    Entry(
        name="supergpqa_economics",
        chapter="ch4-basic",
        items=873,
        accuracy_9b=0.580,
        ci95_9b=(0.44, 0.71),
        budget_signal_9b=0.06,
        source="m-a-p/SuperGPQA · split=train · where discipline='Economics'",
        venue="SuperGPQA, M-A-P, 2025 (285 graduate disciplines)",
        slicing="Server-side `where` on the dataset's OWN `discipline` column — a "
                "named subdivision, not a random sample. 65% `middle` difficulty.",
        grading="Multiple choice, 10 options. Random baseline ~10%, so the usable "
                "band is much wider than a 4-option set.",
        caveat="CI upper bound 0.706 presses against the 0.70 edge.",
    ),
    Entry(
        name="bbh_tracking7",
        chapter="ch4-basic",
        items=250,
        accuracy_9b=0.988,
        ci95_9b=(0.96, 1.00),
        budget_signal_9b=0.02,
        source="lukaemon/bbh · config=tracking_shuffled_objects_seven_objects",
        venue="BIG-Bench Hard, Suzgun et al., ACL Findings 2023",
        slicing="One named BBH task: track seven objects through a swap sequence.",
        grading="Normalised exact match.",
        caveat="DO NOT USE — saturated at 0.988 (full census, n=250), far above "
               "the band's 0.85 ceiling; qwen3.5-2b is higher still. It was "
               "listed at 0.540 because the answer extractor discarded a bare "
               "'(A)' as a format placeholder: 244 of 250 final claims are a "
               "bare '(X)', and regrading flipped 99 FAIL->PASS with 0 the other "
               "way. The old caveat called this the CLEANEST measurement in the "
               "table and cited it as evidence against extrapolation — in fact "
               "the survey agent's 0.93 prediction was closer to the truth than "
               "the measurement was. A grader bug outranks a survey prior.",
    ),
    Entry(
        name="bamboogle",
        chapter="ch4-basic",
        items=125,
        accuracy_9b=0.520,
        ci95_9b=(0.39, 0.65),
        budget_signal_9b=0.08,
        source="chiayewken/bamboogle · split=test",
        venue="Press et al., EMNLP Findings 2023 (self-ask)",
        slicing="Whole test split. Closed-book 2-hop compositional questions.",
        grading="Normalised exact match on a short span.",
        caveat="Smallest usable set, so the cheapest full sweep — but 125 items "
               "puts a wide interval on any subgroup analysis.",
    ),
    Entry(
        name="minervamath",
        chapter="ch1-math",
        items=272,
        accuracy_9b=0.463,
        ci95_9b=(0.40, 0.52),
        budget_signal_9b=0.10,
        source="math-ai/minervamath · split=test",
        venue="Minerva, Lewkowycz et al., NeurIPS 2022",
        slicing="Whole test split. Undergraduate physics/astronomy quantitative "
                "problems, free-response.",
        grading="`_grade_latex`: numeric first, then normalised LaTeX surface "
                "form. NOT a CAS. Scientific notation is normalised across "
                "program form (`4.5e33`) and LaTeX (`4.5 \\times 10^{33}`), with a "
                "1% relative tolerance THAT APPLIES ONLY to exponent-bearing "
                "answers — plain integers still require exact equality.",
        caveat="The one entry here that is NOT an n=50 probe: 0.463 is a regraded "
               "FULL CENSUS (126/272, Wilson [0.40, 0.52]) — the batch was "
               "generated once and relabelled by regrade.py after the "
               "extract_answer fix, which moved it 0.360 -> 0.463 with zero "
               "PASS->FAIL. It replaces a 0.500 probe read off the broken "
               "grader. qwen3.5-2b is 0.419 (114/272) on the same slice under "
               "the same grader — a 4.4-point gap across a 4.5x parameter "
               "difference, so treat this slice as measuring something other "
               "than scale. Measured at a 65k budget. At 40k it read "
               "0.320/budget_limited, and the difference was mostly a grader "
               "bug (23.5% of golds are program-form scientific notation), not "
               "the budget. INFLUENCE EVIDENCE UNCONFIRMED: "
               "`lm_eval/tasks/minerva_math` points at "
               "`EleutherAI/hendrycks_math`, NOT at this dataset.",
        max_tokens=65536,
    ),
    Entry(
        name="supergpqa_law",
        chapter="ch4-basic",
        items=656,
        accuracy_9b=0.460,
        ci95_9b=(0.33, 0.60),
        budget_signal_9b=0.10,
        source="m-a-p/SuperGPQA · split=train · where discipline='Law'",
        venue="SuperGPQA, M-A-P, 2025",
        slicing="Server-side `where` on the `discipline` column. 52% `middle`, "
                "39% `easy`, 9% `hard`.",
        grading="Multiple choice, 10 options.",
        caveat="RECOMMENDED DEFAULT. The only slice in the catalog whose 95% CI "
               "lies entirely inside the band, so neither the point estimate nor "
               "the interval is arguable.",
    ),
    Entry(
        name="supergpqa_medicine_hard",
        chapter="ch4-basic",
        items=217,
        accuracy_9b=0.360,
        ci95_9b=(0.24, 0.50),
        budget_signal_9b=0.10,
        source="m-a-p/SuperGPQA · split=train · "
               "where discipline='Medicine' AND difficulty='hard'",
        venue="SuperGPQA, M-A-P, 2025",
        slicing="Two-column `where`. The ONLY <1000 slice that is entirely `hard`; "
                "every other discipline's hard tier is too small to sample "
                "(History 3 items, Education 1, Sociology 1).",
        grading="Multiple choice, 10 options.",
        caveat="CI lower bound 0.241 falls below 0.30. n=100 is worth it here — on "
               "217 items that is already close to a half census.",
    ),
)

BY_NAME: dict = {e.name: e for e in CATALOG}


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


def get(name: str) -> Entry:
    if name not in BY_NAME:
        raise SystemExit(
            f"unknown dataset {name!r}. Available: {', '.join(sorted(BY_NAME))}"
        )
    return BY_NAME[name]


def resolve_spec(name: str) -> "B.Spec":
    return get(name).spec


def acquisition(name: str) -> dict:
    """Exactly what identifies this slice on the HuggingFace datasets-server.

    Nothing is vendored: every item is fetched at run time from the ids below, so
    these four fields ARE the dataset. A `where` clause is a server-side filter on
    the dataset's own columns -- that is what makes the three SuperGPQA entries
    citable slices rather than private subsamples.
    """
    spec = get(name).spec
    return {
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "where": spec.where or "",
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
        url = B.FILTER_API if acq["where"] else B.ROWS_API
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
    missing = [e.name for e in CATALOG
               if not any(s.name == e.name for s in B.SPECS)]
    print("\nspecs missing from band_locate:", missing or "none")
    print("--acquisition: exact ids | --probe: verify they resolve | "
          "--plan: what an n_cases buys")
