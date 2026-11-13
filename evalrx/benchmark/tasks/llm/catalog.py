"""The LLM task catalog: provenance, slicing and grading notes for each slice.

Each entry was MEASURED on Qwen3.5-9B into the 30-70% usable band (n=50, Qwen
thinking sampling). The band is a property of the (model, dataset) PAIR, not of
the dataset, so ``accuracy_9b`` is a starting point for 2B/4B, never a prediction
-- see README.md.

The slice definitions (fetch, prompt, grader) live in :mod:`.specs`; this
module records WHICH slice and WHY. The measurements come from
``examples/dataset_selection`` (``llm_band_probe`` and ``llm_benchmark``).

.. warning::

   **Every ``accuracy_9b`` below was measured with a BROKEN answer extractor**
   (fixed 2026-08-16 in ``evalrx.analyzers.reasoning._text``).  Two defects,
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

from dataclasses import dataclass

from .specs import SPECS, Spec


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
    def spec(self) -> Spec:
        """How this slice is fetched, prompted and graded."""
        return SPECS[self.name]


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
        name="bbh_word_sorting",
        chapter="ch4-basic",
        items=250,
        accuracy_9b=0.720,
        ci95_9b=(0.58, 0.83),
        budget_signal_9b=0.00,
        source="lukaemon/bbh · config=word_sorting",
        venue="BIG-Bench Hard, Suzgun et al., ACL Findings 2023",
        slicing="One named BBH task: sort a word list into alphabetical order.",
        grading="Normalised exact match on the whole sorted list — one word out "
                "of place is a FAIL, so this is a harsh binary. That is fine for "
                "band purposes but means a FAIL is not evidence of not knowing "
                "how to sort.",
        caveat="ADDED FOR THE 2B, not the 9B: 0.720 above is a 9B number and is "
               "marginal there (CI 0.58-0.83 crosses 0.70). It is here because "
               "qwen3.5-2b measures 0.440 (Wilson [0.31, 0.58], BOTH ends inside "
               "the band) — the most comfortably band-centred entry in this table "
               "for that model. Two budget notes, and they point opposite ways: "
               "the 0.440 was measured at band_locate's 4096, while a census here "
               "runs at the config default 20480, and on bbh_causal_judgement the "
               "same widening moved accuracy +8.8 points. It also carried an 8% "
               "budget signal at 4096, two points under the veto, which the wider "
               "budget should relieve. Expect the census to read higher than "
               "0.440; build_cases WARNS outside [0.15, 0.85] (refuses only with --strict-band).",
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


def get(name: str) -> Entry:
    if name not in BY_NAME:
        raise KeyError(f"unknown dataset {name!r}. Available: {', '.join(sorted(BY_NAME))}")
    return BY_NAME[name]


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
