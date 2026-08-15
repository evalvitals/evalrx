"""The eight band-located datasets, with provenance and slicing criteria.

Each entry was MEASURED on Qwen3.5-9B into the 30-70% usable band (n=50, Qwen
thinking sampling). The band is a property of the (model, dataset) PAIR, not of
the dataset, so ``accuracy_9b`` is a starting point for 2B/4B, never a prediction
-- see README.md.

The dataset plumbing itself lives in ``../llm_band_probe/band_locate.py``; this
module only records WHICH slice and WHY, and re-exports the spec so the two can
never drift apart.
"""

from __future__ import annotations

import os
import sys
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
               "using it as a headline number.",
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
        accuracy_9b=0.540,
        ci95_9b=(0.40, 0.67),
        budget_signal_9b=0.02,
        source="lukaemon/bbh · config=tracking_shuffled_objects_seven_objects",
        venue="BIG-Bench Hard, Suzgun et al., ACL Findings 2023",
        slicing="One named BBH task: track seven objects through a swap sequence.",
        grading="Normalised exact match.",
        caveat="CLEANEST measurement in the table (budget signal 2%). Also the "
               "clearest evidence against extrapolation: a survey agent predicted "
               "0.93 for this slice; it measured 0.540, a 39-point miss.",
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
        accuracy_9b=0.500,
        ci95_9b=(0.37, 0.63),
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
        caveat="Measured at a 65k budget. At 40k it read 0.320/budget_limited, and "
               "the difference was mostly a grader bug (23.5% of golds are "
               "program-form scientific notation), not the budget. INFLUENCE "
               "EVIDENCE UNCONFIRMED: `lm_eval/tasks/minerva_math` points at "
               "`EleutherAI/hendrycks_math`, NOT at this dataset.",
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
        raise SystemExit(
            f"unknown dataset {name!r}. Available: {', '.join(sorted(BY_NAME))}"
        )
    return BY_NAME[name]


def resolve_spec(name: str) -> "B.Spec":
    return get(name).spec


if __name__ == "__main__":
    print(f"{'dataset':26s} {'items':>6s} {'9B acc':>7s} {'signal':>7s}  chapter")
    print("-" * 72)
    for e in CATALOG:
        print(f"{e.name:26s} {e.items:6d} {e.accuracy_9b:7.3f} "
              f"{e.budget_signal_9b:7.2f}  {e.chapter}")
    missing = [e.name for e in CATALOG
               if not any(s.name == e.name for s in B.SPECS)]
    print("\nspecs missing from band_locate:", missing or "none")
