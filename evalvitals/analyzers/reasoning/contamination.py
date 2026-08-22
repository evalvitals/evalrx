"""Contamination probe — is this benchmark measuring reasoning or recall?

Every accuracy number in M1/M2 assumes the model is solving the item, not
remembering it.  When a benchmark leaked into pre-training, PASS/FAIL stops
tracking capability, the mechanism columns correlate with nothing real, and any
fix validated on it will not transfer.  Contamination is therefore a
**confound column**, not a headline: it is measured per case, carried into M2
alongside the mechanism columns, and used to disqualify a finding whose effect
lives entirely on memorised items.

Needs no labels and no reference corpus.  The guided-vs-general design of
Golchin & Surdeanu isolates the effect: both arms ask the model to complete the
second half of the item, but only the guided arm names the benchmark and split.
A model that reconstructs the continuation *better when told which benchmark it
came from* is retrieving, not generalising.

Cost: 2 generations per case (3 with ``ask_provenance=True``).

References:
- Time Travel in LLMs: Tracing Data Contamination in Large Language Models —
  Golchin & Surdeanu, ICLR 2024 — arXiv:2308.08493 (guided vs general prompting)
- Investigating Data Contamination in Modern Benchmarks for LLMs —
  Deng et al., NAACL 2024 — arXiv:2311.09783
- Rethinking Benchmark and Contamination for Language Models with
  Rephrased Samples — Yang et al., 2023 — arXiv:2311.04850
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

from evalvitals.analyzers.reasoning._text import word_ngrams
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

_GENERAL = (
    "Complete the rest of this text exactly as it continues. Output only the "
    "continuation.\n\nText so far:\n{first_half}"
)
_GUIDED = (
    "The text below is the FIRST PIECE of an instance from the {split} split of "
    "the {dataset} dataset. Finish the second piece exactly as it appears in the "
    "dataset. Output only the continuation.\n\nFirst piece:\n{first_half}"
)
_PROVENANCE = (
    "Which public benchmark or dataset is the following item from? Answer with "
    "the dataset name only, or 'unknown'.\n\n{text}"
)


def overlap_score(candidate: Any, reference: Any, n: int = 4) -> float:
    """Fraction of the reference's word ``n``-grams that the candidate reproduces.

    Recall-oriented on purpose: a verbose completion that contains the reference
    span has still reproduced it, and length-penalising that would hide exactly
    the memorisation this probe is looking for.  Falls back to unigrams for
    references too short to have ``n``-grams.
    """
    ref_grams = word_ngrams(reference, n) or word_ngrams(reference, 1)
    if not ref_grams:
        return 0.0
    width = len(ref_grams[0])
    cand_grams = set(word_ngrams(candidate, width))
    if not cand_grams:
        return 0.0
    hits = sum(1 for g in ref_grams if g in cand_grams)
    return round(hits / len(ref_grams), 4)


@register_analyzer("contamination_score")
class ContaminationProbe(Analyzer):
    """Guided-vs-general completion overlap — a per-case memorisation signal.

    Hyper-parameters:
        dataset_name:    benchmark name used by the guided arm (REQUIRED to be
                         meaningful — the guided/general contrast IS the design).
        split:           split name used by the guided arm.
        split_frac:      fraction of the item shown as the first half.
        ngram:           n-gram width for the overlap score.
        flag_threshold:  guided overlap above this ⇒ ``verbatim_flag``.
        ask_provenance:  +1 generation asking the model to name the benchmark.
        max_cases:       label-stratified cap; 0 (the default) = every case.
    """

    name = "contamination_score"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        dataset_name: str = "the evaluation",
        split: str = "test",
        split_frac: float = 0.5,
        ngram: int = 4,
        flag_threshold: float = 0.6,
        ask_provenance: bool = False,
        max_cases: int = 0,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            split_frac=min(max(split_frac, 0.1), 0.9),
            ngram=ngram,
            flag_threshold=flag_threshold,
            ask_provenance=ask_provenance,
            max_cases=max_cases,
        )

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            per_case.append(self._probe_case(model, case))

        scored = [c for c in per_case if "guided_overlap" in c]
        flagged = [c for c in scored if c["verbatim_flag"] == 1]
        labelled = [c for c in scored if c.get("labelled_pass") is not None]
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_scored": len(scored),
            "dataset_name": self.dataset_name,
            "mean_general_overlap": _mean([c["general_overlap"] for c in scored]),
            "mean_guided_overlap": _mean([c["guided_overlap"] for c in scored]),
            "mean_guided_gain": _mean([c["guided_gain"] for c in scored]),
            "verbatim_flag_rate": (
                round(len(flagged) / len(scored), 4) if scored else None
            ),
            # the number that actually matters: does memorisation buy accuracy?
            "accuracy_on_flagged": _mean([c["labelled_pass"] for c in labelled
                                          if c["verbatim_flag"] == 1]),
            "accuracy_on_unflagged": _mean([c["labelled_pass"] for c in labelled
                                            if c["verbatim_flag"] == 0]),
            "per_case": per_case,
            "_caveat": (
                "This is a CONFOUND column, never a headline. A guided_gain > 0 "
                "says the model reconstructs the item better when told which "
                "benchmark it is from — evidence of exposure, not proof, and it "
                "is confounded by items whose continuation is guessable from "
                "format alone (templated benchmarks score high with no leak, so "
                "read general_overlap as the floor). The usable test is the "
                "SPLIT: when accuracy_on_flagged far exceeds "
                "accuracy_on_unflagged, the benchmark's PASS mass is partly "
                "recall and any M3 hypothesis resting on it must be re-run on "
                "unflagged cases. Set dataset_name — with the default the "
                "guided arm names nothing and the contrast is void."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _probe_case(self, model: "Model", case: "FailureCase") -> dict[str, Any]:
        from evalvitals.core.case import Label

        text = str(case.inputs.prompt or "")
        words = text.split()
        entry: dict[str, Any] = {"sample_id": case.id, "n_words": len(words)}
        if len(words) < 2 * self.ngram:
            entry["skipped"] = "prompt too short to split into two halves"
            return entry

        cut = max(1, int(len(words) * self.split_frac))
        first_half, second_half = " ".join(words[:cut]), " ".join(words[cut:])

        general = str(model.generate(_bare(case, _GENERAL.format(first_half=first_half))))
        guided = str(
            model.generate(
                _bare(
                    case,
                    _GUIDED.format(
                        first_half=first_half,
                        dataset=self.dataset_name,
                        split=self.split,
                    ),
                )
            )
        )
        general_overlap = overlap_score(general, second_half, self.ngram)
        guided_overlap = overlap_score(guided, second_half, self.ngram)
        entry.update(
            {
                "general_overlap": general_overlap,
                "guided_overlap": guided_overlap,
                "guided_gain": round(guided_overlap - general_overlap, 4),
                "verbatim_flag": int(guided_overlap >= self.flag_threshold),
                "labelled_pass": (
                    int(case.label == Label.PASS)
                    if case.label in (Label.PASS, Label.FAIL)
                    else None
                ),
            }
        )
        if self.ask_provenance:
            named = str(model.generate(_bare(case, _PROVENANCE.format(text=text[:1500]))))
            entry["named_dataset"] = named.strip()[:80]
            entry["named_correct"] = int(
                self.dataset_name.lower().split()[0] in named.lower()
            )
        return entry


def _bare(case: "FailureCase", prompt: str):
    """Inputs carrying *prompt* alone — media slots are preserved, the task is not."""
    return dataclasses.replace(case.inputs, prompt=prompt)


def _mean(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None
