"""Sampled-passage factual consistency — the text counterpart of POPE/CHAIR (black-box).

SelfCheckGPT's premise: if the model KNOWS a fact, independently sampled
generations agree on it; hallucinated content diverges across samples.  This
port implements the dependency-free unigram-containment variant: each sentence
of the baseline answer is scored by how well the resampled passages cover its
content words.  ``requires=GENERATE`` only, so it runs on API models.

The score is a per-case M2 column, not a truth oracle: correlate
``selfcheck_inconsistency`` with the outcome label downstream instead of
treating a high score as a verdict.

References:
- SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative
  Large Language Models — Manakul et al., EMNLP 2023 — arXiv:2303.08896
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

_WORD = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*|\n+")


def split_sentences(text: str, min_chars: int = 10) -> list[str]:
    """Split into scoreable sentences, dropping fragments below ``min_chars``."""
    parts = [p.strip() for p in _SENT_SPLIT.split(str(text or ""))]
    return [p for p in parts if len(p) >= min_chars]


def _char_bigrams(text: str) -> set[str]:
    chars = [c for c in text if not c.isspace()]
    return {a + b for a, b in zip(chars, chars[1:])} or set(chars)


def containment(sentence: str, passage: str) -> float:
    """Fraction of the sentence's content units that appear in the passage.

    Latin/digit words when the sentence has any; character bigrams otherwise,
    so CJK (or symbol-only) sentences are scored instead of silently passing.
    """
    sent = set(_WORD.findall(sentence.lower()))
    if sent:
        passage_words = set(_WORD.findall(passage.lower()))
        return len(sent & passage_words) / len(sent)
    bigrams = _char_bigrams(sentence.lower())
    if not bigrams:
        return 1.0
    return len(bigrams & _char_bigrams(passage.lower())) / len(bigrams)


@register_analyzer("selfcheck_consistency")
class SelfCheckConsistencyAnalyzer(Analyzer):
    """SelfCheckGPT-style sampled-passage consistency: sentence-level hallucination signal from resampling.

    Hyper-parameters:
        n_samples:  resampled passages per case (baseline answer excluded).
        gen_kwargs: sampling config for the resamples — MUST be stochastic
                    (default ``{"temperature": 1.0}``); at temperature 0 the
                    score is degenerate and reported as such.
        max_cases:  label-stratified cap on probed cases.
        min_sentence_chars: shorter fragments are not scored.
    """

    name = "selfcheck_consistency"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        n_samples: int = 4,
        gen_kwargs: Optional[dict] = None,
        max_cases: int = 32,
        min_sentence_chars: int = 10,
    ) -> None:
        super().__init__(
            n_samples=max(1, int(n_samples)),
            gen_kwargs=dict(gen_kwargs) if gen_kwargs is not None else {"temperature": 1.0},
            max_cases=max_cases,
            min_sentence_chars=min_sentence_chars,
        )

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            baseline = str(case.observed or "").strip()
            baseline_from = "observed"
            if not baseline:
                baseline = str(model.generate(case.inputs))
                baseline_from = "generated"
            sentences = split_sentences(baseline, self.min_sentence_chars)
            entry: dict[str, Any] = {
                "sample_id": case.id,
                "baseline_from": baseline_from,
                "n_sentences": len(sentences),
            }
            if not sentences:
                entry["skipped"] = "no scoreable sentence in the baseline answer"
                per_case.append(entry)
                continue
            samples = [
                str(model.generate(case.inputs, **self.gen_kwargs))
                for _ in range(self.n_samples)
            ]
            # SelfCheckGPT scores a sentence by its support in OTHER samples;
            # a sentence no resample reproduces is the hallucination suspect.
            sentence_scores = [
                1.0 - max(containment(sent, sample) for sample in samples)
                for sent in sentences
            ]
            entry["selfcheck_inconsistency"] = round(
                sum(sentence_scores) / len(sentence_scores), 4
            )
            entry["selfcheck_worst_sentence"] = round(max(sentence_scores), 4)
            worst = sentences[sentence_scores.index(max(sentence_scores))]
            entry["worst_sentence_text"] = worst[:200]
            per_case.append(entry)

        scored = [c["selfcheck_inconsistency"] for c in per_case if "selfcheck_inconsistency" in c]
        temperature = self.gen_kwargs.get("temperature")
        degenerate = (
            (temperature is not None and float(temperature) == 0.0)
            or self.gen_kwargs.get("do_sample") is False
        )
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_samples": self.n_samples,
            "degenerate_sampling": degenerate,
            "gen_kwargs": dict(self.gen_kwargs),
            "mean_inconsistency": round(sum(scored) / len(scored), 4) if scored else None,
            "per_case": per_case,
            "_caveat": (
                "Unigram-containment proxy for SelfCheckGPT (no NLI model): "
                "paraphrases inflate the score, verbatim repetition of a wrong "
                "fact deflates it. Requires stochastic sampling — with "
                "temperature 0 the resamples repeat one greedy generation and the "
                "column degenerates to baseline-vs-greedy disagreement "
                "(degenerate_sampling flags this). "
                "selfcheck_inconsistency / selfcheck_worst_sentence are numeric "
                "M2 columns; treat them as signals to correlate with the outcome "
                "label, never as a hallucination verdict."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
