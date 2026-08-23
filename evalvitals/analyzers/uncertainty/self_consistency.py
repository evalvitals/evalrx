"""Self-consistency — sample N generations and measure agreement.

A near-free black-box uncertainty signal (needs only ``GENERATE``): low agreement
across samples flags brittle/uncertain answers.  Runs on API models too.

Surface agreement is a lower bound, though: "18", "18 apples" and "the answer is
eighteen" are three strings and one answer, so string-level consistency reports
uncertainty the model does not have.  **Semantic entropy** fixes that by
clustering the SAME samples into meaning classes before measuring — no extra
generations, so it is strictly additive over the existing score.  The default
clustering is a lexical proxy (bidirectional content-word containment); inject
``entailment_fn`` to use a real NLI model where it matters.

References:
- Self-Consistency Improves Chain of Thought Reasoning in Language Models
  Wang et al., ICLR 2023 — arXiv:2203.11171
- Detecting Hallucinations in Large Language Models Using Semantic Entropy
  Farquhar et al., Nature 630 (2024) — bidirectional-entailment clustering
- Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
  Kuhn et al., ICLR 2023 — arXiv:2302.09664
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.hallucination.selfcheck import containment
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


def lexical_equivalent(a: str, b: str, threshold: float = 0.8) -> bool:
    """Bidirectional content-word containment ≥ *threshold* — the default proxy.

    Bidirectional on purpose: one-way containment makes every short answer a
    member of every long one ("18" ⊂ "18 apples were left"), which merges
    clusters that disagree and understates entropy.
    """
    if not str(a or "").strip() or not str(b or "").strip():
        return not str(a or "").strip() and not str(b or "").strip()
    return min(containment(a, b), containment(b, a)) >= threshold


def cluster_by_equivalence(
    samples: list[str], equivalent: Callable[[str, str], bool]
) -> list[list[int]]:
    """Greedy single-link clustering against each cluster's first member.

    Greedy rather than transitive-closure: chaining through near-misses collapses
    genuinely different answers into one cluster and silently drives the entropy
    to zero, which is the failure this metric exists to avoid.
    """
    clusters: list[list[int]] = []
    for idx, sample in enumerate(samples):
        for cluster in clusters:
            if equivalent(sample, samples[cluster[0]]):
                cluster.append(idx)
                break
        else:
            clusters.append([idx])
    return clusters


def discrete_entropy(sizes: list[int]) -> float:
    """Shannon entropy (nats) of the cluster-size distribution."""
    total = sum(sizes)
    if total <= 0:
        return 0.0
    return round(
        -sum((n / total) * math.log(n / total) for n in sizes if n > 0), 4
    )


@register_analyzer("self_consistency")
class SelfConsistencyAnalyzer(Analyzer):
    """Sample ``n`` generations and report the modal-answer fraction (consistency).

    Hyper-parameters:
        n:             number of samples.
        gen_kwargs:    passed to ``model.generate`` (e.g. ``{"temperature": 0.7}``).
        semantic:      also cluster the same samples by meaning and report
                       semantic entropy (no extra generations).
        entailment_fn: ``callable(a, b) -> bool`` equivalence test used for that
                       clustering; defaults to :func:`lexical_equivalent`.
        answer_fn:     ``callable(text) -> str`` applied BEFORE comparing.

    ``answer_fn`` is what makes this usable on a reasoning model. Without it the
    comparison is over the whole generation, and two samples of a 3,000-token
    chain of thought are never byte-identical even when they reach the same
    answer — so ``consistency`` reads ``1/n`` by construction and ``n_unique``
    reads ``n``, regardless of whether the model actually agreed with itself.
    That is not a lower bound, it is a constant. Pass
    ``answer_fn=extract_answer`` and the metric measures answers again; the
    raw-text number is still reported alongside as ``raw_text_consistency`` so
    the two are never confused.
    """

    name = "self_consistency"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})
    signal_docs = {
        'consistency': ('Answer stability', 'How often the model gave the same answer when asked the same question repeatedly. Low means unstable reasoning.'),
        'modal_answer': 'The answer the model gave most often.',
        'n_samples': 'How many times each question was re-asked.',
        'n_semantic_clusters': 'How many genuinely different answers there were, after wording differences are ignored.',
        'n_unique': ('Different answers given', 'How many different answers came back for the same question.'),
        'normalized_semantic_entropy': ('Answer spread', 'How spread out the answers were. High means the model has no settled view.'),
    }

    def __init__(
        self,
        n: int = 5,
        gen_kwargs: dict | None = None,
        semantic: bool = True,
        entailment_fn: Optional[Callable[[str, str], bool]] = None,
        answer_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(n=n, gen_kwargs=gen_kwargs or {}, semantic=semantic)
        # ctor name, so sklearn-style get_params() reflection works
        self.entailment_fn = entailment_fn or lexical_equivalent
        self.answer_fn = answer_fn

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        samples = [model.generate(case.inputs, **self.gen_kwargs) for _ in range(self.n)]
        raw = [str(s).strip().lower() for s in samples]
        if self.answer_fn is not None:
            norm = [str(self.answer_fn(s) or "").strip().lower() for s in samples]
        else:
            norm = raw
        counts = Counter(norm)
        modal, modal_n = counts.most_common(1)[0]
        raw_counts = Counter(raw)
        findings: dict[str, Any] = {
            "n_samples": self.n,
            "consistency": round(modal_n / max(len(samples), 1), 4),
            "n_unique": len(counts),
            "compared_on": "answer" if self.answer_fn is not None else "raw_text",
            # Kept alongside so the two are never mistaken for each other. On a
            # reasoning model this one is ~1/n by construction; a downstream
            # stage that sees only the headline number cannot tell.
            "raw_text_consistency": round(
                raw_counts.most_common(1)[0][1] / max(len(samples), 1), 4),
            "modal_answer": (modal if self.answer_fn is not None
                             else samples[norm.index(modal)]),
            # The consistency score is meaningless without the sampling
            # config that produced it (temperature above all): a low score
            # at temperature 0 is a real defect, the same score at 1.0 is
            # expected. Empty dict == the model's own generate() defaults.
            "gen_kwargs": dict(self.gen_kwargs),
        }
        if self.semantic:
            findings.update(self._semantic_findings([str(s) for s in samples]))
        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            artifacts={"samples": samples},
            findings=findings,
        )

    # ------------------------------------------------------------------
    def _semantic_findings(self, samples: list[str]) -> dict[str, Any]:
        clusters = cluster_by_equivalence(samples, self.entailment_fn)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        entropy = discrete_entropy(sizes)
        max_entropy = math.log(len(samples)) if len(samples) > 1 else 0.0
        return {
            "n_semantic_clusters": len(clusters),
            "semantic_entropy": entropy,
            # normalised so batches with different n stay comparable
            "normalized_semantic_entropy": (
                round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0
            ),
            "semantic_consistency": round(sizes[0] / len(samples), 4) if sizes else 0.0,
            "cluster_sizes": sizes,
            "cluster_representatives": [samples[c[0]][:120] for c in clusters[:5]],
            "entailment": (
                "lexical_proxy"
                if self.entailment_fn is lexical_equivalent
                else getattr(self.entailment_fn, "__name__", "injected")
            ),
            "_caveat": (
                "semantic_* uses the SAME samples as consistency — it costs no "
                "extra generations, and the two are meant to be read together: "
                "consistency << semantic_consistency means the disagreement was "
                "only wording. With the default lexical_proxy, clustering is "
                "content-word overlap, NOT entailment: it merges answers that "
                "share vocabulary while contradicting each other ('the claim "
                "holds' / 'the claim does not hold') and splits correct "
                "paraphrases with no shared words. Inject an NLI entailment_fn "
                "before reading semantic_entropy as the Farquhar et al. metric. "
                "At temperature 0 every sample is identical and both scores are "
                "1 by construction, not by measurement."
            ),
        }
