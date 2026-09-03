"""Context-chunk Shapley — which parts of the provided context carry the answer?

The text sibling of :class:`ToolShap` and MM-SHAP: ablate CHUNKS OF THE GIVEN
CONTEXT (paragraphs or sentences) instead of tools or image patches, and
Shapley-attribute the answer to them.  Directly serves RAG/long-context LLM
diagnosis: a failing case whose answer depends on no chunk is answering from
priors; one dominated by a single chunk inherits that chunk's quality.

Black-box (``requires=GENERATE``): the coalition value is the similarity of
the ablated-context answer to the full-context baseline answer.  Uses the
shared permutation-sampling estimator (memoised), so model calls are bounded.

References:
- TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value
  Estimation — Goldshmidt & Horovicz, 2024 — arXiv:2407.10114
- ContextCite: Attributing Model Generation to Context —
  Cohen-Wang et al., NeurIPS 2024 — arXiv:2409.00729
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.analyzers.perturbation._shapley import shapley_values
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model


def _similarity(a: str, b: str) -> float:
    na = " ".join(str(a or "").lower().split())
    nb = " ".join(str(b or "").lower().split())
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def default_context_fn(case: "FailureCase") -> Optional[str]:
    """The ablatable context: ``metadata['context']`` when present."""
    meta = case.metadata if isinstance(case.metadata, dict) else {}
    value = meta.get("context")
    return str(value) if value else None


def split_chunks(context: str, granularity: str, max_chunks: int) -> list[str]:
    if granularity == "sentence":
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", context) if p.strip()]
    else:
        parts = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
        if len(parts) <= 1:  # single paragraph — fall back to sentences
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", context) if p.strip()]
    if len(parts) > max_chunks:  # merge the tail so players stay bounded
        head, tail = parts[: max_chunks - 1], " ".join(parts[max_chunks - 1:])
        parts = head + [tail]
    return parts


@register_analyzer("context_shap")
class ContextShapAnalyzer(Analyzer):
    """Shapley attribution of the answer to provided-context chunks (RAG dependence probe).

    Hyper-parameters:
        granularity: ``"paragraph"`` (default) or ``"sentence"`` chunking.
        n_samples:   permutation samples for the Shapley estimator.
        max_chunks:  chunk cap per case (tail chunks are merged).
        max_cases:   label-stratified cap on probed cases; 0 (the default) = every case.
        seed:        permutation-sampling seed.
        context_fn:  ``callable(case) -> str | None`` supplying the ablatable
                     context (default: ``metadata['context']``). The context
                     must appear verbatim inside the prompt.
    """

    name = "context_shap"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        granularity: str = "paragraph",
        n_samples: int = 16,
        max_chunks: int = 6,
        max_cases: int = 0,
        seed: int = 0,
        context_fn: Optional[Callable[["FailureCase"], Optional[str]]] = None,
    ) -> None:
        super().__init__(
            granularity=granularity,
            n_samples=max(1, int(n_samples)),
            max_chunks=max(2, int(max_chunks)),
            max_cases=max_cases,
            seed=seed,
        )
        # ctor name, so sklearn-style get_params() reflection works
        self.context_fn = context_fn or default_context_fn

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            context = self.context_fn(case)
            prompt = case.inputs.prompt or ""
            entry: dict[str, Any] = {"sample_id": case.id}
            if not context:
                entry["skipped"] = "no context supplied (metadata['context'] or context_fn)"
                per_case.append(entry)
                continue
            if prompt.count(context) != 1:
                entry["skipped"] = (
                    "context must occur exactly once verbatim in the prompt "
                    f"(found {prompt.count(context)} occurrences)"
                )
                per_case.append(entry)
                continue
            chunks = split_chunks(context, self.granularity, self.max_chunks)
            entry["n_chunks"] = len(chunks)
            if len(chunks) < 2:
                entry["skipped"] = "context has a single chunk — nothing to attribute"
                per_case.append(entry)
                continue

            baseline = str(case.observed or "") or str(model.generate(case.inputs))

            def value_fn(kept: set) -> float:
                kept_text = "\n\n".join(chunks[i] for i in sorted(kept))
                ablated = prompt.replace(context, kept_text, 1)
                answer = str(model.generate(dataclasses.replace(case.inputs, prompt=ablated)))
                return _similarity(answer, baseline)

            coalition_cache: dict[frozenset, float] = {}

            def cached_value(kept: set) -> float:
                key = frozenset(kept)
                if key not in coalition_cache:
                    coalition_cache[key] = value_fn(set(key))
                return coalition_cache[key]

            phi = shapley_values(
                range(len(chunks)), cached_value, n_samples=self.n_samples, seed=self.seed
            )
            no_context = cached_value(set())
            top_index = max(phi, key=lambda k: abs(phi[k]))
            total_abs = sum(abs(v) for v in phi.values()) or 1.0
            entry["context_dependence"] = round(1.0 - no_context, 4)
            entry["top_chunk_index"] = int(top_index)
            entry["top_chunk_share"] = round(abs(phi[top_index]) / total_abs, 4)
            entry["shapley"] = {str(k): round(v, 4) for k, v in phi.items()}
            per_case.append(entry)

        deps = [c["context_dependence"] for c in per_case if "context_dependence" in c]
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "granularity": self.granularity,
            "n_samples": self.n_samples,
            "mean_context_dependence": round(sum(deps) / len(deps), 4) if deps else None,
            "per_case": per_case,
            "_caveat": (
                "context_dependence near 0 = the answer survives with the "
                "context removed (answering from priors — a hallucination risk "
                "on context-grounded tasks); top_chunk_share near 1 = one chunk "
                "carries the answer. Value function is answer SIMILARITY to the "
                "full-context baseline, not correctness. Requires the context "
                "verbatim inside the prompt; deterministic decoding assumed "
                "(each coalition sampled once, memoised). INTERVENTIONAL: "
                "held-out verification must RE-RUN the ablations."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
