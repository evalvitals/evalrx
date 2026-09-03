"""CHAIR — Caption Hallucination Assessment with Image Relevance.

Black-box (``GENERATE``): generate a caption, extract mentioned objects from a
fixed object vocabulary, and compare to the image's gold objects
(``metadata["gt_objects"]``).  Reports CHAIR_i (mean fraction of mentioned objects
that are hallucinated) and CHAIR_s (fraction of captions with ≥1 hallucination).
``chair_score`` is the reusable per-instance metric.

Paper: "Object Hallucination in Image Captioning"
       Rohrbach et al., EMNLP 2018 — https://arxiv.org/abs/1809.02156
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


def chair_score(generated_objects: Iterable[str], gt_objects: Iterable[str]) -> dict:
    """Per-instance CHAIR_i: fraction of mentioned objects that are hallucinated."""
    gen = {o.lower() for o in generated_objects}
    gt = {o.lower() for o in gt_objects}
    hallucinated = sorted(gen - gt)
    chair_i = len(hallucinated) / len(gen) if gen else 0.0
    return {"chair_i": round(chair_i, 4), "hallucinated": hallucinated, "n_mentioned": len(gen)}


def extract_objects(caption: str, vocab: Iterable[str]) -> list[str]:
    """Naive object extractor: vocabulary words that appear in the caption."""
    text = f" {caption.lower()} "
    return [w for w in vocab if f" {w.lower()} " in text or f" {w.lower()}s " in text]


@register_analyzer("chair")
class CHAIRAnalyzer(Analyzer):
    """Object-hallucination rate of generated captions vs gold objects."""

    name = "chair"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"image"})
    # An omni model DECLARES image even on an audio benchmark, so the registry
    # match alone still offers this on a batch with no images in it.
    requires_modalities = frozenset({"image"})

    def __init__(self, object_vocab: Iterable[str], gt_key: str = "gt_objects") -> None:
        super().__init__(object_vocab=list(object_vocab), gt_key=gt_key)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case = []
        chair_is = []
        n_with_hallucination = 0
        for case in cases:
            caption = model.generate(case.inputs)
            mentioned = extract_objects(caption, self.object_vocab)
            gt = case.metadata.get(self.gt_key, [])
            sc = chair_score(mentioned, gt)
            per_case.append({"id": case.id, **sc, "mentioned": mentioned})
            chair_is.append(sc["chair_i"])
            if sc["hallucinated"]:
                n_with_hallucination += 1
        n = len(chair_is)
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"per_case": per_case},
            findings={
                "n": n,
                "chair_i": round(sum(chair_is) / n, 4) if n else None,   # mean per-instance hallucination rate
                "chair_s": round(n_with_hallucination / n, 4) if n else None,  # fraction of captions w/ a hallucination
            },
        )
