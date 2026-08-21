"""POPE — Polling-based Object Probing Evaluation.

Black-box (``GENERATE``): pose yes/no object-presence questions about an image and
score the model's answers against gold labels.  Each case carries the gold label
in ``metadata["pope_label"]`` ("yes"/"no") — build a probe set with the datasets
layer.  Reports accuracy / precision / recall / F1 (positive class = "yes").

Paper: "Evaluating Object Hallucination in Large Vision-Language Models"
       Li et al., EMNLP 2023 — https://arxiv.org/abs/2305.10355
Code:  https://github.com/AoiDragon/POPE
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


def parse_yes_no(text: str) -> Optional[str]:
    """Map a free-form answer to 'yes' / 'no' / None (word-boundary, not substring).

    Matching whole words avoids the classic trap where 'not'/'none'/'nothing'
    contain the substring 'no'.
    """
    words = re.findall(r"[a-z]+", str(text).lower())
    for w in words[:6]:  # decision is almost always in the opening words
        if w == "yes":
            return "yes"
        if w == "no":
            return "no"
    if "yes" in words:
        return "yes"
    if "no" in words:
        return "no"
    return None


@register_analyzer("pope")
class POPEAnalyzer(Analyzer):
    """Score yes/no object-presence probes against gold labels."""

    name = "pope"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"image"})

    def __init__(self, label_key: str = "pope_label") -> None:
        super().__init__(label_key=label_key)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        tp = fp = fn = tn = unparsed = 0
        per_case = []
        for case in cases:
            gold = str(case.metadata.get(self.label_key, "")).strip().lower()
            pred = parse_yes_no(model.generate(case.inputs))
            is_correct = pred == gold if gold in {"yes", "no"} and pred is not None else None
            per_case.append({
                "id": case.id,
                "gold": gold,
                "pred": pred,
                # Mechanism signals first: M2's stats layer caps how many per-case
                # signal keys it tests, and these two carry the diagnosis
                # (hallucinated presence vs missed finding) — is_correct alone
                # cannot distinguish them.
                "false_positive": gold == "no" and pred == "yes",
                "false_negative": gold == "yes" and pred == "no",
                # The two MARGINALS those flags are built from. false_positive /
                # false_negative are gold x pred conjunctions -- each a subset of
                # FAIL by construction, so the planner keeps them out of the
                # tested family (descriptive only). answered_yes / gold_yes are
                # not label functions and are what M2/M5 test a directional
                # ("over-affirmation") hypothesis on. Same columns as
                # answer_extraction_audit emits on any yes/no task.
                **({"answered_yes": pred == "yes"} if pred is not None else {}),
                **({"gold_yes": gold == "yes"} if gold in {"yes", "no"} else {}),
                "has_gold": gold in {"yes", "no"},
                "unparsed": pred is None,
                "is_correct": is_correct,
            })
            if pred is None:
                unparsed += 1
                continue
            if gold == "yes" and pred == "yes":
                tp += 1
            elif gold == "no" and pred == "yes":
                fp += 1
            elif gold == "yes" and pred == "no":
                fn += 1
            elif gold == "no" and pred == "no":
                tn += 1
        n = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"per_case": per_case},
            findings={
                "n": n, "unparsed": unparsed,
                "accuracy": round((tp + tn) / n, 4) if n else None,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "yes_rate": round((tp + fp) / n, 4) if n else None,  # >0.5 ⇒ over-affirmation (hallucination)
                "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
                "per_case": per_case,
            },
        )
