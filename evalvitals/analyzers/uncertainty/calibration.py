"""Confidence calibration — ECE / Brier / overconfidence gap against case labels.

Where :class:`LogprobEntropyAnalyzer` and :class:`VerbalizedConfidenceAnalyzer`
emit raw per-case uncertainty, this analyzer closes the loop against PASS/FAIL
labels: it bins confidence, compares each bin's stated confidence with its
actual accuracy, and reports expected calibration error plus the signed
overconfidence gap — for BOTH confidence channels (sequence logprob and
verbalized).  A model that is wrong exactly where it is confident needs a
different fix (abstention / recalibration) than one that is merely inaccurate.

``requires=GENERATE+LOGPROBS`` (OpenAI-style endpoints qualify); labels come
from the batch, unlabeled cases are skipped.

References:
- On Calibration of Modern Neural Networks — Guo et al., ICML 2017 —
  arXiv:1706.04599 (ECE)
- Just Ask for Calibration — Tian et al., EMNLP 2023 — arXiv:2305.14975
- Language Models (Mostly) Know What They Know — Kadavath et al., 2022 —
  arXiv:2207.05221
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import TYPE_CHECKING, Any, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, Label
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.model import Model

_ELICIT = (
    "\n\nAfter answering, state your confidence on a 0-100 scale "
    "on its own line as: Confidence: <number>"
)
_CONF = re.compile(r"confidence[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE)


def _parse_conf(text: str) -> Optional[float]:
    m = _CONF.search(str(text or ""))
    if not m:
        return None
    raw = m.group(1)
    val = float(raw)
    if val > 1.0:
        return max(0.0, min(1.0, val / 100.0))
    # the elicit asks for a 0-100 scale: an integer 0/1 is a percent, while a
    # decimal like 0.8 is taken as an (off-instruction but common) fraction
    return max(0.0, min(1.0, val if "." in raw else val / 100.0))


def expected_calibration_error(pairs: list[tuple[float, bool]], n_bins: int) -> Optional[float]:
    """Standard binned ECE over ``(confidence, correct)`` pairs."""
    if not pairs:
        return None
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        index = min(n_bins - 1, int(conf * n_bins))
        bins[index].append((conf, correct))
    total = len(pairs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_conf - accuracy)
    return round(ece, 4)


@register_analyzer("calibration")
class CalibrationAnalyzer(Analyzer):
    """Label-anchored calibration: ECE, Brier and overconfidence gap for logprob and verbalized confidence.

    Hyper-parameters:
        n_bins:    ECE bins.
        max_cases: label-stratified cap (one logprobs + one generate per case).
        elicit:    verbalized-confidence suffix appended to the prompt.
    """

    name = "calibration"
    requires = frozenset({Capability.GENERATE, Capability.LOGPROBS})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, n_bins: int = 10, max_cases: int = 128, elicit: str = _ELICIT) -> None:
        super().__init__(n_bins=max(2, int(n_bins)), max_cases=max_cases, elicit=elicit)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        lp_pairs: list[tuple[float, bool]] = []
        vb_pairs: list[tuple[float, bool]] = []
        labelled = CaseBatch([c for c in cases if c.label in (Label.PASS, Label.FAIL)])
        n_unlabelled = len(list(cases)) - len(list(labelled))
        for case in labelled.stratified_head(self.max_cases):
            correct = case.label == Label.PASS
            entry: dict[str, Any] = {"sample_id": case.id, "correct": int(correct)}

            toks = model.logprobs(case.inputs)
            lps = [t.logprob for t in toks]
            conf_lp = math.exp(sum(lps) / len(lps)) if lps else None
            if conf_lp is not None:
                conf_lp = max(0.0, min(1.0, conf_lp))
                entry["conf_logprob"] = round(conf_lp, 4)
                lp_pairs.append((conf_lp, correct))

            elicited = dataclasses.replace(
                case.inputs, prompt=(case.inputs.prompt or "") + self.elicit
            )
            raw = str(model.generate(elicited))
            conf_vb = _parse_conf(raw)
            if conf_vb is not None:
                entry["conf_verbal"] = round(conf_vb, 4)
                vb_pairs.append((conf_vb, correct))
            per_case.append(entry)

        def _summary(pairs: list[tuple[float, bool]]) -> dict[str, Any]:
            if not pairs:
                return {"n": 0, "ece": None, "brier": None, "overconfidence_gap": None}
            accuracy = sum(1 for _, ok in pairs if ok) / len(pairs)
            mean_conf = sum(c for c, _ in pairs) / len(pairs)
            brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / len(pairs)
            return {
                "n": len(pairs),
                "ece": expected_calibration_error(pairs, self.n_bins),
                "brier": round(brier, 4),
                "overconfidence_gap": round(mean_conf - accuracy, 4),
            }

        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_unlabelled_skipped": n_unlabelled,
            "n_bins": self.n_bins,
            "logprob_channel": _summary(lp_pairs),
            "verbalized_channel": _summary(vb_pairs),
            "per_case": per_case,
            "_caveat": (
                "Needs PASS/FAIL labels — unlabeled cases are skipped, and ECE "
                "on few labelled cases is unstable (aim for >= ~50). "
                "conf_logprob is the geometric-mean token probability of the "
                "model's own continuation — a proxy, not an answer probability; "
                "on API backends it reflects only the returned top-k. A positive "
                "overconfidence_gap on FAIL-heavy slices argues for abstention/"
                "recalibration fixes rather than capability fixes."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
