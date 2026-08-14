"""Perturbation battery — does the answer track the problem, or the template?

GSM-Symbolic's finding is that accuracy on a fixed benchmark hides two opposite
brittlenesses, and both need the *same* item perturbed rather than a new item:

  * **invariance breaks** — renaming a person or appending a clause that changes
    nothing flips the answer (the model is keying on surface form),
  * **missing sensitivity** — changing the NUMBERS leaves the answer unmoved
    (the model is reciting a remembered result, not computing one).

Neither needs a new gold label, which is what makes the battery cheap: for the
meaning-preserving arms the answer must not move, for the meaning-altering arm
it must.  The GSM-NoOp arm — one true but irrelevant clause — is the single
most diagnostic of the set and gets its own column.

Cost: 1 + ``len(perturbations)`` generations per case (5 by default).

References:
- GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in
  Large Language Models — Mirzadeh et al., ICLR 2025 — arXiv:2410.05229
- Are NLP Models really able to Solve Simple Math Word Problems? — Patel et al.,
  NAACL 2021 — arXiv:2103.07191 (SVAMP: sensitivity to question structure)
- Adaptive Testing / metamorphic robustness for NLP — Ribeiro et al., ACL 2020
  (CheckList) — arXiv:2005.04118
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.reasoning._text import extract_answer, normalize_answer
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

#: Replacement names, chosen to be unambiguous and unlikely to appear already.
_NAME_POOL = ("Kavi", "Rania", "Tomas", "Ingrid", "Naledi", "Yusuf")
_NAMED_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,11})\b", re.MULTILINE)
_INTEGER = re.compile(r"(?<![\w.])(\d{1,6})(?![\w.])")
_STOP_CAPS = frozenset(
    {
        "The", "This", "That", "These", "Those", "There", "How", "What", "Why",
        "When", "Which", "Who", "If", "Answer", "Question", "Step", "Let", "In",
        "On", "At", "For", "And", "But", "So", "Then", "Each", "Every", "All",
        "Reply", "Give", "Think", "Note", "Total", "First", "Next", "Finally",
    }
)
#: A clause that is true, on-topic, and load-bearing for nothing (GSM-NoOp).
_NOOP_CLAUSE = (
    " Note that last year the same quantities were recorded in a different "
    "notebook, which has since been archived."
)

PRESERVING = "preserving"
ALTERING = "altering"


def rename_entities(prompt: str) -> Optional[str]:
    """Swap capitalised mid-sentence names for unused ones (meaning-preserving)."""
    found: list[str] = []
    for match in _NAMED_ENTITY.finditer(prompt):
        token = match.group(1)
        if token not in _STOP_CAPS and token not in found:
            found.append(token)
    if not found:
        return None
    mapping = {name: _NAME_POOL[i % len(_NAME_POOL)] for i, name in enumerate(found[:3])}
    out = prompt
    for old, new in mapping.items():
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out if out != prompt else None


def append_noop_clause(prompt: str) -> Optional[str]:
    """Append one true-but-irrelevant sentence (the GSM-NoOp arm)."""
    lines = prompt.rstrip().splitlines()
    if not lines:
        return None
    # attach to the narrative, not to a trailing instruction line
    idx = 0 if len(lines) == 1 else max(0, len(lines) - 2)
    lines[idx] = lines[idx].rstrip() + _NOOP_CLAUSE
    return "\n".join(lines)


def restate_question(prompt: str) -> Optional[str]:
    """Prepend a neutral framing line (meaning-preserving formatting change)."""
    return f"Read the problem carefully, then solve it.\n\n{prompt}"


def perturb_numbers(prompt: str) -> Optional[str]:
    """Shift every standalone integer (meaning-ALTERING: the answer must move)."""
    seen = 0

    def _bump(match: re.Match) -> str:
        nonlocal seen
        value = int(match.group(1))
        seen += 1
        # +7 avoids the 0/1 identities and keeps magnitudes plausible
        return str(value + 7 if value >= 2 else value + 3)

    out = _INTEGER.sub(_bump, prompt)
    return out if seen and out != prompt else None


#: ``(name, kind, fn)`` — kind decides whether a changed answer is a defect.
DEFAULT_PERTURBATIONS: tuple = (
    ("rename_entities", PRESERVING, rename_entities),
    ("restate_question", PRESERVING, restate_question),
    ("noop_clause", PRESERVING, append_noop_clause),
    ("perturb_numbers", ALTERING, perturb_numbers),
)


@register_analyzer("perturbation_battery")
class PerturbationBattery(Analyzer):
    """Metamorphic battery: invariance breaks and missing sensitivity, per case.

    Hyper-parameters:
        perturbations: ``((name, kind, fn), ...)`` with ``kind`` in
                       ``{"preserving", "altering"}``; ``fn(prompt) -> str | None``
                       returns ``None`` when it does not apply to that prompt.
        max_cases:     label-stratified cap.
        gen_kwargs:    passed to ``model.generate`` — use temperature 0, or the
                       flip rates measure decoding noise instead of brittleness.
        answer_fn:     ``callable(text) -> str`` answer extractor.
    """

    name = "perturbation_battery"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        perturbations: tuple = DEFAULT_PERTURBATIONS,
        max_cases: int = 24,
        gen_kwargs: Optional[dict] = None,
        answer_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(max_cases=max_cases, gen_kwargs=dict(gen_kwargs or {}))
        self.perturbations = tuple(perturbations)
        self.answer_fn = answer_fn or extract_answer

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            per_case.append(self._probe_case(model, case))

        scored = [c for c in per_case if "invariance_break_rate" in c]
        applied = {name for c in scored for name in c.get("applied", [])}
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_scored": len(scored),
            "perturbations": [name for name, _, _ in self.perturbations],
            "applied_perturbations": sorted(applied),
            "gen_kwargs": dict(self.gen_kwargs),
            "mean_invariance_break_rate": _mean(
                [c["invariance_break_rate"] for c in scored]
            ),
            "noop_break_rate": _mean(
                [c["noop_clause_flipped"] for c in scored if "noop_clause_flipped" in c]
            ),
            "mean_sensitivity_rate": _mean(
                [c["sensitivity_rate"] for c in scored if c["sensitivity_rate"] is not None]
            ),
            "n_memorization_suspect": sum(c.get("memorization_suspect", 0) for c in scored),
            "per_case": per_case,
            "_caveat": (
                "Two DIFFERENT defects share this probe and must not be summed: "
                "invariance_break_rate should be 0 (a preserving edit moved the "
                "answer) while sensitivity_rate should be 1 (an altering edit "
                "must move it). sensitivity_rate == 0 with a correct baseline is "
                "the memorisation signature — cross-check contamination_score "
                "before calling it reasoning. Every rate is measured against the "
                "model's OWN baseline answer, not the gold, so a case that was "
                "already wrong still contributes: read these as stability, never "
                "as accuracy. Under sampling, a 'flip' can be decoding noise — "
                "set temperature 0 or repeat. INTERVENTIONAL: re-run to confirm."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _probe_case(self, model: "Model", case: "FailureCase") -> dict[str, Any]:
        prompt = str(case.inputs.prompt or "")
        entry: dict[str, Any] = {"sample_id": case.id}
        if not prompt.strip():
            entry["skipped"] = "empty prompt"
            return entry

        baseline = normalize_answer(
            self.answer_fn(model.generate(case.inputs, **self.gen_kwargs))
        )
        entry["baseline_answer"] = baseline[:80]

        applied: list[str] = []
        preserving_flips: list[int] = []
        altering_flips: list[int] = []
        for name, kind, fn in self.perturbations:
            try:
                variant = fn(prompt)
            except Exception:  # pragma: no cover - user-supplied perturbation
                variant = None
            if not variant or variant == prompt:
                continue
            applied.append(name)
            answer = normalize_answer(
                self.answer_fn(
                    model.generate(
                        dataclasses.replace(case.inputs, prompt=variant), **self.gen_kwargs
                    )
                )
            )
            flipped = int(answer != baseline)
            entry[f"{name}_flipped"] = flipped
            (preserving_flips if kind == PRESERVING else altering_flips).append(flipped)

        entry["applied"] = applied
        if not applied:
            entry["skipped"] = "no perturbation applies to this prompt"
            return entry

        entry["invariance_break_rate"] = (
            round(sum(preserving_flips) / len(preserving_flips), 4)
            if preserving_flips
            else 0.0
        )
        entry["n_preserving"] = len(preserving_flips)
        entry["sensitivity_rate"] = (
            round(sum(altering_flips) / len(altering_flips), 4) if altering_flips else None
        )
        entry["n_altering"] = len(altering_flips)
        if altering_flips and sum(altering_flips) == 0:
            # the numbers changed and the answer did not: recall, not computation
            entry["memorization_suspect"] = 1
        return entry


def _mean(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None
