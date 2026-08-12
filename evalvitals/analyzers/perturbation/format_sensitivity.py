"""Multiple-choice format sensitivity — does the answer follow content or position?

LLMs (and VLMs answering MC benchmarks) carry measurable selection bias: some
models keep the same LETTER when options are reordered, instead of following
the option CONTENT.  This probe rotates the option list, re-asks, and reports
per-case columns separating the two failure signatures:

    format_flip_rate  — answers track neither content nor a fixed position
                        (brittle / guessing);
    positional_bias   — answers keep choosing the same letter across rotations
                        (position prior overrides perception).

Black-box (``requires=GENERATE``); rotations are deterministic, so held-out
verification can re-run the probe exactly.

References:
- Large Language Models Are Not Robust Multiple Choice Selectors —
  Zheng et al., ICLR 2024 — arXiv:2309.03882
- Quantifying Language Models' Sensitivity to Spurious Features in Prompt
  Design (FormatSpread) — Sclar et al., ICLR 2024 — arXiv:2310.11324
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

_OPTION_LINE = re.compile(r"^[ \t]*([A-Z])[.)][ \t]+(.+?)[ \t]*$", re.MULTILINE)
_MAX_LETTERS = 26


def extract_option_block(prompt: str) -> tuple[list[str], Optional[str]]:
    """The LAST contiguous lettered run ``A. ... B. ...`` in the prompt.

    A run must start at ``A``, increment by one letter per line, and contain
    only whitespace between consecutive option lines — this keeps few-shot
    examples and restated blocks from being merged into one bogus list.
    Returns ``(options, exact_block_text)`` or ``([], None)``.
    """
    text = str(prompt or "")
    runs: list[list[re.Match]] = []
    current: list[re.Match] = []
    for match in _OPTION_LINE.finditer(text):
        letter = match.group(1)
        if letter == "A":
            current = [match]
        elif (
            current
            and ord(letter) == ord(current[-1].group(1)) + 1
            and not text[current[-1].end():match.start()].strip()
        ):
            current.append(match)
        else:
            current = []
            continue
        if len(current) >= 2:
            runs.append(list(current))
    if not runs:
        return [], None
    best = runs[-1]  # final state of the last run
    options = [m.group(2) for m in best]
    return options, text[best[0].start():best[-1].end()]


def extract_options(case: "FailureCase") -> tuple[list[str], Optional[str]]:
    """Options from ``metadata['options']`` (no block text) or the prompt's lettered block."""
    meta = case.metadata.get("options") if isinstance(case.metadata, dict) else None
    if isinstance(meta, (list, tuple)) and 2 <= len(meta) <= _MAX_LETTERS:
        return [str(o) for o in meta], None
    options, block = extract_option_block(case.inputs.prompt or "")
    if not 2 <= len(options) <= _MAX_LETTERS:
        return [], None
    return options, block


def render_options(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))


def parse_choice(output: str, n_options: int) -> str:
    """Chosen letter: a tagged answer first, else the last standalone CAPITAL.

    The tag pattern is case-insensitive on the raw text; the fallback scans the
    raw (non-uppercased) output so the article 'a' cannot masquerade as option
    A, and a capital immediately followed by a lowercase word (\"A nice ...\")
    is treated as prose, not a choice.
    """
    n = min(int(n_options), _MAX_LETTERS)
    if n < 1:
        return ""
    last = chr(64 + n)  # 'A' + n - 1
    text = str(output or "")
    marked = re.findall(
        rf"(?:answer|choice|option|final)\s*(?:is)?\s*[:=-]?\s*([A-{last}])\b",
        text,
        re.IGNORECASE,
    )
    if marked:
        return marked[-1].upper()
    letters = re.findall(rf"\b([A-{last}])\b(?!\s+[a-z])", text)
    return letters[-1] if letters else ""


@register_analyzer("format_sensitivity")
class FormatSensitivityAnalyzer(Analyzer):
    """Option-rotation probe: separates positional letter bias from content-tracking on multiple-choice cases.

    Hyper-parameters:
        n_variants: rotations asked per case (rotation k shifts options by k;
                    capped at n_options - 1).
        max_cases:  label-stratified cap on probed cases.
    """

    name = "format_sensitivity"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, n_variants: int = 3, max_cases: int = 48) -> None:
        super().__init__(n_variants=max(1, int(n_variants)), max_cases=max_cases)

    @staticmethod
    def _variant_prompt(
        prompt: str, block: Optional[str], options: list[str], shift: int
    ) -> tuple[str, list[str]]:
        rotated = options[shift:] + options[:shift]
        rotated_block = render_options(rotated)
        if block:
            # identical restatements of the same block rotate together
            return prompt.replace(block, rotated_block), rotated
        return (
            f"{prompt}\n\nChoices:\n{rotated_block}\nAnswer with the option letter.",
            rotated,
        )

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            options, block = extract_options(case)
            entry: dict[str, Any] = {"sample_id": case.id, "n_options": len(options)}
            if len(options) < 2:
                entry["skipped"] = "no multiple-choice options found (2..26 required)"
                per_case.append(entry)
                continue
            prompt = case.inputs.prompt or ""
            shifts = list(range(1, min(self.n_variants, len(options) - 1) + 1))
            chosen_contents: list[str] = []
            chosen_letters: list[str] = []
            n_unparsed = 0
            # Identity variant: reuse the recorded baseline answer only when the
            # prompt contains the lettered block the letters decode against;
            # otherwise every variant (identity included) uses OUR rendered
            # block, so all letters map to options the model actually saw.
            if block is not None and case.observed:
                base_out = str(case.observed)
            else:
                identity_prompt, _ = self._variant_prompt(prompt, block, options, 0)
                base_out = str(
                    model.generate(dataclasses.replace(case.inputs, prompt=identity_prompt))
                )
            base_letter = parse_choice(base_out, len(options))
            if base_letter:
                chosen_letters.append(base_letter)
                chosen_contents.append(options[ord(base_letter) - 65])
            else:
                n_unparsed += 1
            for shift in shifts:
                variant_prompt, rotated = self._variant_prompt(prompt, block, options, shift)
                out = str(model.generate(dataclasses.replace(case.inputs, prompt=variant_prompt)))
                letter = parse_choice(out, len(rotated))
                if not letter:
                    n_unparsed += 1
                    continue
                chosen_letters.append(letter)
                chosen_contents.append(rotated[ord(letter) - 65])
            entry["n_variants"] = 1 + len(shifts)
            entry["n_unparsed"] = n_unparsed
            if chosen_contents:
                content_modal = Counter(chosen_contents).most_common(1)[0][1]
                letter_modal = Counter(chosen_letters).most_common(1)[0]
                entry["format_flip_rate"] = round(1.0 - content_modal / len(chosen_contents), 4)
                entry["positional_bias"] = round(letter_modal[1] / len(chosen_letters), 4)
                entry["modal_letter"] = letter_modal[0]
            per_case.append(entry)

        flips = [c["format_flip_rate"] for c in per_case if "format_flip_rate" in c]
        letters = Counter(c.get("modal_letter") for c in per_case if c.get("modal_letter"))
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_scored": len(flips),
            "mean_flip_rate": round(sum(flips) / len(flips), 4) if flips else None,
            "modal_letter_histogram": dict(letters),
            "per_case": per_case,
            "_caveat": (
                "Rotations only (not all permutations): a model biased toward "
                "'the option after the correct one' can evade detection. "
                "positional_bias near 1 with format_flip_rate near 1 is the "
                "letter-prior signature; both near 0 is content-tracking. "
                "Letter parsing prefers tagged answers and ignores capitals "
                "followed by a lowercase word, but free-prose answers can still "
                "be misread — n_unparsed tracks the abstentions. INTERVENTIONAL "
                "columns: held-out verification must RE-RUN the rotations, "
                "never reuse exploration-set values."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
