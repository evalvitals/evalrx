"""Modality ablation — is the model actually using the modality under test?

Black-box (``GENERATE``): for every media slot a case fills, re-ask the same
question with that slot **ablated**, and compare the answer to the unablated
baseline.  An answer that does not move when the audio is removed or replaced
was not produced by listening to it.

One implementation covers every slot, and that is the point.  The question "is
the image load-bearing" and the question "is the audio load-bearing" are the
same question asked of a different slot, so a per-modality analyzer would be the
same code three times, drifting apart — and in practice the audio and video
copies were simply never written, leaving an ALM run with no way to ask it at
all.  Adding a modality here is adding a member to
:data:`~evalvitals.core.case.MEDIA_SLOTS`.

Two ablation modes, because they answer different questions:

* ``drop`` (default) — remove the slot entirely.  An unchanged answer is strong
  evidence the slot was never read.  A *changed* answer proves only sensitivity,
  not correct use: the input shape changed too.
* ``swap`` — substitute another case's media, keeping the input shape intact and
  breaking only the grounding.  Cleaner evidence, and it needs no extra model
  capability, but it requires at least two cases filling the same slot.

Reported as reliance, never as correctness.  ``ungrounded_rate_<slot>`` is the
share of probed cases whose answer survived the ablation; ``by_strategy`` carries
the paired per-case correctness that M2's McNemar reads, which is what makes this
INTERVENTION-grade rather than one more association.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.perturbation.prompt_contrast import _default_score, _norm_answer
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import MEDIA_SLOTS
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

BASELINE = "baseline"


@register_analyzer("modality_ablation")
class ModalityAblationAnalyzer(Analyzer):
    """Per-slot grounding probe: does ablating a modality move the answer?"""

    name = "modality_ablation"
    description = (
        "Re-asks each case with one media slot removed or swapped; an unchanged "
        "answer means the model never used that modality."
    )
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image", "audio", "video"})
    signal_docs = {
        'ablation_audio_is_swap': "The audio was replaced with another case's rather than removed.",
        'ablation_image_is_swap': "The image was replaced with another case's rather than removed.",
        'ablation_mode_audio': 'How the audio was removed for these cases.',
        'ablation_mode_image': 'How the image was removed for these cases.',
        'ablation_mode_video': 'How the video was removed for these cases.',
        'ablation_video_is_swap': "The video was replaced with another case's rather than removed.",
        'grounded_in_audio': ('Actually listened', 'The answer changed when the audio was taken away — the model was listening.'),
        'grounded_in_image': ('Actually looked', 'The answer changed when the image was taken away — the model was using it.'),
        'grounded_in_video': ('Actually watched', 'The answer changed when the video was taken away — the model was watching.'),
        'mode': "How the modality was removed: dropped entirely, or swapped for another case's.",
        'n_cases_probed': 'How many cases were re-asked with a modality removed.',
        'n_probed_audio': 'How many cases had audio to test.',
        'n_probed_image': 'How many cases had an image to test.',
        'n_probed_video': 'How many cases had video to test.',
        'probed_slots': 'Which modalities were tested this way.',
        'ungrounded_rate_audio': ('Ignored the audio', 'Share of cases whose answer did NOT change when the audio was removed — the model was not listening.'),
        'ungrounded_rate_image': ('Ignored the image', 'Share of cases whose answer did NOT change when the image was removed — the model was not looking.'),
        'ungrounded_rate_video': ('Ignored the video', 'Share of cases whose answer did NOT change when the video was removed — the model was not watching.'),
    }
    #: Needs a filled slot to ablate. Without one there is nothing to remove and
    #: the analyzer would report a grounding rate over zero probes.
    requires_modalities = frozenset(MEDIA_SLOTS)

    def __init__(
        self,
        mode: str = "drop",
        slots: "tuple[str, ...] | None" = None,
        score_fn: Optional[Callable[["FailureCase", str], Optional[bool]]] = None,
        max_cases: int = 0,
    ) -> None:
        """
        Args:
            mode:      ``"drop"`` (remove the slot) or ``"swap"`` (substitute
                       another case's media). ``swap`` falls back to ``drop``
                       per-slot when fewer than two cases fill it.
            slots:     Restrict to these slots; ``None`` probes every filled one.
            score_fn:  ``(case, answer) -> bool | None`` for the paired
                       correctness table. Defaults to the prompt-contrast rubric
                       scorer; ``None`` verdicts are simply not tabulated.
            max_cases: Cap on cases (cost = 1 + n_filled_slots generations each);
                       0 (the default) = every case.
        """
        if mode not in ("drop", "swap"):
            raise ValueError(f"mode must be 'drop' or 'swap'; got {mode!r}")
        super().__init__(mode=mode, slots=slots, score_fn=score_fn, max_cases=max_cases)

    # -- helpers -------------------------------------------------------
    def _slots_of(self, case: "FailureCase") -> "list[str]":
        wanted = self.slots or MEDIA_SLOTS
        return [
            s for s in MEDIA_SLOTS
            if s in wanted and getattr(case.inputs, s, None) is not None
        ]

    @staticmethod
    def _donor(cases: "list[FailureCase]", slot: str, case: "FailureCase") -> Any:
        """Another case's media for *slot*, or ``None`` when there is no donor.

        Deterministic (first non-self filler in batch order) so a rerun of the
        same batch ablates with the same substitute — a random donor would make
        the paired comparison irreproducible.
        """
        for other in cases:
            if other.id != case.id and getattr(other.inputs, slot, None) is not None:
                return getattr(other.inputs, slot)
        return None

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        score = self.score_fn or _default_score
        selected = cases.stratified_head(self.max_cases)
        selected = [c for c in selected if self._slots_of(c)]

        by_strategy: dict[str, dict[str, float]] = {BASELINE: {}}
        per_case: list[dict[str, Any]] = []
        # per-slot tallies: probed cases, and those whose answer never moved
        probed = {s: 0 for s in MEDIA_SLOTS}
        ungrounded = {s: 0 for s in MEDIA_SLOTS}
        modes_used: dict[str, str] = {}

        for case in selected:
            base_answer = str(model.generate(case.inputs))
            base_key = _norm_answer(base_answer)
            base_verdict = score(case, base_answer)
            if base_verdict is not None:
                by_strategy[BASELINE][case.id] = float(bool(base_verdict))

            row: dict[str, Any] = {"sample_id": case.id}
            for slot in self._slots_of(case):
                donor = self._donor(selected, slot, case) if self.mode == "swap" else None
                # swap with no donor degrades to drop rather than skipping: the
                # weaker evidence is still evidence, and silently probing nothing
                # would look identical to a fully-grounded model.
                effective = "swap" if donor is not None else "drop"
                modes_used[slot] = effective
                ablated = dataclasses.replace(case.inputs, **{slot: donor})

                answer = str(model.generate(ablated))
                changed = _norm_answer(answer) != base_key

                probed[slot] += 1
                if not changed:
                    ungrounded[slot] += 1
                # Flat scalars only: the per-case harvester reads one level, so a
                # nested {slot: {...}} here would reach no statistic at all.
                row[f"grounded_in_{slot}"] = bool(changed)
                row[f"ablation_{slot}_is_swap"] = effective == "swap"

                strategy = f"without_{slot}"
                verdict = score(case, answer)
                if verdict is not None:
                    by_strategy.setdefault(strategy, {})[case.id] = float(bool(verdict))
            if len(row) > 1:
                per_case.append(row)

        findings: dict[str, Any] = {
            "n_cases_probed": len(selected),
            "mode": self.mode,
            "probed_slots": sorted(s for s in MEDIA_SLOTS if probed[s]),
            "_note": "measures whether a modality is load-bearing, not correctness",
            "per_case": per_case,
            "by_strategy": by_strategy,
        }
        for slot in MEDIA_SLOTS:
            n = probed[slot]
            # None, not 0.0: "no case filled this slot" and "every case that
            # filled it stayed grounded" are different facts and a reader that
            # cannot separate them will report the wrong one.
            findings[f"n_probed_{slot}"] = n
            findings[f"ungrounded_rate_{slot}"] = round(ungrounded[slot] / n, 4) if n else None
            findings[f"ablation_mode_{slot}"] = modes_used.get(slot)
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            findings=findings,
            metadata={"mode": self.mode, "n_selected": len(selected)},
        )
