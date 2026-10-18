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
:data:`~evalrx.core.case.MEDIA_SLOTS`.

Two ablation modes, because they answer different questions:

* ``drop`` (default) — remove the slot entirely.  An unchanged answer is strong
  evidence the slot was never read.  A *changed* answer proves only sensitivity,
  not correct use: the input shape changed too.
* ``swap`` — substitute another case's media, keeping the input shape intact and
  breaking only the grounding.  Cleaner evidence, and it needs no extra model
  capability, but it requires at least two cases filling the same slot with
  DIFFERENT media: the donor is matched on content, not on case id, and every
  row records which case it came from.

An unchanged answer after a swap is only evidence when the donor's media
should have changed it. On open-answer tasks a different chart or clip almost
always does; on yes/no and few-option tasks it often does not (a random COCO
image usually lacks the asked-about object too, so "No" stays correct).
``donor_rule="counterfactual"`` therefore prefers a donor where the dataset
asks the SAME question (identical prompt) about different media and its gold
answer differs from the model's own unablated answer: answering correctly on
the donor media then requires a different answer. (Relative to the model's
answer, not the case's gold: on a failed case the two differ, and a donor
whose gold equals the wrong answer makes "unchanged" the correct behaviour.)
When no such case exists it falls back to a content-different donor and says
so per row.

Reported as reliance, never as correctness.  ``ungrounded_rate_<slot>`` is the
share of probed cases whose answer survived the ablation; ``by_strategy`` carries
the paired per-case correctness that M2's McNemar reads, which is what makes this
INTERVENTION-grade rather than one more association.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.analyzers.perturbation.prompt_contrast import _default_score, _norm_answer
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.case import MEDIA_SLOTS
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

BASELINE = "baseline"
DONOR_RULES = ("content", "counterfactual")


def _prompt_key(case: Any) -> str:
    return " ".join(str(getattr(case.inputs, "prompt", "") or "").split()).lower()


def _gold_key(case: Any) -> "str | None":
    """The case's gold answer under the same normalisation the answers get."""
    meta = getattr(case, "metadata", None) or {}
    gold = meta.get("gold", getattr(case, "expected", None))
    if gold is None:
        return None
    if isinstance(gold, (list, tuple)):
        gold = "|".join(str(g) for g in gold)
    return _norm_answer(str(gold))


def _media_key(media: Any) -> Any:
    """Content identity of one media value, so a swap never substitutes the same media.

    Files hash their bytes (two rows may name the same file, or copies of it),
    in-memory images/arrays hash their pixels; a URL or unreadable path is its
    own reference; anything else falls back to its repr.
    """
    if isinstance(media, (str, os.PathLike)):
        path = os.fspath(media)
        try:
            with open(path, "rb") as fh:
                return ("file", hashlib.sha1(fh.read()).hexdigest())
        except OSError:
            return ("ref", path)
    if isinstance(media, (bytes, bytearray)):
        return ("bytes", hashlib.sha1(bytes(media)).hexdigest())
    tobytes = getattr(media, "tobytes", None)
    if callable(tobytes):
        try:
            shape = getattr(media, "shape", None) or getattr(media, "size", None)
            return ("pixels", type(media).__name__, getattr(media, "mode", None), str(shape),
                    hashlib.sha1(tobytes()).hexdigest())
        except Exception:  # noqa: BLE001 - an unhashable buffer falls through to repr
            pass
    return ("repr", repr(media))


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
        'ablation_base_answer': 'Swap mode: the unablated answer the ablated ones are compared against.',
        'ablation_audio_answer': 'Swap mode: the answer with the audio swapped (or dropped).',
        'ablation_image_answer': 'Swap mode: the answer with the image swapped (or dropped).',
        'ablation_video_answer': 'Swap mode: the answer with the video swapped (or dropped).',
        'ablation_audio_donor_kind': "Swap mode: 'counterfactual' (same question, different audio, gold differs from the model's answer), 'content' (different audio only) or None (dropped).",
        'ablation_image_donor_kind': "Swap mode: 'counterfactual' (same question, different image, gold differs from the model's answer), 'content' (different image only) or None (dropped).",
        'ablation_video_donor_kind': "Swap mode: 'counterfactual' (same question, different video, gold differs from the model's answer), 'content' (different video only) or None (dropped).",
        'ablation_audio_donor_expected': "Counterfactual donor: the gold answer of the same question on the donor's audio.",
        'ablation_image_donor_expected': "Counterfactual donor: the gold answer of the same question on the donor's image.",
        'ablation_video_donor_expected': "Counterfactual donor: the gold answer of the same question on the donor's video.",
        'n_counterfactual_audio': 'Swap mode: cases whose audio donor is a counterfactual (the answer has to change).',
        'n_counterfactual_image': 'Swap mode: cases whose image donor is a counterfactual (the answer has to change).',
        'n_counterfactual_video': 'Swap mode: cases whose video donor is a counterfactual (the answer has to change).',
        'donor_rule': "Swap mode: how donors were chosen ('content' or 'counterfactual').",
        'ablation_audio_donor': "Swap mode: the case whose audio was put in (None = no different audio, so it was dropped).",
        'ablation_audio_is_swap': "The audio was replaced with another case's rather than removed.",
        'ablation_image_donor': "Swap mode: the case whose image was put in (None = no different image, so it was dropped).",
        'ablation_image_is_swap': "The image was replaced with another case's rather than removed.",
        'ablation_mode_audio': 'How the audio was removed for these cases.',
        'ablation_mode_image': 'How the image was removed for these cases.',
        'ablation_mode_video': 'How the video was removed for these cases.',
        'ablation_video_donor': "Swap mode: the case whose video was put in (None = no different video, so it was dropped).",
        'ablation_video_is_swap': "The video was replaced with another case's rather than removed.",
        'grounded_in_audio': ('Actually listened', 'The answer changed when the audio was taken away — the model was listening.'),
        'grounded_in_image': ('Actually looked', 'The answer changed when the image was taken away — the model was using it.'),
        'grounded_in_video': ('Actually watched', 'The answer changed when the video was taken away — the model was watching.'),
        'mode': "How the modality was removed: dropped entirely, or swapped for another case's.",
        'n_cases_probed': 'How many cases were re-asked with a modality removed.',
        'n_probed_audio': 'How many cases had audio to test.',
        'n_probed_image': 'How many cases had an image to test.',
        'n_probed_video': 'How many cases had video to test.',
        'n_drop_fallback_audio': 'Swap mode: cases whose audio was dropped because no case held different audio.',
        'n_drop_fallback_image': 'Swap mode: cases whose image was dropped because no case held a different image.',
        'n_drop_fallback_video': 'Swap mode: cases whose video was dropped because no case held different video.',
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
        donor_rule: str = "content",
        donor_pool: "CaseBatch | None" = None,
    ) -> None:
        """
        Args:
            mode:       ``"drop"`` (remove the slot) or ``"swap"`` (substitute
                        another case's media). ``swap`` falls back to ``drop``
                        per-slot when no eligible case holds different media.
            slots:      Restrict to these slots; ``None`` probes every filled one.
            score_fn:   ``(case, answer) -> bool | None`` for the paired
                        correctness table. Defaults to the prompt-contrast rubric
                        scorer; ``None`` verdicts are simply not tabulated.
            max_cases:  Cap on cases (cost = 1 + n_filled_slots generations each);
                        0 (the default) = every case.
            donor_rule: swap only — ``"content"``: the next case whose media
                        differs; ``"counterfactual"``: first a case asking the
                        same prompt about different media whose gold differs
                        from the model's unablated answer, else the
                        ``"content"`` donor.
            donor_pool: swap only — the cases donors may come from (e.g. the
                        whole data partition the probed batch was drawn from);
                        ``None`` = the probed batch itself.
        """
        if mode not in ("drop", "swap"):
            raise ValueError(f"mode must be 'drop' or 'swap'; got {mode!r}")
        if donor_rule not in DONOR_RULES:
            raise ValueError(f"donor_rule must be one of {DONOR_RULES}; got {donor_rule!r}")
        super().__init__(mode=mode, slots=slots, score_fn=score_fn, max_cases=max_cases,
                         donor_rule=donor_rule, donor_pool=donor_pool)

    # -- helpers -------------------------------------------------------
    def _slots_of(self, case: "FailureCase") -> "list[str]":
        wanted = self.slots or MEDIA_SLOTS
        return [
            s for s in MEDIA_SLOTS
            if s in wanted and getattr(case.inputs, s, None) is not None
        ]

    @staticmethod
    def _donor(cases: "list[FailureCase]", slot: str, case: "FailureCase",
               key_of: "Callable[[Any], Any]") -> "tuple[str, Any] | None":
        """``(donor id, media)`` for *slot*, or ``None`` when no case holds different media.

        The next case in batch order (wrapping) whose media differs in CONTENT
        from this case's own. Deterministic, so a rerun of the same batch swaps
        with the same donors — a random donor would make the paired comparison
        irreproducible — and spread over the batch instead of one shared donor.
        Comparing case ids is not enough: two rows can carry the same file
        (POPE asks two questions per COCO image, adjacent in the manifest), and
        "swapping" the same image back in reads as a model that ignores it.
        """
        own = key_of(getattr(case.inputs, slot))
        start = next((i for i, c in enumerate(cases) if c.id == case.id), -1)
        for step in range(1, len(cases) + 1):
            other = cases[(start + step) % len(cases)]
            media = getattr(other.inputs, slot, None)
            if media is not None and other.id != case.id and key_of(media) != own:
                return str(other.id), media
        return None

    @staticmethod
    def _counterfactual_donor(same_prompt: "list[FailureCase]", slot: str, case: "FailureCase",
                              answer_key: str, key_of: "Callable[[Any], Any]",
                              ) -> "tuple[str, Any, str] | None":
        """``(donor id, media, donor gold)``: the same question on different media whose
        gold differs from the model's own answer *answer_key*.

        The donor case asks this exact prompt about its own media, so its gold
        is what a correct answer must become after the swap. Searched from the
        case's own position among the same-prompt cases (wrapping), so cases
        sharing a question spread over the available donors.
        """
        if not answer_key:
            return None
        own = key_of(getattr(case.inputs, slot))
        start = next((i for i, c in enumerate(same_prompt) if c.id == case.id), -1)
        for step in range(1, len(same_prompt) + 1):
            other = same_prompt[(start + step) % len(same_prompt)]
            media = getattr(other.inputs, slot, None)
            gold = _gold_key(other)
            if (media is None or other.id == case.id or gold is None or gold == answer_key
                    or key_of(media) == own):
                continue
            return str(other.id), media, gold
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
        modes_used: dict[str, set[str]] = {}
        swap = self.mode == "swap"
        media_keys: dict[int, Any] = {}
        pool = list(self.donor_pool) if self.donor_pool is not None else selected
        by_prompt: dict[str, list] = {}
        if swap and self.donor_rule == "counterfactual":
            for c in pool:
                by_prompt.setdefault(_prompt_key(c), []).append(c)

        def key_of(media: Any) -> Any:  # hash each media value once per run
            if id(media) not in media_keys:
                media_keys[id(media)] = _media_key(media)
            return media_keys[id(media)]

        def find_donor(slot: str, case: Any, answer_key: str) -> "tuple[str, Any, str, str | None] | None":
            if by_prompt:
                cf = self._counterfactual_donor(by_prompt.get(_prompt_key(case), []), slot, case,
                                                answer_key, key_of)
                if cf is not None:
                    return cf[0], cf[1], "counterfactual", cf[2]
            found = self._donor(pool, slot, case, key_of)
            return (found[0], found[1], "content", None) if found else None

        for case in selected:
            base_answer = str(model.generate(case.inputs))
            base_key = _norm_answer(base_answer)
            base_verdict = score(case, base_answer)
            if base_verdict is not None:
                by_strategy[BASELINE][case.id] = float(bool(base_verdict))

            row: dict[str, Any] = {"sample_id": case.id}
            if swap:
                row["ablation_base_answer"] = base_answer[:120]
            for slot in self._slots_of(case):
                found = find_donor(slot, case, base_key) if swap else None
                donor = found[1] if found else None
                # swap with no donor degrades to drop rather than skipping: the
                # weaker evidence is still evidence, and silently probing nothing
                # would look identical to a fully-grounded model.
                effective = "swap" if donor is not None else "drop"
                modes_used.setdefault(slot, set()).add(effective)
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
                if swap:
                    row[f"ablation_{slot}_donor"] = found[0] if found else None
                    row[f"ablation_{slot}_donor_kind"] = found[2] if found else None
                    row[f"ablation_{slot}_donor_expected"] = found[3] if found else None
                    row[f"ablation_{slot}_answer"] = answer[:120]

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
            used = modes_used.get(slot, set())
            findings[f"ablation_mode_{slot}"] = (
                None if not used else next(iter(used)) if len(used) == 1 else "mixed"
            )
            if swap:
                findings[f"n_drop_fallback_{slot}"] = sum(
                    1 for r in per_case if r.get(f"ablation_{slot}_is_swap") is False
                )
                findings[f"n_counterfactual_{slot}"] = sum(
                    1 for r in per_case if r.get(f"ablation_{slot}_donor_kind") == "counterfactual"
                )
        if swap:
            findings["donor_rule"] = self.donor_rule
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            findings=findings,
            metadata={"mode": self.mode, "n_selected": len(selected)},
        )
