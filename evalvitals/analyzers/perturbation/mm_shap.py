"""MM-SHAP — modality-contribution metric, generalised to every media slot.

Black-box (``LOGPROBS``): Shapley over the input's text tokens plus one player
per FILLED media slot (image / audio / video), scored by the model's output
logprob, then aggregated by modality::

    <slot>_contribution / (text_contribution + Σ media contributions)

0 for a slot ⇒ the prediction ignores it; 1 ⇒ it carries the prediction alone.
Measures *reliance*, not correctness — report it as such.

The original metric had exactly one non-text player, the image, and this
implementation hard-coded that: the value function rebuilt ``Inputs(prompt=...,
image=...)`` and dropped audio and video on the floor.  On an audio benchmark it
therefore reported ``mm_score`` computed over a slot that was never filled — the
question "is this model actually listening" was unanswerable by the one analyzer
whose whole job is answering it.  Players are now the slots the case fills, so
an AVLM gets a per-slot breakdown and ``mm_score`` keeps its published meaning
(image reliance) as one member of it.

Paper: "MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal
       Contributions in Vision and Language Models"
       Parcalabescu & Frank, ACL 2022 — https://arxiv.org/abs/2212.08158
Code:  https://github.com/coastalcph/mm-shap
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from evalvitals.analyzers.perturbation._shapley import shapley_values
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import MEDIA_SLOTS, Inputs
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

#: Player key for a media slot. Namespaced so it can never collide with a word
#: index, which is a plain int.
def _slot_player(slot: str) -> str:
    return f"__{slot}__"


#: Kept for backwards compatibility with callers that imported the old constant.
_IMAGE = _slot_player("image")


@register_analyzer("mm_shap")
class MMShapAnalyzer(Analyzer):
    """Per-modality Shapley contribution + the MM-SHAP media-reliance scores."""

    name = "mm_shap"
    requires = frozenset({Capability.LOGPROBS})
    applies_to_modalities = frozenset({"text", "image", "audio", "video"})
    #: Needs at least one media slot filled — with none, every player is a word
    #: and the "modality contribution" it reports is a text-only tautology.
    requires_modalities = frozenset(MEDIA_SLOTS)

    def __init__(
        self,
        score_fn: Optional[Callable[[Inputs], float]] = None,
        n_samples: int = 64,
        mask_token: str = "___",
        top_k: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(score_fn=score_fn, n_samples=n_samples, mask_token=mask_token, top_k=top_k, seed=seed)

    def _default_scorer(self, model: "Model") -> Callable[[Inputs], float]:
        def score(inputs: Inputs) -> float:
            lps = model.logprobs(inputs)
            return sum(t.logprob for t in lps) / len(lps) if lps else 0.0
        return score

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        case = cases[0]
        words = str(case.inputs.prompt).split()
        filled = [s for s in MEDIA_SLOTS if getattr(case.inputs, s, None) is not None]
        score = self.score_fn or self._default_scorer(model)

        players: list = list(range(len(words)))
        players += [_slot_player(s) for s in filled]

        def value(kept: set) -> float:
            masked = " ".join(w if i in kept else self.mask_token for i, w in enumerate(words))
            slots = {
                s: (getattr(case.inputs, s) if _slot_player(s) in kept else None)
                for s in filled
            }
            return score(Inputs(prompt=masked, **slots))

        shap = shapley_values(players, value, n_samples=self.n_samples, seed=self.seed)
        text_contrib = sum(abs(shap[i]) for i in range(len(words)))
        media_contrib = {s: abs(shap.get(_slot_player(s), 0.0)) for s in filled}
        total = text_contrib + sum(media_contrib.values()) or 1.0
        top_text = sorted(
            ({"token": words[i], "shapley": round(shap[i], 4)} for i in range(len(words))),
            key=lambda d: -abs(d["shapley"]),
        )[: self.top_k]

        findings = {
            # Published MM-SHAP: image reliance. 0.0 when no image is present,
            # which is why the per-slot fields below carry the real answer for an
            # audio or video run rather than this one.
            "mm_score": round(media_contrib.get("image", 0.0) / total, 4),
            "media_score": round(sum(media_contrib.values()) / total, 4),
            "text_contribution": round(text_contrib, 4),
            "probed_slots": filled,
            "top_text_tokens": top_text,
            "_note": "measures modality reliance, not correctness",
        }
        for slot in MEDIA_SLOTS:
            # Every slot gets a key, present or not: a reader must be able to tell
            # "measured as zero" from "never in this case", and an absent key
            # collapses the two into whatever their default lookup returns.
            findings[f"has_{slot}"] = slot in filled
            findings[f"{slot}_contribution"] = (
                round(media_contrib[slot], 4) if slot in filled else None
            )
            findings[f"{slot}_score"] = (
                round(media_contrib[slot] / total, 4) if slot in filled else None
            )
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"shapley": shap},
            findings=findings,
        )
