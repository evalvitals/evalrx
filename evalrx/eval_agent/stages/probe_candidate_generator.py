"""VLM probe candidate generator — ProbeLLM Macro/Micro generators (Sec 3.3),
scoped to VLM QA and plugged into :class:`~evalrx.analysis.probe_search.ProbeSearch`.

Scope (v1): both Macro and Micro produce a *paraphrase* of an existing seed's
question over the SAME image, never new imagery and never an altered
semantic target — so the seed's ``expected`` answer stays valid for the new
candidate without needing a vision-capable judge or an image/answer
generation tool (the paper's tool-augmented generation, which invokes web/code
tools to obtain or verify a *new* gold answer, is out of scope here — see
``ProbeSearch``'s docstring: a richer generator with those tools can be
substituted without touching the search algorithm itself).

- **Macro** (broad coverage): picks the seed question least similar (by token
  overlap) to what the macro tree has already explored, then paraphrases it —
  diversifies which part of the fixed image pool gets visited next.
- **Micro** (local refinement): paraphrases the *current search node's own*
  case — same image, same gold answer, different wording — probing surface
  robustness (does the model's correctness flip on a reworded but
  semantically identical question) rather than the paper's full
  entity/attribute substitution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.eval_agent.prompts.probe_candidate_generator import PARAPHRASE_PROMPT

if TYPE_CHECKING:
    from evalrx.analysis.probe_search import ProbeNode
    from evalrx.core.model import Model

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard_distance(a: str, b: str) -> float:
    """1.0 = no shared tokens (maximally distinct), 0.0 = identical token sets."""
    ta, tb = _tokenize(a), _tokenize(b)
    union = ta | tb
    if not union:
        return 0.0
    return 1.0 - len(ta & tb) / len(union)


def _extract_question(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"^(question|paraphrase)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"')


@dataclass
class VLMProbeCandidateGenerator:
    """ProbeLLM-style Macro/Micro candidate generation over a fixed VLM seed pool.

    Args:
        seed_pool: Existing (image, question, expected) cases — e.g. loaded via
                   ``VLMQADataset`` — the only source of images/gold answers
                   (Scope v1: no new imagery, no new gold synthesis).
        judge:     Text-only judge used to paraphrase questions (only the
                   question text is sent — never the image, so no
                   vision-capable judge is required). Required for either
                   regime to produce candidates; ``available`` is False and
                   both ``macro``/``micro`` return ``None`` without one
                   (mirrors ``ProbeGenerator.available``).
        max_repairs: Retries if the judge echoes the question unchanged.
    """

    seed_pool: CaseBatch
    judge: "Model | None" = None
    max_repairs: int = 1

    @property
    def available(self) -> bool:
        return self.judge is not None and len(self.seed_pool) > 0

    def macro(self, node: "ProbeNode", explored: "list[ProbeNode]") -> "FailureCase | None":
        """Diversify: paraphrase the pool seed least similar to what the macro
        tree has already visited (paper Eq.11's "under-represented" frontier,
        approximated here by question-token overlap rather than embeddings)."""
        if not self.available:
            return None
        explored_prompts = [n.case.inputs.prompt for n in explored]
        seed = max(
            self.seed_pool,
            key=lambda c: min(
                (_jaccard_distance(c.inputs.prompt, p) for p in explored_prompts),
                default=1.0,
            ),
        )
        return self._paraphrase(seed, style="a very differently worded question")

    def micro(self, node: "ProbeNode") -> "FailureCase | None":
        """Refine: paraphrase the search node's own case for local
        surface-robustness probing (same image, same gold answer)."""
        if not self.available:
            return None
        return self._paraphrase(
            node.case, style="a lightly reworded variant (synonyms/reordering)"
        )

    def _paraphrase(self, base: FailureCase, *, style: str) -> "FailureCase | None":
        prompt = PARAPHRASE_PROMPT.format(question=base.inputs.prompt, style=style)
        for _attempt in range(self.max_repairs + 1):
            try:
                raw = self.judge.generate(prompt)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 — generation is best-effort
                logger.warning("VLMProbeCandidateGenerator: judge.generate failed: %s", exc)
                return None
            question = _extract_question(str(raw))
            if question and question.strip().lower() != base.inputs.prompt.strip().lower():
                return FailureCase(
                    inputs=Inputs(prompt=question, image=base.inputs.image),
                    expected=base.expected,
                    tags={"probe_search_candidate"},
                    metadata={"seed_case_id": base.id},
                )
        return None
