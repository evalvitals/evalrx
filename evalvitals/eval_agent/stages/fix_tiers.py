"""Intervention-space tiers for the fix module (post-loop repair).

A *fix tier* names the space a candidate repair intervenes in.  The ladder is
ordered by invasiveness into the model — each step gives up some deployability
for more causal reach:

    L0   runtime config     decoding limits / retry and timeout policy
    L1   input space        prompt rewrites, instruction strategies
    L2   scaffold space     agent-designed pipelines around the unchanged
                            model: multi-call, external tools (zoom, contrast
                            enhancement, …), aggregation
    L3a  internals (read)   read attention/logits to guide scaffold actions
                            (attention-guided crop, contrastive decoding,
                            confidence routing)
    L3b  internals (write)  modify the forward pass: attention reweighting,
                            sink suppression, activation steering
    L4   parameter space    build a dataset, fine-tune, re-test

The allowed tier is an **input** to the fix module (default L2) — there is no
automatic escalation.  When every candidate within the allowed tier fails
validation, the fix module *recommends* raising the tier instead, routed from
the verified hypotheses via :func:`route_min_tier`.
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalvitals.eval_agent.hypothesis import Hypothesis


class FixTier(IntEnum):
    """Where a candidate fix intervenes, ordered by invasiveness."""

    L0_RUNTIME_CONFIG = 0
    L1_PROMPT = 1
    L2_SCAFFOLD = 2
    L3A_INTERNALS_READ = 3
    L3B_INTERNALS_WRITE = 4
    L4_PARAMETERS = 5

    @property
    def label(self) -> str:
        return _LABELS[self]

    def describe(self) -> str:
        return f"{self.label} ({_DESCRIPTIONS[self]})"


_LABELS = {
    FixTier.L0_RUNTIME_CONFIG: "L0",
    FixTier.L1_PROMPT: "L1",
    FixTier.L2_SCAFFOLD: "L2",
    FixTier.L3A_INTERNALS_READ: "L3a",
    FixTier.L3B_INTERNALS_WRITE: "L3b",
    FixTier.L4_PARAMETERS: "L4",
}

_DESCRIPTIONS = {
    FixTier.L0_RUNTIME_CONFIG: "runtime configuration: decoding limits / retry policy",
    FixTier.L1_PROMPT: "input space: prompt/instruction changes",
    FixTier.L2_SCAFFOLD: "scaffold space: pipelines + external tools around the model",
    FixTier.L3A_INTERNALS_READ: "internals, read-only: attention/logits guide the scaffold",
    FixTier.L3B_INTERNALS_WRITE: "internals, write: attention/activation modification",
    FixTier.L4_PARAMETERS: "parameter space: dataset construction + fine-tuning",
}

_PARSE = {
    "l0": FixTier.L0_RUNTIME_CONFIG,
    "l1": FixTier.L1_PROMPT,
    "l2": FixTier.L2_SCAFFOLD,
    "l3a": FixTier.L3A_INTERNALS_READ,
    "l3b": FixTier.L3B_INTERNALS_WRITE,
    "l3": FixTier.L3A_INTERNALS_READ,   # bare "L3" means the read side
    "l4": FixTier.L4_PARAMETERS,
}


def parse_tier(value: "str | FixTier") -> FixTier:
    """Parse ``"L1"``/``"l3a"``/… (or pass through a :class:`FixTier`)."""
    if isinstance(value, FixTier):
        return value
    key = str(value).strip().lower()
    if key not in _PARSE:
        raise ValueError(
            f"unknown fix tier {value!r}; expected one of "
            f"{sorted({t.label for t in FixTier})}"
        )
    return _PARSE[key]


# ---------------------------------------------------------------------------
# Hypothesis → minimum effective tier routing
# ---------------------------------------------------------------------------
# A hypothesis's verified mechanism tells you the *space* the failure lives in,
# hence the minimum tier whose interventions can causally address it.  Routing
# prefers an explicit, structured signal (``hypothesis.metadata["fix_tier"]`` or
# ``["intervention_space"]``) — that is the mechanism, not its phrasing.  Only
# when no structured field is present does it fall back to keyword matching.
#
# Keyword fallback, ordered most-invasive-first so specific vocabulary
# ("suppress attention sink") wins over generic vocabulary ("attention").  Two
# precautions distinguish a mechanism from a passing mention of its words:
#   * stems are matched at a word boundary (``\bcrop``), so "constraint" no
#     longer matches the L4 stem "train" mid-word;
#   * negated qualifiers ("training-free", "without fine-tuning", "no retraining")
#     are stripped first — they assert the OPPOSITE of needing that tier.

_TIER_KEYWORDS: "list[tuple[FixTier, tuple[str, ...]]]" = [
    (FixTier.L4_PARAMETERS, (
        "train", "finetun", "fine-tun", "retrain", "pretrain", "lora",
        "knowledge gap", "frequency prior", "language prior", "memoriz",
        "dataset bias",
    )),
    (FixTier.L3B_INTERNALS_WRITE, (
        "suppress", "steer", "reweight", "re-weight", "knockout",
        "activation patch", "modify attention", "edit attention",
        "attention interven",
    )),
    (FixTier.L3A_INTERNALS_READ, (
        "attention", "logit", "logprob", "contrastive decoding", "vcd",
        "hidden state", "internal", "entropy", "sink",
    )),
    (FixTier.L2_SCAFFOLD, (
        "resolution", "downsampl", "image token", "patch grid", "zoom",
        "crop", "contrast", "conspicuity", "small", "subtle", "low-contrast",
        "preprocess", "multi-call", "pipeline", "tool", "enhanc",
    )),
    (FixTier.L0_RUNTIME_CONFIG, (
        "truncat", "finish reason", "finish_reason", "token budget",
        "max token", "max_tokens", "output limit", "decode limit",
        "completion length", "stopped early", "timeout", "rate limit",
    )),
]

# Phrases that NEGATE a tier's vocabulary ("training-free" ≠ "needs training").
# Stripped before keyword matching so the negated stem cannot route upward.
_NEGATION_PATTERNS = (
    re.compile(r"\b(?:training|train|fine[- ]?tun\w*|finetun\w*|retrain\w*)[- ]free\b"),
    re.compile(r"\b(?:without|no|avoid(?:s|ing)?|skip(?:s|ping)?|free of)\s+"
               r"(?:any\s+|re-?\s*)?(?:training|fine[- ]?tuning|finetuning|retraining)\b"),
    re.compile(r"\b(?:does not|doesn't|do not|don't|cannot|can't|won't|will not|"
               r"no need to|without needing to)\s+(?:require\s+|need\s+|involve\s+)?"
               r"(?:any\s+)?(?:re-?\s*)?(?:train\w*|fine[- ]?tun\w*|finetun\w*)\b"),
)

# Compiled stem matchers: a word boundary before the stem, suffix free after.
_TIER_PATTERNS: "list[tuple[FixTier, tuple[tuple[str, re.Pattern[str]], ...]]]" = [
    (tier, tuple((kw, re.compile(r"\b" + re.escape(kw))) for kw in keywords))
    for tier, keywords in _TIER_KEYWORDS
]


def route_min_tier(hypothesis: "Hypothesis") -> "tuple[FixTier, str]":
    """Return the minimum tier that can causally address *hypothesis*.

    Order of precedence:

    1. An explicit ``hypothesis.metadata["fix_tier"]`` / ``["intervention_space"]``
       — the structured mechanism wins over any prose.
    2. Keyword fallback over ``predicted_failure_mode`` + ``statement`` +
       ``test_design`` (negation-stripped, word-boundary-anchored).
    3. L1 (cheapest first) when nothing matches.
    """
    meta = getattr(hypothesis, "metadata", None) or {}
    explicit = meta.get("fix_tier") or meta.get("intervention_space")
    if explicit:
        try:
            tier = parse_tier(explicit)
            return tier, f"explicit metadata fix_tier={explicit!r} -> {tier.label}"
        except ValueError:
            pass  # malformed hint — fall through to keyword routing

    text = " ".join(
        str(getattr(hypothesis, attr, "") or "")
        for attr in ("predicted_failure_mode", "statement", "test_design")
    ).lower()
    for pat in _NEGATION_PATTERNS:
        text = pat.sub(" ", text)
    for tier, patterns in _TIER_PATTERNS:
        hits = [kw for kw, pat in patterns if pat.search(text)]
        if hits:
            return tier, f"matched {hits[:3]} -> {tier.label}"
    return FixTier.L1_PROMPT, "no mechanism keywords matched -> L1 (cheapest first)"
