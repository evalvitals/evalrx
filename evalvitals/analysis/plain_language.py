"""Shared plain-language guardrail for reader-facing headlines.

Both the M2 explorer's takeaway ``plain_title`` and the M3 hypothesis agent's
``PLAIN`` line are meant to be read by someone with no statistics
background — the prompt asking nicely for "plain language" is not enough on
its own (an LLM will readily reuse the jargon-heavy technical line), so this
is the host-side check that catches it and drives a bounded repair turn.
"""

from __future__ import annotations

import re

_JARGON_PATTERN = re.compile(
    # PDF/CDF are excluded: this package's own prompts use "PDF" for the
    # document filetype (see prompts/explorer.py's raw-folder framing), so a
    # perfectly plain headline about scanned PDFs would otherwise false-flag.
    r"\b(AUC|ROC|ECDF|Spearman|Pearson|collinear(?:ity)?|logit|"
    r"logistic|coefficient(?:s)?|p-value|p value|confound(?:ed|ing)?|"
    r"monotonic|quantile|covariance|latent|mediator|"
    r"stratif(?:y|ies|ied)|z-score|t-test|chi-square(?:d)?|FDR|e-value|CI)\b",
    re.IGNORECASE,
)
_JARGON_SYMBOLS = ("→", "ρ", "σ", "µ")


def jargon_violation(plain_text: str, technical_text: str = "") -> str | None:
    """Return a short reason *plain_text* fails the plain-language check, or
    None if it passes. *technical_text* — the paired jargon-heavy line, if
    any — is used only to catch a lazy verbatim copy."""
    plain = (plain_text or "").strip()
    if not plain:
        return "missing"
    if technical_text and plain.lower() == technical_text.strip().lower():
        return "repeats the technical line verbatim"
    hit = _JARGON_PATTERN.search(plain)
    if hit:
        return f"uses jargon ({hit.group(0)!r})"
    for sym in _JARGON_SYMBOLS:
        if sym in plain:
            return f"uses a symbol ({sym!r}) instead of words"
    return None
