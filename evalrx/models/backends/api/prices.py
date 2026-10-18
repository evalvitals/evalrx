"""API prices, dated, for turning recorded token counts into an upper-bound USD cost.

Only prices checked on the provider's page are listed; a model without an
entry costs ``None`` (unknown), never a guess. Prices change, so every entry
carries the date it was read and the page it was read from. The token counts
come from the responses' usage metadata, which does not report implicit cache
hits, so the input part is an upper bound.
"""

from __future__ import annotations

from typing import Any

GEMINI_PRICING_PAGE = "https://ai.google.dev/gemini-api/docs/pricing"

#: USD per 1M tokens, standard paid tier, not batch. Output includes thinking tokens.
PRICES: dict[str, dict[str, Any]] = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "checked": "2026-10-17", "source": GEMINI_PRICING_PAGE},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "checked": "2026-10-17", "source": GEMINI_PRICING_PAGE},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "checked": "2026-10-17", "source": GEMINI_PRICING_PAGE},
}


def price_of(model: "str | None") -> "dict[str, Any] | None":
    """The dated price entry for *model* (a model id, with or without ``models/``), or ``None``."""
    key = str(model or "").removeprefix("models/")
    return PRICES.get(key)


def usd(model: "str | None", usage: "dict[str, Any] | None") -> "float | None":
    """Upper-bound cost of *usage* (``prompt_tokens`` / ``completion_tokens`` /
    ``thought_tokens``) at *model*'s listed price; ``None`` when either is unknown."""
    price = price_of(model)
    if price is None or not usage:
        return None
    tokens_in = int(usage.get("prompt_tokens", 0) or 0)
    tokens_out = int(usage.get("completion_tokens", 0) or 0) + int(usage.get("thought_tokens", 0) or 0)
    return round((tokens_in * price["input"] + tokens_out * price["output"]) / 1e6, 6)
