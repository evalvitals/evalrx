"""e-BH — FDR control over a set of e-values.

Controls FDR under ARBITRARY dependence between the e-values (which the closed
loop has: overlapping prompts, shared tasks) — no independence assumption needed,
unlike BH on p-values.  Procedure: sort e-values descending; reject the largest
k for which ``e_(k) >= m / (alpha * k)``.

Paper: "False Discovery Rate Control with E-values"
       Wang & Ramdas (2022), JRSSB — https://arxiv.org/abs/2009.02824
"""

from __future__ import annotations

from typing import Sequence


def ebh(evalues: Sequence[float], alpha: float = 0.05) -> list[int]:
    """Return the indices (into *evalues*) rejected by e-BH at level *alpha*."""
    m = len(evalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: evalues[i], reverse=True)
    k_star = 0
    for k in range(1, m + 1):
        if evalues[order[k - 1]] >= m / (alpha * k):
            k_star = k
    return sorted(order[:k_star])
