"""stats/api.py — the single entry point. It NEVER returns a bare p-value.

``compare`` returns a :class:`StatResult` with an effect size + CI, an e-value, a
corrected reject decision, and an underpowered flag — so a caller can't ship
"p<0.05" with a CI hugging zero.  Defaults: paired binary -> McNemar + a paired
e-value (anytime-valid), clustered bootstrap CI (cluster at task).

Underlying methods and their papers:
  McNemar test     — McNemar (1947) https://doi.org/10.1007/BF02295996
  Clustered CI     — Efron (1979), Cameron et al. (2008) https://doi.org/10.1162/rest.90.3.414
  E-value          — Grünwald et al. (2022) https://arxiv.org/abs/1906.07801
  e-BH (FDR)       — Wang & Ramdas (2022) https://arxiv.org/abs/2009.02824
  Friedman+Nemenyi — Demšar (2006) https://jmlr.org/papers/v7/demsarar06a.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from evalrx.stats.bootstrap import clustered_bootstrap_diff
from evalrx.stats.evalue import evalue_bernoulli
from evalrx.stats.friedman import friedman_test, nemenyi_cd, nemenyi_pairs
from evalrx.stats.mcnemar import mcnemar


@dataclass
class StatResult:
    """A corrected, effect-sized verdict — the only thing stats hands back."""

    effect: float                       # mean(B) - mean(A)
    ci: tuple[float, float]
    reject: bool                        # corrected decision (e-value / CI), NOT a raw p
    method: str
    alpha: float
    e_value: Optional[float] = None
    underpowered: bool = False
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        arrow = "B>A" if self.effect > 0 else ("A>B" if self.effect < 0 else "A=B")
        verdict = "REJECT H0" if self.reject else "inconclusive"
        warn = "  ⚠ underpowered" if self.underpowered else ""
        e = f", e={self.e_value:.2f}" if self.e_value is not None else ""
        return (f"[{self.method}] effect={self.effect:+.4f} ({arrow}) "
                f"CI={self.ci[0]:+.4f}..{self.ci[1]:+.4f}{e} -> {verdict}{warn}")


def compare(
    success_a: Sequence,
    success_b: Sequence,
    *,
    paired: bool = True,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    cluster_by: Optional[Sequence] = None,
    correction: str = "evalue",   # "evalue" (anytime-valid) | "p" (McNemar exact p)
    n_boot: int = 2000,
    seed: int = 0,
) -> StatResult:
    """Compare two per-example success vectors. Returns a :class:`StatResult`."""
    # The bootstrap CI level must track alpha: on the unpaired path `reject` is
    # "CI excludes 0", so a smaller alpha (e.g. a Bonferroni alpha/m) has to
    # WIDEN the CI. For the default alpha=0.05 this is 1-0.05=0.95 — identical to
    # the previous hard-coded level, so existing callers are unchanged.
    boot = clustered_bootstrap_diff(success_a, success_b, clusters=cluster_by,
                                    n_boot=n_boot, ci=1.0 - alpha, seed=seed,
                                    paired=paired)
    effect = boot["effect"]
    ci = (boot["ci_low"], boot["ci_high"])
    ci_includes_zero = ci[0] <= 0 <= ci[1]

    e_value = None
    if paired:
        mc = mcnemar(success_a, success_b)
        e_value = evalue_bernoulli(mc["b"], mc["n_discordant"], p0=0.5)
        if correction == "p":
            reject = mc["p_value"] < alpha
            method = "mcnemar-exact-p (paired binary)"
        else:
            reject = e_value >= 1.0 / alpha            # anytime-valid
            method = "mcnemar + e-value (paired binary)"
        details = {"p_value": mc["p_value"], "b": mc["b"], "c": mc["c"],
                   "n_discordant": mc["n_discordant"]}
    else:
        reject = not ci_includes_zero                  # unpaired: CI excludes 0
        method = "clustered bootstrap (unpaired)"
        details = {}

    # underpowered: not significant AND the CI is too wide to rule out a min_effect-sized effect
    width = ci[1] - ci[0]
    underpowered = (not reject) and (min_effect > 0) and (width > min_effect)

    return StatResult(effect=effect, ci=ci, reject=reject, method=method, alpha=alpha,
                      e_value=e_value, underpowered=underpowered, details=details)


def compare_paired_rates(
    rate_a: Sequence[float],
    rate_b: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    cluster_by: Optional[Sequence] = None,
) -> StatResult:
    """Paired comparison of per-example PASS RATES (each in [0, 1]) — A vs B.

    The per-arm rates come from k samples per example (a stochastic model at
    T>0 is a coin per example, not a fixed answer), so the paired quantity is
    ``d_i = rate_b_i - rate_a_i`` in [-1, 1]. Effect = mean(d); CI by paired
    (clustered) bootstrap; the decision is the one-sided betting e-value
    :func:`~evalrx.stats.evalue.evalue_bounded_mean` for "B better than A"
    (``e_value``), with the mirror e-value for "A better than B" in
    ``details["e_value_regression"]``. ``reject`` is True when either clears
    ``1/alpha``; read the sign of ``effect`` for the direction.

    With one sample per arm this reduces to d in {-1, 0, +1} — the McNemar
    discordant pairs — but the e-value is the betting one, not the Bernoulli
    mixture; use :func:`compare` for the classic single-sample paired test.
    """
    a = [float(x) for x in rate_a]
    b = [float(x) for x in rate_b]
    if len(a) != len(b):
        raise ValueError("paired rates need equal-length vectors")
    from evalrx.stats.evalue import evalue_bounded_mean

    boot = clustered_bootstrap_diff(a, b, clusters=cluster_by, n_boot=n_boot,
                                    ci=1.0 - alpha, seed=seed, paired=True)
    diffs = [y - x for x, y in zip(a, b)]
    e_up = evalue_bounded_mean(diffs)
    e_down = evalue_bounded_mean([-d for d in diffs])
    threshold = 1.0 / alpha
    reject = e_up >= threshold or e_down >= threshold
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    return StatResult(
        effect=boot["effect"], ci=(boot["ci_low"], boot["ci_high"]), reject=reject,
        method="paired rates + betting e-value (bounded mean)", alpha=alpha,
        e_value=e_up, underpowered=False,
        details={"e_value_regression": e_down, "n": len(diffs),
                 "n_positive": n_pos, "n_negative": n_neg,
                 "mean_a": (sum(a) / len(a)) if a else 0.0,
                 "mean_b": (sum(b) / len(b)) if b else 0.0},
    )


def ab_test(success_a: Sequence, success_b: Sequence, **kwargs) -> StatResult:
    """Back-compat alias: pass per-example success vectors (bools/0-1)."""
    return compare(success_a, success_b, **kwargs)


@dataclass
class MultiCompareResult:
    """Omnibus (Friedman) + post-hoc (Nemenyi) verdict for >2 strategies."""

    reject_global: bool                 # did Friedman find ANY difference?
    avg_ranks: dict                     # name -> mean rank (lower = better)
    significant_pairs: list             # Nemenyi pairs that differ (only if reject_global)
    critical_difference: float
    friedman_stat: float
    p_value: float
    df: int
    n: int
    alpha: float

    def summary(self) -> str:
        order = sorted(self.avg_ranks, key=lambda k: self.avg_ranks[k])
        ranking = " < ".join(f"{k}({self.avg_ranks[k]:.2f})" for k in order)
        verdict = "differ" if self.reject_global else "no global difference"
        return (f"[Friedman χ²={self.friedman_stat:.3f} df={self.df} -> {verdict}] "
                f"ranks: {ranking} | CD={self.critical_difference:.3f} | "
                f"{len(self.significant_pairs)} sig pair(s)")


def compare_multiple(success_by_strategy: dict, *, alpha: float = 0.05) -> MultiCompareResult:
    """Compare 3+ strategies across shared examples (Friedman + Nemenyi).

    ``success_by_strategy``: ``{name: [per-example metric]}`` (equal-length, paired).
    For exactly 2 strategies use :func:`compare` (McNemar) instead.
    """
    if len(success_by_strategy) < 3:
        raise ValueError("compare_multiple is for 3+ strategies; use compare() for 2.")
    fr = friedman_test(success_by_strategy)
    avg = dict(zip(fr["names"], fr["avg_ranks"]))
    cd = nemenyi_cd(fr["k"], fr["n"], alpha)
    reject = fr["p_value"] < alpha
    pairs = nemenyi_pairs(avg, cd) if reject else []  # post-hoc only after a global rejection
    return MultiCompareResult(
        reject_global=reject, avg_ranks=avg, significant_pairs=pairs, critical_difference=cd,
        friedman_stat=fr["statistic"], p_value=fr["p_value"], df=fr["df"], n=fr["n"], alpha=alpha,
    )
