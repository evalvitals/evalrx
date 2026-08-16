"""Render band-sweep results as a table, and say which ones can be trusted.

Separate from `band_locate.py` so a finished sweep can be re-read (and re-judged
against a different threshold) without spending the endpoint again.

    python report_bands.py band_results.json [more_results.json ...]
"""

from __future__ import annotations

import json
import sys

from band_locate import TRUNCATION_ALARM, band_of, bracket_in_band, budget_bracket

_ORDER = {"USABLE": 0, "marginal": 1, "budget_limited": 2, "saturated": 3, "floor": 4}


def load(paths: list[str]) -> list[dict]:
    """Merge sweeps, LAST file wins per dataset.

    Re-running a spec at a bigger budget supersedes the earlier measurement, so
    a plain concatenation would show both and let a stale row be read as current.
    """
    by_name: dict[str, dict] = {}
    for path in paths:
        with open(path) as fh:
            payload = json.load(fh)
        for row in payload.get("results", []):
            if "error" in row:
                print(f"  ! {row['name']}: {row['error']}", file=sys.stderr)
                continue
            row["_source"] = path
            # re-derive so an old file is judged by the CURRENT rule.
            # finish_reason is the real truncation signal; the tag rate is the
            # older proxy and is all an older results file carries.
            row["_budget_signal"] = max(
                row.get("truncated_rate") or 0.0,
                row.get("no_answer_tag_rate") or 0.0,
            )
            row["band"] = band_of(row["accuracy"], *row["ci95"], row["_budget_signal"])
            prior = by_name.get(row["name"])
            if prior is not None:
                row["_superseded"] = {
                    "accuracy": prior["accuracy"],
                    "band": prior["band"],
                    "no_answer_tag_rate": prior.get("no_answer_tag_rate"),
                }
            by_name[row["name"]] = row
    return list(by_name.values())


def main() -> None:
    paths = sys.argv[1:] or ["band_results.json"]
    rows = load(paths)
    rows.sort(key=lambda r: (_ORDER.get(r["band"], 9), -r["accuracy"]))

    header = f"{'dataset':22s} {'chapter':11s} {'n':>3s} {'acc':>6s} {'95% CI':>15s} {'budget':>7s} {'band':14s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        lo, hi = r["ci95"]
        was = r.get("_superseded")
        delta = (
            f"  (was {was['accuracy']:.2f}/{was['no_answer_tag_rate']:.2f} {was['band']})"
            if was else ""
        )
        flags = ""
        if r.get("undersampled"):
            flags += f" [n<{r.get('n_requested')}]"
        if r.get("error_rate"):
            flags += f" [err {r['error_rate']:.0%}]"
        print(
            f"{r['name']:22s} {r.get('chapter', ''):11s} {r['n']:3d} "
            f"{r['accuracy']:6.3f} {f'[{lo:.2f}, {hi:.2f}]':>15s} "
            f"{r['_budget_signal']:7.2f} {r['band']:14s}{flags}{delta}"
        )

    usable = [r for r in rows if r["band"] == "USABLE"]
    budget = [r for r in rows if r["band"] == "budget_limited"]
    print()
    print(f"USABLE (30-70%, low truncation): {len(usable)}/{len(rows)}")
    for r in usable:
        note = f" — {r['note']}" if r.get("note") else ""
        print(f"  * {r['name']} @ {r['accuracy']:.0%}{note}")
    if budget:
        print(
            f"\nBudget-limited ({len(budget)}): >{TRUNCATION_ALARM:.0%} of outputs never "
            "reached an answer tag, so the accuracy measures the token budget, "
            "not the model. Re-run these with a larger budget before reading a band:"
        )
        for r in budget:
            trunc = r.get("truncated_rate")
            how = (
                f"truncated {trunc:.0%}" if trunc is not None
                else f"no-tag {r['no_answer_tag_rate']:.0%}"
            )
            # the bracket sorts the queue: a re-run only answers the question for
            # rows whose ceiling cannot clear the band
            lo, hi = budget_bracket(r["accuracy"], r["_budget_signal"])
            worth = " <- RERUN SETTLES IT" if bracket_in_band(
                r["accuracy"], r["_budget_signal"]
            ) else ""
            print(
                f"  ! {r['name']} @ {r['accuracy']:.0%} "
                f"({how}, mean {r['mean_output_chars']:.0f} chars) "
                f"could land in [{lo:.2f}, {hi:.2f}]{worth}"
            )


if __name__ == "__main__":
    main()
