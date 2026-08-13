"""Render band-sweep results as a table, and say which ones can be trusted.

Separate from `band_locate.py` so a finished sweep can be re-read (and re-judged
against a different threshold) without spending the endpoint again.

    python report_bands.py band_results.json [more_results.json ...]
"""

from __future__ import annotations

import json
import sys

from band_locate import TRUNCATION_ALARM, band_of

_ORDER = {"USABLE": 0, "marginal": 1, "budget_limited": 2, "saturated": 3, "floor": 4}


def load(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path) as fh:
            payload = json.load(fh)
        for row in payload.get("results", []):
            if "error" in row:
                print(f"  ! {row['name']}: {row['error']}", file=sys.stderr)
                continue
            row["_source"] = path
            # re-derive so an old file is judged by the CURRENT rule
            row["band"] = band_of(
                row["accuracy"], *row["ci95"], row.get("no_answer_tag_rate", 0.0)
            )
            rows.append(row)
    return rows


def main() -> None:
    paths = sys.argv[1:] or ["band_results.json"]
    rows = load(paths)
    rows.sort(key=lambda r: (_ORDER.get(r["band"], 9), -r["accuracy"]))

    header = f"{'dataset':22s} {'chapter':11s} {'n':>3s} {'acc':>6s} {'95% CI':>15s} {'no-tag':>7s} {'band':14s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        lo, hi = r["ci95"]
        print(
            f"{r['name']:22s} {r.get('chapter', ''):11s} {r['n']:3d} "
            f"{r['accuracy']:6.3f} {f'[{lo:.2f}, {hi:.2f}]':>15s} "
            f"{r.get('no_answer_tag_rate', 0.0):7.2f} {r['band']:14s}"
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
            print(
                f"  ! {r['name']} @ {r['accuracy']:.0%} "
                f"(no-tag {r['no_answer_tag_rate']:.0%}, "
                f"mean {r['mean_output_chars']:.0f} chars)"
            )


if __name__ == "__main__":
    main()
