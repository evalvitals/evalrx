#!/usr/bin/env python3
"""Re-apply the CURRENT grader to a frozen ``cases.json``.

A batch on disk is two different things welded together: the model's raw
generations, which cost hours of GPU and never go stale, and the PASS/FAIL
labels, which are a function of grading code that does change.  ``SKIP_STAGE0=1``
reuses the file wholesale, so a grader fix silently does nothing for every batch
already written — the loop keeps mining labels produced by the old bug.

That is not hypothetical.  Fixing ``extract_answer`` on 2026-08-16 (a bare
``(A)`` discarded as a format placeholder; ``\\boxed{}`` outranking a LATER
``Answer:`` line) moved the two frozen 9B batches:

    bbh_tracking7  n=250  0.592 -> 0.988   FAIL->PASS 99   PASS->FAIL 0
    minervamath    n=272  0.360 -> 0.463   FAIL->PASS 28   PASS->FAIL 0

Dry-run by default — it prints the delta and touches nothing.  ``--write``
rewrites labels, the accuracy block, and a ``grader_fingerprint`` so a later run
can tell a regraded batch from a stale one.

    python regrade.py                          # every batch under outputs/
    python regrade.py --model qwen3.5-9b --dataset minervamath --write

Regrading NEVER re-generates: the outputs are untouched, so this is free and
repeatable.  It cannot rescue a truncated case (a generation cut off at the
token cap has no final answer to extract) — those stay FAIL and are counted
separately, because that FAIL is a budget artefact either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(HERE.parent / "llm_band_probe"))

import band_locate as B  # noqa: E402


def grader_fingerprint() -> str:
    """Short digest of the grading code, so staleness is detectable, not guessed.

    Hashes the extraction module's source rather than a hand-bumped version
    string: a version constant only records the changes someone remembered to
    record, and the whole failure mode here is a change nobody propagated.
    """
    import hashlib

    from evalrx.analyzers.reasoning import _text

    src = Path(_text.__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


def regrade(report: dict) -> dict:
    """Return a delta summary; does not mutate *report*."""
    spec = next(s for s in B.SPECS if s.name == report["dataset"])
    grade = spec.grader or B.answer_equal

    up, down, up_truncated = [], [], 0
    new_labels = []
    for index, case in enumerate(report["cases"]):
        graded = (case["output"] if spec.grades_raw_output
                  else B.extract_answer(case["output"]))
        ok = bool(grade(graded, case["gold"]))
        new_labels.append("PASS" if ok else "FAIL")
        if ok and case["label"] == "FAIL":
            up.append(index)
            up_truncated += bool(case.get("truncated"))
        elif not ok and case["label"] == "PASS":
            down.append(index)

    n = len(new_labels)
    n_pass = sum(1 for label in new_labels if label == "PASS")
    return {
        "n": n,
        "old_accuracy": report["accuracy"],
        "new_accuracy": n_pass / n if n else 0.0,
        "n_pass": n_pass,
        "n_fail": n - n_pass,
        "fail_to_pass": len(up),
        "pass_to_fail": len(down),
        "fail_to_pass_truncated": up_truncated,
        "labels": new_labels,
    }


def apply(report: dict, delta: dict) -> dict:
    for case, label in zip(report["cases"], delta["labels"]):
        case["label"] = label
    report["accuracy"] = delta["new_accuracy"]
    report["n_pass"] = delta["n_pass"]
    report["n_fail"] = delta["n_fail"]
    report["grader_fingerprint"] = grader_fingerprint()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--write", action="store_true",
                    help="rewrite labels in place (default is a dry run)")
    args = ap.parse_args()

    pattern = f"{args.model or '*'}/{args.dataset or '*'}/cases.json"
    paths = sorted((HERE / "outputs").glob(pattern))
    if not paths:
        raise SystemExit(f"no batches match outputs/{pattern}")

    current = grader_fingerprint()
    print(f"current grader fingerprint: {current}\n")
    changed = 0
    for path in paths:
        report = json.loads(path.read_text())
        delta = regrade(report)
        stale = report.get("grader_fingerprint") != current
        moved = delta["fail_to_pass"] or delta["pass_to_fail"]
        changed += bool(moved)
        rel = path.relative_to(HERE / "outputs")
        print(f"{str(rel.parent):34s} n={delta['n']:4d}  "
              f"{delta['old_accuracy']:.3f} -> {delta['new_accuracy']:.3f}   "
              f"FAIL->PASS {delta['fail_to_pass']:3d}   "
              f"PASS->FAIL {delta['pass_to_fail']:3d}   "
              f"{'STALE' if stale else 'current'}")
        if delta["fail_to_pass_truncated"]:
            # a recovered-but-truncated case is a coincidence, not a rescue
            print(f"  NOTE {delta['fail_to_pass_truncated']} of the recovered "
                  f"cases were TRUNCATED — their answer was cut off, so treat "
                  f"the flip as budget noise rather than a grading fix")
        if args.write:
            path.write_text(json.dumps(apply(report, delta), indent=2))
            print(f"  wrote {path}")

    if not args.write and changed:
        print(f"\n{changed} batch(es) would change. Re-run with --write to apply.")
    elif not changed:
        print("\nall batches already agree with the current grader.")


if __name__ == "__main__":
    main()
