"""Check a persisted run against the wire contract.

    python examples/contract/validate_run.py <run-dir> [<run-dir> ...]

Reports per artifact: accepted, or the first violation. This is the intended
adoption path — point it at existing runs before wiring validation into the
producers, so the contract is calibrated against data that exists rather than
data that was imagined.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from evalvitals.contract.m1 import FindingsWire, ResultWire
from evalvitals.contract.m2 import StatsToolResultWire


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
        return {"__read_error__": str(exc)}


def check_m1(path: Path) -> tuple[str, str]:
    raw = _load(path)
    if "__read_error__" in raw:
        return "READ", raw["__read_error__"]
    # Adapt what the producer writes to what the contract names. Every mapping
    # here is a real drift between code and contract, not a convenience.
    adapted = {
        "analyzer": raw.get("analyzer", ""),
        "model": raw.get("model", ""),
        "n_cases": raw.get("n_cases", 0),
        "findings": raw.get("findings") or {},
        "metadata": raw.get("metadata") or {},
        # producers emit `artifact_names` (names); the contract wants resolvable refs
        "artifact_paths": {},
    }
    try:
        r = ResultWire.model_validate(adapted)
    except ValidationError as exc:
        e = exc.errors()[0]
        return "FAIL", f"{'.'.join(str(x) for x in e['loc'])}: {e['msg'][:110]}"
    n = len(r.findings.per_case)
    return "OK", f"{n} per-case rows, {len(r.signal_names_local())} signals" if hasattr(
        r, "signal_names_local") else f"{n} per-case rows"


def check_m2(path: Path) -> tuple[str, str]:
    raw = _load(path)
    if "__read_error__" in raw:
        return "READ", raw["__read_error__"]
    rows = raw if isinstance(raw, list) else raw.get("stats_results", [])
    ok = imputed = 0
    first_err = ""
    for row in rows:
        d = row.get("details") or {}
        adapted = {
            k: row.get(k) for k in
            ("tool", "config", "ok", "effect", "ci", "p_value", "e_value",
             "underpowered", "reject", "fdr_corrected", "correction_method",
             "correction_family", "analysis_key", "summary", "error",
             "raw_reject", "figure_path")
        }
        adapted["config"] = adapted.get("config") or {}
        adapted["details"] = d
        adapted["n_signal"] = d.get("n_signal")
        adapted["n_control"] = d.get("n_control")
        adapted["ci"] = tuple(row["ci"]) if row.get("ci") else None
        try:
            r = StatsToolResultWire.model_validate({k: v for k, v in adapted.items() if v is not None})
            ok += 1
            if r.n_signal is not None and r.n_control is not None and r.is_decisive():
                pass
        except ValidationError as exc:
            if not first_err:
                e = exc.errors()[0]
                first_err = f"{'.'.join(str(x) for x in e['loc'])}: {e['msg'][:90]}"
    return ("OK", f"{ok}/{len(rows)} tool results") if not first_err else ("FAIL", first_err)


def sweep(run: Path) -> None:
    print(f"\n{'='*78}\n{run}\n{'='*78}")
    arts = sorted(run.rglob("*.result.json"))
    stats = sorted(run.rglob("*_m2_stats_results.json"))
    if not arts and not stats:
        print("  no contract-checkable artifacts found")
        return
    for a in arts:
        state, msg = check_m1(a)
        mark = "ok  " if state == "OK" else "FAIL"
        print(f"  [{mark}] M1 {a.name:44} {msg}")
    for s in stats:
        state, msg = check_m2(s)
        mark = "ok  " if state == "OK" else "FAIL"
        print(f"  [{mark}] M2 {s.name:44} {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    for r in args.runs:
        sweep(r)


if __name__ == "__main__":
    main()
