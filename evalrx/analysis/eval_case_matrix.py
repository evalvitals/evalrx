"""
eval_case_matrix.py
===================
Reconstruct a tidy *per-case* feature matrix from a frozen M1 state pickle
(``outputs/m1_state.pkl``). The explore step only persists aggregates (group
means, a single scatter pair); the raw per-case records needed for distribution
diagnostics, ROC, correlation heatmaps and a confusion matrix live inside the
analyzer Results. This module joins them into one DataFrame, keyed by case id.

No GPU / no re-inference: everything here is read from the pickle the M1 pass
already wrote. Columns that an analyzer did not cover are left as NaN (e.g. the
attention analyzer subsamples, so ``attn_*`` is populated for a subset only).

    from evalrx.analysis.eval_case_matrix import load_case_matrix
    df = load_case_matrix("outputs")        # -> tidy DataFrame or None

Output columns (present when the underlying analyzer ran):
    case_id            str
    label              "FAIL" / "PASS"
    is_fail            1 / 0
    model_yes          1 / 0   (the model's actual Yes/No answer)
    truth_yes          1 / 0   (ground-truth presence)
    attn_max           float   relative_attention.max_relative_weight
    attn_mean          float   relative_attention.mean_relative_weight
    focus_share        float   relative_attention.focus_share
    probe1_fd          0 / 1   generated:probe1.false_detection  (LEAKY — see §4)
    pc_fixed_describe  bool    prompt_contrast: describe-first repaired this case
    pc_broken_describe bool
    pc_fixed_sensitive bool
    pc_broken_sensitive bool
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

# Continuous signals safe to feed to ROC / coefPlot / distribution diagnostics.
# probe1_fd is deliberately excluded — it is the failure label re-measured (leaky).
CONTINUOUS_SIGNALS = ["attn_max", "attn_mean", "focus_share"]


def _yesno(v: Any) -> int | None:
    s = str(v).strip().lower()
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    return None


def build_case_matrix(state: dict[str, Any]) -> pd.DataFrame:
    """Join the analyzers' per-case records into one tidy frame keyed by case id."""
    cases = state.get("cases") or []
    rows: dict[str, dict[str, Any]] = {}
    for c in cases:
        lab = "FAIL" if str(getattr(c, "label", "")).upper().endswith("FAIL") else "PASS"
        rows[c.id] = {
            "case_id": c.id,
            "label": lab,
            "is_fail": 1 if lab == "FAIL" else 0,
            "model_yes": _yesno(getattr(c, "observed", "")),
            "truth_yes": _yesno(getattr(c, "expected", "")),
        }

    pr = state.get("probe_results") or {}

    ra = pr.get("relative_attention")
    if ra is not None:
        for r in (ra.findings.get("per_case") or []):
            cid = r.get("id") or r.get("sample_id")
            if cid in rows:
                rows[cid]["attn_max"] = r.get("max_relative_weight")
                rows[cid]["attn_mean"] = r.get("mean_relative_weight")
                rows[cid]["focus_share"] = r.get("focus_share")

    p1 = pr.get("generated:probe1")
    if p1 is not None:
        for r in (p1.findings.get("per_case") or []):
            cid = r.get("sample_id") or r.get("id")
            if cid in rows:
                rows[cid]["probe1_fd"] = r.get("false_detection")

    pc = pr.get("prompt_contrast")
    if pc is not None:
        per_case = (pc.artifacts or {}).get("per_case") or []
        for r in per_case:
            cid = r.get("sample_id") or r.get("id")
            if cid in rows:
                rows[cid]["pc_fixed_describe"] = r.get("fixed_by_describe_first")
                rows[cid]["pc_broken_describe"] = r.get("broken_by_describe_first")
                rows[cid]["pc_fixed_sensitive"] = r.get("fixed_by_sensitive")
                rows[cid]["pc_broken_sensitive"] = r.get("broken_by_sensitive")

    return pd.DataFrame(list(rows.values()))


def load_case_matrix(outputs_dir: str | Path) -> pd.DataFrame | None:
    """Load ``<outputs_dir>/m1_state.pkl`` and build the per-case matrix.
    Returns None if the pickle is absent or unreadable."""
    pkl = Path(outputs_dir) / "m1_state.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as fh:
            state = pickle.load(fh)
        df = build_case_matrix(state)
        return df if not df.empty else None
    except Exception:
        return None


def continuous_signals(df: pd.DataFrame) -> list[str]:
    """Continuous, non-leaky signal columns that are actually populated."""
    return [c for c in CONTINUOUS_SIGNALS if c in df.columns and df[c].notna().any()]
