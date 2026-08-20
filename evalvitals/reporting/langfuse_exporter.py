"""Langfuse Observability & Trace Exporter for EvalVitals runs.

This module maps EvalVitals diagnosis runs (M1 through M5 and fixes) to
Langfuse Traces, Spans, Generations, and Scores.

Key mapping:
- Trace: The overall diagnostic evaluation run.
- Spans: Individual pipeline stages (M1 Measure, M2 Screen, M3 Explain,
  M5 Adjudicate, M4 Surgery, M4 Fix).
- Generations: LLM judge/agent invocations (prompt, output, latency).
- Scores: Core quantitative vitals (baseline_accuracy, repair_effect,
  e_value, hypothesis_verdict, n_fixed, n_broken).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evalvitals.reporting.html_report import extract_run_data


def export_to_langfuse_bundle(run_dir: str | Path, out_json: str | Path | None = None) -> dict[str, Any]:
    """Convert an EvalVitals run into a Langfuse-compatible ingestion bundle."""
    data = extract_run_data(Path(run_dir))
    run = data["run"]
    m1 = data["m1"]
    m2 = data["m2"]
    m3 = data["m3"]
    m5 = data["m5"]
    m4_s = data["m4_surgery"]
    m4_f = data["m4_fix"]

    trace_id = f"evalvitals_{data['run']['data_fingerprint'][:12] or 'run'}"
    trace_name = f"EvalVitals: {run['model']}"

    # 1. Trace envelope
    trace = {
        "id": trace_id,
        "name": trace_name,
        "release": f"v{run['version']}",
        "tags": ["evalvitals", run["model"], run["stopped_by"]],
        "metadata": {
            "model": run["model"],
            "n_cases": run["n_cases"],
            "cycles": run["cycles"],
            "protocol": run["protocol"],
            "data_fingerprint": run["data_fingerprint"],
            "logs_dir": run["logs_dir"],
        },
    }

    # 2. Spans
    spans = []
    # M1 Span
    spans.append({
        "id": f"{trace_id}_m1",
        "name": "M1: Measurement",
        "type": "span",
        "metadata": {
            "stage": "M1",
            "analyzers": m1["analyzers"],
            "duration_sec": m1["duration"],
            "n_results": len(m1["results"]),
        },
        "input": {"analyzers": m1["analyzers"]},
        "output": {"summary": [f"{r['name']}: {len(r.get('findings', {}))} findings" for r in m1["results"]]},
    })

    # M2 Span
    spans.append({
        "id": f"{trace_id}_m2",
        "name": "M2: Screening & Analysis",
        "type": "span",
        "metadata": {
            "stage": "M2",
            "severity": m2["severity"],
            "duration_sec": m2["duration"],
            "n_tests": len(m2["stats"]),
            "n_rejected": sum(1 for s in m2["stats"] if s["reject"]),
        },
        "input": {"n_signals_screened": len(m2["stats"])},
        "output": {
            "conclusion": m2["conclusion"],
            "significant_signals": [
                s.get("config", {}).get("signal") or s.get("tool")
                for s in m2["stats"] if s["reject"]
            ],
        },
    })

    # M3 Span
    spans.append({
        "id": f"{trace_id}_m3",
        "name": "M3: Hypothesis Generation",
        "type": "span",
        "metadata": {
            "stage": "M3",
            "duration_sec": m3["duration"],
            "n_hypotheses": len(m3["hypotheses"]),
        },
        "input": {"conclusion": m2["conclusion"]},
        "output": {"hypotheses": m3["hypotheses"]},
    })

    # M5 Span
    if m5["ran"]:
        spans.append({
            "id": f"{trace_id}_m5",
            "name": "M5: Hypothesis Validation",
            "type": "span",
            "metadata": {
                "stage": "M5",
                "results_count": len(m5["results"]),
            },
            "input": {"hypotheses_tested": [h.get("statement") for h in m3["hypotheses"]]},
            "output": {"event": m5["event"], "results": m5["results"]},
        })

    # M4 Surgery Span
    if m4_s["ran"]:
        spans.append({
            "id": f"{trace_id}_m4_surgery",
            "name": "M4: Causal Surgery",
            "type": "span",
            "metadata": {"stage": "M4-Surgery"},
            "output": {"surgeries": m4_s["surgeries"]},
        })

    # M4 Fix Span
    if m4_f["ran"]:
        spans.append({
            "id": f"{trace_id}_m4_fix",
            "name": "M4: Repair & Confirmation",
            "type": "span",
            "metadata": {
                "stage": "M4-Fix",
                "fixed": m4_f["fixed"],
                "n_candidates_screened": len(m4_f["selection"]),
            },
            "input": {"candidates": [s.get("name") for s in m4_f["selection"]]},
            "output": {
                "best_candidate": m4_f.get("best", {}).get("name") or (m4_f.get("confirm") or {}).get("name"),
                "confirm": m4_f.get("confirm"),
            },
        })

    # 3. Scores / Metrics
    cfm = m4_f.get("confirm") or {}
    scores = []
    if cfm.get("n_baseline_correct") is not None and cfm.get("n_pairs"):
        scores.append({
            "name": "baseline_accuracy",
            "value": cfm["n_baseline_correct"] / cfm["n_pairs"],
            "comment": f"{cfm['n_baseline_correct']}/{cfm['n_pairs']}",
        })
    if cfm.get("effect") is not None:
        scores.append({
            "name": "repair_effect",
            "value": float(cfm["effect"]),
            "comment": f"Net accuracy shift: {float(cfm['effect']) * 100:+.2f}%",
        })
    if cfm.get("e_value") is not None:
        scores.append({
            "name": "e_value",
            "value": float(cfm["e_value"]),
            "comment": "Evidence strength against H0",
        })
    if cfm.get("n_fixed") is not None:
        scores.append({
            "name": "n_fixed_cases",
            "value": int(cfm["n_fixed"]),
        })
    if cfm.get("n_broken") is not None:
        scores.append({
            "name": "n_broken_cases",
            "value": int(cfm["n_broken"]),
        })

    bundle = {
        "trace": trace,
        "spans": spans,
        "scores": scores,
    }

    if out_json:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[✓] Exported Langfuse bundle to: {out_p}")

    return bundle


def sync_to_langfuse_live(run_dir: str | Path) -> bool:
    """If langfuse SDK is installed and credentials exist, push live to Langfuse."""
    try:
        from langfuse import Langfuse
    except ImportError:
        print("[!] `langfuse` package is not installed. Install with `pip install langfuse` to enable live sync.")
        return False

    bundle = export_to_langfuse_bundle(run_dir)
    trace_meta = bundle["trace"]

    try:
        langfuse = Langfuse()
        trace = langfuse.trace(
            id=trace_meta["id"],
            name=trace_meta["name"],
            release=trace_meta.get("release"),
            tags=trace_meta.get("tags"),
            metadata=trace_meta.get("metadata"),
        )

        for s in bundle["spans"]:
            span = trace.span(
                id=s["id"],
                name=s["name"],
                metadata=s.get("metadata"),
                input=s.get("input"),
                output=s.get("output"),
            )
            span.end()

        for sc in bundle["scores"]:
            trace.score(
                name=sc["name"],
                value=sc["value"],
                comment=sc.get("comment"),
            )

        langfuse.flush()
        print(f"[✓] Successfully synced run to Langfuse: Trace ID '{trace_meta['id']}'")
        return True
    except Exception as exc:
        print(f"[!] Langfuse sync failed: {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Export an EvalVitals run to Langfuse trace format or sync live.")
    ap.add_argument("run_dir", nargs="?", default="outputs", help="Run directory holding run_log.jsonl or logs/")
    ap.add_argument("--out", "-o", default=None, help="Output JSON bundle path")
    ap.add_argument("--sync", action="store_true", help="Sync live to Langfuse Cloud/Server using environment API keys")
    args = ap.parse_args()

    if args.sync:
        sync_to_langfuse_live(args.run_dir)
    else:
        out_p = args.out or (Path(args.run_dir) / "langfuse_trace.json")
        export_to_langfuse_bundle(args.run_dir, out_p)


if __name__ == "__main__":
    main()
