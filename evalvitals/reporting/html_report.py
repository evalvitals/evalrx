"""Self-contained, tabbed, human-understandable HTML report generator for EvalVitals.

This module parses a finished diagnostic run (M1 through M5 and fixes) and
renders a single, zero-dependency, tabbed interactive HTML diagnostic report in English.

Design Philosophy:
- Tabbed Navigation: Clean stage-by-stage tabs (Overview, M1, M2, M3, M5, M4, Cases)
  allowing focused inspection without endless scrolling.
- Human-Friendly Header: Clean model and benchmark titles instead of raw code/debug dumps.
- Plain-English First: Clinical medical-checkup analogy with intuitive explanations.
- 100% Dynamic: Fully parsed from experiment artifacts without hardcoding.
- Multi-Modal: Inlines audio (AAC), images, and structured prompts.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Plain-English Glossary & Metadata
# ---------------------------------------------------------------------------

STAGE_METADATA: dict[str, dict[str, str]] = {
    "overview": {
        "code": "SUMMARY",
        "name": "Executive Summary",
        "short_role": "Overview & Key Findings",
        "plain_desc": "High-level diagnostic summary: what was tested, what failure mechanism was diagnosed, and the final repair outcome.",
    },
    "pre_m1": {
        "code": "PRE-M1",
        "name": "Case Synthesis",
        "short_role": "Adversarial Probe Synthesis",
        "plain_desc": "Evaluates whether the system automatically synthesised challenging edge cases or adversarial probes to uncover hidden blind spots. If skipped, testing ran on fixed benchmark data.",
    },
    "m1": {
        "code": "M1",
        "name": "Checkup & Signals",
        "short_role": "Multi-Dimensional Measurement",
        "plain_desc": "Performs a multi-dimensional clinical checkup on the model, measuring vital signs (such as attention focus, overconfidence, order sensitivity, etc.) without pre-judging guilt.",
    },
    "m2": {
        "code": "M2",
        "name": "Screening & EDA",
        "short_role": "Statistical Association Screening",
        "plain_desc": "Correlates vital signs with actual failure cases. Filters out noise and random coincidence using rigorous multiple-testing corrections (Benjamini–Hochberg FDR) to isolate prime suspects.",
    },
    "m3": {
        "code": "M3",
        "name": "Root-Cause Diagnosis",
        "short_role": "Falsifiable Mechanism Hypothesis",
        "plain_desc": "An AI Diagnostician analyzes the screened evidence to propose specific, falsifiable root-cause hypotheses explaining exactly why and when the model fails.",
    },
    "m5": {
        "code": "M5",
        "name": "Independent Adjudication",
        "short_role": "Blind-Holdout Validation",
        "plain_desc": "Blindly re-tests the AI Doctor's hypothesis on a held-out validation set the model has never seen before, determining whether the mechanism is confirmed or refuted.",
    },
    "m4_surgery": {
        "code": "M4-SURGERY",
        "name": "Causal Surgery",
        "short_role": "Internal Mechanism Intervention",
        "plain_desc": "Directly intervenes in or ablates internal model components (such as attention heads or activation layers) to prove causal necessity.",
    },
    "m4_fix": {
        "code": "M4-FIX",
        "name": "Targeted Repair",
        "short_role": "Repair Search & Paired Confirmation",
        "plain_desc": "Applies targeted treatments across intervention tiers (from prompt patches to decoding scaffolds). Validates on held-out test data by comparing cured cases against broken cases.",
    },
    "case_book": {
        "code": "CASE-STUDIO",
        "name": "Interactive Case Studio",
        "short_role": "Case-Level Inspection & Audio",
        "plain_desc": "Inspect real failure vs repaired cases with before-and-after model answering comparisons and playable audio clips.",
    },
}

ANALYZER_GLOSSARY: dict[str, tuple[str, str, list[tuple[str, str, str]]]] = {
    "answer_extraction_audit": (
        "Answer Extraction Audit",
        "Did the model actually answer wrong, or did our parser fail to extract its response?",
        [
            ("Extraction Failure Rate on Errors", "suspect_rate", "pct"),
            ("Missing Format Tag Rate", "missing_tag_rate", "pct"),
        ],
    ),
    "termination_audit": (
        "Truncation & Early Stop Audit",
        "Did the model finish its thought, or was it abruptly cut off by length caps?",
        [
            ("Suspected Truncation Rate", "truncation_rate", "pct"),
            ("Recovery Rate upon Continuation", "recovered_rate", "pct"),
        ],
    ),
    "calibration": (
        "Confidence & Overconfidence (ECE)",
        "When the model sounds certain, is its actual correctness probability truly high?",
        [
            ("Token Logprob Calibration Error (ECE)", "logprob_channel.ece", "num"),
            ("Verbalized Confidence Error (ECE)", "verbalized_channel.ece", "num"),
        ],
    ),
    "format_sensitivity": (
        "Option Order Sensitivity (Position Bias)",
        "If we shuffle choices (A/B/C/D), does the model arbitrarily flip its answer?",
        [
            ("Answer Flip Rate after Shuffling", "mean_flip_rate", "pct"),
        ],
    ),
    "coverage_verification_gap": (
        "Pass@k Knowledge Blindspot (Coverage)",
        "Over 5 repeat attempts, does the model ever produce the right answer even once?",
        [
            ("Persistent Failure Rate (0/5 correct)", "no_coverage_rate", "pct"),
            ("Pass@5 Coverage Rate", "mean_pass_at_k", "pct"),
        ],
    ),
    "self_consistency": (
        "Sampling Consistency",
        "When sampled repeatedly with temperature, does the answer waver wildly?",
        [
            ("Majority Answer Agreement Rate", "consistency", "pct"),
        ],
    ),
    "logprob_entropy": (
        "Predictive Uncertainty (Output Entropy)",
        "Is the model confident or internally hesitating when generating key tokens?",
        [
            ("Top-Token Prediction Entropy", "mean_top_entropy", "num"),
            ("Model Perplexity", "perplexity", "num"),
        ],
    ),
    "selfcheck_consistency": (
        "Self-Contradiction Detection",
        "Does the model contradict its own claims across repeat samplings?",
        [
            ("Self-Contradiction Index", "mean_inconsistency", "num"),
        ],
    ),
    "hallucination": (
        "Hallucination Probe",
        "Does the model generate details unsupported by the source stimulus?",
        [
            ("Hallucination Deviation Score", "hallucination_score", "num"),
        ],
    ),
    "qwen_attention": (
        "Attention Focus & Sparsity",
        "Does model attention properly focus on critical prompt and media cues?",
        [
            ("Core Cue Attention Share", "focus_share", "pct"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Data Resolution & Extraction
# ---------------------------------------------------------------------------

def clean_model_display_name(raw: str) -> str:
    """Format raw python class repr like HFLocalModel(key='...') into a human-readable title."""
    if not raw:
        return "Target Model"
    m = re.search(r"key=[\'\"]([^\'\"]+)[\'\"]", raw)
    if m:
        key = m.group(1)
        parts = [p.capitalize() if not p.isdigit() else p for p in key.split("-")]
        return "-".join(parts)
    clean = re.sub(r"^HFLocalModel\((.*?)\)$", r"\1", raw).strip()
    return clean or raw


def clean_benchmark_name(protocol: str, manifest_path: str) -> str:
    """Extract a friendly benchmark name from protocol or manifest path."""
    text = (protocol + " " + manifest_path).lower()
    if "mmau" in text:
        return "MMAU (Multi-Modal Audio Understanding)"
    if "chartqa" in text:
        return "ChartQA (Chart Visual Reasoning)"
    if "spatial" in text:
        return "Spatial457 (Spatial Visual Reasoning)"
    if "audiocaps" in text:
        return "AudioCaps Hallucination"
    if "pope" in text:
        return "POPE (Object Probing Evaluation)"
    if "chair" in text:
        return "CHAIR (Captioning Hallucination)"
    return "Evaluation Benchmark"


def _dig(d: dict, path: str) -> Any:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def resolve_run_dirs(run_dir: Path) -> tuple[Path, Path | None, Path | None]:
    """Return (logs_dir, explore_dir, fixes_dir)."""
    run_dir = run_dir.resolve()
    logs_dir = run_dir
    if (run_dir / "logs" / "run_log.jsonl").exists():
        logs_dir = run_dir / "logs"
    elif not (run_dir / "run_log.jsonl").exists():
        for child in run_dir.glob("*/run_log.jsonl"):
            logs_dir = child.parent
            break

    explore_dir = logs_dir / "explore"
    if not explore_dir.is_dir() and (logs_dir.parent / "explore").is_dir():
        explore_dir = logs_dir.parent / "explore"
    if not explore_dir.is_dir():
        explore_dir = None

    fixes_dir = logs_dir / "fixes"
    if not fixes_dir.is_dir() and (logs_dir.parent / "fixes").is_dir():
        fixes_dir = logs_dir.parent / "fixes"
    if not fixes_dir.is_dir():
        fixes_dir = None

    return logs_dir, explore_dir, fixes_dir


def find_manifest(example_dir: Path, logs_dir: Path) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    """Find and load benchmark manifest (.jsonl / .json)."""
    candidates = []
    if example_dir:
        candidates.extend(sorted(example_dir.glob("data/*.jsonl")))
        candidates.extend(sorted(example_dir.glob("data/*.json")))
        candidates.extend(sorted(example_dir.glob("*.jsonl")))
    candidates.extend(sorted(logs_dir.glob("data/*.jsonl")))
    candidates.extend(sorted(logs_dir.parent.glob("data/*.jsonl")))

    for p in candidates:
        if p.name.endswith(".result.json") or p.name.endswith("results.json"):
            continue
        try:
            rows = {}
            if p.suffix == ".jsonl":
                for line in p.open(encoding="utf-8"):
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        cid = str(obj.get("id") or obj.get("case_id") or "")
                        if cid:
                            rows[cid] = obj
            elif p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                recs = data if isinstance(data, list) else (data.get("records") or data.get("cases") or [])
                for obj in recs:
                    if isinstance(obj, dict):
                        cid = str(obj.get("id") or obj.get("case_id") or "")
                        if cid:
                            rows[cid] = obj
            if rows:
                return p, rows
        except Exception:
            continue
    return None, {}


def extract_run_data(run_dir: Path, example_dir: Path | None = None) -> dict[str, Any]:
    """Parse all run artifacts dynamically into a unified dictionary."""
    logs_dir, explore_dir, fixes_dir = resolve_run_dirs(run_dir)
    manifest_path, manifest = find_manifest(example_dir or logs_dir.parent, logs_dir)

    log_path = logs_dir / "run_log.jsonl"
    if not log_path.exists() and (logs_dir.parent / "run_log.jsonl").exists():
        log_path = logs_dir.parent / "run_log.jsonl"

    events: list[dict[str, Any]] = []
    if log_path.exists():
        for line in log_path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    by_event = lambda ev: [e for e in events if e.get("event") == ev]

    run_start = by_event("run_start")[-1] if by_event("run_start") else {}
    loop_end = by_event("loop_end")[-1] if by_event("loop_end") else {}
    manifest_json = logs_dir / "manifest.json"
    cfg = json.loads(manifest_json.read_text()) if manifest_json.exists() else {}

    raw_model_name = (
        run_start.get("model")
        or cfg.get("model")
        or cfg.get("config", {}).get("model")
        or "Target Model"
    )
    clean_model = clean_model_display_name(raw_model_name)
    n_cases = run_start.get("n_cases") or len(manifest) or 0
    label_dist = run_start.get("label_distribution") or {}
    protocol_desc = (run_start.get("protocol") or {}).get("description") or ""
    benchmark_name = clean_benchmark_name(protocol_desc, str(manifest_path or ""))

    # Pre-M1
    probe_searches = by_event("probe_search")
    pre_m1_ran = bool(probe_searches)
    pre_m1_cases = probe_searches[0].get("n_synthesized") if pre_m1_ran else 0

    # M1: Measurements
    probes = by_event("probe")
    p0 = probes[0] if probes else {}
    analyzers = p0.get("analyzers") or p0.get("selected_analyzers") or []
    m1_duration = p0.get("duration_sec")

    m1_results = []
    for name in analyzers:
        p = logs_dir / "artifacts" / f"c0_{name}.result.json"
        findings, n = {}, None
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                findings = raw.get("findings") or {}
                n = findings.get("n_cases") or findings.get("n_scored")
            except Exception:
                pass
        meta = ANALYZER_GLOSSARY.get(
            name, (name.replace("_", " ").title(), "Measures model behavior across this dimension", [])
        )
        headline = []
        for label, path, fmt in meta[2]:
            v = _dig(findings, path)
            if isinstance(v, (int, float)):
                headline.append({
                    "label": label,
                    "value": f"{v:.1%}" if fmt == "pct" else f"{v:.3f}".rstrip("0").rstrip("."),
                })
            else:
                headline.append({"label": label, "value": None})
        m1_results.append({
            "name": name,
            "display_name": meta[0],
            "question": meta[1],
            "n": n,
            "headline": headline,
            "findings": findings,
        })
    m1_results.sort(key=lambda r: -(r["n"] or 0))

    # M2: Explore & Screening
    analyses = by_event("analysis")
    a0 = analyses[0] if analyses else {}
    m2_conclusion = a0.get("conclusion") or ""
    m2_narrative = a0.get("narrative") or ""
    m2_severity = a0.get("severity") or "medium"
    m2_duration = a0.get("duration_sec")

    stats_path = logs_dir / "artifacts" / "c0_m2_stats_results.json"
    raw_stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
    stats = []
    for s in raw_stats:
        stats.append({
            "tool": s.get("tool"),
            "config": s.get("config") or {},
            "summary": s.get("summary") or "",
            "effect": s.get("effect"),
            "ci": s.get("ci") or [None, None],
            "reject": bool(s.get("reject")),
            "p_value": s.get("p_value"),
            "underpowered": bool(s.get("underpowered")),
        })
    stats.sort(key=lambda s: (not s["reject"], -abs(s.get("effect") or 0)))

    explore_data: dict[str, Any] = {}
    if explore_dir and (explore_dir / "exploratory_report.json").exists():
        try:
            explore_data = json.loads((explore_dir / "exploratory_report.json").read_text())
        except Exception:
            pass

    # M3: Hypotheses
    diagnoses = by_event("diagnosis")
    dg = diagnoses[0] if diagnoses else {}
    hypotheses = dg.get("hypotheses") or []
    if not hypotheses and (logs_dir / "report" / "hypotheses.json").exists():
        try:
            hypotheses = json.loads((logs_dir / "report" / "hypotheses.json").read_text())
        except Exception:
            pass

    # M5: Adjudication
    surgeries = by_event("surgery")
    m5_surgeries = [s for s in surgeries if s.get("module") == "m5" or s.get("adjudication")]
    m5_results_file = logs_dir / "report" / "m5_results.json"
    m5_results = json.loads(m5_results_file.read_text()) if m5_results_file.exists() else []
    m5_event = m5_surgeries[0] if m5_surgeries else (surgeries[0] if surgeries else {})

    # M4-Surgery
    m4_surgeries = [s for s in surgeries if s.get("module") != "m5"]
    m4_surgery_ran = bool(m4_surgeries)

    # M4-Fix: Repair
    fixes = by_event("fix")
    f0 = fixes[-1] if fixes else {}
    best_fix = f0.get("best") or {}
    fix_attempts = f0.get("selection_attempted") or f0.get("attempted") or []

    confirmed_fix: dict[str, Any] = {}
    fix_dir = None
    if fixes_dir:
        cand_dirs = sorted(fixes_dir.glob("*/result.json"))
        if cand_dirs:
            picked = cand_dirs[-1]
            for cd in cand_dirs:
                try:
                    res_j = json.loads(cd.read_text())
                    if res_j.get("fixed"):
                        picked = cd
                        break
                except Exception:
                    pass
            fix_dir = picked.parent
            try:
                confirmed_fix = json.loads(picked.read_text())
            except Exception:
                pass

    if not confirmed_fix and best_fix:
        confirmed_fix = best_fix

    cases: list[dict[str, Any]] = []
    output_files = []
    if fix_dir and (fix_dir / "outputs.jsonl").exists():
        output_files.append(fix_dir / "outputs.jsonl")
    output_files.extend(sorted(logs_dir.glob("**/outputs.jsonl")))
    if logs_dir.parent != logs_dir:
        output_files.extend(sorted(logs_dir.parent.glob("**/outputs.jsonl")))

    seen_case_ids = set()
    if output_files:
        target_out = output_files[0]
        for line in target_out.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    cid = str(r.get("case_id") or r.get("id") or "")
                    if not cid:
                        continue
                    seen_case_ids.add(cid)
                    m = manifest.get(cid) or {}
                    instruction = (
                        m.get("instruction")
                        or m.get("question")
                        or m.get("prompt")
                        or m.get("query")
                        or ""
                    )
                    choices = m.get("choices") or m.get("options") or m.get("candidates") or []
                    expected = m.get("expected") or m.get("answer") or m.get("label") or m.get("gt") or ""
                    audio_p = (
                        m.get("audio_path")
                        or m.get("audio")
                        or m.get("wav_path")
                        or m.get("audio_file")
                        or ""
                    )
                    image_p = (
                        m.get("image")
                        or m.get("image_path")
                        or m.get("image_file")
                        or ""
                    )
                    if isinstance(image_p, (list, tuple)):
                        image_p = image_p[0] if image_p else ""

                    cases.append({
                        "id": cid,
                        "status": r.get("status", "unchanged"),
                        "output": str(r.get("output", "")),
                        "instruction": instruction,
                        "choices": choices,
                        "expected": str(expected),
                        "audio_path": audio_p,
                        "image_path": image_p,
                        "duration": float(m.get("duration_sec") or 0.0),
                        "task": (m.get("metadata") or {}).get("mmau_task") or (m.get("metadata") or {}).get("category") or "",
                    })
                except Exception:
                    pass

    # Fallback to manifest + baseline
    if not cases and manifest:
        fixed_set = set(confirmed_fix.get("fixed_cases") or [])
        broken_set = set(confirmed_fix.get("broken_cases") or [])
        baseline_map = {}
        for bp in [logs_dir / "artifacts" / "baseline.json", *logs_dir.glob("**/baseline.json")]:
            if bp.exists():
                try:
                    bdata = json.loads(bp.read_text())
                    for bc in (bdata.get("cases") or []):
                        if isinstance(bc, dict) and bc.get("id"):
                            baseline_map[str(bc["id"])] = bc
                except Exception:
                    pass

        for cid, m in manifest.items():
            st = "unchanged"
            if cid in fixed_set:
                st = "fixed"
            elif cid in broken_set:
                st = "broken"

            b_out = baseline_map.get(cid, {}).get("output", "")
            instruction = (
                m.get("instruction")
                or m.get("question")
                or m.get("prompt")
                or m.get("query")
                or ""
            )
            choices = m.get("choices") or m.get("options") or m.get("candidates") or []
            expected = m.get("expected") or m.get("answer") or m.get("label") or m.get("gt") or ""
            audio_p = (
                m.get("audio_path")
                or m.get("audio")
                or m.get("wav_path")
                or m.get("audio_file")
                or ""
            )
            image_p = (
                m.get("image")
                or m.get("image_path")
                or m.get("image_file")
                or ""
            )
            if isinstance(image_p, (list, tuple)):
                image_p = image_p[0] if image_p else ""

            cases.append({
                "id": cid,
                "status": st,
                "output": str(b_out),
                "instruction": instruction,
                "choices": choices,
                "expected": str(expected),
                "audio_path": audio_p,
                "image_path": image_p,
                "duration": float(m.get("duration_sec") or 0.0),
                "task": (m.get("metadata") or {}).get("mmau_task") or (m.get("metadata") or {}).get("category") or "",
            })

    order = {"fixed": 0, "broken": 1, "unchanged": 2, "untested": 3}
    cases.sort(key=lambda c: (order.get(c["status"], 9), c["id"]))

    return {
        "run": {
            "model": clean_model,
            "raw_model": raw_model_name,
            "benchmark_name": benchmark_name,
            "n_cases": n_cases,
            "label_distribution": label_dist,
            "protocol": protocol_desc,
            "cycles": loop_end.get("cycles", 1),
            "stopped_by": loop_end.get("stopped_by", "completed"),
            "duration_sec": loop_end.get("total_duration_sec", 0),
            "version": run_start.get("evalvitals_version", "0.2.0"),
            "data_fingerprint": run_start.get("data_fingerprint", ""),
            "logs_dir": str(logs_dir),
            "manifest_path": str(manifest_path) if manifest_path else "",
        },
        "pre_m1": {
            "ran": pre_m1_ran,
            "n_cases": pre_m1_cases,
        },
        "m1": {
            "analyzers": analyzers,
            "duration": m1_duration,
            "results": m1_results,
        },
        "m2": {
            "conclusion": m2_conclusion,
            "narrative": m2_narrative,
            "severity": m2_severity,
            "duration": m2_duration,
            "stats": stats,
            "explore": explore_data,
        },
        "m3": {
            "hypotheses": hypotheses,
            "duration": dg.get("duration_sec"),
        },
        "m5": {
            "ran": bool(m5_surgeries or m5_results),
            "results": m5_results,
            "event": m5_event,
        },
        "m4_surgery": {
            "ran": m4_surgery_ran,
            "surgeries": m4_surgeries,
        },
        "m4_fix": {
            "ran": bool(fixes or confirmed_fix),
            "fixed": bool(f0.get("fixed") or confirmed_fix.get("fixed")),
            "selection": fix_attempts,
            "confirm": confirmed_fix,
            "best": best_fix,
            "prompt_template": (confirmed_fix.get("payload") or {}).get("prompt_template") or "",
        },
        "cases": cases,
        "logs_dir": logs_dir,
        "explore_dir": explore_dir,
        "example_dir": example_dir or logs_dir.parent,
    }


# ---------------------------------------------------------------------------
# Media & Figure Embedding
# ---------------------------------------------------------------------------

def embed_figures(explore_dir: Path | None, logs_dir: Path) -> dict[str, str]:
    """Find all PNG charts and convert to base64 Data URIs."""
    out = {}
    paths = []
    if explore_dir and (explore_dir / "figures").is_dir():
        paths += sorted((explore_dir / "figures").glob("*.png"))
    if (logs_dir / "figures").is_dir():
        paths += sorted((logs_dir / "figures").glob("*.png"))

    for p in paths:
        key = p.stem
        clean_key = re.sub(r"^\d+_", "", key)
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            out[clean_key] = f"data:image/png;base64,{b64}"
            out[key] = f"data:image/png;base64,{b64}"
        except Exception:
            pass
    return out


def embed_media(cases: list[dict], example_dir: Path, cache_dir: Path, no_audio: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    """Embed audio and image files as base64 data URIs concurrently."""
    audio_map: dict[str, str] = {}
    image_map: dict[str, str] = {}

    has_ffmpeg = bool(shutil.which("ffmpeg"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_dir = Path("/tmp/evalvitals_media_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

    def process_audio(c: dict) -> tuple[str, str] | None:
        cid = c["id"]
        if not no_audio and c.get("audio_path"):
            src = example_dir / "data" / c["audio_path"]
            if not src.exists() and (example_dir / c["audio_path"]).exists():
                src = example_dir / c["audio_path"]

            if src.exists():
                dst = cache_dir / f"{cid}.m4a"
                if not dst.exists() and has_ffmpeg:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                             "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", str(dst)],
                            capture_output=True,
                            timeout=10,
                        )
                    except Exception:
                        pass
                target = dst if dst.exists() else src
                try:
                    mime = "audio/mp4" if target.suffix in (".m4a", ".aac") else "audio/wav"
                    b64 = base64.b64encode(target.read_bytes()).decode("ascii")
                    return cid, f"data:{mime};base64,{b64}"
                except Exception:
                    pass
        return None

    def process_image(c: dict) -> tuple[str, str] | None:
        cid = c["id"]
        if c.get("image_path"):
            src = example_dir / "data" / c["image_path"]
            if not src.exists() and (example_dir / c["image_path"]).exists():
                src = example_dir / c["image_path"]
            if src.exists():
                try:
                    ext = src.suffix.lower().lstrip(".")
                    mime = f"image/{ext}" if ext in ("png", "jpeg", "webp", "gif") else "image/jpeg"
                    b64 = base64.b64encode(src.read_bytes()).decode("ascii")
                    return cid, f"data:{mime};base64,{b64}"
                except Exception:
                    pass
        return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        for res in pool.map(process_audio, cases):
            if res:
                audio_map[res[0]] = res[1]
        for res in pool.map(process_image, cases):
            if res:
                image_map[res[0]] = res[1]

    return audio_map, image_map


# ---------------------------------------------------------------------------
# HTML Template & Rendering
# ---------------------------------------------------------------------------

def esc(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def generate_html_report(data: dict[str, Any], figures: dict[str, str], audio_map: dict[str, str], image_map: dict[str, str]) -> str:
    """Render the full diagnostic report page with a sleek modern English Tabbed UI."""
    run = data["run"]
    m1 = data["m1"]
    m2 = data["m2"]
    m3 = data["m3"]
    m5 = data["m5"]
    m4_s = data["m4_surgery"]
    m4_f = data["m4_fix"]
    cases = data["cases"]

    n_total = run["n_cases"] or len(cases) or 1
    cfm = m4_f.get("confirm") or {}
    base_correct = cfm.get("n_baseline_correct")
    n_pairs = cfm.get("n_pairs") or n_total
    base_acc = (base_correct / n_pairs) if (base_correct is not None and n_pairs > 0) else None
    repair_effect = cfm.get("effect")
    e_val = cfm.get("e_value")
    n_fixed = cfm.get("n_fixed") or sum(1 for c in cases if c["status"] == "fixed")
    n_broken = cfm.get("n_broken") or sum(1 for c in cases if c["status"] == "broken")

    # M5 Verdict
    m5_status = "Skipped"
    m5_tone = "skip"
    if m5["ran"]:
        m5_r0 = m5["results"][0] if m5["results"] else {}
        is_fixed = m5_r0.get("fixed") or m5["event"].get("fixed")
        if is_fixed:
            m5_status = "Supported"
            m5_tone = "good"
        else:
            m5_status = "Refuted"
            m5_tone = "warn"

    # M4 Status
    m4_status = "Not Tested"
    m4_tone = "skip"
    if m4_f["ran"]:
        if m4_f["fixed"] or (repair_effect is not None and repair_effect > 0):
            m4_status = f"+{repair_effect * 100:.1f}% Net Gain" if repair_effect is not None else "Repair Validated"
            m4_tone = "good"
        else:
            m4_status = "Inconclusive / Neutral"
            m4_tone = "warn"

    # Executive Brief Sentences
    stats_sig = [s for s in m2["stats"] if s["reject"]]
    story_p1 = f"Evaluated <b>{esc(run['model'])}</b> on <b>{esc(run['benchmark_name'])}</b> across <b>{n_total:,} benchmark cases</b>."
    if m1["analyzers"]:
        story_p1 += f" Collected multi-dimensional vital signals across <b>{len(m1['analyzers'])} clinical probes</b>."
    if stats_sig:
        story_p1 += f" Screening isolated <b>{len(stats_sig)} statistically significant anomalous signals</b> strongly associated with failure cases."

    story_p2 = ""
    if m3["hypotheses"]:
        hyp0 = m3["hypotheses"][0]
        hyp_txt = hyp0.get("plain_statement") or hyp0.get("statement") or ""
        story_p2 = f"<b>Diagnosed Mechanism:</b> <i>“{esc(hyp_txt)}”</i>"
        if m5["ran"]:
            if m5_tone == "good":
                story_p2 += " — <b>Confirmed</b> on an independent held-out validation set."
            else:
                story_p2 += " — <b>Refuted</b> on held-out validation (empirical effect contradicted the hypothesized direction)."

    story_p3 = ""
    if m4_f["ran"] and repair_effect is not None:
        cured_str = f"<b>{n_fixed}</b> cured" if n_fixed else "0 cured"
        broken_str = f"<b>{n_broken}</b> broken" if n_broken else "0 broken"
        story_p3 = (
            f"<b>Treatment Outcome:</b> Achieved a <b>{repair_effect * 100:+.2f}% net accuracy gain</b> "
            f"on the test set ({cured_str}, {broken_str})."
        )

    # Hero KPI Tiles
    tiles_data = [
        ("Evaluated Cases", f"{n_total:,}", "Diagnosis & Validation Split", "neutral"),
        (
            "Baseline Accuracy",
            f"{base_acc:.1%}" if base_acc is not None else "—",
            f"{base_correct}/{n_pairs} on test split" if base_correct is not None else "Unmodified baseline",
            "neutral",
        ),
        (
            "M5 Hypothesis Verdict",
            m5_status,
            "Held-out test set adjudication" if m5["ran"] else "No M5 stage configured",
            m5_tone,
        ),
        (
            "Repair Net Effect",
            f"{repair_effect * 100:+.2f}%" if repair_effect is not None else "—",
            f"{n_fixed} Cured · {n_broken} Broken" if (n_fixed or n_broken) else "Paired McNemar outcome",
            m4_tone,
        ),
        (
            "Evidence Strength",
            f"e = {e_val:,.0f}" if (e_val and e_val > 0) else "p < 0.05",
            "Multiplicity-corrected certainty" if e_val else "Paired significance",
            "good" if (e_val and e_val > 10) else "neutral",
        ),
    ]

    tiles_html = "\n".join(
        f'<div class="kpi-card kpi-card--{tone}">'
        f'<div class="kpi-label">{esc(lab)}</div>'
        f'<div class="kpi-val">{esc(val)}</div>'
        f'<div class="kpi-sub">{note}</div></div>'
        for lab, val, note, tone in tiles_data
    )

    # Navigation Tabs
    tab_items = [
        ("tab_overview", "Overview", "Executive Summary", "SUMMARY", "good"),
        ("tab_pre_m1", "PRE-M1", "Case Synthesis", "PRE-M1", "skip" if not data["pre_m1"]["ran"] else "neutral"),
        ("tab_m1", "M1", "Checkup & Signals", f"{len(m1['analyzers'])} Probes", "neutral"),
        ("tab_m2", "M2", "Screening & EDA", f"{len(stats_sig)} Significant" if stats_sig else f"{len(m2['stats'])} Tests", "good" if stats_sig else "neutral"),
        ("tab_m3", "M3", "Diagnosis", f"{len(m3['hypotheses'])} Hypotheses", "neutral" if m3["hypotheses"] else "skip"),
        ("tab_m5", "M5", "Adjudication", m5_status, m5_tone),
        ("tab_m4_surgery", "M4-Surgery", "Causal Surgery", "Skipped" if not m4_s["ran"] else "Executed", "skip" if not m4_s["ran"] else "neutral"),
        ("tab_m4_fix", "M4-Fix", "Targeted Repair", m4_status, m4_tone),
        ("tab_case_book", "Case Studio", "Interactive Cases", f"{len(cases)} Cases", "neutral"),
    ]

    tabs_html = "\n".join(
        f'<button type="button" class="tab-btn {"is-active" if tid == "tab_overview" else ""}" data-tab="{tid}">'
        f'<span class="tab-code">{esc(code)}</span>'
        f'<span class="tab-label">{esc(label)}</span>'
        f'<span class="tab-badge badge--{tone}">{esc(badge)}</span>'
        f'</button>'
        for tid, code, label, badge, tone in tab_items
    )

    # M1 Probes Table
    cov_max = max((r["n"] or 0) for r in m1["results"]) if m1["results"] else 1
    cov_max = max(cov_max, 1)
    m1_rows = []
    for r in m1["results"]:
        n = r["n"] or 0
        hl_items = []
        for h in r["headline"]:
            if h["value"] is not None:
                hl_items.append(f'<div class="hl-pill"><span class="hl-v">{esc(h["value"])}</span><span class="hl-l">{esc(h["label"])}</span></div>')
            else:
                hl_items.append(f'<div class="hl-pill hl-pill--none"><span class="hl-v">—</span><span class="hl-l">{esc(h["label"])}</span></div>')
        hl_html = "".join(hl_items) or '<span class="text-muted">No scalar headlines</span>'

        if r["n"] is None:
            count_cell = '<td class="num text-muted">Batch Aggregate</td>'
            bar_cell = '<td class="barcell"><span class="barpct">Run-level summary</span></td>'
        else:
            count_cell = f'<td class="num">{n:,} cases</td>'
            pct_val = (n / n_total) if n_total > 0 else 0
            bar_cell = (
                f'<td class="barcell"><div class="bar-track"><span class="bar-fill" style="width:{n / cov_max * 100:.1f}%"></span></div>'
                f'<span class="barpct">{pct_val:.0%} coverage</span></td>'
            )

        m1_rows.append(
            f'<tr><td><div class="probe-title">{esc(r["display_name"])} <span class="probe-code mono">{esc(r["name"])}</span></div>'
            f'<div class="probe-q">{esc(r["question"])}</div></td>'
            f'{count_cell}{bar_cell}'
            f'<td><div class="hl-wrap">{hl_html}</div></td></tr>'
        )
    m1_table_html = "\n".join(m1_rows) if m1_rows else "<tr><td colspan='4'>No probes recorded.</td></tr>"

    # M2 Screening Table
    stat_rows_sig = []
    stat_rows_null = []
    for s in m2["stats"]:
        sig_name = (s.get("config") or {}).get("signal") or s.get("tool") or "Signal"
        ci = s.get("ci") or [None, None]
        ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if (ci[0] is not None and ci[1] is not None) else "—"
        p_val = s.get("p_value")
        p_str = f"{p_val:.2e}" if isinstance(p_val, (int, float)) else "—"
        verdict = (
            '<span class="badge badge--good">Significant (BH FDR Passed)</span>'
            if s["reject"]
            else '<span class="badge badge--mute">Not Significant</span>'
        )
        row_h = (
            f'<tr><td class="mono font-semibold">{esc(sig_name)}</td>'
            f'<td class="num font-bold">{(s.get("effect") or 0):+.4f}</td>'
            f'<td class="num text-muted mono">{ci_str}</td><td class="num text-muted mono">{p_str}</td>'
            f'<td>{verdict}</td></tr>'
        )
        if s["reject"]:
            stat_rows_sig.append(row_h)
        else:
            stat_rows_null.append(row_h)

    stats_sig_html = "\n".join(stat_rows_sig) if stat_rows_sig else "<tr><td colspan='5'>No statistically significant signals found.</td></tr>"
    stats_null_html = "\n".join(stat_rows_null) if stat_rows_null else "<tr><td colspan='5'>No null signals recorded.</td></tr>"

    # Figures
    fig_blocks = []
    for k, v in figures.items():
        if k in ("test", "m2_effects"):
            continue
        fig_blocks.append(
            f'<div class="chart-card">'
            f'<div class="chart-head"><span class="chart-title">{esc(k.replace("_", " ").title())}</span></div>'
            f'<img src="{v}" alt="{esc(k)}" loading="lazy">'
            f'</div>'
        )
    all_figs_html = "\n".join(fig_blocks) if fig_blocks else "<p class='text-muted' style='padding:16px;'>No exploratory charts generated for this run.</p>"

    # M4 Candidates Table
    sel_rows = []
    max_eff = max([abs(s.get("effect") or 0) for s in m4_f["selection"]] + [0.01])
    for s in m4_f["selection"]:
        eff = s.get("effect") or 0.0
        w = min(100, abs(eff) / max_eff * 50)
        side = "pos" if eff >= 0 else "neg"
        verdict = s.get("verdict", "tested")
        tone = "good" if verdict == "fixed" else ("warn" if verdict == "partial" else "bad")
        sel_rows.append(
            f'<tr><td class="mono font-semibold">{esc(s.get("tier", "L1"))}</td>'
            f'<td class="mono font-bold">{esc(s.get("name", "candidate"))}</td>'
            f'<td><span class="badge badge--{tone}">{esc(verdict)}</span></td>'
            f'<td class="num text-good">+{s.get("n_fixed", 0)}</td>'
            f'<td class="num text-bad">-{s.get("n_broken", 0)}</td>'
            f'<td class="num eff-cell eff-cell--{side}">{eff * 100:+.2f}%</td>'
            f'<td class="diverge-cell"><div class="diverge-axis"></div>'
            f'<div class="diverge-bar diverge-bar--{side}" style="width:{w:.1f}%"></div></td></tr>'
        )
    sel_table_html = "\n".join(sel_rows) if sel_rows else "<tr><td colspan='7'>No repair candidate sweep recorded.</td></tr>"

    meta_chips = "".join(f"<span class='tag-pill'>{esc(m)}</span>" for m in [
        run["benchmark_name"],
        f"{n_total:,} Cases",
        f"{run['cycles']} Cycles",
        f"{len(m1['analyzers'])} Probes",
        f"{run['duration_sec']:.0f}s Duration" if run['duration_sec'] else "Completed",
        f"EvalVitals v{run['version']}",
    ])

    pre_m1_desc = f"Active probe search generated {data['pre_m1']['n_cases']} synthetic test cases to probe failure mechanisms." if data["pre_m1"]["ran"] else "Testing ran directly on fixed benchmark cases (automated Pre-M1 adversarial probe synthesis was not configured)."
    pre_m1_sub = f"{data['pre_m1']['n_cases']} Probes Synthesized" if data["pre_m1"]["ran"] else "Standard Benchmark"
    pre_m1_cls = "stage--skip" if not data["pre_m1"]["ran"] else ""

    m1_duration_str = f"{m1['duration']:.1f}s" if m1["duration"] else "Completed"
    m2_duration_str = f"{m2['duration']:.1f}s" if m2["duration"] else "Completed"
    m2_conclusion_box = f"<div class='callout callout--accent'><b>Screening Summary:</b> {esc(m2['conclusion'])}</div>" if m2["conclusion"] else ""

    m3_items = []
    for idx, h in enumerate(m3["hypotheses"], 1):
        fm = esc(h.get("failure_mode", "mechanism"))
        st_plain = esc(h.get("plain_statement") or h.get("statement") or h.get("hypothesis"))
        st_raw = esc(h.get("statement"))
        td = esc(h.get("test_design"))
        tech_line = f'<div class="card-detail"><span class="label">Technical Claim:</span> <code>{st_raw}</code></div>' if st_raw and st_raw != st_plain else ""
        test_line = f'<div class="card-detail"><span class="label">Test Design:</span> {td}</div>' if td else ""
        m3_items.append(f"""
        <div class="hypothesis-card">
          <div class="hyp-head">
            <span class="badge badge--brand">{fm}</span>
            <span class="mono text-muted" style="font-size:12px;">Hypothesis #{idx}</span>
          </div>
          <div class="hyp-statement">{st_plain}</div>
          {tech_line}
          {test_line}
        </div>""")
    m3_hypotheses_html = "\n".join(m3_items) if m3_items else "<p class='text-muted'>No M3 hypotheses proposed.</p>"
    m3_cls = "stage--skip" if not m3["hypotheses"] else ""

    m5_cls = "stage--skip" if not m5["ran"] else ""
    m5_tone_cls = "callout--good" if m5_tone == "good" else ("callout--warn" if m5_tone == "warn" else "callout--neutral")
    if m5_tone == "good":
        m5_plain_text = f"<b>Verdict: {m5_status}</b>. Re-evaluated probe signals on an independent held-out split. Observed direction matched the predicted failure mechanism with statistical significance."
    elif m5_tone == "warn":
        m5_plain_text = f"<b>Verdict: {m5_status}</b>. When re-tested on independent validation cases, the empirical data contradicted the AI Doctor's hypothesis."
    else:
        m5_plain_text = "Independent M5 blind validation was not executed for this run."

    m5_detail_html = ""
    if m5["results"] or m5["event"]:
        res0 = m5["results"][0] if m5["results"] else {}
        eff_size = res0.get("effect_size", 0)
        eff_str = f"{eff_size:+.3f}" if isinstance(eff_size, (int, float)) else "—"
        conf_val = res0.get("confidence", 0)
        conf_str = f"{conf_val:.2f}" if isinstance(conf_val, (int, float)) else "—"
        ev_grade = esc(res0.get("evidence", {}).get("evidence_grade", "—"))
        verdict_raw = esc(res0.get("verdict") or m5["event"].get("verdict") or json.dumps(m5["event"], indent=2, ensure_ascii=False))
        status_cls = "text-good font-bold" if m5_tone == "good" else "text-bad font-bold"

        m5_detail_html = f"""
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Verdict</div><div class="kpi-val {status_cls}">{esc(m5_status)}</div></div>
          <div class="kpi-card"><div class="kpi-label">Validation Effect</div><div class="kpi-val">{eff_str}</div></div>
          <div class="kpi-card"><div class="kpi-label">Confidence Score</div><div class="kpi-val">{conf_str}</div></div>
          <div class="kpi-card"><div class="kpi-label">Evidence Grade</div><div class="kpi-val">{ev_grade}</div></div>
        </div>
        <pre class="code-block"><b>Adjudication Audit Log:</b>
{verdict_raw}</pre>"""

    m4_s_cls = "stage--skip" if not m4_s["ran"] else ""
    m4_s_desc = f"Executed {len(m4_s['surgeries'])} causal model interventions / ablations." if m4_s["ran"] else "Focused on black-box prompt and scaffold optimizations (white-box surgery was not invoked)."
    m4_s_sub = "Executed" if m4_s["ran"] else "Skipped"

    m4_f_cls = "stage--skip" if not m4_f["ran"] else ""
    m4_f_tone_cls = "callout--good" if m4_tone == "good" else "callout--neutral"
    if repair_effect is not None and repair_effect > 0:
        cand_name = esc(cfm.get("name", "selected_fix"))
        e_str = f"e = {e_val:,.0f}" if e_val else "Significant"
        m4_f_plain = f"Successfully confirmed repair strategy <b><code>{cand_name}</code></b>! Produced a <b>{repair_effect * 100:+.2f}% net accuracy gain</b> on held-out test data (<b>{n_fixed}</b> cured vs <b>{n_broken}</b> broken, evidence strength <b>{e_str}</b>)."
    else:
        m4_f_plain = "Screened repair candidates across tiers, but no strategy met the statistical threshold for a significant net gain."

    tmpl_code = esc(m4_f["prompt_template"] or cfm.get("summary") or "")
    m4_f_tmpl_html = ""
    if tmpl_code:
        m4_f_tmpl_html = f"""
        <div style="margin-top:16px;">
          <div class="section-subhead">Winning Prompt / Repair Patch</div>
          <pre class="code-block">{tmpl_code}</pre>
        </div>"""

    cases_json = json.dumps(cases, ensure_ascii=False)
    audio_json = json.dumps(audio_map)
    images_json = json.dumps(image_map)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvalVitals · {esc(run['model'])} Diagnostic Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap">
<style>
:root {{
  --bg: #0b0f19;
  --surface: #111827;
  --surface-raised: #1f2937;
  --surface-hover: #374151;
  --border: #374151;
  --border-subtle: #242d3d;
  --text: #f9fafb;
  --text-muted: #9ca3af;
  --text-sub: #6b7280;
  --brand: #6366f1;
  --brand-soft: rgba(99, 102, 241, 0.12);
  --brand-border: rgba(99, 102, 241, 0.35);
  --good: #10b981;
  --good-soft: rgba(16, 185, 129, 0.12);
  --good-border: rgba(16, 185, 129, 0.35);
  --warn: #f59e0b;
  --warn-soft: rgba(245, 158, 11, 0.12);
  --warn-border: rgba(245, 158, 11, 0.35);
  --bad: #ef4444;
  --bad-soft: rgba(239, 68, 68, 0.12);
  --bad-border: rgba(239, 68, 68, 0.35);
  --radius: 10px;
  --radius-sm: 6px;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, Menlo, Monaco, monospace;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 14.5px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}

.container {{ max-width: 1240px; margin: 0 auto; padding: 0 24px; }}

header.app-header {{
  border-bottom: 1px solid var(--border-subtle);
  background: rgba(17, 24, 39, 0.95);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 100;
}}
.header-inner {{
  padding: 16px 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}}
.brand-group {{
  display: flex;
  align-items: center;
  gap: 14px;
}}
.brand-icon {{
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--brand), #8b5cf6);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 16px;
  font-weight: 700;
}}
.brand-titles {{ display: flex; flex-direction: column; }}
.brand-model {{ font-size: 17px; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }}
.brand-benchmark {{ font-size: 12.5px; color: var(--text-muted); }}

.header-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}

nav.tabs-nav {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 65px;
  z-index: 90;
}}
.tabs-scroll {{
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding: 10px 0;
  scrollbar-width: none;
}}
.tabs-scroll::-webkit-scrollbar {{ display: none; }}

.tab-btn {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--text-muted);
  font-family: var(--font-sans);
  font-size: 13.5px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s ease;
}}
.tab-btn:hover {{
  background: var(--surface-raised);
  color: var(--text);
}}
.tab-btn.is-active {{
  background: var(--surface-raised);
  border-color: var(--border);
  color: #ffffff;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}}
.tab-code {{
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  color: var(--brand);
}}
.tab-badge {{
  font-family: var(--font-mono);
  font-size: 10.5px;
  padding: 2px 6px;
  border-radius: 99px;
}}

.tab-pane {{
  display: none;
  padding: 32px 0 80px;
  animation: fadeIn 0.15s ease-in-out;
}}
.tab-pane.is-active {{ display: block; }}
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(4px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}}
.card-head {{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 12px;
  padding-bottom: 16px;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border-subtle);
}}
.card-tag {{
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  color: var(--brand);
  background: var(--brand-soft);
  border: 1px solid var(--brand-border);
  padding: 3px 8px;
  border-radius: 4px;
}}
.card-title {{ font-size: 20px; font-weight: 700; letter-spacing: -0.02em; color: #ffffff; }}
.card-subtext {{ flex: 1 1 100%; font-size: 13.5px; color: var(--text-muted); }}

.story-box {{
  background: linear-gradient(180deg, rgba(31, 41, 55, 0.6) 0%, rgba(17, 24, 39, 0.9) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 26px;
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.story-box p {{ font-size: 15px; color: var(--text); line-height: 1.6; }}
.story-box b {{ color: #ffffff; }}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
  margin-bottom: 24px;
}}
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}}
.kpi-label {{
  font-size: 11.5px;
  font-family: var(--font-mono);
  text-transform: uppercase;
  color: var(--text-sub);
  letter-spacing: 0.05em;
}}
.kpi-val {{
  font-size: 26px;
  font-weight: 700;
  font-family: var(--font-mono);
  letter-spacing: -0.02em;
}}
.kpi-sub {{ font-size: 12px; color: var(--text-muted); }}

.kpi-card--good {{ border-color: var(--good-border); }}
.kpi-card--good .kpi-val {{ color: var(--good); }}
.kpi-card--warn {{ border-color: var(--warn-border); }}
.kpi-card--warn .kpi-val {{ color: var(--warn); }}
.kpi-card--bad {{ border-color: var(--bad-border); }}
.kpi-card--bad .kpi-val {{ color: var(--bad); }}

.callout {{
  padding: 16px 20px;
  border-radius: var(--radius-sm);
  background: var(--surface-raised);
  border-left: 4px solid var(--border);
  font-size: 14px;
  color: var(--text);
  margin-bottom: 20px;
}}
.callout--accent {{ border-left-color: var(--brand); background: var(--brand-soft); }}
.callout--good {{ border-left-color: var(--good); background: var(--good-soft); }}
.callout--warn {{ border-left-color: var(--warn); background: var(--warn-soft); }}
.callout b {{ color: #ffffff; }}

.table-wrapper {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 20px;
}}
table.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  text-align: left;
}}
table.data-table th {{
  font-family: var(--font-mono);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-sub);
  background: var(--surface-raised);
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
}}
table.data-table td {{
  padding: 11px 14px;
  border-bottom: 1px solid var(--border-subtle);
  vertical-align: middle;
}}
table.data-table tr:hover td {{
  background: rgba(255, 255, 255, 0.02);
}}
.num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: var(--font-mono); }}
.mono {{ font-family: var(--font-mono); font-size: 0.92em; }}
.font-semibold {{ font-weight: 600; }}
.font-bold {{ font-weight: 700; }}
.text-muted {{ color: var(--text-muted); }}
.text-good {{ color: var(--good); }}
.text-warn {{ color: var(--warn); }}
.text-bad {{ color: var(--bad); }}

.badge {{
  display: inline-flex;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 99px;
  border: 1px solid transparent;
}}
.badge--brand {{ color: #818cf8; background: var(--brand-soft); border-color: var(--brand-border); }}
.badge--good {{ color: var(--good); background: var(--good-soft); border-color: var(--good-border); }}
.badge--warn {{ color: var(--warn); background: var(--warn-soft); border-color: var(--warn-border); }}
.badge--bad {{ color: var(--bad); background: var(--bad-soft); border-color: var(--bad-border); }}
.badge--mute {{ color: var(--text-muted); background: var(--surface-raised); border-color: var(--border); }}

.tag-pill {{
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  background: var(--surface-raised);
  border: 1px solid var(--border);
  padding: 4px 12px;
  border-radius: 99px;
}}

.probe-title {{ font-weight: 600; font-size: 14px; color: #ffffff; }}
.probe-code {{ color: var(--brand); font-size: 11px; margin-left: 6px; }}
.probe-q {{ font-size: 12.5px; color: var(--text-muted); margin-top: 3px; }}

.barcell {{ width: 150px; }}
.bar-track {{ height: 6px; border-radius: 3px; background: var(--surface-raised); overflow: hidden; margin-bottom: 3px; }}
.bar-fill {{ display: block; height: 100%; background: var(--brand); border-radius: 3px; }}
.barpct {{ font-family: var(--font-mono); font-size: 11px; color: var(--text-sub); }}

.hl-wrap {{ display: flex; flex-direction: column; gap: 4px; }}
.hl-pill {{ display: inline-flex; align-items: baseline; gap: 6px; font-size: 12px; }}
.hl-v {{ font-family: var(--font-mono); font-weight: 700; color: var(--text); min-width: 44px; text-align: right; }}
.hl-l {{ color: var(--text-muted); font-size: 11.5px; }}
.hl-pill--none .hl-v {{ color: var(--text-sub); }}

.diverge-cell {{ position: relative; width: 120px; height: 20px; }}
.diverge-axis {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--border); }}
.diverge-bar {{ position: absolute; top: 6px; height: 8px; border-radius: 2px; }}
.diverge-bar--pos {{ left: 50%; background: var(--good); }}
.diverge-bar--neg {{ right: 50%; background: var(--bad); }}
.eff-cell--pos {{ color: var(--good); font-weight: 700; }}
.eff-cell--neg {{ color: var(--bad); font-weight: 700; }}

.hypothesis-card {{
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-raised);
  padding: 18px 22px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 14px;
}}
.hyp-head {{ display: flex; justify-content: space-between; align-items: center; }}
.hyp-statement {{ font-size: 16px; font-weight: 600; color: #ffffff; line-height: 1.45; }}
.card-detail {{ font-size: 13px; color: var(--text-muted); }}
.card-detail .label {{ font-weight: 600; color: var(--text); margin-right: 6px; }}

.charts-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 18px;
}}
.chart-card {{
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-raised);
  overflow: hidden;
}}
.chart-head {{
  padding: 11px 16px;
  background: rgba(0, 0, 0, 0.25);
  border-bottom: 1px solid var(--border-subtle);
}}
.chart-title {{ font-size: 13.5px; font-weight: 600; color: var(--text); }}
.chart-card img {{ display: block; width: 100%; height: auto; }}

details.collapsible-box {{
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  margin-bottom: 16px;
}}
details.collapsible-box summary {{
  padding: 13px 18px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  color: var(--text-muted);
  user-select: none;
}}
details.collapsible-box summary:hover {{ color: var(--text); }}
details.collapsible-box .content {{ padding: 0 18px 18px; }}

pre.code-block {{
  background: #06090e;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 16px;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.55;
  color: #e2e8f0;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
}}
.section-subhead {{
  font-size: 12px;
  font-family: var(--font-mono);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-sub);
  margin-bottom: 8px;
}}

.case-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;
}}
.filter-btn-group {{
  display: flex;
  gap: 6px;
  background: var(--surface-raised);
  padding: 4px;
  border-radius: 99px;
  border: 1px solid var(--border);
}}
.filter-btn {{
  font-family: var(--font-mono);
  font-size: 12.5px;
  padding: 6px 14px;
  border-radius: 99px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.15s ease;
}}
.filter-btn:hover {{ color: var(--text); }}
.filter-btn.is-active {{
  background: var(--brand);
  color: white;
  font-weight: 600;
}}
.search-box {{
  margin-left: auto;
  padding: 8px 16px;
  border-radius: 99px;
  border: 1px solid var(--border);
  background: var(--surface-raised);
  color: var(--text);
  font-size: 13.5px;
  outline: none;
  width: 280px;
}}
.search-box:focus {{ border-color: var(--brand); }}

.case-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 18px;
}}
.case-item {{
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-raised);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}
.case-item-header {{
  padding: 11px 16px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  gap: 8px;
}}
.case-dur {{ margin-left: auto; font-family: var(--font-mono); font-size: 11px; color: var(--text-sub); }}
.case-q {{
  padding: 16px;
  font-size: 14.5px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.45;
}}
.case-media-box {{ padding: 0 16px 12px; }}
.case-media-box audio {{ width: 100%; height: 34px; }}
.case-media-box img {{ width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px; border: 1px solid var(--border); }}
.case-choices {{
  list-style: none;
  padding: 0 16px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.case-choices li {{
  font-size: 13px;
  padding: 7px 12px;
  border-radius: 6px;
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  display: flex;
  gap: 8px;
  align-items: baseline;
}}
.case-choices li.is-gold {{
  background: var(--good-soft);
  border-color: var(--good-border);
  color: #ffffff;
  font-weight: 600;
}}
.case-item-foot {{
  margin-top: auto;
  padding: 11px 16px;
  background: rgba(0, 0, 0, 0.3);
  border-top: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
}}
.case-id-tag {{
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-sub);
}}

footer.app-footer {{
  border-top: 1px solid var(--border-subtle);
  padding: 32px 0 48px;
  color: var(--text-sub);
  font-size: 13px;
}}
</style>
</head>
<body>

<header class="app-header">
  <div class="container header-inner">
    <div class="brand-group">
      <div class="brand-icon">⚡</div>
      <div class="brand-titles">
        <span class="brand-model">{esc(run['model'])}</span>
        <span class="brand-benchmark">{esc(run['benchmark_name'])}</span>
      </div>
    </div>
    <div class="header-badges">{meta_chips}</div>
  </div>
</header>

<nav class="tabs-nav">
  <div class="container">
    <div class="tabs-scroll">
      {tabs_html}
    </div>
  </div>
</nav>

<div class="container">

  <!-- TAB: OVERVIEW -->
  <div class="tab-pane is-active" id="tab_overview">
    <div class="story-box">
      <p>{story_p1}</p>
      {f"<p>{story_p2}</p>" if story_p2 else ""}
      {f"<p>{story_p3}</p>" if story_p3 else ""}
    </div>

    <div class="kpi-grid">{tiles_html}</div>

    <div class="card">
      <div class="card-head">
        <span class="card-tag">PIPELINE SUMMARY</span>
        <h2 class="card-title">Diagnostic Stages Breakdown</h2>
        <p class="card-subtext">Click on any tab above to inspect deep-dive evidence, statistical tests, or interactive case media.</p>
      </div>

      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px;">
        <div class="hypothesis-card" style="cursor:pointer;" onclick="switchTab('tab_m1')">
          <div class="hyp-head"><span class="badge badge--brand">STAGE M1</span><span class="mono text-muted">{len(m1['analyzers'])} Probes</span></div>
          <div class="font-bold">Multi-Modal Checkup</div>
          <div class="text-muted" style="font-size:13px;">Clinical vital signs (order sensitivity, pass@5 coverage, uncertainty, calibration).</div>
        </div>

        <div class="hypothesis-card" style="cursor:pointer;" onclick="switchTab('tab_m2')">
          <div class="hyp-head"><span class="badge badge--good">STAGE M2</span><span class="mono text-muted">{len(stats_sig)} Significant</span></div>
          <div class="font-bold">Screening & Confirmatory Signals</div>
          <div class="text-muted" style="font-size:13px;">Identified key anomaly signals strongly correlated with errors under Benjamini–Hochberg FDR control.</div>
        </div>

        <div class="hypothesis-card" style="cursor:pointer;" onclick="switchTab('tab_m3')">
          <div class="hyp-head"><span class="badge badge--brand">STAGE M3 & M5</span><span class="mono text-muted">{m5_status}</span></div>
          <div class="font-bold">Diagnosis & Validation</div>
          <div class="text-muted" style="font-size:13px;">Falsifiable root-cause mechanism hypotheses verified on held-out test data.</div>
        </div>

        <div class="hypothesis-card" style="cursor:pointer;" onclick="switchTab('tab_m4_fix')">
          <div class="hyp-head"><span class="badge badge--good">STAGE M4</span><span class="mono text-muted">{m4_status}</span></div>
          <div class="font-bold">Targeted Repair & Case Studio</div>
          <div class="text-muted" style="font-size:13px;">Validated repair strategies with paired McNemar confirmation (+{repair_effect * 100:.1f}% net gain).</div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB: PRE-M1 -->
  <div class="tab-pane" id="tab_pre_m1">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">PRE-M1</span>
        <h2 class="card-title">Case Synthesis & Adversarial Probes</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{pre_m1_sub}</span>
        <p class="card-subtext">{STAGE_METADATA['pre_m1']['plain_desc']}</p>
      </div>
      <div class="callout">
        <b>Stage Status:</b> {pre_m1_desc}
      </div>
    </div>
  </div>

  <!-- TAB: M1 -->
  <div class="tab-pane" id="tab_m1">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">M1</span>
        <h2 class="card-title">Checkup & Vital Signals</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m1_duration_str}</span>
        <p class="card-subtext">{STAGE_METADATA['m1']['plain_desc']}</p>
      </div>

      <div class="callout callout--accent">
        <b>Measurement Summary:</b> Executed <b>{len(m1['analyzers'])} clinical probes</b>. Each probe measures a specific behavioral dimension without making premature failure attributions.
      </div>

      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>Probe & Diagnostic Question</th>
              <th class="num">Cases Scored</th>
              <th>Dataset Coverage</th>
              <th>Core Measured Headlines</th>
            </tr>
          </thead>
          <tbody>{m1_table_html}</tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB: M2 -->
  <div class="tab-pane" id="tab_m2">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">M2</span>
        <h2 class="card-title">Screening & Confirmatory Signals</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m2_duration_str}</span>
        <p class="card-subtext">{STAGE_METADATA['m2']['plain_desc']}</p>
      </div>

      <div class="callout callout--good">
        <b>Screening Outcome:</b> Out of {len(m2['stats'])} tested feature associations, <b>{len(stat_rows_sig)} signals passed rigorous Benjamini–Hochberg statistical significance correction</b>.
      </div>

      {m2_conclusion_box}

      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>Anomalous Feature Signal</th>
              <th class="num">Effect Size</th>
              <th class="num">95% Conf. Interval</th>
              <th class="num">p-value</th>
              <th>FDR Screening Verdict</th>
            </tr>
          </thead>
          <tbody>{stats_sig_html}</tbody>
        </table>
      </div>

      <details class="collapsible-box">
        <summary>View {len(stat_rows_null)} Non-Significant Feature Signals</summary>
        <div class="content">
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Feature Signal</th>
                  <th class="num">Effect Size</th>
                  <th class="num">95% Conf. Interval</th>
                  <th class="num">p-value</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>{stats_null_html}</tbody>
            </table>
          </div>
        </div>
      </details>

      <details class="collapsible-box" open>
        <summary>Exploratory Data Analysis Charts ({len(figures)} plots)</summary>
        <div class="content">
          <div class="charts-grid">{all_figs_html}</div>
        </div>
      </details>
    </div>
  </div>

  <!-- TAB: M3 -->
  <div class="tab-pane" id="tab_m3">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">M3</span>
        <h2 class="card-title">Root-Cause Diagnosis (AI Doctor)</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{len(m3['hypotheses'])} Hypotheses</span>
        <p class="card-subtext">{STAGE_METADATA['m3']['plain_desc']}</p>
      </div>

      <div class="callout">
        <b>Diagnostician Rationale:</b> Proposed mechanisms must be <b>falsifiable</b> and pre-register their expected direction of effect to be verified on holdout data.
      </div>

      {m3_hypotheses_html}
    </div>
  </div>

  <!-- TAB: M5 -->
  <div class="tab-pane" id="tab_m5">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">M5</span>
        <h2 class="card-title">Independent Blind Adjudication</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m5_status}</span>
        <p class="card-subtext">{STAGE_METADATA['m5']['plain_desc']}</p>
      </div>

      <div class="callout {m5_tone_cls}">
        {m5_plain_text}
      </div>

      {m5_detail_html}
    </div>
  </div>

  <!-- TAB: M4-SURGERY -->
  <div class="tab-pane" id="tab_m4_surgery">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">M4-SURGERY</span>
        <h2 class="card-title">Causal Surgery & Interventions</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m4_s_sub}</span>
        <p class="card-subtext">{STAGE_METADATA['m4_surgery']['plain_desc']}</p>
      </div>

      <div class="callout">
        <b>Stage Status:</b> {m4_s_desc}
      </div>
    </div>
  </div>

  <!-- TAB: M4-FIX -->
  <div class="tab-pane" id="tab_m4_fix">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">M4-FIX</span>
        <h2 class="card-title">Targeted Repair & Paired Confirmation</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m4_status}</span>
        <p class="card-subtext">{STAGE_METADATA['m4_fix']['plain_desc']}</p>
      </div>

      <div class="callout {m4_f_tone_cls}">
        {m4_f_plain}
      </div>

      <details class="collapsible-box" open>
        <summary>Repair Candidate Sweep ({len(m4_f['selection'])} candidates evaluated)</summary>
        <div class="content">
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Tier</th>
                  <th>Candidate Strategy</th>
                  <th>Screening Verdict</th>
                  <th class="num">Cured (+)</th>
                  <th class="num">Broken (-)</th>
                  <th class="num">Net Accuracy Shift</th>
                  <th>Balance Direction</th>
                </tr>
              </thead>
              <tbody>{sel_table_html}</tbody>
            </table>
          </div>
        </div>
      </details>

      {m4_f_tmpl_html}
    </div>
  </div>

  <!-- TAB: CASE STUDIO -->
  <div class="tab-pane" id="tab_case_book">
    <div class="card">
      <div class="card-head">
        <span class="card-tag">CASE-STUDIO</span>
        <h2 class="card-title">Interactive Case Studio</h2>
        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{len(cases)} Cases Available</span>
        <p class="card-subtext">{STAGE_METADATA['case_book']['plain_desc']}</p>
      </div>

      <div class="case-controls">
        <div class="filter-btn-group">
          <button type="button" class="filter-btn is-active" data-f="all">All ({len(cases)})</button>
          <button type="button" class="filter-btn" data-f="fixed">Cured (+{sum(1 for c in cases if c['status'] == 'fixed')})</button>
          <button type="button" class="filter-btn" data-f="broken">Broken (-{sum(1 for c in cases if c['status'] == 'broken')})</button>
          <button type="button" class="filter-btn" data-f="unchanged">Unchanged ({sum(1 for c in cases if c['status'] == 'unchanged')})</button>
        </div>
        <input type="text" class="search-box" id="csearch" placeholder="Search prompt, question, or ID...">
      </div>

      <div class="case-grid" id="cases_grid"></div>
    </div>
  </div>

</div>

<footer class="app-footer">
  <div class="container" style="display:flex; flex-direction:column; gap:6px;">
    <div>Generated by <b>EvalVitals Diagnostic Engine</b> · Single-page tabbed interactive report · Zero external runtime dependencies</div>
    <div>Model: <code>{esc(run['model'])}</code> · Fingerprint: <code>{esc(run['data_fingerprint'])}</code> · Path: <code>{esc(run['logs_dir'])}</code></div>
  </div>
</footer>

<script>
const CASES = {cases_json};
const AUDIO = {audio_json};
const IMAGES = {images_json};

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('is-active', b.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('is-active', p.id === tabId);
  });
  if (history.replaceState) {
    history.replaceState(null, null, '#' + tabId.replace('tab_', ''));
  }
}

document.querySelectorAll('.tab-btn').forEach(b => {
  b.addEventListener('click', () => switchTab(b.dataset.tab));
});

if (window.location.hash) {
  const targetId = 'tab_' + window.location.hash.replace('#', '');
  if (document.getElementById(targetId)) {
    switchTab(targetId);
  }
}

const grid = document.getElementById('cases_grid');
const searchInput = document.getElementById('csearch');
let currentFilter = 'all';
let searchKeyword = '';

const esc = s => String(s || '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function formatChoices(c) {
  if (!Array.isArray(c.choices) || c.choices.length === 0) return '';
  return c.choices.map(raw => {
    let letter = '', text = String(raw);
    const m = text.match(/^\s*\(?([A-Za-z])\)?[\.:)]?\s*(.*)$/s);
    if (m) {
      letter = m[1].toUpperCase();
      text = m[2];
    }
    const isGold = letter && (letter === String(c.expected).trim().toUpperCase());
    return `<li class="${isGold ? 'is-gold' : ''}"><span class="mono font-bold">${esc(letter || '·')}</span> <span>${esc(text)}</span> ${isGold ? '<span class="badge badge--good" style="margin-left:auto; font-size:10px;">Ground Truth</span>' : ''}</li>`;
  }).join('');
}

function renderCard(c) {
  let badgeClass = 'badge--mute', badgeText = 'Unchanged';
  if (c.status === 'fixed') { badgeClass = 'badge--good'; badgeText = 'Cured (Fixed)'; }
  else if (c.status === 'broken') { badgeClass = 'badge--bad'; badgeText = 'Regression (Broken)'; }

  const audioSrc = AUDIO[c.id];
  const imgSrc = IMAGES[c.id];

  return `
  <article class="case-item">
    <div class="case-item-header">
      <span class="badge ${badgeClass}">${badgeText}</span>
      ${c.task ? `<span class="mono text-muted" style="font-size:11px;">${esc(c.task)}</span>` : ''}
      ${c.duration ? `<span class="case-dur">${c.duration.toFixed(1)}s</span>` : ''}
    </div>
    <div class="case-q">${esc(c.instruction)}</div>
    <div class="case-media-box">
      ${audioSrc ? `<audio controls preload="none" src="${audioSrc}"></audio>` : ''}
      ${imgSrc ? `<img src="${imgSrc}" alt="Stimulus" loading="lazy">` : ''}
    </div>
    <ul class="case-choices">${formatChoices(c)}</ul>
    <div class="case-item-foot">
      <span class="text-muted font-semibold" style="font-size:11px; text-transform:uppercase;">Repaired Output:</span>
      <span class="mono font-bold" style="color:#ffffff;">${esc(c.output || '—')}</span>
      <span class="case-id-tag">ID: ${esc(c.id.slice(0, 10))}</span>
    </div>
  </article>`;
}

function updateGrid() {
  let rows = CASES;
  if (currentFilter !== 'all') {
    rows = rows.filter(c => c.status === currentFilter);
  }
  if (searchKeyword) {
    const kw = searchKeyword.toLowerCase();
    rows = rows.filter(c =>
      c.id.toLowerCase().includes(kw) ||
      (c.instruction && c.instruction.toLowerCase().includes(kw)) ||
      (c.task && c.task.toLowerCase().includes(kw))
    );
  }
  grid.innerHTML = rows.slice(0, 100).map(renderCard).join('') || '<p class="text-muted" style="padding:24px;">No matching cases found.</p>';
}

document.querySelectorAll('.filter-btn').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(o => o.classList.toggle('is-active', o === b));
    currentFilter = b.dataset.f;
    updateGrid();
  });
});

if (searchInput) {
  searchInput.addEventListener('input', e => {
    searchKeyword = e.target.value.trim();
    updateGrid();
  });
}

updateGrid();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_html_report(
    run_dir: str | Path,
    example_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    no_audio: bool = False,
) -> Path:
    run_dir = Path(run_dir).resolve()
    example_dir = Path(example_dir).resolve() if example_dir else run_dir.parent
    out_path = Path(out_path).resolve() if out_path else run_dir / "report.html"

    print(f"[*] Reading run artifacts from: {run_dir}")
    data = extract_run_data(run_dir, example_dir)
    logs_dir = Path(data["run"]["logs_dir"])
    explore_dir = data["explore_dir"]

    figures = embed_figures(explore_dir, logs_dir)
    print(f"[*] Embedded {len(figures)} figures/charts")

    cache_dir = out_path.parent / ".media_cache"
    audio_map, image_map = embed_media(data["cases"], example_dir, cache_dir, no_audio=no_audio)
    print(f"[*] Embedded {len(audio_map)} audio clips and {len(image_map)} images ({len(data["cases"])} joined cases)")

    html_content = generate_html_report(data, figures, audio_map, image_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Wrote self-contained HTML report to: {out_path} ({size_mb:.2f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a finished EvalVitals diagnostic run as an interactive HTML page.")
    ap.add_argument("run_dir", nargs="?", default="outputs", help="Run directory holding run_log.jsonl or logs/")
    ap.add_argument("--example-dir", default=None, help="Root holding data/ (default: parent of run_dir)")
    ap.add_argument("--out", "-o", default=None, help="Output HTML file path (default: <run_dir>/report.html)")
    ap.add_argument("--no-audio", action="store_true", help="Skip audio clip compression and embedding")
    args = ap.parse_args()

    build_html_report(
        run_dir=args.run_dir,
        example_dir=args.example_dir,
        out_path=args.out,
        no_audio=args.no_audio,
    )


if __name__ == "__main__":
    main()
