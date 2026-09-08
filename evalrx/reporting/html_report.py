"""Self-contained, tabbed, human-understandable HTML report generator for EvalRX.

This module parses a finished diagnostic run (M1 through M4 and fixes) and
renders a single, zero-dependency, tabbed interactive HTML diagnostic report in English.

Design Philosophy:
- Tabbed Navigation: Clean stage-by-stage tabs (Overview, M1, M2, M3, M4, M5, Cases)
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
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
    "m4": {
        "code": "M4",
        "name": "Independent Adjudication",
        "short_role": "Blind-Holdout Validation",
        "plain_desc": "Blindly re-tests the AI Doctor's hypothesis on a held-out validation set the model has never seen before, determining whether the mechanism is confirmed or refuted.",
    },
    "m5_surgery": {
        "code": "M5-SURGERY",
        "name": "Causal Surgery",
        "short_role": "Internal Mechanism Intervention",
        "plain_desc": "Directly intervenes in or ablates internal model components (such as attention heads or activation layers) to prove causal necessity.",
    },
    "m5_fix": {
        "code": "M5-FIX",
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
    "agents": {
        "code": "AGENTS",
        "name": "Agent Trajectories",
        "short_role": "Judge & Coder Agent I/O",
        "plain_desc": "Every agent invocation behind the diagnosis, layer by layer: verbatim judge prompts and responses (M1 selection, M2 screening, M3 diagnosis, M4 adjudication), the explore coder-agent's raw CLI trajectory, and every synthesized tool's codegen attempt. This is the audit trail for debugging and case studies.",
    },
}

ANALYZER_GLOSSARY: dict[str, tuple[str, str, str, list[tuple[str, str, str]]]] = {
    "answer_extraction_audit": (
        "Answer Extraction Audit",
        "Did the model actually answer wrong, or did our parser fail to extract its response?",
        "Checks whether formatting quirks (e.g. missing tags, preamble chatter) prevented the parser from extracting the answer even when the model understood the problem.",
        [
            ("Extraction Failure Rate on Errors", "suspect_rate", "pct"),
            ("Missing Format Tag Rate", "missing_tag_rate", "pct"),
        ],
    ),
    "termination_audit": (
        "Truncation & Early Stop Audit",
        "Did the model finish its thought, or was it abruptly cut off by length caps?",
        "Identifies whether premature token limits or sudden EOS tokens clipped the model's reasoning chain mid-sentence.",
        [
            ("Suspected Truncation Rate", "truncation_rate", "pct"),
            ("Recovery Rate upon Continuation", "recovered_rate", "pct"),
        ],
    ),
    "calibration": (
        "Confidence & Overconfidence (ECE)",
        "When the model sounds certain, is its actual correctness probability truly high?",
        "Measures Expected Calibration Error (ECE) across token logprobs and verbalized confidence. High ECE indicates dangerous overconfidence on incorrect answers.",
        [
            ("Token Logprob Calibration Error (ECE)", "logprob_channel.ece", "num"),
            ("Verbalized Confidence Error (ECE)", "verbalized_channel.ece", "num"),
        ],
    ),
    "format_sensitivity": (
        "Option Order Sensitivity (Position Bias)",
        "If we shuffle choices (A/B/C/D), does the model arbitrarily flip its answer?",
        "Permutes multiple-choice options across variants. If answer flips across orderings, the model suffers from severe positional shortcutting rather than semantic understanding.",
        [
            ("Answer Flip Rate after Shuffling", "mean_flip_rate", "pct"),
        ],
    ),
    "coverage_verification_gap": (
        "Pass@k Knowledge Blindspot (Coverage)",
        "Over 5 repeat attempts, does the model ever produce the right answer even once?",
        "Samples temperature completions k=5 times. If pass@5 is 0%, the capability is completely missing from weights; if pass@5 is high but greedy fails, the model suffers from search/ranking misalignment.",
        [
            ("Persistent Failure Rate (0/5 correct)", "no_coverage_rate", "pct"),
            ("Pass@5 Coverage Rate", "mean_pass_at_k", "pct"),
        ],
    ),
    "self_consistency": (
        "Sampling Consistency",
        "When sampled repeatedly with temperature, does the answer waver wildly?",
        "Measures whether model predictions stabilize on a single consensus choice or disperse across inconsistent alternatives.",
        [
            ("Majority Answer Agreement Rate", "consistency", "pct"),
        ],
    ),
    "self_repair": (
        "Self-Repair on Re-ask",
        "When asked to check its own answer, does the model catch and fix its mistakes?",
        "Asks the model to critique and revise its first answer. Reports how often it detects a real error, how often it raises a false alarm on a correct answer, and how often the revision actually repairs a failure.",
        [
            ("Error Detection Accuracy", "detection_accuracy", "pct"),
            ("False Alarm Rate on Correct Answers", "false_alarm_rate", "pct"),
            ("Repair Rate on Failures", "repair_rate", "pct"),
        ],
    ),
    "cot_faithfulness": (
        "Chain-of-Thought Faithfulness",
        "Does the written reasoning actually drive the final answer, or is it decoration?",
        "Compares the answer the model commits to early in its reasoning with the one it ends on: reasoning that drifts away from a correct early answer, or rescues a wrong one late, is unfaithful to the final output.",
        [
            ("Early Answer Matches Final", "mean_early_match_rate", "pct"),
            ("Drift-Away Rate", "drift_away_rate", "pct"),
            ("Late Rescue Rate", "late_rescue_rate", "pct"),
        ],
    ),
    "perturbation_battery": (
        "Perturbation Invariance",
        "Do meaning-preserving edits to the prompt change the answer?",
        "Rewrites each prompt in ways that should not change the answer (paraphrase, whitespace, ordering) and counts how often the answer breaks anyway; a no-op edit that breaks the answer points at memorisation or brittleness.",
        [
            ("Invariance Break Rate", "mean_invariance_break_rate", "pct"),
            ("No-op Break Rate", "noop_break_rate", "pct"),
        ],
    ),
    "step_rollout_value": (
        "Step Rollout Value",
        "At which reasoning step does the model's chance of finishing correctly collapse?",
        "Rolls out completions from successive prefixes of the reasoning and scores each; the point where the success rate drops is where the reasoning went wrong.",
        [
            ("Success Rate from the First Step", "mean_initial_value", "pct"),
            ("Success Rate from the Last Step", "mean_final_value", "pct"),
            ("Mean Break Depth", "mean_break_depth", "num"),
        ],
    ),
    "logprob_entropy": (
        "Predictive Uncertainty (Output Entropy)",
        "Is the model confident or internally hesitating when generating key tokens?",
        "Computes Shannon entropy across the top candidate tokens to measure decision ambivalence.",
        [
            ("Top-Token Prediction Entropy", "mean_top_entropy", "num"),
            ("Model Perplexity", "perplexity", "num"),
        ],
    ),
    "selfcheck_consistency": (
        "Self-Contradiction Detection",
        "Does the model contradict its own claims across repeat samplings?",
        "Probes whether the model generates assertions in one sample that directly contradict its claims in another sample.",
        [
            ("Self-Contradiction Index", "mean_inconsistency", "num"),
        ],
    ),
    "hallucination": (
        "Hallucination Probe",
        "Does the model generate details unsupported by the source stimulus?",
        "Measures semantic grounding against the audible stimulus to catch fabricated details.",
        [
            ("Hallucination Deviation Score", "hallucination_score", "num"),
        ],
    ),
    "qwen_attention": (
        "Attention Focus & Sparsity",
        "Does model attention properly focus on critical prompt and media cues?",
        "Examines cross-attention maps between audio tokens and query tokens to verify sensory uptake.",
        [
            ("Core Cue Attention Share", "focus_share", "pct"),
        ],
    ),
}


def clean_model_display_name(raw: str) -> str:
    """Format raw python class repr like HFLocalModel(key='...') into a human-readable title."""
    if not raw:
        return "Target Model"
    m = re.search(r"key=[\'\"]([^\'\"]+)[\'\"]", raw)
    if m:
        key = m.group(1)
        parts = [p.capitalize() if not p.isdigit() else p for p in key.split("-")]
        return "-".join(parts)
    # Any Model subclass stringifies as ClassName(<name>) — EndpointModel(qwen3.5-2b)
    # for the vLLM chain — and the class is not something a reader needs.
    clean = re.sub(r"^[A-Za-z_]\w*\((.*)\)$", r"\1", raw.strip()).strip().strip("'\"")
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
    elif (run_dir / "logs" / "run.json").exists():
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


def _extract_agent_layer(logs_dir: Path, explore_dir: "Path | None") -> dict[str, Any]:
    """The agent layer of a run: verbatim judge I/O, the explore coder-agent's
    trajectory, synthesized tool codegen attempts, and the Langfuse trace bundle.

    This is what makes the report drill below the pipeline stages: a reader who
    sees a probe or a hypothesis and asks "what did the agent actually see and
    say?" can expand down to the raw prompt/response text.
    """
    agents: dict[str, Any] = {
        "judge_calls": [],
        "explore": {},
        "tool_codegen": [],
        "langfuse": {},
    }

    # -- Verbatim judge calls (prompts/<stem>.prompt.txt + .response.txt) ----
    def _stage_of(stem: str) -> str:
        if "_m1_" in stem:
            return "M1 Analyzer Selection"
        if "_m2_" in stem:
            return "M2 Statistical Screening"
        if "_m3_" in stem:
            return "M3 AI Doctor Diagnosis"
        if "_m4_" in stem:
            return "M4 Protocol Consistency"
        if "_m5_" in stem:
            return "M5 Intervention"
        if "_agent_decision" in stem:
            return "Agentic Loop Decision"
        return "Other"

    prompts_dir = logs_dir / "prompts"
    if prompts_dir.exists():
        for pf in sorted(prompts_dir.glob("*.prompt.txt")):
            stem = pf.name[: -len(".prompt.txt")]
            rf = pf.with_name(stem + ".response.txt")
            try:
                prompt = pf.read_text(encoding="utf-8")
            except Exception:
                prompt = ""
            try:
                response = rf.read_text(encoding="utf-8") if rf.exists() else ""
            except Exception:
                response = ""
            if not prompt and not response:
                continue
            agents["judge_calls"].append({
                "stem": stem,
                "stage": _stage_of(stem),
                "prompt": prompt,
                "response": response,
                "prompt_chars": len(prompt),
                "raw_chars": len(response),
            })

    # -- Explore coder-agent trajectory --------------------------------------
    exp: dict[str, Any] = {}
    if explore_dir is not None and Path(explore_dir).exists():
        ed = Path(explore_dir)
        for key, fname in (
            ("raw_output", "agent_raw_output.txt"),
            ("raw_streams", "agent_raw_streams.txt"),
            ("code", "analysis.py"),
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
        ):
            p = ed / fname
            if p.exists():
                try:
                    exp[key] = p.read_text(encoding="utf-8")
                except Exception:
                    pass
        audit_p = ed / "agent_audit.json"
        if audit_p.exists():
            try:
                exp["audit"] = json.loads(audit_p.read_text(encoding="utf-8"))
            except Exception:
                pass
    agents["explore"] = exp

    # -- Synthesized tool codegen attempts (tools/) --------------------------
    tools_dir = logs_dir / "tools"
    if tools_dir.exists():
        groups: dict[str, dict[str, str]] = {}
        order: list[str] = []
        for f in sorted(tools_dir.iterdir()):
            if not f.is_file():
                continue
            name = f.name
            for suffix, kind in (
                ("_code.py", "code"), ("_prompt.txt", "prompt"),
                ("_agent_thinking.txt", "agent_thinking"), ("_stdout.txt", "stdout"),
                ("_agent_raw_stream.txt", "raw_stream"),
            ):
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    if stem not in groups:
                        groups[stem] = {}
                        order.append(stem)
                    try:
                        groups[stem][kind] = f.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    break
        for stem in order:
            if groups[stem]:
                agents["tool_codegen"].append({"stem": stem, "files": groups[stem]})

    # -- Langfuse trace bundle summary ---------------------------------------
    lf = logs_dir / "langfuse_trace.json"
    if lf.exists():
        try:
            b = json.loads(lf.read_text(encoding="utf-8"))
            agents["langfuse"] = {
                "trace_id": (b.get("trace") or {}).get("id"),
                "n_spans": len(b.get("spans") or []),
                "n_generations": len(b.get("generations") or []),
                "n_scores": len(b.get("scores") or []),
                "spans": [
                    {
                        "name": s.get("name"),
                        "stage": s.get("stage"),
                        "status": s.get("status", "completed"),
                    }
                    for s in (b.get("spans") or [])
                ],
            }
        except Exception:
            pass

    return agents


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
    elif (logs_dir / "run.json").exists():
        from evalrx.reporting.run_events import read_v2_events

        events = read_v2_events(logs_dir)

    all_run_starts = [e for e in events if e.get("event") == "run_start"]
    run_start = all_run_starts[-1] if all_run_starts else {}
    active_trace_id = run_start.get("trace_id")
    # Directories are commonly reused.  Keep a report internally consistent by
    # selecting the events from the latest run's trace only.
    if active_trace_id:
        scoped_events = [e for e in events if e.get("trace_id") == active_trace_id]
        # Keep compatibility with pre-trace legacy logs, where only run_start
        # may have a trace id (or none of the events do).
        if len(scoped_events) > 1:
            events = scoped_events

    def by_event(event: str) -> list[dict[str, Any]]:
        return [e for e in events if e.get("event") == event]

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
    protocol = run_start.get("protocol") or {}
    protocol_desc = (
        protocol.get("description") or ""
        if isinstance(protocol, dict)
        else str(protocol)
    )
    benchmark_name = clean_benchmark_name(protocol_desc, str(manifest_path or ""))
    # The run's own summary.json (written by run_pipeline beside logs/) names
    # the model and dataset the way the user typed them — "qwen3.5-2b",
    # "bbh_word_sorting". It beats both a stringified Model repr and the
    # keyword table above, which only knows a handful of benchmarks and
    # otherwise says "Evaluation Benchmark". The run directory's own name is
    # the last resort for the dataset, since run_all.sh names it after one.
    run_summary: dict[str, Any] = {}
    try:
        summary_path = logs_dir.parent / "summary.json"
        loaded = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        run_summary = loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        run_summary = {}
    if run_summary.get("model"):
        clean_model = str(run_summary["model"])
    if benchmark_name == "Evaluation Benchmark":
        folder = logs_dir.parent.name
        benchmark_name = (
            str(run_summary.get("dataset") or "")
            or (folder if folder not in {"outputs", "logs", ""} else "")
            or benchmark_name
        )

    # Pre-M1
    probe_searches = by_event("probe_search")
    pre_m1_ran = bool(probe_searches)
    pre_m1_cases = probe_searches[0].get("n_synthesized") if pre_m1_ran else 0

    # Probe Case Map for reverse-lookup of anomalies per sample
    probe_case_flags: dict[str, list[str]] = {}

    # M1: Measurements & Per-case probe drilldowns
    probes = by_event("probe")
    p0 = probes[-1] if probes else {}
    m1_cycle = int(p0.get("cycle", 0) or 0)
    analyzers = p0.get("analyzers") or p0.get("selected_analyzers") or []
    m1_duration = p0.get("duration_sec")

    # An agent-written probe ("generated:probe1") has no glossary entry; the
    # need it was written for is the only description that exists, and the
    # codegen event carries it verbatim.
    generated_need: dict[str, str] = {}
    for event in by_event("tool_codegen"):
        tool = str(event.get("tool_name") or "")
        if tool and event.get("need"):
            generated_need[tool] = str(event["need"])
    m1_results = []
    for name in analyzers:
        result_paths = p0.get("result_paths") or {}
        p = logs_dir / str(result_paths.get(name) or f"artifacts/c{m1_cycle}_{name}.result.json")
        findings, n = {}, None
        per_case_rows = []
        # V1 externalised each analyzer's result to artifacts/; RunLoggerV2
        # keeps the findings inline on the probe entry itself.
        inline = (p0.get("findings") or {}).get(name)
        if not p.exists() and isinstance(inline, dict):
            findings = inline
            n = findings.get("n_cases") or findings.get("n_scored")
            per_case_rows = findings.get("per_case") or []
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                findings = raw.get("findings") or {}
                n = findings.get("n_cases") or findings.get("n_scored")
                per_case_rows = findings.get("per_case") or []
                for pc in per_case_rows:
                    sid = str(pc.get("sample_id", ""))
                    if sid:
                        if sid not in probe_case_flags:
                            probe_case_flags[sid] = []
                        if pc.get("format_flip_rate", 0) > 0:
                            probe_case_flags[sid].append(f"Option Order Flip ({pc['format_flip_rate']:.0%})")
                        if pc.get("conf_logprob", 0) > 0.8 and pc.get("correct") == 0:
                            probe_case_flags[sid].append(f"Overconfident on Error ({pc['conf_logprob']:.0%})")
                        if pc.get("pass_at_k") == 1.0:
                            probe_case_flags[sid].append("Pass@5 Capable")
                        elif pc.get("pass_at_k") == 0.0:
                            probe_case_flags[sid].append("Pass@5 Zero-Coverage")
            except Exception:
                pass
        meta = ANALYZER_GLOSSARY.get(
            name, (name.replace("_", " ").title(), "Measures model behavior across this dimension", "Standard diagnostic probe.", [])
        )
        if name.startswith("generated:"):
            # No glossary can know a probe the agent wrote during this run: the
            # need it was written for is its question (first sentence up front,
            # the whole brief as the description), and its headline is whatever
            # scalar findings it reported, under their own names.
            need = generated_need.get(name.split(":", 1)[1], "")
            first = re.split(r"(?<=[.;])\s+", need.strip(), maxsplit=1)[0] if need else "Agent-written probe"
            if len(first) > 180:
                first = first[:177].rstrip() + "…"
            scalar = [
                (key.replace("_", " "), key, "pct" if key.endswith(("_rate", "_frac")) else "num")
                for key, value in findings.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and key not in {"n_cases", "n_scored", "n_graded", "n_samples"}
            ][:5]
            meta = (f"Agent-written probe · {name.split(':', 1)[1]}", first, need or "Agent-written probe.", scalar)
        headline = []
        for label, path, fmt in meta[3]:
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
            "description": meta[2],
            "n": n,
            "headline": headline,
            "findings": findings,
            "per_case": per_case_rows,
        })
    m1_results.sort(key=lambda r: -(r["n"] or 0))

    # M2: Explore & Screening
    analyses = by_event("analysis")
    a0 = analyses[-1] if analyses else {}
    m2_conclusion = a0.get("conclusion") or ""
    m2_narrative = a0.get("narrative") or ""
    m2_severity = a0.get("severity") or "medium"
    m2_duration = a0.get("duration_sec")

    m2_cycle = int(a0.get("cycle", m1_cycle) or 0)
    stats_path = logs_dir / f"artifacts/c{m2_cycle}_m2_stats_results.json"
    stats_ref = a0.get("stats_results")
    if isinstance(stats_ref, dict) and stats_ref.get("path"):
        stats_path = logs_dir / str(stats_ref["path"])
    try:
        raw_stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
    except (OSError, json.JSONDecodeError):
        raw_stats = []
    if not raw_stats and isinstance(stats_ref, list):
        # RunLoggerV2 keeps the M2 rows inline on the analysis entry rather
        # than externalising them to artifacts/.
        raw_stats = [row for row in stats_ref if isinstance(row, dict)]
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
    if not explore_data:
        # RunLoggerV2 records the explore step as an M2 event: observations and
        # caveats as sentences, figures as paths under logs/. Shape it the way
        # the explore report is shaped, so the M2 record renders either.
        explores = by_event("explore")
        e0 = explores[-1] if explores else {}
        takeaways = []
        for text in a0.get("findings") or []:
            if isinstance(text, str) and text.strip():
                takeaways.append({"title": "Screening finding", "plain_title": "Screening finding",
                                  "analysis": text.strip(), "chart_names": [], "table_names": []})
        for text in e0.get("observations") or []:
            if isinstance(text, str) and text.strip():
                head = text.strip().split(";")[0].split(". ")[0]
                takeaways.append({"title": head[:90], "plain_title": head[:90],
                                  "analysis": text.strip(), "chart_names": [], "table_names": []})
        if takeaways or e0:
            explore_data = {
                "takeaways": takeaways,
                "caveats": [c for c in e0.get("caveats") or [] if isinstance(c, str)],
                "observations": [o for o in e0.get("observations") or [] if isinstance(o, str)],
                "figures": [f for f in e0.get("figures") or [] if isinstance(f, str)],
                "adjudication": e0.get("adjudication") or {},
                "candidate_signals": [], "hypotheses": [],
            }

    # M3: Hypotheses
    diagnoses = by_event("diagnosis")
    dg = diagnoses[-1] if diagnoses else {}
    hypotheses = dg.get("hypotheses") or []
    if not hypotheses and (logs_dir / "report" / "hypotheses.json").exists():
        try:
            hypotheses = json.loads((logs_dir / "report" / "hypotheses.json").read_text())
        except Exception:
            pass

    # M4: Adjudication
    surgeries = by_event("surgery")
    m4_surgeries = [s for s in surgeries if s.get("module") == "m4" or s.get("adjudication")]
    m4_results_file = logs_dir / "report" / "m4_results.json"
    try:
        m4_results = json.loads(m4_results_file.read_text()) if m4_results_file.exists() else []
    except (OSError, json.JSONDecodeError):
        m4_results = []
    m4_event = m4_surgeries[-1] if m4_surgeries else {}
    if m4_surgeries:
        # The JSONL event is trace-scoped; a report artifact can be stale when
        # the same run directory is appended to later. One hypothesis can be
        # tested more than once across cycles — every hypothesis this run
        # actually adjudicated belongs here, not just the most recent verdict,
        # otherwise a 3-hypothesis M4 pass reports as if only 1 ran.
        m4_results = [{
            "hypothesis_id": event.get("hypothesis_id"),
            "hypothesis": event.get("hypothesis"),
            "status": event.get("status"),
            "effect_size": (event.get("evidence") or {}).get("m4_effect_size"),
            "confidence": event.get("confidence_score"),
            "verdict": (event.get("evidence") or {}).get("m4_verdict"),
            "evidence_grade": (event.get("evidence") or {}).get("m4_evidence_grade"),
            "protocol_consistent": (event.get("evidence") or {}).get("m4_protocol_consistent"),
        } for event in m4_surgeries]

    # M5-Surgery
    m5_surgeries = [s for s in surgeries if s.get("module") != "m4"]
    m5_surgery_ran = bool(m5_surgeries)

    # M5-Fix: Repair
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

    if not confirmed_fix and isinstance(best_fix, dict):
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
                        "probe_flags": probe_case_flags.get(cid, []),
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
                "probe_flags": probe_case_flags.get(cid, []),
            })

    order = {"fixed": 0, "broken": 1, "unchanged": 2, "untested": 3}
    cases.sort(key=lambda c: (order.get(c["status"], 9), c["id"]))

    data = {
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
            "version": run_start.get("evalrx_version", "0.2.0"),
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
            # The analysis step's own figures (V2 paths under logs/).
            "figures": [f for f in a0.get("figures") or [] if isinstance(f, str)],
        },
        "m3": {
            "hypotheses": hypotheses,
            # The proposer's own list, kept beside the (possibly reordered and
            # critic-annotated) accepted list. It is the only place a run
            # recorded before the log writer carried `plain_statement` on
            # `hypotheses` still has the judge's plain sentence, and the report
            # compiler joins the two on the statement to recover it.
            "proposed_hypotheses": dg.get("proposed_hypotheses") or [],
            "duration": dg.get("duration_sec"),
        },
        "m4": {
            "ran": bool(m4_surgeries or m4_results),
            "results": m4_results,
            "event": m4_event,
        },
        "m5_surgery": {
            "ran": m5_surgery_ran,
            "surgeries": m5_surgeries,
        },
        "m5_fix": {
            "ran": bool(fixes or confirmed_fix),
            "fixed": bool(f0.get("fixed") or confirmed_fix.get("fixed")),
            "selection": fix_attempts,
            "confirm": confirmed_fix,
            "best": best_fix,
            "prompt_template": (confirmed_fix.get("payload") or {}).get("prompt_template") or "",
        },
        "cases": cases,
        "agents": _extract_agent_layer(logs_dir, explore_dir),
        "logs_dir": logs_dir,
        "explore_dir": explore_dir,
        "example_dir": example_dir or logs_dir.parent,
    }
    # Keep the audience-first semantic layer independent from HTML.  The same
    # object can later drive a different renderer without reinterpreting logs.
    from evalrx.reporting.compiler import compile_reader_report

    data["reader_report"] = compile_reader_report(data).to_dict()
    return data


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
        cache_dir = Path("/tmp/evalrx_media_cache")
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


def esc(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def _json_for_script(value: Any) -> str:
    """Serialize untrusted values safely inside an inline ``<script>`` block."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def generate_html_report(data: dict[str, Any], figures: dict[str, str], audio_map: dict[str, str], image_map: dict[str, str]) -> str:
    run = data["run"]
    m1 = data["m1"]
    m2 = data["m2"]
    m3 = data["m3"]
    m4 = data["m4"]
    m5_s = data["m5_surgery"]
    m5_f = data["m5_fix"]
    cases = data["cases"]
    reader = data.get("reader_report") or {}

    n_total = run["n_cases"] or len(cases) or 1
    cfm = m5_f.get("confirm") or {}
    repair_effect = cfm.get("effect")
    e_val = cfm.get("e_value")
    n_fixed = cfm.get("n_fixed") or sum(1 for c in cases if c["status"] == "fixed")
    n_broken = cfm.get("n_broken") or sum(1 for c in cases if c["status"] == "broken")

    # M4 Verdict
    m4_status = "Skipped"
    m4_tone = "skip"
    if m4["ran"]:
        m4_r0 = m4["results"][0] if m4["results"] else {}
        verdict_status = str(m4_r0.get("status") or m4["event"].get("status") or "").lower()
        if verdict_status == "supported":
            m4_status = "Supported"
            m4_tone = "good"
        elif verdict_status == "refuted":
            m4_status = "Refuted"
            m4_tone = "warn"
        else:
            m4_status = "Inconclusive"
            m4_tone = "neutral"

    # M5 Status
    m5_status = "Not Tested"
    m5_tone = "skip"
    if m5_f["ran"]:
        if m5_f["fixed"] or (repair_effect is not None and repair_effect > 0):
            m5_status = f"+{repair_effect * 100:.1f}% Net Gain" if repair_effect is not None else "Repair Validated"
            m5_tone = "good"
        else:
            m5_status = "Inconclusive / Neutral"
            m5_tone = "warn"

    stats_sig = [s for s in m2["stats"] if s["reject"]]

    # Compact visual summary.  Values come only from the canonical run stages,
    # so this remains valid for any task that emits the standard artifacts.
    tiles_data = [
        ("Cases", f"{n_total:,}", "evaluated", "neutral"),
        (
            "Patterns",
            str(len(stats_sig)),
            "worth checking",
            "good" if stats_sig else "neutral",
        ),
        (
            "Independent check",
            m4_status,
            "new cases" if m4["ran"] else "not run",
            m4_tone,
        ),
        (
            "Repair result",
            f"{repair_effect * 100:+.2f}%" if repair_effect is not None else "—",
            f"{n_fixed} improved · {n_broken} worse" if (n_fixed or n_broken) else "not tested",
            m5_tone,
        ),
    ]

    tiles_html = "\n".join([
        f'<div class="kpi-card kpi-card--{tone}">'
        f'<div class="kpi-label">{esc(lab)}</div>'
        f'<div class="kpi-val">{esc(val)}</div>'
        f'<div class="kpi-sub">{note}</div></div>'
        for lab, val, note, tone in tiles_data
    ])

    reader_findings_html = "\n".join(
        f'''<article class="reader-finding">
          <div class="reader-finding-head"><h3>{esc(finding.get("title"))}</h3><span class="badge badge--neutral">{esc(finding.get("evidence_level"))}</span></div>
          <p>{esc(finding.get("summary"))}</p>
        </article>'''
        for finding in (reader.get("key_findings") or [])
    ) or '<p class="text-muted">No reader-ready findings are available for this run.</p>'
    reader_method = " ".join(str(step) for step in (reader.get("what_we_did") or []))
    reader_caveat = next(iter(reader.get("open_questions") or []), "")
    reader_next_step = next(iter(reader.get("next_steps") or []), "")
    setting_html = f'''<section class="setting-card">
      <div>
        <div class="setting-kicker">EvalRX · failure investigation</div>
        <h1 class="setting-title">Why did this model fail this evaluation?</h1>
        <p class="setting-copy">We evaluate <b>{esc(run["model"])}</b> on <b>{esc(run["benchmark_name"])}</b>, then trace a failure from targeted probes to a verified repair.</p>
      </div>
      <div class="setting-facts">
        <div class="setting-fact"><span class="setting-fact-label">Evaluation</span><span class="setting-fact-value">{n_total:,} cases</span></div>
        <div class="setting-fact"><span class="setting-fact-label">Investigation question</span><span class="setting-fact-value">{esc(reader.get("question") or "Find the mechanism behind observed failures.")}</span></div>
      </div>
    </section>'''
    run_map_html = "\n".join([
        f'<div class="run-node run-node--neutral"><span class="run-node-label">Start</span><span class="run-node-title">Failure observed</span><span class="run-node-value">{n_total:,} cases evaluated</span></div>',
        f'<div class="run-node run-node--good"><span class="run-node-label">M1 · Probe</span><span class="run-node-title">Characterize the failure</span><span class="run-node-value">{len(m1["analyzers"])} targeted probes</span></div>',
        f'<div class="run-node run-node--good"><span class="run-node-label">M2 · Analyze</span><span class="run-node-title">Find recurring structure</span><span class="run-node-value">{len(stats_sig)} patterns retained</span></div>',
        f'<div class="run-node run-node--{"good" if m3["hypotheses"] else "skip"}"><span class="run-node-label">M3 · Diagnose</span><span class="run-node-title">Propose a testable mechanism</span><span class="run-node-value">{len(m3["hypotheses"])} hypotheses</span></div>',
        f'<div class="run-node run-node--{m4_tone if m4["ran"] else "skip"}"><span class="run-node-label">M4 · Verify</span><span class="run-node-title">Check on unseen cases</span><span class="run-node-value">{esc(m4_status)}</span></div>',
        f'<div class="run-node run-node--{m5_tone}"><span class="run-node-label">M5 · Repair</span><span class="run-node-title">Beat the original baseline</span><span class="run-node-value">{esc(m5_status)}</span></div>',
    ])

    # ── Agent Trajectory layer ────────────────────────────────────────────
    agents = data.get("agents") or {}
    agent_judges = agents.get("judge_calls") or []
    agent_explore = agents.get("explore") or {}
    agent_codegen = agents.get("tool_codegen") or []
    agent_langfuse = agents.get("langfuse") or {}
    agents_n = len(agent_judges) + len(agent_codegen) + (1 if agent_explore else 0)
    agents_summary = f"{len(agent_judges) + len(agent_codegen) + (1 if agent_explore else 0)} Calls"

    judge_boxes = []
    for jc in agent_judges:
        judge_boxes.append(f"""
      <details class="collapsible-box">
        <summary>{esc(jc['stage'])} — <span class="mono" style="font-size:12px;">{esc(jc['stem'])}</span>
          <span class="text-muted" style="font-weight:400; font-size:12px;">&nbsp;(prompt {jc['prompt_chars']:,} chars · response {jc['raw_chars']:,} chars)</span></summary>
        <div class="content">
          <div class="section-subhead">Agent Input (Prompt)</div>
          <pre class="code-block" style="max-height:260px;">{esc(jc['prompt'])}</pre>
          <div class="section-subhead" style="margin-top:12px;">Agent Output (Response)</div>
          <pre class="code-block" style="max-height:260px;">{esc(jc['response']) or '<span class="text-muted">(empty)</span>'}</pre>
        </div>
      </details>""")
    judge_boxes_html = "\n".join(judge_boxes) or '<p class="text-muted">No judge calls were recorded for this run.</p>'

    explore_boxes = []
    if agent_explore:
        # Prefer the UNTRUNCATED streams when present; fall back to the
        # capped rendering.
        explore_stream_text = agent_explore.get("raw_streams") or agent_explore.get("raw_output") or ""
        n_attempts = explore_stream_text.count("--- attempt") or (1 if explore_stream_text else 0)
        if explore_stream_text:
            explore_boxes.append(f"""
      <details class="collapsible-box">
        <summary>Explore Coder Agent — Raw CLI Trajectory ({n_attempts} attempt{'s' if n_attempts != 1 else ''}{' · full untruncated streams' if agent_explore.get('raw_streams') else ' · truncated rendering'})</summary>
        <div class="content"><pre class="code-block" style="max-height:340px;">{esc(explore_stream_text)}</pre></div>
      </details>""")
        if agent_explore.get("code"):
            explore_boxes.append(f"""
      <details class="collapsible-box">
        <summary>Explore Synthesized Analysis Code (analysis.py)</summary>
        <div class="content"><pre class="code-block" style="max-height:340px;">{esc(agent_explore['code'])}</pre></div>
      </details>""")
        if agent_explore.get("audit"):
            explore_boxes.append(f"""
      <details class="collapsible-box">
        <summary>Explore Agent Audit (provider, commands, artifacts)</summary>
        <div class="content"><pre class="code-block" style="max-height:260px;">{esc(json.dumps(agent_explore['audit'], indent=2, ensure_ascii=False))}</pre></div>
      </details>""")
        if agent_explore.get("stdout"):
            explore_boxes.append(f"""
      <details class="collapsible-box">
        <summary>Explore Execution stdout</summary>
        <div class="content"><pre class="code-block" style="max-height:260px;">{esc(agent_explore['stdout'])}</pre></div>
      </details>""")
    explore_boxes_html = "\n".join(explore_boxes) or '<p class="text-muted">The explore step did not run or left no trajectory for this run.</p>'

    codegen_boxes = []
    for cg in agent_codegen:
        files = cg.get("files") or {}
        inner = []
        for kind, label in (
            ("prompt", "Agent Input (Task Prompt)"),
            ("raw_stream", "Agent Output (Full Raw CLI Trajectory)"),
            ("agent_thinking", "Agent Output (Raw CLI Trajectory)"),
            ("code", "Synthesized Code"),
            ("stdout", "Validation stdout"),
        ):
            if files.get(kind):
                inner.append(
                    f'<div class="section-subhead" style="margin-top:10px;">{label}</div>'
                    f'<pre class="code-block" style="max-height:260px;">{esc(files[kind])}</pre>'
                )
        codegen_boxes.append(f"""
      <details class="collapsible-box">
        <summary>Tool Codegen — <span class="mono" style="font-size:12px;">{esc(cg['stem'])}</span>
          <span class="text-muted" style="font-weight:400; font-size:12px;">&nbsp;({len(files)} artifacts)</span></summary>
        <div class="content">{''.join(inner)}</div>
      </details>""")
    codegen_boxes_html = "\n".join(codegen_boxes) or '<p class="text-muted">No tools were synthesized in this run.</p>'

    if agent_langfuse:
        span_rows = "".join(
            f'<tr><td class="mono font-semibold" style="font-size:12px;">{esc(s.get("stage") or "")}</td>'
            f'<td style="font-size:12.5px;">{esc(s.get("name") or "")}</td>'
            f'<td><span class="badge badge--{"good" if s.get("status", "completed") == "completed" else "mute"}">{esc(s.get("status", "completed"))}</span></td></tr>'
            for s in agent_langfuse.get("spans") or []
        )
        langfuse_html = f"""
      <details class="collapsible-box" open>
        <summary>Langfuse Trace Bundle — {agent_langfuse.get('n_spans', 0)} spans · {agent_langfuse.get('n_generations', 0)} generations · {agent_langfuse.get('n_scores', 0)} scores</summary>
        <div class="content">
          <p class="text-muted" style="font-size:12.5px;">Trace id <span class="mono">{esc(agent_langfuse.get('trace_id') or '')}</span> — the same id used by <span class="mono">run_log.jsonl</span> and (when live-synced) the Langfuse dashboard. Every span below is also mirrored there.</p>
          <div class="table-wrapper" style="max-height:300px; overflow-y:auto;">
            <table class="data-table"><thead><tr><th>Stage</th><th>Span</th><th>Status</th></tr></thead><tbody>{span_rows}</tbody></table>
          </div>
        </div>
      </details>"""
    else:
        langfuse_html = '<p class="text-muted">No Langfuse trace bundle (langfuse_trace.json) was written for this run.</p>'

    tab_items = [
        ("tab_overview", "START", "What this report says", reader.get("confidence", "Summary"), "good"),
        ("tab_m1", "1", "What we checked", f"{len(m1['analyzers'])} checks", "neutral"),
        ("tab_m2", "2", "What we found", f"{len(stats_sig)} leads" if stats_sig else "Patterns", "neutral"),
        ("tab_m3", "3", "Possible explanation", "Not run" if not m3["hypotheses"] else "To test", "skip" if not m3["hypotheses"] else "neutral"),
        ("tab_m4", "4", "Independent check", "Not run" if not m4["ran"] else m4_status, "skip" if not m4["ran"] else m4_tone),
        ("tab_m5_fix", "5", "Repair attempt", m5_status, m5_tone),
        ("tab_case_book", "EXAMPLES", "Listen to real cases", f"{len(cases)} cases", "neutral"),
        ("tab_agents", "DETAILS", "Research details", f"{agents_summary}", "neutral" if agents_n else "skip"),
    ]
    if data["pre_m1"]["ran"]:
        tab_items.insert(1, ("tab_pre_m1", "PREP", "Extra test cases", "Completed", "neutral"))
    if m5_s["ran"]:
        tab_items.insert(-2, ("tab_m5_surgery", "RESEARCH", "Internal intervention", "Completed", "neutral"))

    tabs_html = "\n".join([
        f'<button type="button" class="tab-btn {"is-active" if tid == "tab_overview" else ""}" data-tab="{tid}">'
        f'<span class="tab-code">{esc(code)}</span>'
        f'<span class="tab-label">{esc(label)}</span>'
        f'<span class="tab-badge badge--{tone}">{esc(badge)}</span>'
        f'</button>'
        for tid, code, label, badge, tone in tab_items
    ])

    cov_max = max((r["n"] or 0) for r in m1["results"]) if m1["results"] else 1
    cov_max = max(cov_max, 1)
    m1_blocks = []
    for idx, r in enumerate(m1["results"], 1):
        n = r["n"] or 0
        hl_items = []
        for h in r["headline"]:
            if h["value"] is not None:
                hl_items.append(f'<div class="hl-pill"><span class="hl-v">{esc(h["value"])}</span><span class="hl-l">{esc(h["label"])}</span></div>')
            else:
                hl_items.append(f'<div class="hl-pill hl-pill--none"><span class="hl-v">—</span><span class="hl-l">{esc(h["label"])}</span></div>')
        hl_html = "".join(hl_items) or '<span class="text-muted">No scalar headlines</span>'

        pct_val = (n / n_total) if n_total > 0 else 0
        cov_bar = (
            f'<div class="bar-track"><span class="bar-fill" style="width:{n / cov_max * 100:.1f}%"></span></div>'
            f'<span class="barpct">{pct_val:.0%} coverage ({n:,} cases scored)</span>'
        )

        # Build Per-Case Drilldown Table
        per_case_rows = r.get("per_case") or []
        case_rows_html = []
        for pc in per_case_rows[:30]:
            sid = str(pc.get("sample_id", "case"))
            metrics_summary = ", ".join(
                f"<b>{esc(k)}</b>: {esc(v)}" for k, v in pc.items() if k != "sample_id"
            )
            sid_js = esc(json.dumps(sid))
            case_rows_html.append(
                f'<tr><td class="mono font-bold" style="font-size:12px; color:var(--brand);">{esc(sid[:12])}</td>'
                f'<td style="font-size:12.5px;">{metrics_summary}</td>'
                f'<td><button type="button" class="mini-btn" onclick="openCaseModal({sid_js})">Inspect Case & Audio</button></td></tr>'
            )
        per_case_table = (
            f'<div class="table-wrapper" style="margin-top:12px; max-height:280px; overflow-y:auto;">'
            f'<table class="data-table"><thead><tr><th>Sample ID</th><th>Probe Model Output & Decision Metrics</th><th>Action</th></tr></thead>'
            f'<tbody>{"".join(case_rows_html)}</tbody></table></div>'
            if case_rows_html else '<p class="text-muted" style="font-size:12px; margin-top:8px;">No per-case drilldown records for this probe.</p>'
        )

        findings_json_str = esc(json.dumps(r.get("findings", {}), indent=2, ensure_ascii=False))

        m1_blocks.append(f"""
        <div class="probe-card">
          <div class="probe-header" onclick="toggleProbe('probe_detail_{idx}')">
            <div style="flex:1;">
              <div class="probe-title">{esc(r["display_name"])} <span class="probe-code mono">{esc(r["name"])}</span></div>
              <div class="probe-q">{esc(r["question"])}</div>
            </div>
            <div style="width:160px; margin:0 16px;">{cov_bar}</div>
            <div class="hl-wrap" style="margin-right:16px;">{hl_html}</div>
            <div class="expand-icon" id="icon_probe_detail_{idx}">▼</div>
          </div>
          <div class="probe-body" id="probe_detail_{idx}" style="display:none;">
            <div class="probe-desc-box">
              <div class="section-subhead">Clinical Methodology & Rationale</div>
              <p>{esc(r.get("description", ""))}</p>
            </div>
            <div style="margin-top:14px;">
              <div class="section-subhead">Sample-Level Probe Execution & Model Decisions ({len(per_case_rows)} samples recorded)</div>
              {per_case_table}
            </div>
            <details class="collapsible-box" style="margin-top:14px;">
              <summary>View Complete Probe Findings JSON & Attention Parameters</summary>
              <div class="content"><pre class="code-block">{findings_json_str}</pre></div>
            </details>
          </div>
        </div>""")

    m1_cards_html = "\\n".join(m1_blocks) if m1_blocks else "<p class='text-muted'>No probes recorded.</p>"

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
    # A small, deterministic visual sample on the landing page.  Do not infer
    # which plot is "best" from its filename; keep the first two unique
    # generated artifacts in their stable order and leave the full set in M2.
    hero_fig_blocks = []
    seen_figure_data: set[str] = set()
    for k, v in figures.items():
        if k in ("test", "m2_effects") or v in seen_figure_data:
            continue
        seen_figure_data.add(v)
        hero_fig_blocks.append(
            f'<div class="chart-card"><div class="chart-head"><span class="chart-title">{esc(k.replace("_", " ").title())}</span></div>'
            f'<img src="{v}" alt="{esc(k)}" loading="lazy"></div>'
        )
        if len(hero_fig_blocks) == 2:
            break
    hero_figs_html = "\n".join(hero_fig_blocks)

    sel_rows = []
    max_eff = max([abs(s.get("effect") or 0) for s in m5_f["selection"]] + [0.01])
    for s in m5_f["selection"]:
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

    meta_chips = "".join([f"<span class='tag-pill'>{esc(m)}</span>" for m in [
        run["benchmark_name"],
        f"{n_total:,} Cases",
        f"{run['cycles']} Cycles",
        f"{len(m1['analyzers'])} Probes",
        f"{run['duration_sec']:.0f}s Duration" if run['duration_sec'] else "Completed",
        f"EvalRX v{run['version']}",
    ]])

    pre_m1_desc = f"Active probe search generated {data['pre_m1']['n_cases']} synthetic test cases to probe failure mechanisms." if data["pre_m1"]["ran"] else "Testing ran directly on fixed benchmark cases (automated Pre-M1 adversarial probe synthesis was not configured)."
    pre_m1_sub = f"{data['pre_m1']['n_cases']} Probes Synthesized" if data["pre_m1"]["ran"] else "Standard Benchmark"
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
    m4_tone_cls = "callout--good" if m4_tone == "good" else ("callout--warn" if m4_tone == "warn" else "callout--neutral")
    if m4_tone == "good":
        m4_plain_text = f"<b>Verdict: {m4_status}</b>. Re-evaluated probe signals on an independent held-out split. Observed direction matched the predicted failure mechanism with statistical significance."
    elif m4_tone == "warn":
        m4_plain_text = f"<b>Verdict: {m4_status}</b>. When re-tested on independent validation cases, the empirical data contradicted the AI Doctor's hypothesis."
    else:
        m4_plain_text = "Independent M4 blind validation was not executed for this run."

    m4_detail_html = ""
    if m4["results"] or m4["event"]:
        res0 = m4["results"][0] if m4["results"] else {}
        eff_size = res0.get("effect_size", 0)
        eff_str = f"{eff_size:+.3f}" if isinstance(eff_size, (int, float)) else "—"
        conf_val = res0.get("confidence", 0)
        conf_str = f"{conf_val:.2f}" if isinstance(conf_val, (int, float)) else "—"
        ev_grade = esc(res0.get("evidence_grade") or res0.get("evidence", {}).get("evidence_grade") or "—")
        verdict_raw = esc(res0.get("verdict") or m4["event"].get("verdict") or json.dumps(m4["event"], indent=2, ensure_ascii=False))
        status_cls = "text-good font-bold" if m4_tone == "good" else "text-bad font-bold"

        m4_detail_html = f"""
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Verdict</div><div class="kpi-val {status_cls}">{esc(m4_status)}</div></div>
          <div class="kpi-card"><div class="kpi-label">Validation Effect</div><div class="kpi-val">{eff_str}</div></div>
          <div class="kpi-card"><div class="kpi-label">Confidence Score</div><div class="kpi-val">{conf_str}</div></div>
          <div class="kpi-card"><div class="kpi-label">Evidence Grade</div><div class="kpi-val">{ev_grade}</div></div>
        </div>
        <pre class="code-block"><b>Adjudication Audit Log:</b>\n{verdict_raw}</pre>"""

    m5_s_desc = f"Executed {len(m5_s['surgeries'])} causal model interventions / ablations." if m5_s["ran"] else "Focused on black-box prompt and scaffold optimizations (white-box surgery was not invoked)."
    m5_s_sub = "Executed" if m5_s["ran"] else "Skipped"

    m5_f_tone_cls = "callout--good" if m5_tone == "good" else "callout--neutral"
    if repair_effect is not None and repair_effect > 0:
        cand_name = esc(cfm.get("name", "selected_fix"))
        e_str = f"e = {e_val:,.0f}" if e_val else "Significant"
        m5_f_plain = f"Successfully confirmed repair strategy <b><code>{cand_name}</code></b>! Produced a <b>{repair_effect * 100:+.2f}% net accuracy gain</b> on held-out test data (<b>{n_fixed}</b> cured vs <b>{n_broken}</b> broken, evidence strength <b>{e_str}</b>)."
    else:
        m5_f_plain = "Screened repair candidates across tiers, but no strategy met the statistical threshold for a significant net gain."

    tmpl_code = esc(m5_f["prompt_template"] or cfm.get("summary") or "")
    m5_f_tmpl_html = ""
    if tmpl_code:
        m5_f_tmpl_html = f"""
        <div style="margin-top:16px;">
          <div class="section-subhead">Winning Prompt / Repair Patch</div>
          <pre class="code-block">{tmpl_code}</pre>
        </div>"""

    cases_json = _json_for_script(cases)
    audio_json = _json_for_script(audio_map)
    images_json = _json_for_script(image_map)

    html_lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>EvalRX · {esc(run["model"])} Diagnostic Report</title>',
        '<style>',
        ':root {',
        '  --bg: #0b0f19;',
        '  --surface: #111827;',
        '  --surface-raised: #1f2937;',
        '  --surface-hover: #374151;',
        '  --border: #374151;',
        '  --border-subtle: #242d3d;',
        '  --text: #f9fafb;',
        '  --text-muted: #9ca3af;',
        '  --text-sub: #6b7280;',
        '  --brand: #6366f1;',
        '  --brand-soft: rgba(99, 102, 241, 0.12);',
        '  --brand-border: rgba(99, 102, 241, 0.35);',
        '  --good: #10b981;',
        '  --good-soft: rgba(16, 185, 129, 0.12);',
        '  --good-border: rgba(16, 185, 129, 0.35);',
        '  --warn: #f59e0b;',
        '  --warn-soft: rgba(245, 158, 11, 0.12);',
        '  --warn-border: rgba(245, 158, 11, 0.35);',
        '  --bad: #ef4444;',
        '  --bad-soft: rgba(239, 68, 68, 0.12);',
        '  --bad-border: rgba(239, 68, 68, 0.35);',
        '  --radius: 10px;',
        '  --radius-sm: 6px;',
        '  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;',
        '  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;',
        '}',
        '* { box-sizing: border-box; margin: 0; padding: 0; }',
        'body { background: var(--bg); color: var(--text); font-family: var(--font-sans); font-size: 14.5px; line-height: 1.6; -webkit-font-smoothing: antialiased; }',
        '.container { max-width: 1240px; margin: 0 auto; padding: 0 24px; }',
        'header.app-header { border-bottom: 1px solid var(--border-subtle); background: rgba(17, 24, 39, 0.95); backdrop-filter: blur(12px); position: sticky; top: 0; z-index: 100; }',
        '.header-inner { padding: 16px 0; display: flex; justify-content: space-between; align-items: center; gap: 16px; }',
        '.brand-group { display: flex; align-items: center; gap: 14px; }',
        '.brand-icon { width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, var(--brand), #8b5cf6); display: flex; align-items: center; justify-content: center; color: white; font-size: 16px; font-weight: 700; }',
        '.brand-titles { display: flex; flex-direction: column; }',
        '.brand-model { font-size: 17px; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }',
        '.brand-benchmark { font-size: 12.5px; color: var(--text-muted); }',
        '.header-badges { display: flex; gap: 8px; flex-wrap: wrap; }',
        'nav.tabs-nav { background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 65px; z-index: 90; }',
        '.tabs-scroll { display: flex; gap: 6px; overflow-x: auto; padding: 10px 0; scrollbar-width: none; }',
        '.tabs-scroll::-webkit-scrollbar { display: none; }',
        '.tab-btn { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 8px; border: 1px solid transparent; background: transparent; color: var(--text-muted); font-family: var(--font-sans); font-size: 13.5px; font-weight: 500; cursor: pointer; white-space: nowrap; transition: all 0.15s ease; }',
        '.tab-btn:hover { background: var(--surface-raised); color: var(--text); }',
        '.tab-btn.is-active { background: var(--surface-raised); border-color: var(--border); color: #ffffff; font-weight: 600; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4); }',
        '.tab-code { font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: var(--brand); }',
        '.tab-badge { font-family: var(--font-mono); font-size: 10.5px; padding: 2px 6px; border-radius: 99px; }',
        '.tab-pane { display: none; padding: 32px 0 80px; animation: fadeIn 0.15s ease-in-out; }',
        '.tab-pane.is-active { display: block; }',
        '@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }',
        '.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25); }',
        '.card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; padding-bottom: 16px; margin-bottom: 20px; border-bottom: 1px solid var(--border-subtle); }',
        '.card-tag { font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: var(--brand); background: var(--brand-soft); border: 1px solid var(--brand-border); padding: 3px 8px; border-radius: 4px; }',
        '.card-title { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; color: #ffffff; }',
        '.card-subtext { flex: 1 1 100%; font-size: 13.5px; color: var(--text-muted); }',
        '.story-box { background: linear-gradient(180deg, rgba(31, 41, 55, 0.6) 0%, rgba(17, 24, 39, 0.9) 100%); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px 26px; margin-bottom: 24px; display: flex; flex-direction: column; gap: 10px; }',
        '.story-box p { font-size: 15px; color: var(--text); line-height: 1.6; }',
        '.story-box b { color: #ffffff; }',
        '.setting-card { display:grid; grid-template-columns: minmax(0, 1.6fr) minmax(220px, .8fr); gap:22px; border:1px solid var(--brand-border); background:linear-gradient(135deg, var(--brand-soft), transparent 62%), var(--surface); border-radius:var(--radius); padding:26px; margin-bottom:22px; }',
        '.setting-kicker { color:var(--brand); font-family:var(--font-mono); font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }',
        '.setting-title { color:#fff; font-size:clamp(24px, 3vw, 34px); line-height:1.12; letter-spacing:-.03em; margin:7px 0 10px; }',
        '.setting-copy { color:var(--text-muted); max-width:760px; font-size:15px; }',
        '.setting-facts { display:grid; gap:9px; align-content:center; }',
        '.setting-fact { border-left:2px solid var(--brand); padding:4px 0 4px 11px; }',
        '.setting-fact-label { display:block; color:var(--text-sub); font-size:10px; letter-spacing:.07em; text-transform:uppercase; }',
        '.setting-fact-value { display:block; color:var(--text); font-weight:600; font-size:13px; overflow-wrap:anywhere; }',
        '.reader-headline { font-size: clamp(28px, 4vw, 42px); line-height: 1.12; margin: 0 0 12px; letter-spacing: -0.035em; }',
        '.reader-question { color: var(--text-muted); font-size: 16px; line-height: 1.6; margin: 0; }',
        '.reader-section { margin: 30px 0; }',
        '.reader-section h2 { margin: 0 0 12px; font-size: 22px; letter-spacing: -0.02em; }',
        '.reader-list { margin: 0; padding-left: 20px; color: var(--text-muted); line-height: 1.75; }',
        '.reader-findings { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }',
        '.reader-finding { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-raised); padding: 20px; position: relative; overflow: hidden; }',
        '.reader-finding::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--brand); }',
        '.reader-finding-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }',
        '.reader-finding h3 { font-size: 17px; line-height: 1.35; margin: 0; }',
        '.reader-finding p { color: var(--text-muted); line-height: 1.55; font-size: 14px; }',
        '.reader-finding .reader-why { color: var(--text); }',
        '.journey-title { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin: 28px 0 10px; }',
        '.journey-title h2 { font-size:20px; letter-spacing:-.02em; }',
        '.run-map { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; margin: 12px 0 10px; }',
        '.run-node { position: relative; min-height: 112px; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 15px; background: var(--surface); }',
        '.run-node:not(:last-child)::after { content: "→"; position: absolute; right: -13px; top: 42px; z-index: 2; color: var(--text-sub); font-size: 18px; }',
        '.run-node--good { border-color: var(--good-border); }',
        '.run-node--neutral { border-color: var(--brand-border); }',
        '.run-node--skip { opacity: .62; }',
        '.run-node-label { display: block; color: var(--brand); font-size: 11px; font-weight:700; text-transform: uppercase; letter-spacing: .08em; }',
        '.run-node-title { display:block; margin-top:5px; color:#fff; font-size:14px; font-weight:700; line-height:1.2; }',
        '.run-node-value { display: block; margin-top: 8px; color: var(--text-muted); font-size:12px; line-height: 1.25; }',
        '.journey-loop { color:var(--text-sub); font-size:12px; text-align:right; margin-bottom:26px; }',
        '.hero-charts { margin: 28px 0; }',
        '.hero-charts .chart-card { min-height: 240px; }',
        '@media (max-width: 780px) { .setting-card { grid-template-columns:1fr; } .run-map { grid-template-columns: repeat(2, minmax(0, 1fr)); } .run-node:not(:last-child)::after { display: none; } }',
        '.reader-finding .reader-limit { font-size: 12.5px; color: var(--text-sub); margin-bottom: 0; }',
        '.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 24px; }',
        '.kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px 20px; display: flex; flex-direction: column; gap: 5px; }',
        '.kpi-label { font-size: 11.5px; font-family: var(--font-mono); text-transform: uppercase; color: var(--text-sub); letter-spacing: 0.05em; }',
        '.kpi-val { font-size: 26px; font-weight: 700; font-family: var(--font-mono); letter-spacing: -0.02em; }',
        '.kpi-sub { font-size: 12px; color: var(--text-muted); }',
        '.kpi-card--good { border-color: var(--good-border); } .kpi-card--good .kpi-val { color: var(--good); }',
        '.kpi-card--warn { border-color: var(--warn-border); } .kpi-card--warn .kpi-val { color: var(--warn); }',
        '.kpi-card--bad { border-color: var(--bad-border); } .kpi-card--bad .kpi-val { color: var(--bad); }',
        '.callout { padding: 16px 20px; border-radius: var(--radius-sm); background: var(--surface-raised); border-left: 4px solid var(--border); font-size: 14px; color: var(--text); margin-bottom: 20px; }',
        '.callout--accent { border-left-color: var(--brand); background: var(--brand-soft); }',
        '.callout--good { border-left-color: var(--good); background: var(--good-soft); }',
        '.callout--warn { border-left-color: var(--warn); background: var(--warn-soft); }',
        '.callout b { color: #ffffff; }',
        '.table-wrapper { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius-sm); margin-bottom: 20px; }',
        'table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }',
        'table.data-table th { font-family: var(--font-mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-sub); background: var(--surface-raised); padding: 11px 14px; border-bottom: 1px solid var(--border); }',
        'table.data-table td { padding: 11px 14px; border-bottom: 1px solid var(--border-subtle); vertical-align: middle; }',
        'table.data-table tr:hover td { background: rgba(255, 255, 255, 0.02); }',
        '.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--font-mono); }',
        '.mono { font-family: var(--font-mono); font-size: 0.92em; }',
        '.font-semibold { font-weight: 600; }',
        '.font-bold { font-weight: 700; }',
        '.text-muted { color: var(--text-muted); }',
        '.text-good { color: var(--good); }',
        '.text-warn { color: var(--warn); }',
        '.text-bad { color: var(--bad); }',
        '.badge { display: inline-flex; align-items: center; font-family: var(--font-mono); font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 99px; border: 1px solid transparent; }',
        '.badge--brand { color: #818cf8; background: var(--brand-soft); border-color: var(--brand-border); }',
        '.badge--good { color: var(--good); background: var(--good-soft); border-color: var(--good-border); }',
        '.badge--warn { color: var(--warn); background: var(--warn-soft); border-color: var(--warn-border); }',
        '.badge--bad { color: var(--bad); background: var(--bad-soft); border-color: var(--bad-border); }',
        '.badge--mute { color: var(--text-muted); background: var(--surface-raised); border-color: var(--border); }',
        '.tag-pill { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); background: var(--surface-raised); border: 1px solid var(--border); padding: 4px 12px; border-radius: 99px; }',
        '.probe-card { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface); margin-bottom: 16px; overflow: hidden; }',
        '.probe-header { padding: 16px 20px; display: flex; align-items: center; cursor: pointer; transition: background 0.15s ease; user-select: none; }',
        '.probe-header:hover { background: var(--surface-raised); }',
        '.probe-title { font-weight: 700; font-size: 15px; color: #ffffff; display: flex; align-items: center; }',
        '.probe-code { color: var(--brand); font-size: 11.5px; margin-left: 8px; }',
        '.probe-q { font-size: 13px; color: var(--text-muted); margin-top: 3px; }',
        '.probe-body { padding: 20px 24px; background: rgba(0, 0, 0, 0.2); border-top: 1px solid var(--border-subtle); }',
        '.probe-desc-box { background: var(--surface-raised); padding: 14px 18px; border-radius: var(--radius-sm); border-left: 3px solid var(--brand); font-size: 13.5px; }',
        '.expand-icon { font-size: 12px; color: var(--text-sub); transition: transform 0.2s ease; margin-left: 12px; }',
        '.expand-icon.is-open { transform: rotate(180deg); }',
        '.mini-btn { padding: 4px 10px; font-size: 11px; font-family: var(--font-mono); border-radius: 4px; border: 1px solid var(--border); background: var(--surface-raised); color: var(--text); cursor: pointer; }',
        '.mini-btn:hover { background: var(--brand); color: white; border-color: var(--brand); }',
        '.bar-track { height: 6px; border-radius: 3px; background: var(--surface-raised); overflow: hidden; margin-bottom: 3px; }',
        '.bar-fill { display: block; height: 100%; background: var(--brand); border-radius: 3px; }',
        '.barpct { font-family: var(--font-mono); font-size: 11px; color: var(--text-sub); }',
        '.hl-wrap { display: flex; flex-direction: column; gap: 4px; }',
        '.hl-pill { display: inline-flex; align-items: baseline; gap: 6px; font-size: 12px; }',
        '.hl-v { font-family: var(--font-mono); font-weight: 700; color: var(--text); min-width: 44px; text-align: right; }',
        '.hl-l { color: var(--text-muted); font-size: 11.5px; }',
        '.hl-pill--none .hl-v { color: var(--text-sub); }',
        '.diverge-cell { position: relative; width: 120px; height: 20px; }',
        '.diverge-axis { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--border); }',
        '.diverge-bar { position: absolute; top: 6px; height: 8px; border-radius: 2px; }',
        '.diverge-bar--pos { left: 50%; background: var(--good); }',
        '.diverge-bar--neg { right: 50%; background: var(--bad); }',
        '.eff-cell--pos { color: var(--good); font-weight: 700; }',
        '.eff-cell--neg { color: var(--bad); font-weight: 700; }',
        '.hypothesis-card { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-raised); padding: 18px 22px; display: flex; flex-direction: column; gap: 12px; margin-bottom: 14px; }',
        '.hyp-head { display: flex; justify-content: space-between; align-items: center; }',
        '.hyp-statement { font-size: 16px; font-weight: 600; color: #ffffff; line-height: 1.45; }',
        '.card-detail { font-size: 13px; color: var(--text-muted); }',
        '.card-detail .label { font-weight: 600; color: var(--text); margin-right: 6px; }',
        '.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }',
        '.chart-card { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-raised); overflow: hidden; cursor: pointer; }',
        '.chart-head { padding: 11px 16px; background: rgba(0, 0, 0, 0.25); border-bottom: 1px solid var(--border-subtle); }',
        '.chart-title { font-size: 13.5px; font-weight: 600; color: var(--text); }',
        '.chart-card img { display: block; width: 100%; height: auto; transition: transform 0.2s ease; }',
        '.chart-card:hover img { transform: scale(1.02); }',
        'details.collapsible-box { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface); margin-bottom: 16px; }',
        'details.collapsible-box summary { padding: 13px 18px; cursor: pointer; font-weight: 600; font-size: 14px; color: var(--text-muted); user-select: none; }',
        'details.collapsible-box summary:hover { color: var(--text); }',
        'details.collapsible-box .content { padding: 0 18px 18px; }',
        'pre.code-block { background: #06090e; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 16px; font-family: var(--font-mono); font-size: 12.5px; line-height: 1.55; color: #e2e8f0; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }',
        '.section-subhead { font-size: 12px; font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-sub); margin-bottom: 8px; }',
        '.case-controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 20px; }',
        '.filter-btn-group { display: flex; gap: 6px; background: var(--surface-raised); padding: 4px; border-radius: 99px; border: 1px solid var(--border); }',
        '.filter-btn { font-family: var(--font-mono); font-size: 12.5px; padding: 6px 14px; border-radius: 99px; border: none; background: transparent; color: var(--text-muted); cursor: pointer; transition: all 0.15s ease; }',
        '.filter-btn:hover { color: var(--text); }',
        '.filter-btn.is-active { background: var(--brand); color: white; font-weight: 600; }',
        '.search-box { margin-left: auto; padding: 8px 16px; border-radius: 99px; border: 1px solid var(--border); background: var(--surface-raised); color: var(--text); font-size: 13.5px; outline: none; width: 280px; }',
        '.search-box:focus { border-color: var(--brand); }',
        '.case-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }',
        '.case-item { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-raised); display: flex; flex-direction: column; overflow: hidden; }',
        '.case-item-header { padding: 11px 16px; background: rgba(0, 0, 0, 0.3); border-bottom: 1px solid var(--border-subtle); display: flex; align-items: center; gap: 8px; }',
        '.case-dur { margin-left: auto; font-family: var(--font-mono); font-size: 11px; color: var(--text-sub); }',
        '.case-q { padding: 16px; font-size: 14.5px; font-weight: 600; color: var(--text); line-height: 1.45; }',
        '.case-media-box { padding: 0 16px 12px; }',
        '.case-media-box audio { width: 100%; height: 34px; }',
        '.case-media-box img { width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px; border: 1px solid var(--border); }',
        '.case-choices { list-style: none; padding: 0 16px 14px; display: flex; flex-direction: column; gap: 6px; }',
        '.case-choices li { font-size: 13px; padding: 7px 12px; border-radius: 6px; background: var(--surface); border: 1px solid var(--border-subtle); display: flex; gap: 8px; align-items: baseline; }',
        '.case-choices li.is-gold { background: var(--good-soft); border-color: var(--good-border); color: #ffffff; font-weight: 600; }',
        '.case-item-foot { margin-top: auto; padding: 11px 16px; background: rgba(0, 0, 0, 0.3); border-top: 1px solid var(--border-subtle); display: flex; align-items: center; gap: 8px; font-size: 12.5px; }',
        '.case-id-tag { margin-left: auto; font-family: var(--font-mono); font-size: 11px; color: var(--text-sub); }',
        '.case-probe-tags { padding: 0 16px 10px; display: flex; flex-wrap: wrap; gap: 4px; }',
        '.case-probe-tag { font-family: var(--font-mono); font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--brand-soft); color: var(--brand); border: 1px solid var(--brand-border); }',
        '.modal-backdrop { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(8px); z-index: 1000; display: none; align-items: center; justify-content: center; padding: 24px; }',
        '.modal-backdrop.is-open { display: flex; animation: fadeIn 0.15s ease; }',
        '.modal-content { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); max-width: 820px; width: 100%; max-height: 90vh; overflow-y: auto; padding: 28px; position: relative; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6); }',
        '.modal-close { position: absolute; top: 16px; right: 16px; background: transparent; border: none; font-size: 22px; color: var(--text-muted); cursor: pointer; line-height: 1; padding: 4px 8px; }',
        '.modal-close:hover { color: #ffffff; }',
        'footer.app-footer { border-top: 1px solid var(--border-subtle); padding: 32px 0 48px; color: var(--text-sub); font-size: 13px; }',
        '</style>',
        '</head>',
        '<body>',
        '<header class="app-header">',
        '  <div class="container header-inner">',
        '    <div class="brand-group">',
        '      <div class="brand-icon">⚡</div>',
        '      <div class="brand-titles">',
        f'        <span class="brand-model">{esc(run["model"])}</span>',
        f'        <span class="brand-benchmark">{esc(run["benchmark_name"])}</span>',
        '      </div>',
        '    </div>',
        f'    <div class="header-badges">{meta_chips}</div>',
        '  </div>',
        '</header>',
        '<nav class="tabs-nav">',
        '  <div class="container">',
        f'    <div class="tabs-scroll">{tabs_html}</div>',
        '  </div>',
        '</nav>',
        '<div class="container">',
        '  <div class="tab-pane is-active" id="tab_overview">',
        f'    {setting_html}',
        '    <div class="story-box">',
        f'      <h1 class="reader-headline">{esc(reader.get("headline") or "What this report says")}</h1>',
        f'      <p class="reader-question"><b>Short answer:</b> {esc(reader.get("answer") or "")}</p>',
        f'      <p class="reader-question">{esc(reader_method)}</p>',
        '    </div>',
        f'    <div class="kpi-grid">{tiles_html}</div>',
        '    <div class="journey-title"><h2>Failure-to-fix journey</h2><span class="text-muted">Each card is this run’s actual state</span></div>',
        f'    <div class="run-map">{run_map_html}</div>',
        '    <div class="journey-loop">If verification refutes the mechanism or repair misses the baseline, the next cycle returns to M1.</div>',
        '    <section class="reader-section">',
        '      <h2>Key findings</h2>',
        f'      <div class="reader-findings">{reader_findings_html}</div>',
        '    </section>',
        f'    <div class="callout callout--accent"><b>Limit:</b> {esc(reader_caveat or "See the evidence details for the scope of this result.")} {("<b>Next:</b> " + esc(reader_next_step)) if reader_next_step else ""}</div>',
        (f'    <section class="hero-charts"><h2 class="card-title" style="margin-bottom:12px;">Selected charts</h2><div class="charts-grid">{hero_figs_html}</div></section>' if hero_figs_html else ''),
        '    <details class="collapsible-box"><summary>Numbers and study details</summary><div class="content">Full methods, statistical tables, and every chart are in the stage tabs.</div></details>',
        '  </div>',
        '  <div class="tab-pane" id="tab_pre_m1">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">PRE-M1</span>',
        '        <h2 class="card-title">Case Synthesis & Adversarial Probes</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{pre_m1_sub}</span>',
        f'        <p class="card-subtext">{STAGE_METADATA["pre_m1"]["plain_desc"]}</p>',
        '      </div>',
        f'      <div class="callout"><b>Stage Status:</b> {pre_m1_desc}</div>',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_m1">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">M1</span>',
        '        <h2 class="card-title">What we checked</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m1_duration_str}</span>',
        '        <p class="card-subtext">We looked for observable behaviors that might be connected to mistakes. Open a check only if you want its method and raw records.</p>',
        '      </div>',
        f'      <div class="callout callout--accent"><b>Summary:</b> We ran <b>{len(m1["analyzers"])} checks</b> across {n_total:,} cases. These checks describe behavior; they do not by themselves prove a cause.</div>',
        f'      <div style="display:flex; flex-direction:column; gap:14px;">{m1_cards_html}</div>',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_m2">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">M2</span>',
        '        <h2 class="card-title">What patterns we found</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m2_duration_str}</span>',
        '        <p class="card-subtext">These are patterns that appeared more often around mistakes. They are leads for further tests, not final explanations.</p>',
        '      </div>',
        f'      <div class="callout callout--good"><b>What this means:</b> Out of {len(m2["stats"])} measured patterns, <b>{len(stat_rows_sig)}</b> were strong enough to keep investigating. This does not prove they caused the errors.</div>',
        f'      {m2_conclusion_box}',
        '      <details class="collapsible-box">',
        '        <summary>Research details: statistical results and effect sizes</summary>',
        '        <div class="content"><div class="table-wrapper">',
        '        <table class="data-table">',
        '          <thead>',
        '            <tr>',
        '              <th>Anomalous Feature Signal</th>',
        '              <th class="num">Effect Size</th>',
        '              <th class="num">95% Conf. Interval</th>',
        '              <th class="num">p-value</th>',
        '              <th>FDR Screening Verdict</th>',
        '            </tr>',
        '          </thead>',
        f'          <tbody>{stats_sig_html}</tbody>',
        '        </table>',
        '        </div></div>',
        '      </details>',
        '      <details class="collapsible-box">',
        f'        <summary>View {len(stat_rows_null)} Non-Significant Feature Signals</summary>',
        '        <div class="content">',
        '          <div class="table-wrapper">',
        '            <table class="data-table">',
        '              <thead>',
        '                <tr>',
        '                  <th>Feature Signal</th>',
        '                  <th class="num">Effect Size</th>',
        '                  <th class="num">95% Conf. Interval</th>',
        '                  <th class="num">p-value</th>',
        '                  <th>Verdict</th>',
        '                </tr>',
        '              </thead>',
        f'              <tbody>{stats_null_html}</tbody>',
        '            </table>',
        '          </div>',
        '        </div>',
        '      </details>',
        '      <details class="collapsible-box" open>',
        f'        <summary>Exploratory Data Analysis Charts ({len(figures)} plots)</summary>',
        f'        <div class="content"><div class="charts-grid">{all_figs_html}</div></div>',
        '      </details>',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_m3">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">M3</span>',
        '        <h2 class="card-title">Possible explanation to test</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{len(m3["hypotheses"])} Hypotheses</span>',
        '        <p class="card-subtext">A possible explanation must predict what should happen on new cases before we can treat it as evidence.</p>',
        '      </div>',
        '      <div class="callout"><b>Important:</b> A possible explanation is not a confirmed root cause. It must be checked on cases that were not used to find it.</div>',
        f'      {m3_hypotheses_html}',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_m4">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">M4</span>',
        '        <h2 class="card-title">Independent check</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m4_status}</span>',
        '        <p class="card-subtext">This is where we ask whether a proposed explanation still holds on new cases.</p>',
        '      </div>',
        f'      <div class="callout {m4_tone_cls}">{m4_plain_text}</div>',
        f'      {m4_detail_html}',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_m5_surgery">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">M5-SURGERY</span>',
        '        <h2 class="card-title">Causal Surgery & Interventions</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m5_s_sub}</span>',
        f'        <p class="card-subtext">{STAGE_METADATA["m5_surgery"]["plain_desc"]}</p>',
        '      </div>',
        f'      <div class="callout"><b>Stage Status:</b> {m5_s_desc}</div>',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_m5_fix">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">M5-FIX</span>',
        '        <h2 class="card-title">Repair attempt</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{m5_status}</span>',
        '        <p class="card-subtext">We test a focused change and count both improvements and newly introduced mistakes.</p>',
        '      </div>',
        f'      <div class="callout {m5_f_tone_cls}">{m5_f_plain}</div>',
        '      <details class="collapsible-box" open>',
        f'        <summary>Repair Candidate Sweep ({len(m5_f["selection"])} candidates evaluated)</summary>',
        '        <div class="content">',
        '          <div class="table-wrapper">',
        '            <table class="data-table">',
        '              <thead>',
        '                <tr>',
        '                  <th>Tier</th>',
        '                  <th>Candidate Strategy</th>',
        '                  <th>Screening Verdict</th>',
        '                  <th class="num">Cured (+)</th>',
        '                  <th class="num">Broken (-)</th>',
        '                  <th class="num">Net Accuracy Shift</th>',
        '                  <th>Balance Direction</th>',
        '                </tr>',
        '              </thead>',
        f'              <tbody>{sel_table_html}</tbody>',
        '            </table>',
        '          </div>',
        '        </div>',
        '      </details>',
        f'      {m5_f_tmpl_html}',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_agents">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">AGENTS</span>',
        '        <h2 class="card-title">Agent Trajectories — Layer-by-Layer Audit</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{agents_n} Agent Invocations</span>',
        f'        <p class="card-subtext">{STAGE_METADATA["agents"]["plain_desc"]}</p>',
        '      </div>',
        '      <div class="callout neutral">Every pipeline stage above is driven by agent calls (LLM judges and CLI coder agents). This tab is the bottom layer of the drill-down: pipeline stage → probe / hypothesis → the raw agent input and output that produced it.</div>',
        '      <div class="section-subhead" style="margin-top:18px;">Judge Calls (verbatim prompt → response)</div>',
        f'      {judge_boxes_html}',
        '      <div class="section-subhead" style="margin-top:18px;">Explore Coder Agent</div>',
        f'      {explore_boxes_html}',
        '      <div class="section-subhead" style="margin-top:18px;">Tool Codegen Attempts</div>',
        f'      {codegen_boxes_html}',
        '      <div class="section-subhead" style="margin-top:18px;">Unified Trace (Langfuse Bundle)</div>',
        f'      {langfuse_html}',
        '    </div>',
        '  </div>',
        '  <div class="tab-pane" id="tab_case_book">',
        '    <div class="card">',
        '      <div class="card-head">',
        '        <span class="card-tag">CASE-STUDIO</span>',
        '        <h2 class="card-title">Interactive Case Studio</h2>',
        f'        <span class="text-muted mono" style="margin-left:auto; font-size:12px;">{len(cases)} Cases Available</span>',
        f'        <p class="card-subtext">{STAGE_METADATA["case_book"]["plain_desc"]}</p>',
        '      </div>',
        '      <div class="case-controls">',
        '        <div class="filter-btn-group">',
        f'          <button type="button" class="filter-btn is-active" data-f="all">All ({len(cases)})</button>',
        f'          <button type="button" class="filter-btn" data-f="fixed">Cured (+{sum(1 for c in cases if c["status"] == "fixed")})</button>',
        f'          <button type="button" class="filter-btn" data-f="broken">Broken (-{sum(1 for c in cases if c["status"] == "broken")})</button>',
        f'          <button type="button" class="filter-btn" data-f="unchanged">Unchanged ({sum(1 for c in cases if c["status"] == "unchanged")})</button>',
        '        </div>',
        '        <input type="text" class="search-box" id="csearch" placeholder="Search prompt, question, or ID...">',
        '      </div>',
        '      <div class="case-grid" id="cases_grid"></div>',
        '    </div>',
        '  </div>',
        '</div>',
        '<!-- MODAL LIGHTBOX & CASE AUDIT DRAWER -->',
        '<div class="modal-backdrop" id="app_modal" onclick="closeModal(event)">',
        '  <div class="modal-content" onclick="event.stopPropagation()">',
        '    <button type="button" class="modal-close" onclick="closeModal()">&times;</button>',
        '    <div id="modal_body"></div>',
        '  </div>',
        '</div>',
        '<footer class="app-footer">',
        '  <div class="container" style="display:flex; flex-direction:column; gap:6px;">',
        '    <div>Generated by <b>EvalRX Diagnostic Engine</b> · Native Langfuse OpenTelemetry Schema · Single-page zero external dependencies</div>',
        f'    <div>Model: <code>{esc(run["model"])}</code> · Fingerprint: <code>{esc(run["data_fingerprint"])}</code> · Path: <code>{esc(run["logs_dir"])}</code></div>',
        '  </div>',
        '</footer>',
        '<script>',
        f'const CASES = {cases_json};',
        f'const AUDIO = {audio_json};',
        f'const IMAGES = {images_json};',
        r"""
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

function toggleProbe(detailId) {
  const el = document.getElementById(detailId);
  const icon = document.getElementById('icon_' + detailId);
  if (!el) return;
  const isHidden = el.style.display === 'none';
  el.style.display = isHidden ? 'block' : 'none';
  if (icon) icon.classList.toggle('is-open', isHidden);
}

function openLightbox(imgSrc, title) {
  const modal = document.getElementById('app_modal');
  const body = document.getElementById('modal_body');
  body.innerHTML = '<div style="font-size:18px; font-weight:700; margin-bottom:12px;">' + esc(title) + '</div>' +
    '<img src="' + imgSrc + '" style="width:100%; border-radius:6px; border:1px solid var(--border);">';
  modal.classList.add('is-open');
}

function openCaseModal(caseId) {
  const c = CASES.find(item => item.id === caseId);
  if (!c) return;
  const modal = document.getElementById('app_modal');
  const body = document.getElementById('modal_body');
  const audioSrc = AUDIO[c.id];
  const imgSrc = IMAGES[c.id];
  const probeFlagsHtml = (c.probe_flags && c.probe_flags.length > 0) ?
    '<div style="margin-bottom:14px;"><div class="section-subhead">Anomalies Detected on This Case</div><div style="display:flex; flex-wrap:wrap; gap:6px;">' +
    c.probe_flags.map(f => '<span class="badge badge--warn" style="font-size:11px;">⚠️ ' + esc(f) + '</span>').join('') +
    '</div></div>' : '';

  body.innerHTML = '<div style="font-size:18px; font-weight:700; margin-bottom:4px;">Case Drilldown: <span class="mono" style="color:var(--brand);">' + esc(c.id) + '</span></div>' +
    '<div style="font-size:13px; color:var(--text-muted); margin-bottom:16px;">Task Domain: ' + esc(c.task || 'General Audio Understanding') + '</div>' +
    probeFlagsHtml +
    '<div class="callout"><b>Prompt / Instruction:</b> ' + esc(c.instruction) + '</div>' +
    (audioSrc ? '<div style="margin-bottom:16px;"><div class="section-subhead">Audible Stimulus (AAC 16kHz)</div><audio controls src="' + audioSrc + '" style="width:100%;"></audio></div>' : '') +
    (imgSrc ? '<div style="margin-bottom:16px;"><img src="' + imgSrc + '" style="max-height:300px; width:100%; object-fit:contain;"></div>' : '') +
    '<div class="section-subhead">Options & Ground Truth</div>' +
    '<ul class="case-choices" style="padding:0; margin-bottom:16px;">' + formatChoices(c) + '</ul>' +
    '<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">' +
    '<div class="kpi-card"><div class="kpi-label">Ground Truth Expected</div><div class="kpi-val text-good">' + esc(c.expected || '—') + '</div></div>' +
    '<div class="kpi-card"><div class="kpi-label">Repaired Model Output</div><div class="kpi-val">' + esc(c.output || '—') + '</div></div>' +
    '</div>';
  modal.classList.add('is-open');
}

function closeModal(e) {
  const modal = document.getElementById('app_modal');
  modal.classList.remove('is-open');
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

const esc = s => String(s || '').replace(/[&<>"']/g, c => {
  if (c === '&') return '&amp;';
  if (c === '<') return '&lt;';
  if (c === '>') return '&gt;';
  if (c === '"') return '&quot;';
  return '&#39;';
});

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
    const goldBadge = isGold ? '<span class="badge badge--good" style="margin-left:auto; font-size:10px;">Ground Truth</span>' : '';
    const goldClass = isGold ? 'is-gold' : '';
    return '<li class="' + goldClass + '"><span class="mono font-bold">' + esc(letter || '·') + '</span> <span>' + esc(text) + '</span> ' + goldBadge + '</li>';
  }).join('');
}

function renderCard(c) {
  let badgeClass = 'badge--mute', badgeText = 'Unchanged';
  if (c.status === 'fixed') { badgeClass = 'badge--good'; badgeText = 'Cured (Fixed)'; }
  else if (c.status === 'broken') { badgeClass = 'badge--bad'; badgeText = 'Regression (Broken)'; }

  const audioSrc = AUDIO[c.id];
  const imgSrc = IMAGES[c.id];
  const audioHtml = audioSrc ? '<audio controls preload="none" src="' + audioSrc + '"></audio>' : '';
  const imgHtml = imgSrc ? '<img src="' + imgSrc + '" alt="Stimulus" loading="lazy">' : '';
  const taskHtml = c.task ? '<span class="mono text-muted" style="font-size:11px;">' + esc(c.task) + '</span>' : '';
  const durHtml = c.duration ? '<span class="case-dur">' + c.duration.toFixed(1) + 's</span>' : '';
  const probeTagsHtml = (c.probe_flags && c.probe_flags.length > 0) ?
    '<div class="case-probe-tags">' + c.probe_flags.map(f => '<span class="case-probe-tag">' + esc(f) + '</span>').join('') + '</div>' : '';

  return '<article class="case-item">' +
    '<div class="case-item-header"><span class="badge ' + badgeClass + '">' + badgeText + '</span>' + taskHtml + durHtml + '</div>' +
    '<div class="case-q">' + esc(c.instruction) + '</div>' +
    probeTagsHtml +
    '<div class="case-media-box">' + audioHtml + imgHtml + '</div>' +
    '<ul class="case-choices">' + formatChoices(c) + '</ul>' +
    '<div class="case-item-foot"><span class="text-muted font-semibold" style="font-size:11px; text-transform:uppercase;">Repaired Output:</span>' +
    '<span class="mono font-bold" style="color:#ffffff;">' + esc(c.output || '—') + '</span>' +
    '<button type="button" class="mini-btn" style="margin-left:auto;" onclick="openCaseModal(' + JSON.stringify(c.id) + ')">Deep Audit</button></div>' +
    '</article>';
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
""",
        '</script>',
        '</body>',
        '</html>',
    ]
    return '\n'.join(html_lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_html_report(
    run_dir: str | Path,
    example_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    no_audio: bool = False,
) -> Path:
    """Collect data from *run_dir* and write a self-contained HTML report."""
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
    print(f"[*] Embedded {len(audio_map)} audio clips and {len(image_map)} images ({len(data['cases'])} joined cases)")

    html_content = generate_html_report(data, figures, audio_map, image_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A report may be tens or hundreds of MB when it embeds case media.  Write
    # beside the destination and replace atomically so a browser never sees a
    # partially generated diagnosis after an interruption or disk-full error.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.stem}.", suffix=".tmp", dir=out_path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(html_content)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Wrote self-contained HTML report to: {out_path} ({size_mb:.2f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a finished EvalRX diagnostic run as an interactive HTML page.")
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
