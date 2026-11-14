"""M1's probe results as a report shows them.

One entry per analyzer the probe step ran: what it asks, its headline numbers
and its findings, read from the probe event or from the result file an older
run wrote beside the log.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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


def _dig(d: dict, path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _flag_case(row: dict[str, Any], flags: dict[str, list[str]]) -> None:
    """Note what one per-case probe row says about its case."""
    sid = str(row.get("sample_id", ""))
    if not sid:
        return
    if sid not in flags:
        flags[sid] = []
    if row.get("format_flip_rate", 0) > 0:
        flags[sid].append(f"Option Order Flip ({row['format_flip_rate']:.0%})")
    if row.get("conf_logprob", 0) > 0.8 and row.get("correct") == 0:
        flags[sid].append(f"Overconfident on Error ({row['conf_logprob']:.0%})")
    if row.get("pass_at_k") == 1.0:
        flags[sid].append("Pass@5 Capable")
    elif row.get("pass_at_k") == 0.0:
        flags[sid].append("Pass@5 Zero-Coverage")


def _read_findings(
    path: Path, inline: Any, flags: dict[str, list[str]],
) -> tuple[dict[str, Any], Any, list[Any]]:
    """One analyzer's (findings, case count, per-case rows).

    V1 externalised each analyzer's result to artifacts/; RunLoggerV2 keeps
    the findings inline on the probe entry itself. Only a result file feeds
    the per-case ``flags``.
    """
    findings: dict[str, Any] = {}
    n = None
    per_case: list[Any] = []
    if not path.exists() and isinstance(inline, dict):
        findings = inline
        n = findings.get("n_cases") or findings.get("n_scored")
        per_case = findings.get("per_case") or []
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            findings = raw.get("findings") or {}
            n = findings.get("n_cases") or findings.get("n_scored")
            per_case = findings.get("per_case") or []
            for row in per_case:
                _flag_case(row, flags)
        except Exception:
            pass
    return findings, n, per_case


def _generated_meta(tool: str, need: str, findings: dict[str, Any]) -> tuple[str, str, str, list[tuple[str, str, str]]]:
    """Glossary-shaped entry for a probe the agent wrote during this run.

    No glossary can know it: the need it was written for is its question
    (first sentence up front, the whole brief as the description), and its
    headline is whatever scalar findings it reported, under their own names.
    """
    first = re.split(r"(?<=[.;])\s+", need.strip(), maxsplit=1)[0] if need else "Agent-written probe"
    if len(first) > 180:
        first = first[:177].rstrip() + "…"
    scalar = [
        (key.replace("_", " "), key, "pct" if key.endswith(("_rate", "_frac")) else "num")
        for key, value in findings.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and key not in {"n_cases", "n_scored", "n_graded", "n_samples"}
    ][:5]
    return (f"Agent-written probe · {tool}", first, need or "Agent-written probe.", scalar)


def _headline(findings: dict[str, Any], fields: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    headline: list[dict[str, Any]] = []
    for label, path, fmt in fields:
        v = _dig(findings, path)
        if isinstance(v, (int, float)):
            headline.append({
                "label": label,
                "value": f"{v:.1%}" if fmt == "pct" else f"{v:.3f}".rstrip("0").rstrip("."),
            })
        else:
            headline.append({"label": label, "value": None})
    return headline


def probe_section(
    probe: dict[str, Any], codegen_events: list[dict[str, Any]], logs_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """(the M1 section, per-case flags by sample id) for a run's probe event."""
    # An agent-written probe ("generated:probe1") has no glossary entry; the
    # need it was written for is the only description that exists, and the
    # codegen event carries it verbatim.
    generated_need: dict[str, str] = {}
    for event in codegen_events:
        tool = str(event.get("tool_name") or "")
        if tool and event.get("need"):
            generated_need[tool] = str(event["need"])

    cycle = int(probe.get("cycle", 0) or 0)
    analyzers = probe.get("analyzers") or probe.get("selected_analyzers") or []
    flags: dict[str, list[str]] = {}
    results = []
    for name in analyzers:
        result_paths = probe.get("result_paths") or {}
        path = logs_dir / str(result_paths.get(name) or f"artifacts/c{cycle}_{name}.result.json")
        findings, n, per_case = _read_findings(path, (probe.get("findings") or {}).get(name), flags)
        if name.startswith("generated:"):
            tool = name.split(":", 1)[1]
            meta = _generated_meta(tool, generated_need.get(tool, ""), findings)
        else:
            meta = ANALYZER_GLOSSARY.get(
                name, (name.replace("_", " ").title(), "Measures model behavior across this dimension", "Standard diagnostic probe.", [])
            )
        results.append({
            "name": name,
            "display_name": meta[0],
            "question": meta[1],
            "description": meta[2],
            "n": n,
            "headline": _headline(findings, meta[3]),
            "findings": findings,
            "per_case": per_case,
        })
    results.sort(key=lambda r: -(r["n"] or 0))
    return {"analyzers": analyzers, "duration": probe.get("duration_sec"), "results": results}, flags
