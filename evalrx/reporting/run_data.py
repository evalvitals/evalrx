"""Read a finished run's artifacts into one dictionary.

:func:`extract_run_data` is the single reader behind every report (the UI's
data in :mod:`evalrx.reporting.dynamic`, the Langfuse bundle export). It knows
the on-disk layout of a loop run and of an explore output, and the run log; it
knows nothing about HTML.
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
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


#: What an explore step writes as its report: ``evalrx explore`` / run-codebase
#: write the first, a fused discover→confirm run the second.
EXPLORE_REPORTS = ("exploratory_report.json", "fused_report.json")


def resolve_run_dirs(run_dir: Path) -> tuple[Path, Path | None, Path | None]:
    """Return (logs_dir, explore_dir, fixes_dir).

    An explore output (``evalrx explore``, run-codebase, a fused run) has no
    run log and holds its report at the top: it is its own explore directory.
    """
    run_dir = run_dir.resolve()
    logs_dir = run_dir
    if (run_dir / "logs" / "run.json").exists():
        logs_dir = run_dir / "logs"
    elif not (run_dir / "run.json").exists():
        for child in run_dir.glob("*/run.json"):
            logs_dir = child.parent
            break

    own_report = any((run_dir / name).is_file() for name in EXPLORE_REPORTS)
    explore_dir: Path | None
    if own_report and not (logs_dir / "run.json").exists():
        explore_dir = run_dir  # an explore output; a sibling explore/ is another run's
    else:
        explore_dir = logs_dir / "explore"
        if not explore_dir.is_dir() and (logs_dir.parent / "explore").is_dir():
            explore_dir = logs_dir.parent / "explore"
        if not explore_dir.is_dir():
            explore_dir = run_dir if own_report else None

    fixes_dir: Path | None
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


def _snapshot_explore_report(explore_event: dict[str, Any]) -> dict[str, Any]:
    """The explore step's own report, read back out of a V2 explore event.

    RunLoggerV2 keeps no explore/ directory beside the logs. It inlines the
    coder-agent's whole working directory on the explore event
    (``workspace_snapshot.files``) and copies every figure into M2/artifacts/
    under a content-hashed name, recording the original -> copied path pairs
    as ``media_files``. The inlined ``exploratory_report.json`` is the very
    document V1 wrote to explore/: the ranked takeaways with the charts each
    one cites, the visual plan, the chart readings. Without it the M2 record
    fell back to the event's observation sentences, which cite nothing, and
    every finding rendered as "referenced visual evidence was not found" with
    all of its charts filed under supporting material.

    Returns {} when the event carries no parseable report (a run recorded
    before the snapshot existed, or a report over the inline cap), leaving the
    caller to that sentence-level fallback.
    """
    snapshot = explore_event.get("workspace_snapshot")
    if not isinstance(snapshot, dict):
        return {}
    files = snapshot.get("files")
    text = files.get("exploratory_report.json") if isinstance(files, dict) else None
    if not isinstance(text, str) or not text.lstrip().startswith("{"):
        return {}
    try:
        report = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(report, dict) or not (report.get("takeaways") or report.get("observations")):
        return {}
    report = dict(report)
    # The report names its figures at sandbox paths that no longer exist; the
    # copies under logs/ are what the event lists, and ``media_files`` is the
    # join between the two.
    report["figures"] = [f for f in explore_event.get("figures") or [] if isinstance(f, str)]
    media_files = snapshot.get("media_files")
    report["media_files"] = {
        str(name): str(copied)
        for name, copied in (media_files.items() if isinstance(media_files, dict) else [])
        if isinstance(name, str) and isinstance(copied, str)
    }
    if not report.get("adjudication") and isinstance(explore_event.get("adjudication"), dict):
        report["adjudication"] = explore_event["adjudication"]
    return report


def extract_run_data(run_dir: Path, example_dir: Path | None = None) -> dict[str, Any]:
    """Parse all run artifacts dynamically into a unified dictionary."""
    logs_dir, explore_dir, fixes_dir = resolve_run_dirs(run_dir)
    manifest_path, manifest = find_manifest(example_dir or logs_dir.parent, logs_dir)

    from evalrx.reporting.run_events import read_v2_events

    events: list[dict[str, Any]] = read_v2_events(logs_dir)

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
        per_case_rows: list[Any] = []
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
    report_path = next((explore_dir / name for name in EXPLORE_REPORTS
                        if explore_dir and (explore_dir / name).exists()), None)
    if report_path is not None:
        try:
            explore_data = json.loads(report_path.read_text())
        except Exception:
            pass
    # RunLoggerV2 records the explore step as an M2 event; both fallbacks
    # below read from it.
    explores = by_event("explore")
    e0 = explores[-1] if explores else {}
    if not explore_data:
        # The explorer's own report -- takeaways and the charts they cite --
        # rides along inlined in the event's workspace snapshot; read that
        # back first.
        explore_data = _snapshot_explore_report(e0)
    if not explore_data:
        # No snapshot report: the event still carries observations and caveats
        # as sentences and figures as paths under logs/. Shape those the way
        # the explore report is shaped, so the M2 record renders either.
        takeaways = []
        for text in a0.get("findings") or []:
            if isinstance(text, str) and text.strip():
                # "[MEDIUM] self_consistency.consistency=0.2 < 0.5: low self-consistency — …"
                body = re.sub(r"^\[[A-Z]+\]\s*", "", text.strip())
                head = body.split(": ", 1)[1] if ": " in body else body
                takeaways.append({"title": head[:110], "plain_title": head[:110],
                                  "analysis": body, "chart_names": [], "table_names": []})
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

    data: dict[str, Any] = {
        "run": {
            # "loop" for a diagnose loop's run log, "explore" for the report
            # an explore step wrote on its own.
            "kind": "loop",
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
    if not events and explore_data and explore_dir == logs_dir:
        # An explore output: its report at the top and no run log, so the
        # stages that ran, the question and the data explored are all read
        # off the report. (A loop run's explore/ child is not one.)
        data["m2"]["ran"] = True
        data["m3"]["ran"] = bool(explore_data.get("hypotheses"))
        profile = explore_data.get("data_profile") or {}
        source = Path(str(profile.get("source_path") or "")).name
        data["run"].update(
            kind="explore",
            benchmark_name=source or data["run"]["benchmark_name"],
            protocol=data["run"]["protocol"] or str(
                explore_data.get("plain_question") or explore_data.get("question") or ""),
        )
    # Keep the audience-first semantic layer independent from HTML.  The same
    # object can later drive a different renderer without reinterpreting logs.
    from evalrx.reporting.compiler import compile_reader_report

    data["reader_report"] = compile_reader_report(data).to_dict()
    return data
