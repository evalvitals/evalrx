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

from evalrx.reporting.run_probes import probe_section


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


def clean_benchmark_name(protocol: str) -> str:
    """A friendly benchmark name for the benchmarks a protocol text names."""
    text = protocol.lower()
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


#: What an explore step writes as its report: ``evalrx explore`` / run-codebase
#: write the first, a fused discover→confirm run the second.
EXPLORE_REPORTS = ("exploratory_report.json", "fused_report.json")


def resolve_run_dirs(run_dir: Path) -> tuple[Path, Path | None]:
    """Return (logs_dir, explore_dir).

    A loop run keeps its log under ``logs/`` (or at the top). An explore
    output (``evalrx explore``, run-codebase, a fused run) holds its report at
    the top and is its own explore directory; a run without one has none.
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
    return logs_dir, run_dir if own_report else None


#: The explore coder-agent's own files, by the key the report shows them under.
_EXPLORE_AGENT_FILES = {
    "raw_output": "agent_raw_output.txt",
    "raw_streams": "agent_raw_streams.txt",
    "code": "analysis.py",
    "stdout": "stdout.txt",
    "stderr": "stderr.txt",
    "audit": "agent_audit.json",
}


def _explore_agent_files(explore_dir: "Path | None", explore_event: dict[str, Any]) -> dict[str, str]:
    """The text of each explore-agent file that exists, by file name.

    On disk for an explore output; for a loop run they are the files the
    logger inlined on the explore event (``workspace_snapshot.files``).
    """
    files: dict[str, str] = {}
    if explore_dir is not None:
        for name in _EXPLORE_AGENT_FILES.values():
            try:
                files[name] = (Path(explore_dir) / name).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                pass
        return files
    snapshot = explore_event.get("workspace_snapshot")
    inlined = snapshot.get("files") if isinstance(snapshot, dict) else None
    return {name: text for name, text in (inlined or {}).items()
            if name in _EXPLORE_AGENT_FILES.values() and isinstance(text, str)}


def _extract_agent_layer(
    logs_dir: Path, explore_dir: "Path | None", explore_event: dict[str, Any],
) -> dict[str, Any]:
    """What sits below the pipeline stages: the explore coder-agent's code,
    output and audit, and a summary of the Langfuse trace bundle."""
    files = _explore_agent_files(explore_dir, explore_event)
    exp: dict[str, Any] = {key: files[name] for key, name in _EXPLORE_AGENT_FILES.items() if name in files}
    if "audit" in exp:
        try:
            exp["audit"] = json.loads(exp["audit"])
        except ValueError:
            del exp["audit"]
    agents: dict[str, Any] = {"explore": exp, "langfuse": {}}

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


def _latest_run_events(logs_dir: Path) -> list[dict[str, Any]]:
    """The run log's events, from the latest run only when a directory was reused."""
    from evalrx.reporting.run_events import read_v2_events

    events: list[dict[str, Any]] = read_v2_events(logs_dir)
    run_starts = [e for e in events if e.get("event") == "run_start"]
    if not run_starts:
        return events
    trace_id = run_starts[-1].get("trace_id")
    return [e for e in events if e.get("trace_id") == trace_id]


def _run_section(
    run_start: dict[str, Any], loop_end: dict[str, Any], logs_dir: Path,
) -> dict[str, Any]:
    """What ran: the model, the benchmark, its size and how the loop ended."""
    raw_model_name = run_start.get("model") or "Target Model"
    clean_model = clean_model_display_name(raw_model_name)
    protocol = run_start.get("protocol") or {}
    protocol_desc = (
        protocol.get("description") or ""
        if isinstance(protocol, dict)
        else str(protocol)
    )
    benchmark_name = clean_benchmark_name(protocol_desc)
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
    return {
        # "loop" for a diagnose loop's run log, "explore" for the report
        # an explore step wrote on its own.
        "kind": "loop",
        "model": clean_model,
        "raw_model": raw_model_name,
        "benchmark_name": benchmark_name,
        "n_cases": run_start.get("n_cases") or 0,
        "label_distribution": run_start.get("label_distribution") or {},
        "protocol": protocol_desc,
        "cycles": loop_end.get("cycles", 1),
        "stopped_by": loop_end.get("stopped_by", "completed"),
        "duration_sec": loop_end.get("total_duration_sec", 0),
        "version": run_start.get("evalrx_version", "0.2.0"),
        "data_fingerprint": run_start.get("data_fingerprint", ""),
        "coder": run_start.get("coder") or "",
        "logs_dir": str(logs_dir),
    }


def _m2_stats(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """The analysis step's test rows, rejected nulls first, largest effect first."""
    rows = analysis.get("stats_results")
    stats: list[dict[str, Any]] = [
        {
            "tool": s.get("tool"),
            "config": s.get("config") or {},
            "summary": s.get("summary") or "",
            "effect": s.get("effect"),
            "ci": s.get("ci") or [None, None],
            "reject": bool(s.get("reject")),
            "p_value": s.get("p_value"),
            "underpowered": bool(s.get("underpowered")),
        }
        for s in (rows if isinstance(rows, list) else [])
        if isinstance(s, dict)
    ]
    stats.sort(key=lambda s: (not s["reject"], -abs(s.get("effect") or 0)))
    return stats


def _explore_from_sentences(explore_event: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """An explore report shaped from what the run log kept as sentences.

    With no report to read, the events still carry observations and caveats
    as sentences and figures as paths under logs/. Shaping those the way the
    explore report is shaped lets the M2 record render either.
    """
    takeaways = []
    for text in analysis.get("findings") or []:
        if isinstance(text, str) and text.strip():
            # "[MEDIUM] self_consistency.consistency=0.2 < 0.5: low self-consistency — …"
            body = re.sub(r"^\[[A-Z]+\]\s*", "", text.strip())
            head = body.split(": ", 1)[1] if ": " in body else body
            takeaways.append({"title": head[:110], "plain_title": head[:110],
                              "analysis": body, "chart_names": [], "table_names": []})
    for text in explore_event.get("observations") or []:
        if isinstance(text, str) and text.strip():
            head = text.strip().split(";")[0].split(". ")[0]
            takeaways.append({"title": head[:90], "plain_title": head[:90],
                              "analysis": text.strip(), "chart_names": [], "table_names": []})
    if not (takeaways or explore_event):
        return {}
    return {
        "takeaways": takeaways,
        "caveats": [c for c in explore_event.get("caveats") or [] if isinstance(c, str)],
        "observations": [o for o in explore_event.get("observations") or [] if isinstance(o, str)],
        "figures": [f for f in explore_event.get("figures") or [] if isinstance(f, str)],
        "adjudication": explore_event.get("adjudication") or {},
        "candidate_signals": [], "hypotheses": [],
    }


def _explore_report(
    explore_dir: Path | None, explore_event: dict[str, Any], analysis: dict[str, Any],
) -> dict[str, Any]:
    """The explore step's report, from the best source the run has.

    The report file it wrote, else the copy inlined in the explore event's
    workspace snapshot (RunLoggerV2 records the explore step as an M2 event),
    else the events' own sentences.
    """
    report: dict[str, Any] = {}
    report_path = next((explore_dir / name for name in EXPLORE_REPORTS
                        if explore_dir and (explore_dir / name).exists()), None)
    if report_path is not None:
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            pass
    return report or _snapshot_explore_report(explore_event) or _explore_from_sentences(explore_event, analysis)


def _m4_results(m4_surgeries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One verdict row per hypothesis M4 adjudicated.

    One hypothesis can be tested more than once across cycles — every
    hypothesis this run actually adjudicated belongs here, not just the most
    recent verdict, otherwise a 3-hypothesis M4 pass reports as if only 1 ran.
    """
    return [{
        "hypothesis_id": event.get("hypothesis_id"),
        "hypothesis": event.get("hypothesis"),
        "status": event.get("status"),
        "effect_size": (event.get("evidence") or {}).get("m4_effect_size"),
        "confidence": event.get("confidence_score"),
        "verdict": (event.get("evidence") or {}).get("m4_verdict"),
        "evidence_grade": (event.get("evidence") or {}).get("m4_evidence_grade"),
        "protocol_consistent": (event.get("evidence") or {}).get("m4_protocol_consistent"),
    } for event in m4_surgeries]


def _mark_explore_output(data: dict[str, Any], report: dict[str, Any]) -> None:
    """Describe ``data`` as an explore output rather than a loop run.

    Its report sits at the top with no run log, so the stages that ran, the
    question and the data explored are all read off the report.
    """
    data["m2"]["ran"] = True
    data["m3"]["ran"] = bool(report.get("hypotheses"))
    profile = report.get("data_profile") or {}
    source = Path(str(profile.get("source_path") or "")).name
    data["run"].update(
        kind="explore",
        benchmark_name=source or data["run"]["benchmark_name"],
        protocol=data["run"]["protocol"] or str(
            report.get("plain_question") or report.get("question") or ""),
    )


def extract_run_data(run_dir: Path) -> dict[str, Any]:
    """Read a run's log, or an explore output's report, into one dictionary."""
    logs_dir, explore_dir = resolve_run_dirs(run_dir)
    events = _latest_run_events(logs_dir)

    def named(event: str) -> list[dict[str, Any]]:
        return [e for e in events if e.get("event") == event]

    def last(event: str) -> dict[str, Any]:
        found = named(event)
        return found[-1] if found else {}

    run = _run_section(last("run_start"), last("loop_end"), logs_dir)
    probe_searches = named("probe_search")
    probe, analysis, diagnosis, fix = last("probe"), last("analysis"), last("diagnosis"), last("fix")
    m1 = probe_section(probe, named("tool_codegen"))
    explore_report = _explore_report(explore_dir, last("explore"), analysis)
    surgeries = named("surgery")
    m4_surgeries = [s for s in surgeries if s.get("module") == "m4" or s.get("adjudication")]
    m4_results = _m4_results(m4_surgeries)
    m5_surgeries = [s for s in surgeries if s.get("module") != "m4"]
    best_fix = fix.get("best") or {}
    confirmed_fix = best_fix if isinstance(best_fix, dict) else {}

    data: dict[str, Any] = {
        "run": run,
        "pre_m1": {
            "ran": bool(probe_searches),
            "n_cases": probe_searches[0].get("n_synthesized") if probe_searches else 0,
        },
        "m1": m1,
        "m2": {
            "conclusion": analysis.get("conclusion") or "",
            "narrative": analysis.get("narrative") or "",
            "severity": analysis.get("severity") or "medium",
            "duration": analysis.get("duration_sec"),
            "stats": _m2_stats(analysis),
            "explore": explore_report,
            # The analysis step's own figures (V2 paths under logs/).
            "figures": [f for f in analysis.get("figures") or [] if isinstance(f, str)],
        },
        "m3": {
            "hypotheses": diagnosis.get("hypotheses") or [],
            # The proposer's own list, kept beside the (possibly reordered and
            # critic-annotated) accepted list. It is the only place a run
            # recorded before the log writer carried `plain_statement` on
            # `hypotheses` still has the judge's plain sentence, and the report
            # compiler joins the two on the statement to recover it.
            "proposed_hypotheses": diagnosis.get("proposed_hypotheses") or [],
            "duration": diagnosis.get("duration_sec"),
        },
        "m4": {
            "ran": bool(m4_surgeries or m4_results),
            "results": m4_results,
            "event": m4_surgeries[-1] if m4_surgeries else {},
        },
        "m5_surgery": {
            "ran": bool(m5_surgeries),
            "surgeries": m5_surgeries,
        },
        "m5_fix": {
            "ran": bool(fix or confirmed_fix),
            "fixed": bool(fix.get("fixed") or confirmed_fix.get("fixed")),
            "selection": fix.get("selection_attempted") or fix.get("attempted") or [],
            "confirm": confirmed_fix,
            "best": best_fix,
            "prompt_template": (confirmed_fix.get("payload") or {}).get("prompt_template") or "",
        },
        "agents": _extract_agent_layer(logs_dir, explore_dir, last("explore")),
        "logs_dir": logs_dir,
        "explore_dir": explore_dir,
    }
    if not events and explore_report and explore_dir == logs_dir:
        # A loop run's explore/ child is not an explore output; only a report
        # at the top of a directory with no run log is.
        _mark_explore_output(data, explore_report)
    # Keep the audience-first semantic layer independent from HTML.  The same
    # object can later drive a different renderer without reinterpreting logs.
    from evalrx.reporting.compiler import compile_reader_report

    data["reader_report"] = compile_reader_report(data).to_dict()
    return data
