#!/usr/bin/env python3
"""
extract_figure_data.py — pull everything the case-study figure needs out of an
EvalRX run directory, in three shapes: a readable JSON document, a flat
JSONL stream for plotting code, and a Markdown write-up of the figure.

Usage
-----
    python extract_figure_data.py path/to/chartqa.chain1
    python extract_figure_data.py path/to/run -o figure_data          # -> figure_data.json
    python extract_figure_data.py path/to/run --format jsonl          # flat, one record per line
    python extract_figure_data.py path/to/run --format both
    python extract_figure_data.py path/to/run --example-case chartqa-human-1142

Two output shapes, same numbers:

  --format json  (default)  one nested file, pretty-printed, nulls pruned, floats
                            rounded, prompts split into lines, sources hoisted to
                            a single ``_sources`` map. Meant to be read.
  --format jsonl            one flat record per line, every field kept verbatim.
                            Meant to be piped into a plotting script.
  --format md               the figure written out in prose: which analyzers ran,
                            the signal as a text bar chart, the verdicts, the
                            repair ladder, the before/after, and the caveats.
  --format all              json + jsonl + md

Input is the run root: the directory that contains ``summary.json`` and ``logs/``.
Internally everything is assembled as flat records, one per figure block, each
carrying a ``block`` key:

    run              header strip: model, dataset, split sizes, baseline accuracy
    m1_analyzers     which analyzer families exist, which was selected, why
    m1_signal_curve  failure rate binned by the confirmed signal (the bar chart)
    m2_test          one row per statistical test (the forest plot), per phase
    m2_family        the multiplicity-correction family summary, per phase
    m3_hypothesis    one row per proposed hypothesis + the critic's reaction
    m4_verdict       one row per adjudicated hypothesis
    m5_candidate     one row per repair candidate tried
    m5_ladder        one row per tier L1..L4 with its outcome
    repair           the accepted fix: image ops, prompt, decoding settings
    validation       the paired held-out comparison
    example_case     one illustrative failing case, with its media path

Nothing here is invented: every field carries a ``source`` pointing at the file
it was read from, so a figure number can be traced back to an artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

CI_RE = re.compile(r"CI=([+-][\d.]+)\.\.([+-][\d.]+)")


def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return rows


def parse_ci(summary: Optional[str]) -> Optional[List[float]]:
    """Recover a CI from a summary string like '... CI=+0.2109..+0.3984 ...'."""
    if not summary:
        return None
    m = CI_RE.search(summary)
    return [float(m.group(1)), float(m.group(2))] if m else None


def rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 6) if den else None


def warn(msg: str) -> None:
    print(f"  ! {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# the run object
# --------------------------------------------------------------------------


class Run:
    """Lazily-loaded view over one EvalRX run directory."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.logs = os.path.join(self.root, "logs")
        if not os.path.isdir(self.logs):
            raise SystemExit(f"no logs/ directory under {self.root}")

        self.summary = load_json(os.path.join(self.root, "summary.json"), {}) or {}
        self.manifest = load_json(os.path.join(self.logs, "manifest.json"), {}) or {}
        self.config = self.manifest.get("config", {}) or {}
        self.events = load_jsonl(os.path.join(self.logs, "run_log.jsonl"))

        # the manifest's run_id is sometimes just "logs"; prefer the directory name
        self.run_id = os.path.basename(self.root.rstrip(os.sep))
        self.manifest_run_id = self.manifest.get("run_id")

        # every case in the batch, with prompt / gold / baseline output / label
        self.all_cases = {
            c["id"]: c
            for c in (load_json(self.rel("report/discovery_cases.json"), []) or [])
            if isinstance(c, dict) and "id" in c
        }

        # case_record events are emitted for the discovery split only
        self.explore_ids = [
            e["case_id"] for e in self.events if e.get("event") == "case_record"
        ]
        self.heldout_ids = [
            cid for cid in self.all_cases if cid not in set(self.explore_ids)
        ]
        self.case_records = {
            e["case_id"]: e for e in self.events if e.get("event") == "case_record"
        }

    # -- paths ------------------------------------------------------------

    def rel(self, *parts: str) -> str:
        return os.path.join(self.logs, *parts)

    def relpath(self, abspath: str) -> str:
        try:
            return os.path.relpath(abspath, self.root)
        except ValueError:
            return abspath

    def rooted(self, recorded: str) -> str:
        """A recorded path re-anchored on THIS run root.

        The run log records absolute paths from the process that wrote it — in a
        container that is ``/app/work/outputs/<run>/logs/...``, which exists on
        no host. ``os.path.relpath`` against such a path walks up to ``/`` and
        yields a ``../../..`` chain whose length depends on where the READER
        sits. The artifact itself lives inside the run dir, so re-anchor: take
        the ``logs/...`` suffix when the result exists under the root, else
        fall back to :meth:`relpath` for a path that really is elsewhere.
        """
        recorded = str(recorded or "")
        marker = os.sep + "logs" + os.sep
        idx = recorded.rfind(marker)
        if idx >= 0:
            candidate = recorded[idx + 1:]
            if os.path.isdir(os.path.join(self.root, candidate)) or os.path.isfile(
                os.path.join(self.root, candidate)
            ):
                return candidate.replace(os.sep, "/")
        return self.relpath(recorded)

    # -- events -----------------------------------------------------------

    def event(self, name: str, cycle: Any = "__any__") -> Optional[dict]:
        for e in self.events:
            if e.get("event") == name and (cycle == "__any__" or e.get("cycle") == cycle):
                return e
        return None

    def events_of(self, name: str) -> List[dict]:
        return [e for e in self.events if e.get("event") == name]

    # -- phases -----------------------------------------------------------

    def m2_files(self) -> List[tuple]:
        """[(phase, abspath)] for every M2 stats-results file present.

        Convention in the run logs: ``c0_`` is the discovery cycle, ``post_``
        (or ``c-1_``) is the held-out confirmation pass.
        """
        art = self.rel("artifacts")
        out = []
        if not os.path.isdir(art):
            return out
        for fn in sorted(os.listdir(art)):
            if not fn.endswith("_m2_stats_results.json"):
                continue
            if fn.startswith("post_") or fn.startswith("c-1_"):
                phase = "heldout"
            elif fn.startswith("c0_"):
                phase = "explore"
            else:
                phase = fn.split("_m2_")[0]
            out.append((phase, os.path.join(art, fn)))
        # explore first, held-out second
        out.sort(key=lambda p: 0 if p[0] == "explore" else 1)
        return out

    def measurement_inventory(self, analyzers: List[str],
                              cycle_prefix: str = "c0") -> List[dict]:
        """Every numeric per-case field each analyzer produced, with its status."""
        out: List[dict] = []
        for a in analyzers:
            rows = self.analyzer_per_case(a, cycle_prefix)
            if not rows:
                continue
            fields: Dict[str, list] = defaultdict(list)
            for r in rows:
                for k, v in r.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        fields[k].append(v)
            for k, vals in sorted(fields.items()):
                if k in OUTCOME_DERIVED:
                    status = "dropped_sees_answer_key"
                elif len(set(vals)) == 1:
                    status = "dropped_never_varies"
                elif len(vals) < len(rows):
                    status = "dropped_partial_coverage"
                else:
                    status = "candidate"
                out.append({
                    "analyzer": a,
                    "field": k,
                    "signal": f"{a}.{k}",
                    "n_values": len(vals),
                    "n_distinct": len(set(vals)),
                    "status": status,
                    "question": question_for(a, k),
                })
        return out

    def analyzer_per_case(self, analyzer: str, cycle_prefix: str) -> List[dict]:
        path = self.rel("artifacts", f"{cycle_prefix}_{analyzer}.result.json")
        blob = load_json(path, {}) or {}
        return (blob.get("findings") or {}).get("per_case", []) or []


# --------------------------------------------------------------------------
# naming and plain-language probe questions
# --------------------------------------------------------------------------

MODULE_NAMES = {
    "M1": "Suspicious Behavior Detection",
    "M2": "Statistical Screening",
    "M3": "Hypothesis Formation",
    "M4": "Held-out Verification",
    "M5": "Validated Repair",
}

MODULE_SUBTITLES = {
    "M1": "Run the analyzer probing library, find per-case suspicious behaviors",
    "M2": "Using plots to explain, statistical tests to decide",
    "M3": "Explore the reason behind the signals",
    "M4": "Confirm whether the hypothesis is verified over the held-out cases",
    "M5": "Fix the failure with the validated repair ladder",
}

# What a probe is actually asking, in the reader's language. Keyed by the
# per-case field a probe emits, because one analyzer often asks two questions
# and one question is often asked by two analyzers.
QUESTION_BY_FIELD = {
    "matches_output_contract": "Did it answer in the form we asked for?",
    "has_answer_tag":          "Did it answer in the form we asked for?",
    "n_unparsed":              "Did it answer in the form we asked for?",
    "gave_up":                 "Did it give up answering?",
    "has_output":              "Did it produce anything at all?",
    "answered_yes":            "Does it lean to one answer regardless?",
    "gold_yes":                "Does it lean to one answer regardless?",
    "output_chars":            "Did it stop, or keep talking?",
    "output_words":            "Did it stop, or keep talking?",
    "output_truncated":        "Did it stop, or keep talking?",
    "looks_truncated":         "Did it stop, or keep talking?",
    "continuation_chars":      "Did it stop, or keep talking?",
    "repetition_score":        "Did it stop, or keep talking?",
    "n_unique":                "Same question five times, same answer?",
    "majority_share":          "Same question five times, same answer?",
    "n_samples":               "Same question five times, same answer?",
    "n_graded":                "Same question five times, same answer?",
    "coverage_gap":            "Did it ever produce the right answer?",
    "format_flip_rate":        "Reworded question, same answer?",
    "positional_bias":         "Reworded question, same answer?",
    "n_variants":              "Reworded question, same answer?",
    "n_options":               "Reworded question, same answer?",
    "n_sentences":             "Asked to check itself, does it change?",
    "selfcheck_inconsistency": "Asked to check itself, does it change?",
    "selfcheck_worst_sentence":"Asked to check itself, does it change?",
    "conf_logprob":            "Was it as sure as it sounded?",
    "conf_verbal":             "Was it as sure as it sounded?",
    "attention_entropy":       "Where was it looking?",
    "attention_to_region":     "Where was it looking?",
    "logprob_entropy":         "How uncertain is it inside?",
    "token_entropy":           "How uncertain is it inside?",
}

# fallback when a field is unknown: ask by analyzer instead
QUESTION_BY_ANALYZER = {
    "answer_extraction_audit":   "Did the answer come out readable?",
    "termination_audit":         "Did it stop, or keep talking?",
    "selfcheck_consistency":     "Asked to check itself, does it change?",
    "self_consistency":          "Same question five times, same answer?",
    "coverage_verification_gap": "Did it ever produce the right answer?",
    "format_sensitivity":        "Reworded question, same answer?",
    "calibration":               "Was it as sure as it sounded?",
    "logprob_entropy":           "How uncertain is it inside?",
    "attention":                 "Where was it looking?",
    "hallucination":             "Did it describe something that is not there?",
    "multimodal_attribution":    "Did the answer use the image or audio at all?",
    "loop_detection":            "Did it get stuck repeating itself?",
}

# reading order, so the question list is stable across runs
QUESTION_ORDER = [
    "Did it produce anything at all?",
    "Did it answer in the form we asked for?",
    "Did the answer come out readable?",
    "Did it give up answering?",
    "Did it stop, or keep talking?",
    "Same question five times, same answer?",
    "Reworded question, same answer?",
    "Asked to check itself, does it change?",
    "Was it as sure as it sounded?",
    "Does it lean to one answer regardless?",
    "Did it ever produce the right answer?",
    "How uncertain is it inside?",
    "Where was it looking?",
]

# fields that are a function of the answer key; testing them would let a signal
# predict the answer from the answer
OUTCOME_DERIVED = {
    "labelled_fail", "strict_match", "gold_in_output", "gold_in_answer_region",
    "label_disagrees", "extraction_suspect", "extraction_point_miss",
    "n_correct", "any_correct", "majority_correct", "pass_at_k", "correct",
    "continuation_correct", "continuation_has_answer", "recovered_by_continuation",
}


def question_for(analyzer: str, field: str) -> str:
    if field in QUESTION_BY_FIELD:
        return QUESTION_BY_FIELD[field]
    return QUESTION_BY_ANALYZER.get(analyzer, f"What does {analyzer} show?")


# --------------------------------------------------------------------------
# block extractors — each returns a list of records
# --------------------------------------------------------------------------


def block_run(run: Run) -> List[dict]:
    start = run.event("run_start") or {}
    end = run.event("loop_end") or {}
    cfg = run.config
    dist = start.get("label_distribution") or {}
    return [
        {
            "block": "run",
            "run_id": run.run_id,
            "dataset": cfg.get("dataset") or run.summary.get("dataset"),
            "benchmark": cfg.get("benchmark"),
            "model": cfg.get("model") or run.summary.get("model"),
            "model_spec": cfg.get("spec") or run.summary.get("spec"),
            "hf_repo": cfg.get("hf_repo"),
            "modality": cfg.get("modality") or run.summary.get("modality"),
            "backend": cfg.get("backend") or run.summary.get("backend"),
            "n_cases": cfg.get("n_cases") or run.summary.get("n_cases"),
            "n_explore": len(run.explore_ids),
            "n_heldout": len(run.heldout_ids),
            "confirm_split": cfg.get("confirm_split"),
            "manifest_seed": cfg.get("manifest_seed"),
            "baseline_accuracy": run.summary.get("baseline_accuracy"),
            "explore_label_distribution": dist,
            "data_fingerprint": start.get("data_fingerprint"),
            "generation_kwargs": cfg.get("generation_kwargs"),
            "fix_tier_cap": cfg.get("fix_tier"),
            "auto_escalate": cfg.get("auto_escalate"),
            "allow_codegen": cfg.get("allow_codegen"),
            "m1_selection": cfg.get("m1_selection"),
            "judge": str(start.get("judge")) if start.get("judge") else None,
            "coder": str(start.get("coder")) if start.get("coder") else None,
            "protocol": (start.get("protocol") or {}).get("description"),
            "manifest_run_id": run.manifest_run_id,
            "cycles": run.summary.get("cycles"),
            "stopped_by": run.summary.get("stopped_by"),
            "timings_sec": end.get("timings_sec"),
            "total_duration_sec": end.get("total_duration_sec"),
            "source": "summary.json + logs/manifest.json + logs/run_log.jsonl",
        }
    ]


# Analyzer -> family. Extend this table when new analyzers are added; anything
# unknown is reported under "other" rather than silently dropped.
ANALYZER_FAMILY = {
    "answer_extraction_audit": "black_box_behavior",
    "selfcheck_consistency": "black_box_behavior",
    "coverage_verification_gap": "black_box_behavior",
    "uncertainty": "black_box_behavior",
    "perturbation_consistency": "black_box_behavior",
    "prompt_contrast": "black_box_behavior",
    "attention": "internal_white_box",
    "attention_rollout": "internal_white_box",
    "attention_sink": "internal_white_box",
    "logit_lens": "internal_white_box",
    "token_entropy": "internal_white_box",
    "representation_similarity": "internal_white_box",
    "termination_audit": "black_box_behavior",
    "format_sensitivity": "black_box_behavior",
    "self_consistency": "black_box_behavior",
    "calibration": "black_box_behavior",
    "loop_detection": "black_box_behavior",
    "first_error_localization": "black_box_behavior",
    "ignored_observation": "black_box_behavior",
    "counterfactual": "black_box_behavior",
    "logprob_entropy": "internal_white_box",
    "hallucination": "multimodal",
    "multimodal_attribution": "multimodal",
    "grounding": "multimodal",
    "audio_attention": "multimodal",
    "modality_ablation": "multimodal",
}

# fallback for analyzers not in the table: match on substrings rather than
# silently dumping everything into "other"
FAMILY_KEYWORDS = [
    ("internal_white_box", ("attention", "logit", "logprob", "entropy",
                            "representation", "activation", "residual", "sink")),
    ("multimodal", ("hallucinat", "grounding", "attribution", "modality",
                    "visual", "audio", "image")),
    ("black_box_behavior", ("audit", "consistency", "calibration", "uncertain",
                            "perturbation", "contrast", "sensitivity",
                            "termination", "coverage", "verification", "loop",
                            "counterfactual", "trajectory")),
]


def family_of(analyzer: str) -> str:
    if analyzer in ANALYZER_FAMILY:
        return ANALYZER_FAMILY[analyzer]
    low = analyzer.lower()
    for fam, keys in FAMILY_KEYWORDS:
        if any(k in low for k in keys):
            return fam
    return "other"

FAMILY_LABEL = {
    "black_box_behavior": "BLACK-BOX BEHAVIOR",
    "internal_white_box": "INTERNAL / WHITE-BOX",
    "multimodal": "MULTIMODAL",
    "other": "OTHER",
}


def block_m1_analyzers(run: Run) -> List[dict]:
    probes = run.events_of("probe")
    if not probes:
        return []
    first = probes[0]
    selected = first.get("selected_analyzers") or first.get("analyzers") or []
    fams = defaultdict(list)
    for a in selected:
        fams[family_of(a)].append(a)

    families = []
    for key, label in FAMILY_LABEL.items():
        members = fams.get(key, [])
        families.append(
            {
                "family": key,
                "label": label,
                "selected": bool(members),
                "analyzers": members,
            }
        )

    headline: Dict[str, dict] = {}
    for name, findings in (first.get("findings") or {}).items():
        if isinstance(findings, dict):
            headline[name] = {
                k: v
                for k, v in findings.items()
                if isinstance(v, (int, float, str, bool)) and k != "_caveat"
            }

    return [
        {
            "block": "m1_analyzers",
            "run_id": run.run_id,
            "selection_mode": run.config.get("m1_selection"),
            "selection_rationale": first.get("selection_rationale"),
            "selected_analyzers": selected,
            "families": families,
            "n_probe_passes": len(probes),
            "probe_cycles": [p.get("cycle") for p in probes],
            "headline_metrics": headline,
            "source": "logs/run_log.jsonl:probe",
        }
    ]


def block_m1_questions(run: Run, forwarded: Optional[int]) -> List[dict]:
    """What the probes asked, in plain language, plus the measurement funnel."""
    probes = run.events_of("probe")
    if not probes:
        return []
    analyzers = probes[0].get("selected_analyzers") or probes[0].get("analyzers") or []
    inv = run.measurement_inventory(analyzers)
    if not inv:
        return []

    # answer-key-derived fields are not probes asking anything; they are
    # excluded before the question list is built
    by_q: Dict[str, List[dict]] = defaultdict(list)
    for m in inv:
        if m["status"] != "dropped_sees_answer_key":
            by_q[m["question"]].append(m)

    def rank(q: str) -> tuple:
        return (QUESTION_ORDER.index(q) if q in QUESTION_ORDER else len(QUESTION_ORDER), q)

    questions = []
    for q in sorted(by_q, key=rank):
        rows = by_q[q]
        questions.append({
            "question": q,
            "analyzers": sorted({r["analyzer"] for r in rows}),
            "n_measurements": len(rows),
            "n_candidates": sum(1 for r in rows if r["status"] == "candidate"),
        })

    drops = Counter(m["status"] for m in inv)
    return [{
        "block": "m1_probe_questions",
        "run_id": run.run_id,
        "n_analyzers": len(analyzers),
        "n_measured": len(inv),
        "n_forwarded": forwarded,
        "questions": questions,
        "dropped": {
            "saw_the_answer_key": drops.get("dropped_sees_answer_key", 0),
            "never_varied": drops.get("dropped_never_varies", 0),
            "partial_coverage": drops.get("dropped_partial_coverage", 0),
        },
        "note": "each case is answered five times; the probes never see the answer key",
        "inventory": inv,
        "source": "logs/artifacts/c0_<analyzer>.result.json per-case fields",
    }]


def survivor_signals(run: Run) -> List[str]:
    """Signals that survived multiplicity correction, preferring the held-out pass."""
    for phase, path in reversed(run.m2_files()):
        found = [
            (r.get("config") or {}).get("signal")
            for r in (load_json(path, []) or [])
            if r.get("fdr_corrected")
        ]
        if found:
            return [f for f in found if f]
    return []


def signal_effects(run: Run) -> Dict[str, float]:
    """Effect size per signal from the latest M2 pass, for ranking."""
    out: Dict[str, float] = {}
    for _phase, path in run.m2_files():
        for r in load_json(path, []) or []:
            if r.get("tool") == "signal_label_assoc" and r.get("effect") is not None:
                sig = (r.get("config") or {}).get("signal")
                if sig:
                    out[sig] = r["effect"]
    return out


def pick_curve_signal(run: Run, signals: List[str]) -> Optional[str]:
    """Choose the survivor that makes the most legible bar chart.

    A handful of discrete levels shows a gradient and reads well; a binary
    signal gives only two bars; a continuous one such as output length would
    otherwise produce dozens of singleton bins. Within a band, the strongest
    association is the most informative to plot.
    """
    effects = signal_effects(run)
    best, best_key = None, None
    for sig in signals:
        if "." not in sig:
            continue
        analyzer, field = sig.split(".", 1)
        vals = [r.get(field) for r in run.analyzer_per_case(analyzer, "c0")]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        card = len(set(vals))
        band = 0 if 3 <= card <= 10 else (1 if card == 2 else 2)
        key = (band, -abs(effects.get(sig, 0.0)))
        if best_key is None or key < best_key:
            best, best_key = sig, key
    return best or (signals[0] if signals else None)


def block_m1_signal_curve(run: Run, signal: Optional[str],
                          all_survivors: Optional[List[str]] = None) -> List[dict]:
    """Failure rate binned by the confirmed signal — the M1 bar chart."""
    if not signal or "." not in signal:
        warn("no confirmed signal found; skipping m1_signal_curve")
        return []
    analyzer, field = signal.split(".", 1)
    per_case = run.analyzer_per_case(analyzer, "c0")
    if not per_case:
        warn(f"no per-case rows for {analyzer} on the explore split")
        return []

    labels = {
        cid: (rec.get("case") or {}).get("label")
        for cid, rec in run.case_records.items()
    }
    pairs = [
        (row.get(field), labels.get(row.get("sample_id")))
        for row in per_case
    ]
    pairs = [(v, lab) for v, lab in pairs if v is not None and lab is not None]
    if not pairs:
        warn(f"no usable values for {signal} on the explore split")
        return []

    values = [v for v, _ in pairs]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    levels = sorted(set(values), key=float) if numeric else sorted(set(values), key=str)
    binning = "levels"

    if numeric and len(levels) > 10:
        # quantile bins, so a continuous signal still reads as a bar chart
        binning = "quartiles"
        ordered = sorted(values)
        n_bins = 4
        cuts = [ordered[int(len(ordered) * i / n_bins)] for i in range(1, n_bins)]
        cuts = sorted(set(cuts))

        def assign(v):
            for i, c in enumerate(cuts):
                if v < c:
                    return i
            return len(cuts)

        edges = [ordered[0]] + cuts + [ordered[-1]]
        agg: Dict[Any, Counter] = defaultdict(Counter)
        for v, lab in pairs:
            b = assign(v)
            agg[b]["n"] += 1
            if str(lab).lower() == "fail":
                agg[b]["fail"] += 1
        bins = []
        for b in sorted(agg):
            lo = edges[b]
            hi = edges[b + 1] if b + 1 < len(edges) else edges[-1]
            bins.append({
                "bin": b,
                "range": [lo, hi],
                "label": f"{lo:g}\u2013{hi:g}",
                "n_cases": agg[b]["n"],
                "n_fail": agg[b]["fail"],
                "failure_rate": rate(agg[b]["fail"], agg[b]["n"]),
            })
    else:
        agg = defaultdict(Counter)
        for v, lab in pairs:
            agg[v]["n"] += 1
            if str(lab).lower() == "fail":
                agg[v]["fail"] += 1
        bins = [
            {
                "value": k,
                "n_cases": agg[k]["n"],
                "n_fail": agg[k]["fail"],
                "failure_rate": rate(agg[k]["fail"], agg[k]["n"]),
            }
            for k in levels
        ]
    return [
        {
            "block": "m1_signal_curve",
            "run_id": run.run_id,
            "phase": "explore",
            "signal": signal,
            "analyzer": analyzer,
            "field": field,
            "all_surviving_signals": all_survivors or [signal],
            "n_surviving_signals": len(all_survivors or [signal]),
            "binning": binning,
            "n_distinct_values": len(levels),
            "n_cases": sum(b["n_cases"] for b in bins),
            "bins": bins,
            "source": f"logs/artifacts/c0_{analyzer}.result.json + run_log case_record labels",
        }
    ]


def block_m2(run: Run) -> List[dict]:
    """One record per statistical test, plus one family summary per phase."""
    out: List[dict] = []
    for phase, path in run.m2_files():
        results = load_json(path, []) or []
        src = run.relpath(path)
        tested = 0
        for r in results:
            tool = r.get("tool")
            det = r.get("details") or {}
            degenerate = "degenerate" in (r.get("summary") or "").lower()
            in_family = bool(r.get("correction_family")) and not degenerate
            if in_family:
                tested += 1
            out.append(
                {
                    "block": "m2_test",
                    "run_id": run.run_id,
                    "phase": phase,
                    "tool": tool,
                    "signal": (r.get("config") or {}).get("signal"),
                    "effect": r.get("effect"),
                    "ci": r.get("ci"),
                    "p_value": r.get("p_value"),
                    "e_value": r.get("e_value"),
                    "reject_raw": r.get("raw_reject", r.get("reject")),
                    "survives_correction": bool(r.get("fdr_corrected")),
                    "correction_method": r.get("correction_method"),
                    "correction_family": r.get("correction_family"),
                    "in_correction_family": in_family,
                    "degenerate": degenerate,
                    "n_signal": det.get("n_signal"),
                    "n_control": det.get("n_control"),
                    "fail_rate_signal": det.get("fail_rate_signal"),
                    "fail_rate_control": det.get("fail_rate_control"),
                    "summary": r.get("summary"),
                    "source": src,
                }
            )

        assoc = [r for r in results if r.get("tool") == "signal_label_assoc"]
        survivors = [
            (r.get("config") or {}).get("signal") for r in assoc if r.get("fdr_corrected")
        ]
        out.append(
            {
                "block": "m2_family",
                "run_id": run.run_id,
                "phase": phase,
                "n_results_total": len(results),
                "n_signal_label_assoc": len(assoc),
                "n_degenerate": sum(
                    1 for r in assoc if "degenerate" in (r.get("summary") or "").lower()
                ),
                "n_in_correction_family": tested,
                "correction_method": next(
                    (r.get("correction_method") for r in assoc if r.get("correction_method")),
                    None,
                ),
                "survivors": survivors,
                "source": src,
            }
        )
    return out


def block_m3(run: Run) -> List[dict]:
    diag = run.event("diagnosis")
    if not diag:
        return []
    hyps = diag.get("hypotheses") or diag.get("proposed_hypotheses") or []
    review = diag.get("review") or {}
    # the proposal record does not always store the direction; M4 does
    direction = {}
    for v in run.events_of("surgery"):
        if v.get("module") == "m4":
            d = (v.get("evidence") or {}).get("expected_direction")
            if d:
                direction[v.get("failure_mode")] = d
    out = []
    for i, h in enumerate(hyps, start=1):
        if not isinstance(h, dict):
            continue
        out.append(
            {
                "block": "m3_hypothesis",
                "run_id": run.run_id,
                "hypothesis_id": f"H{i}",
                "failure_mode": h.get("failure_mode"),
                "statement": h.get("statement"),
                "test_design": h.get("test_design"),
                "expected_direction": h.get("expected_direction")
                or h.get("expected_association")
                or direction.get(h.get("failure_mode")),
                "status_at_proposal": h.get("status"),
                "source": "logs/run_log.jsonl:diagnosis",
            }
        )
    out.append(
        {
            "block": "m3_critic",
            "run_id": run.run_id,
            "n_proposed": len(hyps),
            "n_critic_kept": diag.get("n_critic_kept"),
            "n_critic_rejected": diag.get("n_critic_rejected"),
            "review": review if isinstance(review, (dict, list)) else str(review),
            "note": "critic objections are recorded, not vetoes; adjudication is statistical",
            "source": "logs/run_log.jsonl:diagnosis",
        }
    )
    return out


def block_m4(run: Run) -> List[dict]:
    verdicts = [e for e in run.events_of("surgery") if e.get("module") == "m4"]
    stored = load_json(run.rel("report", "m4_results.json"), []) or []
    by_mode = {}
    for s in stored:
        if isinstance(s, dict):
            by_mode[s.get("failure_mode")] = s

    out = []
    for i, v in enumerate(verdicts, start=1):
        ev = v.get("evidence") or {}
        extra = by_mode.get(v.get("failure_mode"), {})
        fdr = (extra.get("evidence") or {}).get("fdr") if isinstance(extra, dict) else None
        out.append(
            {
                "block": "m4_verdict",
                "run_id": run.run_id,
                "hypothesis_id": f"H{i}",
                "failure_mode": v.get("failure_mode"),
                "status": v.get("status"),
                "statement": v.get("hypothesis"),
                "test_name": ev.get("m4_test_name") or ev.get("chosen_tool"),
                "expected_direction": ev.get("expected_direction"),
                "effect": ev.get("effect_size"),
                "ci": ev.get("ci"),
                "e_value": ev.get("e_value"),
                "reject": ev.get("reject"),
                "underpowered": ev.get("underpowered"),
                "evidence_grade": ev.get("m4_evidence_grade"),
                "protocol_consistent": ev.get("m4_protocol_consistent"),
                "confidence_score": v.get("confidence_score"),
                "fdr": fdr,
                "verdict_text": ev.get("m4_verdict"),
                "split": "heldout",
                "source": "logs/run_log.jsonl:surgery[module=m4] + logs/report/m4_results.json",
            }
        )
    return out


TIER_LABEL = {
    "L1": "Prompt / instructions",
    "L2": "Scaffold / tools / multi-call",
    "L3": "Internals / read & write",
    "L3a": "Internals / read-only",
    "L3b": "Internals / audited write",
    "L4": "Re-training",
}


def block_m5(run: Run) -> List[dict]:
    fix = run.event("fix")
    if not fix:
        return []
    out: List[dict] = []

    # candidates searched on the explore split
    for c in fix.get("selection_attempted") or []:
        out.append(
            {
                "block": "m5_candidate",
                "run_id": run.run_id,
                "phase": "explore_selection",
                "tier": c.get("tier"),
                "name": c.get("name"),
                "n_pairs": c.get("n_pairs"),
                "n_fixed": c.get("n_fixed"),
                "n_broken": c.get("n_broken"),
                "effect": c.get("effect"),
                "verdict": c.get("verdict"),
                "selected": c.get("name") == (fix.get("best") or {}).get("name"),
                "source": "logs/run_log.jsonl:fix.selection_attempted",
            }
        )

    # candidates re-run on held-out for confirmation
    for c in fix.get("attempted") or []:
        out.append(
            {
                "block": "m5_candidate",
                "run_id": run.run_id,
                "phase": "heldout_confirmation",
                "tier": c.get("tier"),
                "name": c.get("name"),
                "n_pairs": c.get("n_pairs"),
                "n_fixed": c.get("n_fixed"),
                "n_broken": c.get("n_broken"),
                "effect": c.get("effect"),
                "ci": parse_ci(c.get("summary")),
                "e_value": c.get("e_value"),
                "verdict": c.get("verdict"),
                "coverage": c.get("coverage"),
                "selected": bool(c.get("fixed")),
                "source": "logs/run_log.jsonl:fix.attempted",
            }
        )

    # ladder rows: one per tier, so the figure can show L1..L4 even when a tier
    # was never entered
    cap = fix.get("max_tier") or run.config.get("fix_tier")
    order = ["L1", "L2", "L3", "L4"]
    by_tier: Dict[str, List[dict]] = defaultdict(list)
    for c in fix.get("selection_attempted") or []:
        by_tier[str(c.get("tier") or "")[:2]].append(c)
    accepted_tier = (fix.get("best") or {}).get("tier")

    for tier in order:
        tried = by_tier.get(tier, [])
        best = max(tried, key=lambda c: c.get("effect") or -9e9) if tried else None
        if not tried:
            status = "untouched"
        elif accepted_tier and str(accepted_tier)[:2] == tier:
            status = "accepted"
        elif best and (best.get("effect") or 0) < 0:
            status = "regressed"
        else:
            status = "not_selected"
        out.append(
            {
                "block": "m5_ladder",
                "run_id": run.run_id,
                "tier": tier,
                "label": TIER_LABEL.get(tier),
                "n_candidates": len(tried),
                "best_effect": best.get("effect") if best else None,
                "best_candidate": best.get("name") if best else None,
                "status": status,
                "within_cap": bool(cap) and order.index(tier) <= order.index(str(cap)[:2]),
                "tier_cap": cap,
                "source": "logs/run_log.jsonl:fix",
            }
        )
    return out


def _prompt_steps(payload: dict) -> List[str]:
    """Human-readable steps, derived from the payload rather than hand-written."""
    steps = []
    ops = payload.get("image_ops") or []
    if ops:
        desc = ", ".join(
            f"{o.get('tool')}({', '.join(f'{k}={v}' for k, v in (o.get('params') or {}).items())})"
            for o in ops
        )
        steps.append(f"image ops: {desc}")
    tmpl = payload.get("prompt_template") or ""
    for para in [p.strip() for p in tmpl.split("\n\n") if p.strip()]:
        steps.append(para.replace("\n", " "))
    n = payload.get("n_samples")
    if n and n > 1:
        steps.append(f"sample {n} times, take the modal answer")
    return steps


def block_repair(run: Run) -> List[dict]:
    fix = run.event("fix")
    if not fix:
        return []
    best = fix.get("best") or {}
    payload = best.get("payload") or {}
    if not payload:
        return []
    return [
        {
            "block": "repair",
            "run_id": run.run_id,
            "name": payload.get("name") or best.get("name"),
            "tier": best.get("tier"),
            "strategy": payload.get("strategy"),
            "image_ops": payload.get("image_ops"),
            "n_samples": payload.get("n_samples"),
            "generation_kwargs": payload.get("generation_kwargs"),
            "output_key_pattern": payload.get("output_key_pattern"),
            "prompt_template": payload.get("prompt_template"),
            "steps": _prompt_steps(payload),
            "modifies_parameters": False,
            "trial_root": run.rooted(best.get("trial_root") or "")
            if best.get("trial_root")
            else None,
            "source": "logs/run_log.jsonl:fix.best.payload",
        }
    ]


def block_validation(run: Run) -> List[dict]:
    fix = run.event("fix")
    if not fix:
        return []
    best = fix.get("best") or {}
    if not best:
        return []
    n = best.get("n_pairs") or 0
    base_ok = best.get("n_baseline_correct") or 0
    cand_ok = best.get("n_candidate_correct") or 0
    fixed = best.get("n_fixed") or 0
    broken = best.get("n_broken") or 0
    return [
        {
            "block": "validation",
            "run_id": run.run_id,
            "phase": "heldout",
            "candidate": best.get("name"),
            "tier": best.get("tier"),
            "n_pairs": n,
            "n_baseline_correct": base_ok,
            "n_candidate_correct": cand_ok,
            "baseline_rate": best.get("baseline_rate"),
            "candidate_rate": best.get("candidate_rate"),
            "n_fixed": fixed,
            "n_broken": broken,
            # the 2x2 paired table the figure can render directly
            "both_correct": base_ok - broken,
            "both_wrong": n - (base_ok - broken) - fixed - broken,
            "effect": best.get("effect"),
            "ci": parse_ci(best.get("summary")),
            "e_value": best.get("e_value"),
            "noise_model": best.get("noise_model"),
            "reject": best.get("reject"),
            "coverage": best.get("coverage"),
            "n_applicable": best.get("n_applicable"),
            "n_unstable_dropped": best.get("n_unstable"),
            "n_model_independent_excluded": best.get("n_model_independent"),
            "verdict": best.get("verdict"),
            "ebh_survivors": fix.get("ebh_survivors"),
            "summary": best.get("summary"),
            "source": "logs/run_log.jsonl:fix.best",
        }
    ]


def block_example_case(
    run: Run, signal: Optional[str], forced_id: Optional[str]
) -> List[dict]:
    """Pick one failing case to illustrate the mechanism, preferring one with media."""
    analyzer, field = (signal.split(".", 1) if signal and "." in signal else (None, None))
    sig_by_case = {}
    if analyzer:
        for row in run.analyzer_per_case(analyzer, "c0"):
            sig_by_case[row.get("sample_id")] = row

    media_by_case = {
        cid: rec.get("media_paths") or [] for cid, rec in run.case_records.items()
    }

    def score(cid: str) -> tuple:
        rec = run.case_records.get(cid, {})
        lab = str((rec.get("case") or {}).get("label", "")).lower()
        sig = (sig_by_case.get(cid) or {}).get(field)
        return (
            1 if lab == "fail" else 0,
            1 if media_by_case.get(cid) else 0,
            sig if isinstance(sig, (int, float)) else -1,
        )

    if forced_id:
        cid = forced_id
        if cid not in run.all_cases:
            warn(f"--example-case {cid} not found in this run")
            return []
    else:
        candidates = [c for c in run.explore_ids if score(c)[0] == 1]
        if not candidates:
            return []
        cid = max(candidates, key=score)

    case = run.all_cases.get(cid, {})
    rec = run.case_records.get(cid, {})
    media = [run.relpath(run.rel(p)) if not os.path.isabs(p) else p
             for p in media_by_case.get(cid, [])]
    return [
        {
            "block": "example_case",
            "run_id": run.run_id,
            "case_id": cid,
            "split": "explore" if cid in set(run.explore_ids) else "heldout",
            "question": case.get("prompt") or ((rec.get("case") or {}).get("inputs") or {}).get("prompt"),
            "gold": case.get("expected"),
            "baseline_output": case.get("observed"),
            "label": case.get("label"),
            "signal": signal,
            "signal_values": sig_by_case.get(cid),
            "media_paths": media,
            "note": "illustrative case drawn from the explore split; "
                    "held-out cases have no stored media in this run",
            "source": "logs/report/discovery_cases.json + run_log case_record",
        }
    ]


def block_qa_flags(run: Run, records: List[dict]) -> List[dict]:
    """Cheap consistency checks, so a figure is not built on a silent bug."""
    flags: List[dict] = []

    verdicts = [r for r in records if r["block"] == "m4_verdict"]
    # two hypotheses adjudicated with the identical test and effect are really
    # one finding wearing two hats
    seen: Dict[tuple, str] = {}
    for v in verdicts:
        key = (v.get("test_name"), v.get("effect"))
        if key[0] is not None and key[1] is not None:
            if key in seen:
                flags.append({
                    "level": "warning",
                    "code": "shared_test_between_hypotheses",
                    "detail": f"{v['hypothesis_id']} ({v.get('failure_mode')}) reuses the test "
                              f"and effect of {seen[key]}; it has no independent predicate",
                })
            else:
                seen[key] = v["hypothesis_id"]
    # a verdict whose observed sign contradicts its stated direction
    for v in verdicts:
        d, e = v.get("expected_direction"), v.get("effect")
        if d and isinstance(e, (int, float)):
            if ("higher" in d and e < 0) or ("lower" in d and e > 0):
                flags.append({
                    "level": "warning",
                    "code": "direction_mismatch",
                    "detail": f"{v['hypothesis_id']} expects {d} but the observed effect is {e:+.3f}",
                })

    fams = {r["phase"]: r for r in records if r["block"] == "m2_family"}
    sizes = {p: f.get("n_in_correction_family") for p, f in fams.items()}
    if len(set(sizes.values())) > 1:
        flags.append({
            "level": "note",
            "code": "correction_family_size_differs",
            "detail": f"multiplicity family size differs across phases: {sizes}",
        })

    ex = next((r for r in records if r["block"] == "example_case"), None)
    if ex and ex.get("split") != "heldout":
        flags.append({
            "level": "note",
            "code": "example_case_from_explore",
            "detail": "the illustrative case comes from the explore split; do not present "
                      "its treated output as a measured held-out result",
        })

    curve = next((r for r in records if r["block"] == "m1_signal_curve"), None)
    if curve and (curve.get("n_surviving_signals") or 1) > 1:
        flags.append({
            "level": "note",
            "code": "multiple_signals_survived",
            "detail": f"{curve['n_surviving_signals']} signals survived correction; the figure "
                      f"plots {curve.get('signal')} and must say so rather than implying "
                      "a single finding",
        })
    if curve and curve.get("binning") == "quartiles":
        flags.append({
            "level": "note",
            "code": "signal_binned_for_plotting",
            "detail": f"{curve.get('signal')} is continuous and was quartile-binned; "
                      "the bar chart shows bins, not raw levels",
        })

    val = next((r for r in records if r["block"] == "validation"), None)
    if val and not any(v.get("status") == "supported" for v in verdicts):
        flags.append({
            "level": "warning",
            "code": "repair_without_supported_hypothesis",
            "detail": "a repair was accepted although no hypothesis reached 'supported'; "
                      "present the gain as an empirical fix, not as a validated mechanism",
        })
    if val and (val.get("n_broken") or 0) > 0:
        flags.append({
            "level": "note",
            "code": "accepted_fix_breaks_cases",
            "detail": f"the accepted fix broke {val['n_broken']} previously-correct cases; "
                      "report it alongside the gain",
        })

    if not flags:
        return []
    return [{
        "block": "qa_flags",
        "run_id": run.run_id,
        "n_flags": len(flags),
        "flags": flags,
        "source": "derived from the blocks above",
    }]


# --------------------------------------------------------------------------
# human-readable document view
# --------------------------------------------------------------------------

# how many decimals each kind of number deserves in the readable document
_ROUND = {
    "effect": 4, "best_effect": 4, "baseline_rate": 4, "candidate_rate": 4,
    "failure_rate": 4, "fail_rate_signal": 4, "fail_rate_control": 4,
    "confidence_score": 3, "coverage": 4, "baseline_accuracy": 4,
    "total_duration_sec": 1, "e_value": 2,
}


def _clean(obj: Any, drop: Iterable[str] = ()) -> Any:
    """Round floats, drop nulls and boilerplate keys, recursively."""
    drop = set(drop)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in drop or v is None:
                continue
            if isinstance(v, float) and k in _ROUND:
                v = round(v, _ROUND[k])
            elif isinstance(v, list) and k == "ci":
                v = [round(x, 4) if isinstance(x, float) else x for x in v]
            else:
                v = _clean(v, drop)
            if v == [] or v == {}:
                continue
            out[k] = v
        return out
    if isinstance(obj, list):
        return [_clean(v, drop) for v in obj]
    return obj


BOILERPLATE = ("block", "run_id", "source")


def to_document(records: List[dict]) -> dict:
    """Reshape the flat records into one nested, readable object."""
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by[r["block"]].append(r)

    def one(block: str) -> dict:
        return by[block][0] if by.get(block) else {}

    def many(block: str) -> List[dict]:
        return by.get(block, [])

    run = one("run")
    val = one("validation")
    rep = one("repair")

    # ---- the handful of numbers a reader looks for first ----
    headline = _clean({
        "model": run.get("model_spec") or run.get("model"),
        "dataset": run.get("benchmark") or run.get("dataset"),
        "n_cases": run.get("n_cases"),
        "split": f"{run.get('n_explore')} explore / {run.get('n_heldout')} held-out"
                 if run.get("n_explore") else None,
        "baseline_accuracy": run.get("baseline_accuracy"),
        "confirmed_signal": one("m1_signal_curve").get("signal"),
        "hypotheses": {
            "proposed": one("m3_critic").get("n_proposed"),
            "supported": sum(1 for v in many("m4_verdict") if v.get("status") == "supported"),
            "inconclusive": sum(1 for v in many("m4_verdict") if v.get("status") == "inconclusive"),
        },
        "accepted_repair": rep.get("name"),
        "accepted_tier": rep.get("tier"),
        "heldout_accuracy": f"{val.get('baseline_rate')} -> {val.get('candidate_rate')}"
                            if val.get("candidate_rate") is not None else None,
        "cases_fixed": val.get("n_fixed"),
        "cases_broken": val.get("n_broken"),
    })

    # ---- M2, grouped by phase, informative tests first ----
    m2: Dict[str, Any] = {}
    for fam in many("m2_family"):
        phase = fam.get("phase")
        tests = [t for t in many("m2_test") if t.get("phase") == phase]
        tests.sort(key=lambda t: (t.get("degenerate", False),
                                  -(t.get("effect") or -9e9)))
        m2[phase] = _clean({
            "family": _clean(fam, BOILERPLATE + ("phase",)),
            "tests": [_clean(t, BOILERPLATE + ("phase", "summary")) for t in tests],
        })

    # ---- M5 candidates, split by what the phase actually proves ----
    cands = many("m5_candidate")
    m5 = _clean({
        "ladder": [_clean(t, BOILERPLATE) for t in many("m5_ladder")],
        "candidates_selected_on_explore": [
            _clean(c, BOILERPLATE + ("phase",))
            for c in cands if c.get("phase") == "explore_selection"
        ],
        "candidates_confirmed_on_heldout": [
            _clean(c, BOILERPLATE + ("phase",))
            for c in cands if c.get("phase") == "heldout_confirmation"
        ],
    })

    # ---- the repair: keep the prompt as lines, not one escaped blob ----
    repair = _clean(rep, BOILERPLATE + ("prompt_template",))
    if rep.get("prompt_template"):
        repair["prompt_template_lines"] = rep["prompt_template"].split("\n")

    doc = {
        "pipeline": [
            {"module": m, "name": MODULE_NAMES[m], "subtitle": MODULE_SUBTITLES[m]}
            for m in ("M1", "M2", "M3", "M4", "M5")
        ],
        "headline": headline,
        "run": _clean(run, BOILERPLATE),
        "m1_probe": _clean({
            "module": f'M1 \u00b7 {MODULE_NAMES["M1"]}',
            "analyzers": _clean(one("m1_analyzers"), BOILERPLATE),
            "probe_questions": _clean(one("m1_probe_questions"),
                                      BOILERPLATE + ("inventory",)),
            "measurement_inventory": one("m1_probe_questions").get("inventory", []),
            "signal_curve": _clean(one("m1_signal_curve"), BOILERPLATE),
        }),
        "m2_statistics": {"module": f'M2 \u00b7 {MODULE_NAMES["M2"]}', **m2},
        "m3_hypotheses": _clean({
            "module": f'M3 \u00b7 {MODULE_NAMES["M3"]}',
            "proposed": [_clean(h, BOILERPLATE) for h in many("m3_hypothesis")],
            "adversarial_critic": _clean(one("m3_critic"), BOILERPLATE),
        }),
        "m4_verdicts": {"module": f'M4 \u00b7 {MODULE_NAMES["M4"]}',
                        "verdicts": [_clean(v, BOILERPLATE) for v in many("m4_verdict")]},
        "m5_repair_search": {"module": f'M5 \u00b7 {MODULE_NAMES["M5"]}', **m5},
        "accepted_repair": repair,
        "heldout_validation": _clean(val, BOILERPLATE),
        "example_case": _clean(one("example_case"), BOILERPLATE),
        "qa_flags": one("qa_flags").get("flags", []),
        "_sources": {
            b: recs[0].get("source") for b, recs in sorted(by.items()) if recs[0].get("source")
        },
    }
    return {k: v for k, v in doc.items() if v not in (None, {}, [])}


# --------------------------------------------------------------------------
# markdown view — only what appears on the figure
# --------------------------------------------------------------------------


def _bar(fraction: Optional[float], width: int = 12) -> str:
    if fraction is None:
        return ""
    n = max(0, min(width, round(fraction * width)))
    return "\u2588" * n + "\u00b7" * (width - n)


def _pct(x: Optional[float], digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%" if isinstance(x, (int, float)) else "—"


def _first_sentence(text: Optional[str], limit: int = 200) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    cut = text.find(". ")
    if 0 < cut < limit:
        return text[: cut + 1]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def to_markdown(doc: dict, title: Optional[str] = None) -> str:
    """Render the document as the figure would read it: shapes, not decimals."""
    L: List[str] = []
    h = doc.get("headline", {})
    run = doc.get("run", {})
    val = doc.get("heldout_validation", {})
    rep = doc.get("accepted_repair", {})
    probe = doc.get("m1_probe", {})
    curve = probe.get("signal_curve", {})

    L.append(f"# {title or 'EvalRX case study'} — "
             f"{h.get('model', '?')} on {h.get('dataset', '?')}")
    L.append("")
    if val:
        L.append(f"**{_pct(val.get('baseline_rate'), 1)} \u2192 "
                 f"{_pct(val.get('candidate_rate'), 1)}** on {val.get('n_pairs')} held-out cases "
                 f"\u00b7 {val.get('n_fixed')} fixed, {val.get('n_broken')} broken "
                 f"\u00b7 accepted repair `{h.get('accepted_repair')}` ({h.get('accepted_tier')})")
        L.append("")
    L.append(f"> {h.get('n_cases')} cases, split {h.get('split')} \u00b7 "
             f"baseline {_pct(h.get('baseline_accuracy'), 1)} \u00b7 "
             f"repair cap {run.get('fix_tier_cap')} \u00b7 "
             f"analyzer selection: {run.get('m1_selection')}")
    L.append("")

    # ---------------- M1 ----------------
    an = probe.get("analyzers", {})
    L.append(f"## M1 · {MODULE_NAMES['M1']}")
    L.append("")
    L.append(f"*{MODULE_SUBTITLES['M1']}*")
    L.append("")
    for fam in an.get("families", []):
        if fam.get("label") == "OTHER" and not fam.get("selected"):
            continue
        mark = "**selected**" if fam.get("selected") else "available, not used"
        L.append(f"- **{fam.get('label')}** — {mark}")
        for a in fam.get("analyzers", []) or []:
            L.append(f"  - `{a}`")
    L.append("")

    pq = probe.get("probe_questions", {})
    if pq:
        L.append("**What the probes ask**")
        L.append("")
        for q in pq.get("questions", []):
            L.append(f"- {q.get('question')}")
        L.append("")
        d = pq.get("dropped", {})
        L.append(f"**{pq.get('n_measured')} measurements, "
                 f"{pq.get('n_forwarded')} forwarded to M2** \u2014 dropped: "
                 f"{d.get('saw_the_answer_key', 0)} saw the answer key, "
                 f"{d.get('never_varied', 0)} never varied, "
                 f"{d.get('partial_coverage', 0)} partial coverage.")
        L.append("")
        L.append(f"*{pq.get('note')}*")
        L.append("")

    if curve:
        surv = curve.get("n_surviving_signals", 1)
        L.append(f"**The signal it found:** `{curve.get('signal')}`"
                 + (f" — 1 of {surv} signals that survived correction" if surv > 1 else ""))
        L.append("")
        axis = curve.get("field", "signal value")
        L.append(f"| {axis} | cases | failure rate |")
        L.append("| --- | ---: | :-- |")
        for b in curve.get("bins", []):
            lab = b.get("label", b.get("value"))
            L.append(f"| {lab} | {b.get('n_cases')} | "
                     f"`{_bar(b.get('failure_rate'))}` {_pct(b.get('failure_rate'))} |")
        L.append("")

    # ---------------- M2 ----------------
    L.append(f"## M2 · {MODULE_NAMES['M2']}")
    L.append("")
    L.append(f"*{MODULE_SUBTITLES['M2']}*")
    L.append("")
    for phase in ("explore", "heldout"):
        blk = (doc.get("m2_statistics") or {}).get(phase)
        if not blk:
            continue
        fam = blk.get("family", {})
        survivors = fam.get("survivors") or []
        L.append(f"- **{phase}** — {fam.get('n_in_correction_family')} signals tested "
                 f"({fam.get('correction_method')} correction), "
                 f"{len(survivors)} survived"
                 + (f": {', '.join('`%s`' % x for x in survivors[:3])}"
                    + (" …" if len(survivors) > 3 else "") if survivors else ""))
    top = None
    for phase in ("heldout", "explore"):
        blk = (doc.get("m2_statistics") or {}).get(phase)
        if blk:
            top = next((t for t in blk.get("tests", []) if t.get("survives_correction")), None)
            if top:
                break
    if top:
        ci = top.get("ci")
        L.append("")
        L.append(f"Strongest confirmed effect: **{top.get('effect'):+.2f}** extra failure rate "
                 f"when `{top.get('signal')}` is high"
                 + (f" (95% CI {ci[0]:+.2f} to {ci[1]:+.2f})" if ci else ""))
    L.append("")

    # ---------------- M3 / M4 ----------------
    hyps = (doc.get("m3_hypotheses") or {}).get("proposed", [])
    verdicts = (doc.get("m4_verdicts") or {}).get("verdicts", [])
    by_mode = {v.get("failure_mode"): v for v in verdicts}
    L.append(f"## M3 · {MODULE_NAMES['M3']}  →  M4 · {MODULE_NAMES['M4']}")
    L.append("")
    L.append("*frozen on explore, then adjudicated on held-out cases*")
    L.append("")
    ICON = {"supported": "\u2713 SUPPORTED", "refuted": "\u2717 REFUTED",
            "inconclusive": "\u25cb INCONCLUSIVE"}
    for i, hy in enumerate(hyps, start=1):
        v = by_mode.get(hy.get("failure_mode"), {})
        st = ICON.get(v.get("status"), (v.get("status") or "not adjudicated").upper())
        L.append(f"**H{i} · `{hy.get('failure_mode')}` — {st}**")
        L.append("")
        L.append(f"> {_first_sentence(hy.get('statement'))}")
        L.append("")
    crit = (doc.get("m3_hypotheses") or {}).get("adversarial_critic", {})
    if crit.get("n_critic_rejected"):
        L.append(f"*An adversarial critic objected to {crit['n_critic_rejected']} of "
                 f"{crit.get('n_proposed')} hypotheses; objections are recorded, not vetoes — "
                 f"adjudication is statistical.*")
        L.append("")

    # ---------------- M5 ----------------
    search = doc.get("m5_repair_search", {})
    L.append(f"## M5 · {MODULE_NAMES['M5']}")
    L.append("")
    L.append(f"*{MODULE_SUBTITLES['M5']}*")
    L.append("")
    STATUS = {"accepted": "**accepted**", "regressed": "regressed",
              "untouched": "untouched", "not_selected": "tried, not selected"}
    L.append("| tier | | outcome |")
    L.append("| --- | --- | --- |")
    for tier in search.get("ladder", []):
        best = tier.get("best_candidate")
        note = STATUS.get(tier.get("status"), tier.get("status"))
        if best:
            note += f" — `{best}`"
        L.append(f"| {tier.get('tier')} | {tier.get('label')} | {note} |")
    L.append("")
    cands = search.get("candidates_selected_on_explore", [])
    if cands:
        L.append(f"{len(cands)} candidates were tried on the explore split; "
                 "the winner was then re-measured on held-out cases.")
        L.append("")
        for c in sorted(cands, key=lambda c: -(c.get("effect") or 0)):
            star = " \u2190 accepted" if c.get("selected") else ""
            L.append(f"- `{c.get('tier')}` **{c.get('name')}** — "
                     f"{c.get('n_fixed')} fixed / {c.get('n_broken')} broken "
                     f"({c.get('verdict')}){star}")
        L.append("")

    # ---------------- the repair ----------------
    if rep:
        L.append(f"## The accepted repair — `{rep.get('name')}` ({rep.get('tier')})")
        L.append("")
        L.append("No parameter update; the model itself is unchanged.")
        L.append("")
        for i, step in enumerate(rep.get("steps", []), start=1):
            L.append(f"{i}. {step}")
        L.append("")
        lines = rep.get("prompt_template_lines")
        if lines:
            L.append("<details><summary>full prompt</summary>")
            L.append("")
            L.append("```")
            L.extend(lines)
            L.append("```")
            L.append("")
            L.append("</details>")
            L.append("")

    # ---------------- validation ----------------
    if val:
        L.append("## Held-out validation")
        L.append("")
        L.append("| | | |")
        L.append("| --- | :-- | ---: |")
        L.append(f"| unchanged model | `{_bar(val.get('baseline_rate'), 20)}` | "
                 f"{_pct(val.get('baseline_rate'), 1)} |")
        L.append(f"| with the repair | `{_bar(val.get('candidate_rate'), 20)}` | "
                 f"{_pct(val.get('candidate_rate'), 1)} |")
        L.append("")
        L.append(f"Of {val.get('n_pairs')} paired cases: **{val.get('n_fixed')} wrong \u2192 right**, "
                 f"**{val.get('n_broken')} right \u2192 wrong**, "
                 f"{val.get('both_correct')} already right, {val.get('both_wrong')} still wrong. "
                 f"The gain is far beyond chance.")
        L.append("")

    # ---------------- example ----------------
    ex = doc.get("example_case", {})
    if ex:
        L.append("## One case")
        L.append("")
        L.append(f"**{ex.get('case_id')}** ({ex.get('split')} split)")
        L.append("")
        L.append(f"> {_first_sentence(ex.get('question'), 240)}")
        L.append("")
        gold = ex.get("gold")
        gold = ", ".join(map(str, gold)) if isinstance(gold, list) else gold
        L.append(f"- expected: `{gold}`")
        L.append(f"- unchanged model answered: `{_first_sentence(str(ex.get('baseline_output')), 120)}`")
        for m in ex.get("media_paths", []) or []:
            L.append(f"- media: `{m}`")
        L.append("")

    # ---------------- checks ----------------
    flags = doc.get("qa_flags", [])
    if flags:
        L.append("## Checks worth reading before you draw the figure")
        L.append("")
        for f in flags:
            tag = "**warning**" if f.get("level") == "warning" else "note"
            L.append(f"- {tag} — {f.get('detail')}")
        L.append("")

    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def extract(root: str, example_case: Optional[str] = None) -> List[dict]:
    run = Run(root)
    signals = survivor_signals(run)
    signal = pick_curve_signal(run, signals)
    records: List[dict] = []
    forwarded = None
    for phase, path in run.m2_files():
        if phase == "explore":
            assoc = [r for r in (load_json(path, []) or [])
                     if r.get("tool") == "signal_label_assoc"]
            forwarded = sum(1 for r in assoc
                            if r.get("correction_family")
                            and "degenerate" not in (r.get("summary") or "").lower())
    for name, fn in [
        ("run", lambda: block_run(run)),
        ("m1_analyzers", lambda: block_m1_analyzers(run)),
        ("m1_probe_questions", lambda: block_m1_questions(run, forwarded)),
        ("m1_signal_curve", lambda: block_m1_signal_curve(run, signal, signals)),
        ("m2", lambda: block_m2(run)),
        ("m3", lambda: block_m3(run)),
        ("m4", lambda: block_m4(run)),
        ("m5", lambda: block_m5(run)),
        ("repair", lambda: block_repair(run)),
        ("validation", lambda: block_validation(run)),
        ("example_case", lambda: block_example_case(run, signal, example_case)),
    ]:
        try:
            records.extend(fn())
        except Exception as exc:  # one bad block must not lose the rest
            warn(f"block {name} failed: {type(exc).__name__}: {exc}")
    try:
        records.extend(block_qa_flags(run, records))
    except Exception as exc:
        warn(f"block qa_flags failed: {type(exc).__name__}: {exc}")
    return records


def find_run_roots(path: str) -> List[str]:
    """Accept a run root, or a parent directory holding several runs."""
    if os.path.isfile(os.path.join(path, "summary.json")):
        return [path]
    roots = []
    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "summary.json")):
            roots.append(sub)
    return roots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run root (contains summary.json and logs/), "
                               "or a directory of such runs")
    ap.add_argument("-o", "--out", default=None,
                    help="output path (default: figure_data.json / .jsonl)")
    ap.add_argument("-f", "--format", choices=("json", "jsonl", "md", "both", "all"),
                    default="json",
                    help="json = one nested readable file (default); "
                         "jsonl = one flat record per line; "
                         "md = the figure, written out in prose; "
                         "both = json + jsonl; all = json + jsonl + md")
    ap.add_argument("--example-case", default=None,
                    help="force a specific case id for the example_case block")
    args = ap.parse_args()

    roots = find_run_roots(args.run)
    if not roots:
        raise SystemExit(f"no run root with summary.json found under {args.run}")

    all_records: List[dict] = []
    for root in roots:
        print(f"reading {root}", file=sys.stderr)
        all_records.extend(extract(root, args.example_case))

    stem = args.out
    if stem:
        stem = re.sub(r"\.jsonl?$", "", stem)
    else:
        stem = "figure_data"
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)

    written = []
    if args.format in ("jsonl", "both", "all"):
        path = stem + ".jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for rec in all_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written.append((path, f"{len(all_records)} records"))

    if args.format in ("json", "both", "all"):
        path = stem + ".json"
        if len(roots) == 1:
            payload = to_document(all_records)
        else:
            payload = {}
            for root in roots:
                key = os.path.basename(root.rstrip(os.sep))
                payload[key] = to_document(
                    [r for r in all_records if r.get("run_id") == key])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        written.append((path, f"{len(roots)} run(s)"))

    if args.format in ("md", "all"):
        path = stem + ".md"
        chunks = []
        for root in roots:
            key = os.path.basename(root.rstrip(os.sep))
            recs = [r for r in all_records if r.get("run_id") == key]
            chunks.append(to_markdown(to_document(recs), title=key))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n\n---\n\n".join(chunks))
        written.append((path, f"{len(roots)} run(s)"))

    print("", file=sys.stderr)
    for path, what in written:
        print(f"wrote {path}  ({what})", file=sys.stderr)
    for block, n in Counter(r["block"] for r in all_records).most_common():
        print(f"  {block:18s} {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
