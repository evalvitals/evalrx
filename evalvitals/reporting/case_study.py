"""The case-study sheet: one run's failure-to-repair story, as report data.

The served report already carries every stage's evidence, but spread across
five views a reader has to assemble themselves. This module compiles the same
run into the one-page shape the case-study figure uses — what the probes asked,
which signals survived correction, which hypotheses held up on held-out cases,
which rung of the repair ladder held — so the UI can render it as a single
sheet under the overview.

It reads the run's own artifacts and nothing else, and it is the reporting-side
twin of ``examples/benchmark/tools/extract_figure_data.py``: same block names,
same field names, so a figure drawn from the CLI tool and this sheet cannot
disagree about what a number means.

Every block degrades on its own. A run that never reached M4 still gets its M1
and M2 sections; a run with no probe artifacts at all yields ``None`` and the
section is dropped rather than rendered empty.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CI_RE = re.compile(r"CI=([+-][\d.]+)\.\.([+-][\d.]+)")

# Analyzer -> family, so M1 reads as three kinds of probe rather than a list of
# module names. Unknown analyzers fall back to keyword matching, then "other".
ANALYZER_FAMILY = {
    "answer_extraction_audit": "black_box_behavior",
    "termination_audit": "black_box_behavior",
    "selfcheck_consistency": "black_box_behavior",
    "self_consistency": "black_box_behavior",
    "calibration": "black_box_behavior",
    "coverage_verification_gap": "black_box_behavior",
    "format_sensitivity": "black_box_behavior",
    "perturbation_battery": "black_box_behavior",
    "prompt_contrast": "black_box_behavior",
    "cot_faithfulness": "black_box_behavior",
    "arith_audit": "black_box_behavior",
    "knowledge_split": "black_box_behavior",
    "self_repair": "black_box_behavior",
    "contamination": "black_box_behavior",
    "step_rollout_value": "black_box_behavior",
    "logprob_entropy": "internal_white_box",
    "logit_lens": "internal_white_box",
    "layer_contrast": "internal_white_box",
    "linear_probe": "internal_white_box",
    "relative_attn": "internal_white_box",
    "context_shap": "internal_white_box",
    "mm_shap": "multimodal",
    "vl_shap": "multimodal",
    "modality_ablation": "multimodal",
    "pope": "multimodal",
    "chair": "multimodal",
    "opera": "multimodal",
    "vcd": "multimodal",
}
FAMILY_KEYWORDS = (
    ("attn", "internal_white_box"), ("attention", "internal_white_box"),
    ("logit", "internal_white_box"), ("logprob", "internal_white_box"),
    ("lens", "internal_white_box"), ("probe", "internal_white_box"),
    ("shap", "multimodal"), ("hallu", "multimodal"), ("modality", "multimodal"),
    ("visual", "multimodal"), ("audio", "multimodal"),
)
FAMILY_LABEL = {
    "black_box_behavior": "BLACK-BOX",
    "internal_white_box": "INTERNAL / WHITE-BOX",
    "multimodal": "MULTIMODAL",
    "other": "OTHER",
}

# What a probe is asking, in the reader's language, keyed by the per-case field
# it emits — one analyzer often asks two questions, and one question is often
# asked by two analyzers.
QUESTION_BY_FIELD = {
    "matches_output_contract": "Did it answer in the form we asked for?",
    "has_answer_tag": "Did it answer in the form we asked for?",
    "n_unparsed": "Did it answer in the form we asked for?",
    "gave_up": "Did it give up answering?",
    "has_output": "Did it produce anything at all?",
    "answered_yes": "Does it lean to one answer regardless?",
    "gold_yes": "Does it lean to one answer regardless?",
    "output_chars": "Did it stop, or keep talking?",
    "output_words": "Did it stop, or keep talking?",
    "output_truncated": "Did it stop, or keep talking?",
    "looks_truncated": "Did it stop, or keep talking?",
    "continuation_chars": "Did it stop, or keep talking?",
    "repetition_score": "Did it stop, or keep talking?",
    "n_unique": "Same question five times, same answer?",
    "majority_share": "Same question five times, same answer?",
    "n_samples": "Same question five times, same answer?",
    "n_graded": "Same question five times, same answer?",
    "coverage_gap": "Did it ever produce the right answer?",
    "format_flip_rate": "Reworded question, same answer?",
    "positional_bias": "Reworded question, same answer?",
    "n_variants": "Reworded question, same answer?",
    "n_options": "Reworded question, same answer?",
    "noop_clause_flipped": "Reworded question, same answer?",
    "restate_question_flipped": "Reworded question, same answer?",
    "invariance_break_rate": "Reworded question, same answer?",
    "n_sentences": "Asked to check itself, does it change?",
    "selfcheck_inconsistency": "Asked to check itself, does it change?",
    "selfcheck_worst_sentence": "Asked to check itself, does it change?",
    "conf_logprob": "Was it as sure as it sounded?",
    "conf_verbal": "Was it as sure as it sounded?",
    "attention_entropy": "Where was it looking?",
    "attention_to_region": "Where was it looking?",
    "logprob_entropy": "How uncertain is it inside?",
    "token_entropy": "How uncertain is it inside?",
}
QUESTION_BY_ANALYZER = {
    "answer_extraction_audit": "Did the answer come out readable?",
    "termination_audit": "Did it stop, or keep talking?",
    "selfcheck_consistency": "Asked to check itself, does it change?",
    "self_consistency": "Same question five times, same answer?",
    "coverage_verification_gap": "Did it ever produce the right answer?",
    "format_sensitivity": "Reworded question, same answer?",
    "perturbation_battery": "Reworded question, same answer?",
    "calibration": "Was it as sure as it sounded?",
    "logprob_entropy": "How uncertain is it inside?",
    "attention": "Where was it looking?",
    "hallucination": "Did it describe something that is not there?",
    "multimodal_attribution": "Did the answer use the image or audio at all?",
    "loop_detection": "Did it get stuck repeating itself?",
}
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
# Fields computed from the answer key. Testing one would let a "signal" predict
# the label from the label, so they are excluded before anything is counted.
OUTCOME_DERIVED = {
    "labelled_fail", "strict_match", "gold_in_output", "gold_in_answer_region",
    "label_disagrees", "extraction_suspect", "extraction_point_miss",
    "n_correct", "any_correct", "majority_correct", "pass_at_k", "correct",
    "continuation_correct", "continuation_has_answer", "recovered_by_continuation",
}
TIER_LABEL = {
    "L1": "Prompt / instructions",
    "L2": "Scaffold / tools / multi-call",
    "L3": "Internals / read & write",
    "L3a": "Internals / read & write",
    "L3b": "Internals / read & write",
    "L4": "Re-training",
}
MODULES = [
    ("M1", "Suspicious behavior detection",
     "Run the probing library, find per-case suspicious behaviors."),
    ("M2", "Statistical screening",
     "Using plots to explain, statistical tests to decide."),
    ("M3", "Hypothesis formation",
     "Explore the reason behind the signals."),
    ("M5", "Held-out verification",
     "Confirm the hypothesis over cases it never saw."),
    ("M4", "Validated repair",
     "Climb the repair ladder until something holds."),
]


# What each probe measures, as a phrase rather than a module name: the card is
# read by someone who does not know the analyzer registry. Unknown analyzers
# fall back to their identifier with the underscores taken out, which is still
# readable and never wrong.
ANALYZER_PHRASE = {
    "answer_extraction_audit": "answer-extraction audit",
    "termination_audit": "termination / truncation audit",
    "selfcheck_consistency": "self-check consistency",
    "self_consistency": "consistency across 5 resamples",
    "calibration": "confidence calibration",
    "coverage_verification_gap": "coverage / verification gap",
    "format_sensitivity": "format sensitivity",
    "perturbation_battery": "perturbation battery",
    "prompt_contrast": "prompt contrast",
    "cot_faithfulness": "chain-of-thought faithfulness",
    "arith_audit": "arithmetic audit",
    "knowledge_split": "knowledge vs reasoning split",
    "self_repair": "self-repair on re-ask",
    "contamination": "contamination check",
    "step_rollout_value": "step rollout value",
    "logprob_entropy": "logits & representations",
    "logit_lens": "logit lens",
    "layer_contrast": "layer contrast",
    "linear_probe": "linear probe",
    "relative_attn": "attention & attribution",
    "context_shap": "context attribution",
    "mm_shap": "modality attribution",
    "vl_shap": "vision-language attribution",
    "modality_ablation": "modality ablation",
    "pope": "grounding & hallucination",
    "chair": "caption hallucination",
    "opera": "over-trust decoding",
    "vcd": "contrastive decoding",
}
# The probe menu each family offers, as the figure prints it: a short fixed
# vocabulary rather than whatever this run happened to select, so two runs are
# comparable card to card. A row lights up when the run used any probe under it;
# the funnel below the list is what accounts for every measurement taken.
FAMILY_MENU = {
    "black_box_behavior": [
        ("consistency across 5 resamples", {"self_consistency", "selfcheck_consistency"}),
        ("coverage / verification gap", {"coverage_verification_gap"}),
        ("answer-extraction audit", {"answer_extraction_audit", "termination_audit",
                                     "format_sensitivity", "perturbation_battery"}),
    ],
    "internal_white_box": [
        ("attention & attribution", {"relative_attn", "context_shap", "linear_probe"}),
        ("logits & representations", {"logprob_entropy", "logit_lens", "layer_contrast"}),
    ],
    "multimodal": [
        ("grounding & hallucination", {"pope", "chair", "opera", "vcd"}),
        ("modality attribution", {"mm_shap", "vl_shap", "modality_ablation"}),
    ],
    "other": [],
}


def phrase_for(analyzer: str) -> str:
    return ANALYZER_PHRASE.get(analyzer, analyzer.replace("_", " "))


def family_of(analyzer: str) -> str:
    if analyzer in ANALYZER_FAMILY:
        return ANALYZER_FAMILY[analyzer]
    lowered = analyzer.lower()
    for needle, family in FAMILY_KEYWORDS:
        if needle in lowered:
            return family
    return "other"


def question_for(analyzer: str, field: str) -> str:
    if field in QUESTION_BY_FIELD:
        return QUESTION_BY_FIELD[field]
    return QUESTION_BY_ANALYZER.get(analyzer, f"What does {analyzer} show?")


ANSWER_LINE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*[::]\s*(.+)", re.IGNORECASE)


def final_answer(output: Any) -> "str | None":
    """The answer a long generation ended on, not the reasoning it opened with.

    A card has room for a line or two, and the head of a chain of thought is the
    least informative part of it: the claim the case turns on is the last thing
    written. Returns None when the output declares no answer, so the caller can
    fall back rather than present a guess as the model's answer.
    """
    text = str(output or "").strip()
    if not text:
        return None
    matches = ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].strip().strip("*` ")
    tail = [line.strip() for line in text.splitlines() if line.strip()]
    return tail[-1] if tail else None


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _parse_ci(summary: Any) -> "list[float] | None":
    match = CI_RE.search(str(summary or ""))
    return [float(match.group(1)), float(match.group(2))] if match else None


def _rate(numerator: int, denominator: int) -> "float | None":
    return round(numerator / denominator, 4) if denominator else None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Run:
    """The artifacts one run wrote, whichever level of it was zipped up.

    ``root`` is whatever the report was loaded from: the directory holding
    ``run_log.jsonl`` (what the server finds in an archive) or the run directory
    above it. Both are accepted, because both are what people actually zip.
    """

    def __init__(self, root: Path, events: Sequence[Mapping[str, Any]]):
        root = Path(root)
        if (root / "run_log.jsonl").exists() or (root / "artifacts").is_dir():
            self.logs, self.run_dir = root, root.parent
        else:
            self.logs = next((p for p in sorted(root.glob("logs*")) if p.is_dir()), root)
            self.run_dir = root
        self.events = list(events)
        self.summary = _load(self.run_dir / "summary.json", {}) or {}
        self.manifest = _load(self.logs / "manifest.json", {}) or {}
        self.config = self.manifest.get("config") or {}
        self.case_records = {
            event["case_id"]: event
            for event in self.events
            if event.get("event") == "case_record" and event.get("case_id")
        }
        self.all_cases = {
            case["id"]: case
            for case in (_load(self.logs / "report" / "discovery_cases.json", []) or [])
            if isinstance(case, dict) and "id" in case
        }

    def event(self, name: str) -> "Mapping[str, Any] | None":
        return next((e for e in self.events if e.get("event") == name), None)

    def events_of(self, name: str) -> "list[Mapping[str, Any]]":
        return [e for e in self.events if e.get("event") == name]

    def per_case(self, analyzer: str, cycle_prefix: str = "c0") -> "list[dict[str, Any]]":
        blob = _load(self.logs / "artifacts" / f"{cycle_prefix}_{analyzer}.result.json", {}) or {}
        rows = (blob.get("findings") or {}).get("per_case") or []
        return [row for row in rows if isinstance(row, dict)]

    def m2_files(self) -> "list[tuple[str, Path]]":
        """[(phase, path)] for every M2 results file, explore before held-out."""
        artifacts = self.logs / "artifacts"
        if not artifacts.is_dir():
            return []
        found = []
        for path in sorted(artifacts.glob("*_m2_stats_results.json")):
            name = path.name
            if name.startswith(("post_", "c-1_")):
                phase = "heldout"
            elif name.startswith("c0_"):
                phase = "explore"
            else:
                phase = name.split("_m2_")[0]
            found.append((phase, path))
        found.sort(key=lambda item: 0 if item[0] == "explore" else 1)
        return found

    def measurements(self, analyzers: Iterable[str]) -> "list[dict[str, Any]]":
        """Every numeric per-case field a probe produced, with why it was kept."""
        out: list[dict[str, Any]] = []
        for analyzer in analyzers:
            rows = self.per_case(analyzer)
            if not rows:
                continue
            fields: dict[str, list[Any]] = defaultdict(list)
            for row in rows:
                for key, value in row.items():
                    if _number(value):
                        fields[key].append(value)
            for field, values in sorted(fields.items()):
                if field in OUTCOME_DERIVED:
                    status = "dropped_sees_answer_key"
                elif len(set(values)) == 1:
                    status = "dropped_never_varies"
                elif len(values) < len(rows):
                    status = "dropped_partial_coverage"
                else:
                    status = "candidate"
                out.append({
                    "analyzer": analyzer, "field": field, "signal": f"{analyzer}.{field}",
                    "n_values": len(values), "n_distinct": len(set(values)),
                    "status": status, "question": question_for(analyzer, field),
                })
        return out


def _m1(run: _Run) -> "dict[str, Any] | None":
    probes = run.events_of("probe")
    if not probes:
        return None
    selected = probes[0].get("selected_analyzers") or probes[0].get("analyzers") or []
    grouped: dict[str, list[str]] = defaultdict(list)
    for analyzer in selected:
        grouped[family_of(analyzer)].append(analyzer)
    families = []
    for key, label in FAMILY_LABEL.items():
        members = grouped.get(key, [])
        if key == "other" and not members:
            continue
        probes = []
        for phrase, owners in FAMILY_MENU.get(key, []):
            used = sorted(owners.intersection(members))
            probes.append({"phrase": phrase, "used": bool(used), "analyzers": used})
        # a selected analyzer the menu does not name still has to appear, or the
        # card would say the run probed something it did not
        named = {analyzer for _phrase, owners in FAMILY_MENU.get(key, []) for analyzer in owners}
        for analyzer in members:
            if analyzer not in named:
                probes.append({"phrase": phrase_for(analyzer), "used": True, "analyzers": [analyzer]})
        families.append({
            "family": key, "label": label, "selected": bool(members),
            "analyzers": members, "probes": probes,
        })

    inventory = run.measurements(selected)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        if row["status"] != "dropped_sees_answer_key":
            by_question[row["question"]].append(row)

    def rank(question: str) -> tuple:
        index = QUESTION_ORDER.index(question) if question in QUESTION_ORDER else len(QUESTION_ORDER)
        return (index, question)

    questions = []
    for question in sorted(by_question, key=rank):
        rows = by_question[question]
        questions.append({
            "question": question,
            "analyzers": sorted({row["analyzer"] for row in rows}),
            "n_measurements": len(rows),
            "n_candidates": sum(1 for row in rows if row["status"] == "candidate"),
        })
    dropped = Counter(row["status"] for row in inventory)
    return {
        "selection_mode": run.config.get("m1_selection"),
        "selected_analyzers": list(selected),
        "families": families,
        "questions": questions,
        "n_analyzers": len(selected),
        "n_measured": len(inventory),
        "dropped": {
            "saw_the_answer_key": dropped.get("dropped_sees_answer_key", 0),
            "never_varied": dropped.get("dropped_never_varies", 0),
            "partial_coverage": dropped.get("dropped_partial_coverage", 0),
        },
        "note": "each case is answered five times; the probes never see the answer key",
    }


def _survivors(run: _Run) -> "list[str]":
    """Signals that survived correction, preferring the held-out pass."""
    for _phase, path in reversed(run.m2_files()):
        found = [
            (row.get("config") or {}).get("signal")
            for row in (_load(path, []) or [])
            if row.get("fdr_corrected")
        ]
        if found:
            return [signal for signal in found if signal]
    return []


def _signal_curve(run: _Run, signals: Sequence[str]) -> "dict[str, Any] | None":
    """Failure rate binned by the confirmed signal — M1's bar chart."""
    effects: dict[str, float] = {}
    for _phase, path in run.m2_files():
        for row in _load(path, []) or []:
            if row.get("tool") == "signal_label_assoc" and row.get("effect") is not None:
                signal = (row.get("config") or {}).get("signal")
                if signal:
                    effects[signal] = row["effect"]
    best, best_key = None, None
    for signal in signals:
        if "." not in signal:
            continue
        analyzer, field = signal.split(".", 1)
        values = [row.get(field) for row in run.per_case(analyzer)]
        values = [value for value in values if value is not None]
        if not values:
            continue
        cardinality = len(set(values))
        band = 0 if 3 <= cardinality <= 10 else (1 if cardinality == 2 else 2)
        key = (band, -abs(effects.get(signal, 0.0)))
        if best_key is None or key < best_key:
            best, best_key = signal, key
    signal = best or (signals[0] if signals else None)
    if not signal or "." not in signal:
        return None
    analyzer, field = signal.split(".", 1)
    rows = run.per_case(analyzer)
    if not rows:
        return None
    labels = {
        case_id: (record.get("case") or {}).get("label")
        for case_id, record in run.case_records.items()
    }
    pairs = [
        (row.get(field), labels.get(row.get("sample_id")))
        for row in rows
    ]
    pairs = [(value, label) for value, label in pairs if value is not None and label is not None]
    if not pairs:
        return None
    observed: list[Any] = [value for value, _ in pairs]  # Nones filtered out above
    numeric = all(_number(value) for value in observed)
    levels = sorted(set(observed), key=(lambda v: float(v)) if numeric else str)
    binning = "levels"
    aggregate: dict[Any, Counter] = defaultdict(Counter)
    if numeric and len(levels) > 10:
        binning = "quartiles"
        ordered = sorted(float(value) for value in observed)
        cuts = sorted({ordered[int(len(ordered) * i / 4)] for i in (1, 2, 3)})
        edges = [ordered[0]] + cuts + [ordered[-1]]

        def assign(value):
            for index, cut in enumerate(cuts):
                if value < cut:
                    return index
            return len(cuts)

        for value, label in pairs:
            bucket = assign(value)
            aggregate[bucket]["n"] += 1
            if str(label).lower() == "fail":
                aggregate[bucket]["fail"] += 1
        bins = [
            {"label": f"{edges[bucket]:g}–{edges[min(bucket + 1, len(edges) - 1)]:g}",
             "n_cases": aggregate[bucket]["n"], "n_fail": aggregate[bucket]["fail"],
             "failure_rate": _rate(aggregate[bucket]["fail"], aggregate[bucket]["n"])}
            for bucket in sorted(aggregate)
        ]
    else:
        for value, label in pairs:
            aggregate[value]["n"] += 1
            if str(label).lower() == "fail":
                aggregate[value]["fail"] += 1
        bins = [
            {"label": f"{field} = {level}", "value": level,
             "n_cases": aggregate[level]["n"], "n_fail": aggregate[level]["fail"],
             "failure_rate": _rate(aggregate[level]["fail"], aggregate[level]["n"])}
            for level in levels
        ]
    return {
        "signal": signal, "analyzer": analyzer, "field": field, "phase": "explore",
        "binning": binning, "bins": bins,
        "n_cases": sum(item["n_cases"] for item in bins),
        "all_surviving_signals": list(signals),
        "n_surviving_signals": len(signals),
    }


def _m2(run: _Run) -> "dict[str, Any] | None":
    files = run.m2_files()
    if not files:
        return None
    phases = {}
    for phase, path in files:
        results = _load(path, []) or []
        tests = []
        family_size = 0
        for row in results:
            degenerate = "degenerate" in str(row.get("summary") or "").lower()
            in_family = bool(row.get("correction_family")) and not degenerate
            family_size += int(in_family)
            if row.get("tool") != "signal_label_assoc":
                continue
            tests.append({
                "signal": (row.get("config") or {}).get("signal"),
                "effect": row.get("effect"),
                "ci": row.get("ci") or _parse_ci(row.get("summary")),
                "p_value": row.get("p_value"),
                "survives_correction": bool(row.get("fdr_corrected")),
                "degenerate": degenerate,
                "in_correction_family": in_family,
            })
        survivors = [t["signal"] for t in tests if t["survives_correction"] and t["signal"]]
        phases[phase] = {
            "tests": tests,
            "n_in_correction_family": family_size,
            "n_degenerate": sum(1 for t in tests if t["degenerate"]),
            "correction_method": next(
                (row.get("correction_method") for row in results if row.get("correction_method")), None),
            "survivors": survivors,
        }
    return phases or None


def _m3(run: _Run) -> "list[dict[str, Any]]":
    diagnosis = run.event("diagnosis")
    if not diagnosis:
        return []
    proposed = diagnosis.get("hypotheses") or diagnosis.get("proposed_hypotheses") or []
    return [
        {"id": f"H{index}",
         "failure_mode": str(item.get("failure_mode") or "").strip("`"),
         "statement": item.get("statement"),
         "expected_direction": item.get("expected_direction") or item.get("expected_association")}
        for index, item in enumerate(proposed, start=1)
        if isinstance(item, dict)
    ]


def _m5(run: _Run) -> "list[dict[str, Any]]":
    verdicts = [e for e in run.events_of("surgery") if e.get("module") == "m5"]
    out = []
    for index, verdict in enumerate(verdicts, start=1):
        evidence = verdict.get("evidence") or {}
        out.append({
            "id": f"H{index}",
            "failure_mode": str(verdict.get("failure_mode") or "").strip("`"),
            "status": verdict.get("status"),
            "statement": verdict.get("hypothesis"),
            "test_name": evidence.get("m5_test_name") or evidence.get("chosen_tool"),
            "effect": evidence.get("effect_size"),
            "ci": evidence.get("ci"),
            "evidence_grade": evidence.get("m5_evidence_grade"),
            "underpowered": evidence.get("underpowered"),
        })
    return out


def _m4(run: _Run) -> "dict[str, Any] | None":
    """The repair ladder, and every candidate the search actually tried.

    Two lists with different authority: ``selection_attempted`` is the search on
    the explore split (which rung to climb), ``attempted`` is the confirmation on
    held-out cases (whether the winner holds). The ladder is built from the first,
    because that is the search; the second is what the validation strip reports.
    """
    fix = run.event("fix")
    if not fix:
        return None
    selection = [c for c in (fix.get("selection_attempted") or []) if isinstance(c, dict)]
    confirmation = [c for c in (fix.get("attempted") or []) if isinstance(c, dict)]
    best = fix.get("best") or {}
    accepted = best.get("name")

    def row(candidate: Mapping[str, Any], selected: bool) -> "dict[str, Any]":
        return {
            "name": candidate.get("name"), "tier": str(candidate.get("tier") or ""),
            "effect": candidate.get("effect"), "n_fixed": candidate.get("n_fixed"),
            "n_broken": candidate.get("n_broken"), "verdict": candidate.get("verdict"),
            "selected": selected,
        }

    candidates = [row(c, c.get("name") == accepted) for c in selection]
    confirmed = [row(c, bool(c.get("fixed"))) for c in confirmation]

    cap = str(fix.get("max_tier") or run.config.get("fix_tier") or "")
    order = ["L1", "L2", "L3", "L4"]
    by_tier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in selection:
        by_tier[str(candidate.get("tier") or "")[:2]].append(candidate)
    accepted_tier = str(best.get("tier") or "")[:2]
    ladder = []
    for tier in order:
        tried = by_tier.get(tier, [])
        top = max(tried, key=lambda c: c.get("effect") or -9e9) if tried else None
        if not tried:
            status = "untouched"
        elif accepted_tier == tier:
            status = "accepted"
        elif top and (top.get("effect") or 0) < 0:
            status = "regressed"
        else:
            status = "not_selected"
        ladder.append({
            "tier": tier, "label": TIER_LABEL[tier], "n_candidates": len(tried),
            "best_effect": top.get("effect") if top else None,
            "best_candidate": top.get("name") if top else None,
            "status": status,
            "within_cap": bool(cap) and order.index(tier) <= order.index(cap[:2]),
            "tier_cap": cap or None,
        })
    if not candidates and not confirmed and not best:
        return None
    return {"ladder": ladder, "candidates": candidates, "confirmed": confirmed}


def _repair(run: _Run) -> "dict[str, Any] | None":
    """The accepted repair, as steps a reader can follow.

    The steps are derived from the payload rather than written out, so a repair
    the pipeline invents next month still describes itself correctly here.
    """
    fix: Mapping[str, Any] = run.event("fix") or {}
    best: Mapping[str, Any] = fix.get("best") or {}
    raw_payload = best.get("payload")
    payload: Mapping[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    if not best:
        return None
    steps: list[dict[str, Any]] = []
    ops = payload.get("image_ops") or []
    if ops:
        steps.append({"title": "Image operation", "lines": [
            f"{op.get('tool')}({', '.join(f'{k}={v}' for k, v in (op.get('params') or {}).items())})"
            for op in ops if isinstance(op, dict)][:4]})
    template = str(payload.get("prompt_template") or "")
    paragraphs = [p.strip().replace("\n", " ") for p in template.split("\n\n") if p.strip()]
    if paragraphs:
        steps.append({"title": "Prompt", "lines": paragraphs[:4], "mono": True})
    samples = payload.get("n_samples")
    if samples and samples > 1:
        temperature = (payload.get("generation_kwargs") or {}).get("temperature")
        lines = [f"{samples} independent answers", "keep the modal one"]
        if temperature is not None:
            lines.insert(0, f"temperature {temperature}")
        steps.append({"title": f"Sample \u00d7{samples}, take the mode", "lines": lines})
    return {
        "name": payload.get("name") or best.get("name"),
        "tier": best.get("tier"),
        "strategy": payload.get("strategy"),
        "n_samples": samples,
        "generation_kwargs": payload.get("generation_kwargs"),
        "steps": steps,
    }


def _validation(run: _Run) -> "dict[str, Any] | None":
    fix: Mapping[str, Any] = run.event("fix") or {}
    best: Mapping[str, Any] = fix.get("best") or {}
    if not best:
        return None
    pairs = best.get("n_pairs") or 0
    baseline_correct = best.get("n_baseline_correct") or 0
    fixed = best.get("n_fixed") or 0
    broken = best.get("n_broken") or 0
    return {
        "candidate": best.get("name"), "tier": best.get("tier"), "n_pairs": pairs,
        "baseline_rate": best.get("baseline_rate"), "candidate_rate": best.get("candidate_rate"),
        "n_fixed": fixed, "n_broken": broken,
        "both_correct": baseline_correct - broken,
        "both_wrong": pairs - (baseline_correct - broken) - fixed - broken,
        "effect": best.get("effect"), "ci": best.get("ci") or _parse_ci(best.get("summary")),
        "e_value": best.get("e_value"), "verdict": best.get("verdict"),
    }


def _example_case(run: _Run, curve: "Mapping[str, Any] | None") -> "dict[str, Any] | None":
    field = str(curve.get("field") or "") if curve else ""
    analyzer = str(curve.get("analyzer") or "") if curve else ""
    signals = {row.get("sample_id"): row for row in run.per_case(analyzer)} if analyzer else {}
    explore_ids = list(run.case_records)

    def score(case_id: str) -> tuple:
        record = run.case_records.get(case_id, {})
        label = str((record.get("case") or {}).get("label", "")).lower()
        value = (signals.get(case_id) or {}).get(field)
        return (1 if label == "fail" else 0,
                1 if record.get("media_paths") else 0,
                value if _number(value) else -1)

    failing = [case_id for case_id in explore_ids if score(case_id)[0] == 1]
    if not failing:
        return None
    case_id = max(failing, key=score)
    case = run.all_cases.get(case_id, {})
    record = run.case_records.get(case_id, {})
    return {
        "case_id": case_id,
        "split": "explore",
        "question": case.get("prompt") or ((record.get("case") or {}).get("inputs") or {}).get("prompt"),
        "gold": case.get("expected"),
        "baseline_output": case.get("observed"),
        # what it answered, separately from everything it said on the way there
        "baseline_answer": final_answer(case.get("observed")),
        "label": case.get("label") or (record.get("case") or {}).get("label"),
        "signal": curve.get("signal") if curve else None,
        "signal_values": signals.get(case_id),
        "media_paths": record.get("media_paths") or [],
    }


def _qa_flags(curve, m5, validation, example, m2) -> "list[dict[str, str]]":
    """The caveats the sheet is required to carry, from the run's own numbers."""
    flags: list[dict[str, str]] = []
    seen: dict[tuple, str] = {}
    for verdict in m5:
        key = (verdict.get("test_name"), verdict.get("effect"))
        if key[0] is not None and key[1] is not None:
            if key in seen:
                flags.append({"level": "warning", "code": "shared_test_between_hypotheses",
                              "detail": f"{verdict['id']} rests on the same test and effect as "
                                        f"{seen[key]}; they are one finding, not two"})
            else:
                seen[key] = verdict["id"]
    if curve and (curve.get("n_surviving_signals") or 1) > 1:
        flags.append({"level": "note", "code": "multiple_signals_survived",
                      "detail": f"{curve['n_surviving_signals']} signals survived correction; "
                                f"this sheet plots {curve.get('signal')}"})
    if curve and curve.get("binning") == "quartiles":
        flags.append({"level": "note", "code": "signal_binned_for_plotting",
                      "detail": f"{curve.get('signal')} is continuous and was quartile-binned; "
                                "the bars are bins, not raw levels"})
    if m2 and len({phase.get("n_in_correction_family") for phase in m2.values()}) > 1:
        flags.append({"level": "note", "code": "correction_family_size_differs",
                      "detail": "the two phases corrected over different numbers of signals"})
    if example and example.get("split") != "heldout":
        flags.append({"level": "note", "code": "example_case_from_explore",
                      "detail": "the illustrative case comes from the explore split; its output is "
                                "the unchanged model's, not a measured repaired result"})
    if validation and not any(v.get("status") == "supported" for v in m5):
        flags.append({"level": "warning", "code": "repair_without_supported_hypothesis",
                      "detail": "a repair was accepted although no hypothesis reached 'supported'; "
                                "the gain is empirical, not a validated mechanism"})
    if validation and (validation.get("n_broken") or 0) > 0:
        flags.append({"level": "note", "code": "accepted_fix_breaks_cases",
                      "detail": f"the accepted repair broke {validation['n_broken']} previously-correct "
                                "cases; the gain is net of them"})
    return flags


def build_case_study(
    root: str | Path,
    events: Sequence[Mapping[str, Any]],
    *,
    setting: "Mapping[str, Any] | None" = None,
) -> "dict[str, Any] | None":
    """Compile one run into the case-study sheet, or ``None`` if it has no probes.

    ``events`` is the already-parsed run log, so this costs one pass over the
    artifacts the reader has anyway.
    """
    run = _Run(Path(root), events)
    m1 = _m1(run)
    m2 = _m2(run)
    if not m1 and not m2:
        return None
    signals = _survivors(run)
    curve = _signal_curve(run, signals) if signals else None
    m3 = _m3(run)
    m5 = _m5(run)
    m4 = _m4(run)
    repair = _repair(run)
    validation = _validation(run)
    example = _example_case(run, curve)
    if m1 is not None:
        explore = (m2 or {}).get("explore") or {}
        m1["n_forwarded"] = explore.get("n_in_correction_family")
        m1["signal_curve"] = curve
        # The probe that produced the confirmed signal leads its family: of the
        # eight that ran, this is the one the rest of the sheet is about.
        owner = curve.get("analyzer") if curve else None
        for family in m1["families"]:
            for probe in family["probes"]:
                probe["confirmed"] = bool(owner) and owner in probe["analyzers"]
    start = run.event("run_start") or {}
    baseline = run.summary.get("baseline_accuracy")
    setting = setting or {}

    def prefer(key: str) -> Any:
        """The run's own summary wins over the report's generic placeholders.

        ``build_report_data`` falls back to "Evaluation dataset" / "Target model"
        when it cannot name either, and a sheet headed "Target model on
        Evaluation dataset" tells a reader nothing the run itself could have.
        """
        placeholders = {"evaluation dataset", "evaluation benchmark", "target model"}
        candidates = [run.summary.get(key), setting.get(key)]
        for value in candidates:
            if value and str(value).strip().lower() not in placeholders:
                return value
        return next((value for value in candidates if value), None)

    return {
        "modules": [{"code": code, "name": name, "subtitle": subtitle}
                    for code, name, subtitle in MODULES],
        "headline": {
            "model": prefer("model"),
            "dataset": prefer("dataset"),
            "n_cases": run.summary.get("n_cases") or setting.get("n_cases") or len(run.all_cases),
            "baseline_accuracy": baseline,
            "n_explore": len(run.case_records),
            "n_heldout": max(len(run.all_cases) - len(run.case_records), 0),
            "judge": str(start.get("judge")) if start.get("judge") else None,
            "repair": (repair or {}).get("name"),
            "repair_tier": (repair or {}).get("tier"),
            "delta": (None if not validation or validation.get("candidate_rate") is None
                      or validation.get("baseline_rate") is None
                      else round(validation["candidate_rate"] - validation["baseline_rate"], 4)),
        },
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m5": m5,
        "m4": m4,
        "repair": repair,
        "validation": validation,
        "example_case": example,
        "qa_flags": _qa_flags(curve, m5, validation, example, m2),
    }
