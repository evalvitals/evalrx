"""Case-level study view for paper-method bench reports (audio and image).

A bench runner under ``examples/`` — ``examples/agent_loop/qwen2_audio_tcd_mmau/
run.py`` (TCD x MMAU, audio) or ``examples/vlm_paper_benchmark/run_hf_autofix.py``
(VCD/PAI/ViCrop x POPE/V*Bench, images) — writes ONE report JSON shaped::

    {paper, model, splits, hypothesis,
     baseline: {diagnosis, selection: {cases: [{id, output, correct}]}, confirmation},
     auto_fix: {selection: FixOutcome.to_dict(), confirmation}}

That report records per-case *outcomes* (which case ids a repair candidate
flipped) but never the stimulus: the question, its options and the media file
live in the benchmark manifest the runner read, beside the report. Nothing in
the product joined the two, so a repaired case was only ever a bare uuid — you
could not listen to the clip, read the question, or answer it yourself.

This module is that join, and it is deliberately Streamlit-free (same split as
``workbench.py``): it resolves the manifest, normalizes rows from either
benchmark's field names into one :class:`CaseView`, and labels every case with
what each repair candidate did to it. ``dashboard_app.py`` renders the result.

Manifest resolution prefers an explicit ``report["dataset"]["manifest"]``
pointer (written by newer runners); older reports fall back to a search of the
conventional sibling locations, scored by how many of the report's own case ids
each candidate manifest actually covers — a directory holding five benchmark
manifests (vlm_paper_benchmark/data/) then resolves to the right one instead of
the alphabetically-first one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Manifest field aliases, most specific first. MMAU rows carry
#: ``instruction``/``choices``/``audio_path``; the VLM slices carry
#: ``question``/``options``/``image``. Neither runner should have to rename its
#: data to get a case view.
QUESTION_FIELDS = ("question", "instruction", "prompt", "query")
CHOICES_FIELDS = ("choices", "options", "candidates")
EXPECTED_FIELDS = ("expected", "answer", "label", "gt")
AUDIO_FIELDS = ("audio_path", "audio", "wav_path", "audio_file")
IMAGE_FIELDS = ("image", "image_path", "images", "image_file")

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

#: Split names in report["baseline"], in reading order. ``diagnosis`` is
#: summary-only in both runners (no per-case list) and simply yields nothing.
SPLITS = ("diagnosis", "selection", "confirmation")

#: What each repair did to one case, relative to the unmodified baseline.
FLIP_REPAIRED = "repaired"
FLIP_BROKE = "broke"
FLIP_UNCHANGED = "unchanged"
FLIP_UNTESTED = "untested"

# Only ever matches an explicit option marker — "(A) foo", "A. foo", "B) foo" —
# never a bare answer that happens to start with a letter ("A minor chord").
_CHOICE_MARKER = re.compile(r"^\s*(?:\(([A-Za-z])\)|([A-Za-z])[.):])\s+(.*)$", re.S)


# ---------------------------------------------------------------------------
# Report detection
# ---------------------------------------------------------------------------


def is_bench_report(obj: Any) -> bool:
    """True for a paper-method bench report (see the module docstring)."""
    if not isinstance(obj, dict):
        return False
    baseline, auto_fix = obj.get("baseline"), obj.get("auto_fix")
    if not isinstance(baseline, dict) or not isinstance(auto_fix, dict):
        return False
    # A baseline split carrying per-case rows is what makes a case study
    # possible at all; a report without one is a summary, not a case book.
    return any(
        isinstance(baseline.get(name), dict) and isinstance(baseline[name].get("cases"), list)
        for name in SPLITS
    )


def find_bench_reports(root: str | Path) -> list[Path]:
    """Bench reports under *root*: its own ``*.json`` and ``outputs/*.json``."""
    base = Path(root)
    seen: set[Path] = set()
    found: list[Path] = []
    for pattern in ("*.json", "outputs/*.json", "*/outputs/*.json"):
        for path in sorted(base.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if is_bench_report(_read_json(path)):
                found.append(path)
    return found


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------


@dataclass
class MediaRef:
    """The stimulus file for one case, resolved against the manifest's dir."""

    kind: str = "none"  # "audio" | "image" | "none"
    path: str = ""      # absolute path, "" when unresolved
    declared: str = ""  # what the manifest said, for the "missing file" message

    @property
    def exists(self) -> bool:
        return bool(self.path) and Path(self.path).is_file()


@dataclass
class CaseView:
    """One benchmark case: the stimulus, the question, and every arm's answer."""

    id: str
    split: str
    question: str = ""
    choices: "list[tuple[str, str]]" = field(default_factory=list)  # [(letter, text)]
    expected: str = ""
    media: MediaRef = field(default_factory=MediaRef)
    metadata: "dict[str, Any]" = field(default_factory=dict)
    duration_sec: "float | None" = None
    baseline_output: str = ""
    baseline_correct: "bool | None" = None
    #: candidate name -> FLIP_* (what that repair did to THIS case)
    flips: "dict[str, str]" = field(default_factory=dict)
    #: True when the manifest had no row for this id (outcome-only case).
    unresolved: bool = False

    @property
    def expected_text(self) -> str:
        for letter, text in self.choices:
            if letter == self.expected.strip().upper():
                return text
        return ""

    def flipped_by(self) -> "list[str]":
        return [n for n, v in self.flips.items() if v == FLIP_REPAIRED]

    def broken_by(self) -> "list[str]":
        return [n for n, v in self.flips.items() if v == FLIP_BROKE]


@dataclass
class CaseStudy:
    """Everything the case-study UI renders for one bench report."""

    report_path: Path
    report: "dict[str, Any]"
    cases: "list[CaseView]"
    candidates: "list[dict[str, Any]]"
    manifest_path: "Path | None" = None
    manifest_coverage: float = 0.0
    #: Reader-facing warnings (missing manifest, unresolved media, ...).
    notes: "list[str]" = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.report_path.stem

    @property
    def media_kind(self) -> str:
        for case in self.cases:
            if case.media.kind != "none":
                return case.media.kind
        return "none"

    def case_by_id(self, case_id: str) -> "CaseView | None":
        return next((c for c in self.cases if c.id == case_id), None)


# ---------------------------------------------------------------------------
# Manifest resolution
# ---------------------------------------------------------------------------


def _first(row: "dict[str, Any]", fields: Iterable[str]) -> Any:
    for name in fields:
        if row.get(name) not in (None, "", [], {}):
            return row[name]
    return None


def read_manifest(path: Path) -> "dict[str, dict[str, Any]]":
    """Read a ``.jsonl`` (or ``.json`` list) benchmark manifest, keyed by id."""
    rows: "dict[str, dict[str, Any]]" = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    if path.suffix == ".json":
        data = _read_json(path)
        records = data if isinstance(data, list) else (data or {}).get("records") or []
    else:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    for record in records:
        if isinstance(record, dict) and record.get("id") is not None:
            rows[str(record["id"])] = record
    return rows


def _manifest_search_paths(report_path: Path) -> "list[Path]":
    """Conventional manifest locations relative to a report file.

    Both runners write ``<example>/outputs/<name>.json`` beside
    ``<example>/data/<manifest>.jsonl``, so the example root is the report's
    grandparent; the flatter layouts are included for hand-assembled dirs.
    """
    parents = [report_path.parent, report_path.parent.parent, report_path.parent.parent.parent]
    out: "list[Path]" = []
    seen: set[Path] = set()
    for parent in parents:
        for sub in (parent / "data", parent):
            if not sub.is_dir() or sub in seen:
                continue
            seen.add(sub)
            out.extend(sorted(sub.glob("*.jsonl")))
            out.extend(sorted(sub.glob("*.json")))
    return [p for p in out if p != report_path]


def resolve_manifest(
    report: "dict[str, Any]", report_path: Path, case_ids: "set[str]"
) -> "tuple[Path | None, dict[str, dict[str, Any]], float]":
    """Find the manifest whose ids best cover *case_ids*.

    An explicit ``report["dataset"]["manifest"]`` pointer wins outright (it is
    the runner stating what it read); otherwise every conventional sibling is
    scored by id coverage, which is what disambiguates a data dir holding
    several benchmarks' manifests.
    """
    declared = ((report.get("dataset") or {}).get("manifest") or "").strip()
    if declared:
        path = Path(declared)
        if not path.is_absolute():
            for base in (report_path.parent, report_path.parent.parent):
                if (base / declared).is_file():
                    path = base / declared
                    break
        if path.is_file():
            rows = read_manifest(path)
            covered = len(case_ids & set(rows)) / max(1, len(case_ids))
            return path, rows, covered

    best: "tuple[Path | None, dict[str, dict[str, Any]], float]" = (None, {}, 0.0)
    for path in _manifest_search_paths(report_path):
        rows = read_manifest(path)
        if not rows:
            continue
        covered = len(case_ids & set(rows)) / max(1, len(case_ids))
        if covered > best[2]:
            best = (path, rows, covered)
        if covered == 1.0:
            break
    return best


def _media_ref(row: "dict[str, Any]", media_root: Path) -> MediaRef:
    """Resolve a manifest row's stimulus file. Field name decides the kind
    first (``audio_path`` is audio even if the suffix is unusual); the file
    suffix decides for a generic field name."""
    raw = _first(row, AUDIO_FIELDS)
    kind = "audio" if raw is not None else ""
    if raw is None:
        raw = _first(row, IMAGE_FIELDS)
        kind = "image" if raw is not None else ""
    if raw is None:
        return MediaRef()
    if isinstance(raw, (list, tuple)):  # multi-image rows: the first is the view
        raw = raw[0] if raw else None
    if not isinstance(raw, str) or not raw:
        return MediaRef()

    suffix = Path(raw).suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        kind = "audio"
    elif suffix in IMAGE_SUFFIXES:
        kind = "image"
    path = Path(raw)
    if not path.is_absolute():
        path = media_root / raw
    return MediaRef(kind=kind or "none", path=str(path), declared=raw)


def split_choice(raw: Any, index: int) -> "tuple[str, str]":
    """``"(A) siren"`` -> ``("A", "siren")``; an unmarked option gets its
    positional letter, so ``["red", "blue"]`` still reads as A/B."""
    text = str(raw or "").strip()
    match = _CHOICE_MARKER.match(text)
    if match:
        letter = (match.group(1) or match.group(2) or "").upper()
        return letter, match.group(3).strip()
    return chr(ord("A") + index), text


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _flip_map(attempted: "list[dict[str, Any]]", tested_ids: "set[str]") -> "dict[str, dict[str, str]]":
    """case_id -> {candidate name: FLIP_*} for every case in the sweep."""
    per_case: "dict[str, dict[str, str]]" = {cid: {} for cid in tested_ids}
    for attempt in attempted:
        name = str(attempt.get("name") or "")
        if not name:
            continue
        repaired = {str(c) for c in attempt.get("fixed_cases") or []}
        broke = {str(c) for c in attempt.get("broken_cases") or []}
        for cid in tested_ids:
            per_case.setdefault(cid, {})[name] = (
                FLIP_REPAIRED if cid in repaired
                else FLIP_BROKE if cid in broke
                else FLIP_UNCHANGED
            )
    return per_case


def build_case_study(report_path: str | Path) -> "CaseStudy | None":
    """Join a bench report with its benchmark manifest into a case book."""
    path = Path(report_path)
    report = _read_json(path)
    if not is_bench_report(report):
        return None

    baseline = report.get("baseline") or {}
    outcomes: "list[tuple[str, dict[str, Any]]]" = []
    for split in SPLITS:
        block = baseline.get(split)
        if isinstance(block, dict):
            for case in block.get("cases") or []:
                if isinstance(case, dict) and case.get("id") is not None:
                    outcomes.append((split, case))

    case_ids = {str(c["id"]) for _, c in outcomes}
    manifest_path, rows, coverage = resolve_manifest(report, path, case_ids)
    media_root = manifest_path.parent if manifest_path else path.parent

    selection = (report.get("auto_fix") or {}).get("selection") or {}
    attempted = [a for a in selection.get("attempted") or [] if isinstance(a, dict)]
    # Candidates are validated on the SELECTION split only — a confirmation-split
    # case was never in the sweep and must not be shown as "unchanged by" a
    # repair that never ran on it.
    tested_ids = {str(c["id"]) for split, c in outcomes if split == "selection"}
    flips = _flip_map(attempted, tested_ids)

    cases: "list[CaseView]" = []
    for split, outcome in outcomes:
        cid = str(outcome["id"])
        row = rows.get(cid) or {}
        raw_choices = _first(row, CHOICES_FIELDS) or []
        choices = [split_choice(c, i) for i, c in enumerate(raw_choices)]
        duration = row.get("duration_sec")
        cases.append(CaseView(
            id=cid,
            split=split,
            question=str(_first(row, QUESTION_FIELDS) or ""),
            choices=choices,
            expected=str(_first(row, EXPECTED_FIELDS) or "").strip(),
            media=_media_ref(row, media_root),
            metadata={**(row.get("metadata") or {}),
                      **({"task": row["task"]} if row.get("task") else {})},
            duration_sec=float(duration) if isinstance(duration, (int, float)) else None,
            baseline_output=str(outcome.get("output", "")),
            baseline_correct=(
                bool(outcome["correct"]) if outcome.get("correct") is not None else None
            ),
            flips=dict(flips.get(cid, {})) if split == "selection" else {},
            unresolved=not row,
        ))

    notes: "list[str]" = []
    if manifest_path is None:
        notes.append(
            "No benchmark manifest was found next to this report, so questions, "
            "options and media cannot be shown — only per-case outcomes. Expected "
            "a `.jsonl` manifest under the example's `data/` directory."
        )
    elif coverage < 1.0:
        missing = sum(1 for c in cases if c.unresolved)
        notes.append(
            f"{manifest_path.name} covers {coverage:.0%} of this report's cases; "
            f"{missing} case(s) have outcomes but no question/media row."
        )
    missing_media = [c for c in cases if c.media.kind != "none" and not c.media.exists]
    if missing_media:
        notes.append(
            f"{len(missing_media)} case(s) reference a media file that is not on "
            f"disk (e.g. `{missing_media[0].media.declared}`) — re-run the "
            "example's download step to restore playback."
        )
    if attempted:
        notes.append(
            "Repair candidates record which cases they flipped, not what they "
            "answered: FixAgent stores per-case pass/fail, so a repaired case "
            "shows the flip, not the candidate's own output text."
        )

    return CaseStudy(
        report_path=path,
        report=report,
        cases=cases,
        candidates=[describe_candidate(a) for a in attempted],
        manifest_path=manifest_path,
        manifest_coverage=coverage,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Repair-method description
# ---------------------------------------------------------------------------

#: One line per candidate ``kind`` saying what the intervention actually does.
#: Sourced from this repo's own implementations (``models/paper_methods/*.py``,
#: ``eval_agent/stages/fix_agent.py``) so the UI never invents a mechanism.
METHOD_NOTES: "dict[str, str]" = {
    "tcd": (
        "Temporal Contrastive Decoding (Li et al. 2026, arXiv:2604.15383): re-encodes "
        "a Hann-window-blurred copy of the waveform once before decoding, then adds a "
        "positive-rectified contrast between the clean and blurred logits over a small "
        "candidate set. Blur window and update scale are per-example adaptive."
    ),
    "vcd": (
        "Visual Contrastive Decoding: contrasts the clean-image logits against a "
        "diffusion-noised copy of the same image to subtract language-prior mass."
    ),
    "icd": (
        "Instruction Contrastive Decoding: contrasts against a negatively-instructed "
        "forward pass instead of a corrupted input."
    ),
    "ifcd": (
        "Internal Fact-Contrastive Decoding: contrasts against a representation-edited "
        "forward pass produced by a trained editor."
    ),
    "pai": (
        "Pay Attention to Image: scales image-attention logits during decoding "
        "(an internals WRITE, not a read-only decode contrast)."
    ),
    "vicrop": (
        "ViCrop (MLLMs Know Where to Look, ICLR 2025): reads task-vs-general attention "
        "to pick a crop, then answers from the original image plus that crop."
    ),
    "vicrop_consensus": (
        "ViCrop with a label-free consensus guard: keeps the crop answer only when it "
        "agrees with the baseline answer."
    ),
    "opera": (
        "OPERA's binary specialization: penalises next-token candidates that neglect "
        "the image, scoped to one-token yes/no answers."
    ),
    "template": (
        "Prompt template (L1): wraps the unchanged task prompt in extra instruction "
        "text. Nothing about the model, the input media or the decoder changes."
    ),
    "spec": (
        "Input/scaffold spec (L2): a declarative pipeline — media ops, prompt template, "
        "sampling settings and a multi-call strategy — run around the same model."
    ),
    "code": "Coded pipeline (L2): agent-written Python run in a sandbox around the model.",
    "primitive": "Pre-audited internals primitive, parameterised by the judge.",
    "finetune_spec": "LoRA repair (L4): parameter-space fine-tuning on a repair pool.",
}

#: Multi-call strategies used by L2 specs, named in ``payload["strategy"]``.
STRATEGY_NOTES: "dict[str, str]" = {
    "self_refine": "answer once, critique that answer, then revise it (deterministic, single path).",
    "least_to_most": "decompose the question into sub-questions, answer them in order, then answer the original.",
    "self_consistency": "sample several independent answers and take the majority vote.",
    "describe_first": "describe the input before answering the question about it.",
}


def describe_candidate(attempt: "dict[str, Any]") -> "dict[str, Any]":
    """Normalize one ``attempted`` entry into what the Repair-methods UI shows."""
    payload = attempt.get("payload") if isinstance(attempt.get("payload"), dict) else {}
    kind = str(attempt.get("kind") or "spec")
    note = METHOD_NOTES.get(kind, "")
    strategy = str(payload.get("strategy") or "")
    if strategy and strategy in STRATEGY_NOTES:
        note = f"{note} Strategy `{strategy}`: {STRATEGY_NOTES[strategy]}".strip()

    prompt_template = str(payload.get("prompt_template") or "")
    knobs = {
        k: v for k, v in payload.items()
        if k not in {"prompt_template", "name", "strategy"} and v not in (None, "", [], {})
    }
    return {
        "name": str(attempt.get("name") or ""),
        "tier": str(attempt.get("tier") or ""),
        "kind": kind,
        "source": str(attempt.get("source") or ""),
        "note": note,
        "strategy": strategy,
        "prompt_template": prompt_template,
        "knobs": knobs,
        "defaults_only": not payload,
        "n_fixed": attempt.get("n_fixed"),
        "n_broken": attempt.get("n_broken"),
        "n_pairs": attempt.get("n_pairs"),
        "effect": attempt.get("effect"),
        "e_value": attempt.get("e_value"),
        "verdict": str(attempt.get("verdict") or ""),
        "summary": str(attempt.get("summary") or ""),
        "fixed_cases": [str(c) for c in attempt.get("fixed_cases") or []],
        "broken_cases": [str(c) for c in attempt.get("broken_cases") or []],
    }


# ---------------------------------------------------------------------------
# Facets / aggregation
# ---------------------------------------------------------------------------


def facet_keys(cases: "list[CaseView]", *, max_values: int = 20) -> "list[str]":
    """Metadata keys worth offering as filters: present, categorical, not unique."""
    values: "dict[str, set[str]]" = {}
    for case in cases:
        for key, value in (case.metadata or {}).items():
            if isinstance(value, (dict, list)):
                continue
            values.setdefault(key, set()).add(str(value))
    return sorted(
        k for k, vals in values.items()
        if 2 <= len(vals) <= max_values and len(vals) < max(2, len(cases))
    )


def facet_values(cases: "list[CaseView]", key: str) -> "list[str]":
    return sorted({str(c.metadata.get(key)) for c in cases if c.metadata.get(key) is not None})


def accuracy_by_facet(cases: "list[CaseView]", key: str) -> "list[dict[str, Any]]":
    """Baseline accuracy grouped by one metadata key, most-cases first."""
    buckets: "dict[str, list[bool]]" = {}
    for case in cases:
        if case.baseline_correct is None or case.metadata.get(key) is None:
            continue
        buckets.setdefault(str(case.metadata[key]), []).append(case.baseline_correct)
    rows = [
        {"value": value, "n": len(hits), "correct": sum(hits),
         "accuracy": sum(hits) / len(hits)}
        for value, hits in buckets.items()
    ]
    rows.sort(key=lambda r: (-r["n"], r["value"]))
    return rows


def case_label(case: CaseView, *, blind: bool = False) -> str:
    """Compact picker label: outcome badge, split, then the question stem."""
    if blind:
        badge = "•"
    elif case.baseline_correct is None:
        badge = "?"
    else:
        badge = "✓" if case.baseline_correct else "✗"
    if not blind and case.flipped_by():
        badge = "🛠"
    elif not blind and case.broken_by():
        badge = "💥"
    stem = (case.question or case.id).strip().replace("\n", " ")
    if len(stem) > 68:
        stem = stem[:65] + "…"
    return f"{badge} [{case.split[:4]}] {stem}"
