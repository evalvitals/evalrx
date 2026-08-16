"""Music-AVQA -> FailureCase adapter + protocol + scorer.

Music-AVQA (https://gewu-lab.github.io/MUSIC-AVQA/) question records look like::

    {"video_id": "00000028", "question_id": 7,
     "type": "[\"Audio-Visual\", \"Existential\"]",
     "question_content": "Is the <Object> in the video always playing?",
     "templ_values": "[\"ukulele\"]", "question_deleted": 0, "anser": "yes"}

``type`` and ``templ_values`` are JSON *strings* (double-encoded) in the
released files, and the answer key is spelled ``anser`` (dataset's own typo,
kept as-is on read). This module is local to this example — it does not touch
``evalvitals/`` — matching the "no architecture changes" constraint: it only
builds ``FailureCase``/``CaseBatch`` (public core types) from raw JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Raw record -> question text / gold answer
# ---------------------------------------------------------------------------

def fill_template(question_content: str, templ_values_raw: str) -> str:
    """Substitute each ``<Object>`` placeholder in order with a templ_values entry."""
    try:
        values = json.loads(templ_values_raw) if templ_values_raw else []
    except json.JSONDecodeError:
        values = []
    q = question_content
    for v in values:
        q = q.replace("<Object>", v, 1)
    return q


def parse_record(raw: dict) -> dict:
    """Normalize one raw Music-AVQA record into a flat dict."""
    modality, qtype = json.loads(raw["type"])
    return {
        "video_id": raw["video_id"],
        "question_id": raw["question_id"],
        "modality": modality,  # "Audio-Visual" | "Audio" | "Visual"
        "qtype": qtype,  # "Counting" | "Comparative" | "Existential" | "Location" | "Temporal"
        "question": fill_template(raw["question_content"], raw.get("templ_values", "[]")),
        "answer": str(raw["anser"]).strip(),
    }


def load_records(json_path: "str | Path", videos_dir: "str | Path",
                  modalities: "set[str] | None" = None,
                  limit: "int | None" = None) -> list[dict]:
    """Read an ``avqa-*.json`` split, drop deleted/missing-video rows, resolve video paths.

    ``modalities``: keep only these Music-AVQA modality tags (default: all three).
    Audio-Visual and Audio-only questions are the ones an audio-blind model
    (or an audio-visual model that ignores its audio branch) should fail
    disproportionately — that contrast is what the diagnosis loop needs.
    """
    videos_dir = Path(videos_dir)
    raw_records = json.loads(Path(json_path).read_text())
    out = []
    for raw in raw_records:
        if raw.get("question_deleted"):
            continue
        rec = parse_record(raw)
        if modalities and rec["modality"] not in modalities:
            continue
        video_path = videos_dir / f"{rec['video_id']}.mp4"
        if not video_path.exists():
            continue
        rec["video_path"] = str(video_path)
        out.append(rec)
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Answer scoring — short categorical/numeric/yes-no answers, substring match
# ---------------------------------------------------------------------------

def normalize_answer(a: str) -> str:
    a = a.strip().lower().replace("_", " ")
    a = re.sub(r"[^a-z0-9 ]+", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def answers_match(predicted: str, gold: str) -> bool:
    """True if the (usually free-text) predicted answer contains the gold phrase.

    Gold answers are short (yes/no, a number word, an instrument name); the
    model's raw output is often a full sentence ("Yes, the ukulele is always
    playing."), so containment is the right match rule, not equality.
    """
    p, g = normalize_answer(predicted), normalize_answer(gold)
    if not g:
        return False
    return f" {g} " in f" {p} " or p == g


def make_avqa_score_fn():
    """(case, output) -> bool|None for FixAgent — True if output still matches gold."""

    def score_fn(case, output: str):
        gold = (getattr(case, "metadata", {}) or {}).get("gold_answer")
        if not gold:
            return None
        return answers_match(str(output), gold)

    return score_fn


# ---------------------------------------------------------------------------
# Frozen manifest (built by mine_cases.py) -> CaseBatch
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: "str | Path"):
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    path = Path(manifest_path)
    if not path.exists():
        raise SystemExit(f"{path} missing — run mine_cases.py first")
    raw = json.loads(path.read_text())
    cases = []
    for row in raw["cases"]:
        cases.append(FailureCase(
            inputs=Inputs(prompt=row["question"], video=row["video_path"]),
            expected=row["answer"],
            observed=row.get("observed"),
            label=Label.FAIL if row["label"] == "fail" else Label.PASS,
            tags={"audio-visual-qa", row["modality"].lower(), row["qtype"].lower()},
            metadata={
                "video_id": row["video_id"],
                "question_id": row["question_id"],
                "modality": row["modality"],
                "qtype": row["qtype"],
                "gold_answer": row["answer"],
                "video_path": row["video_path"],
            },
        ))
    return CaseBatch(cases), raw


# ---------------------------------------------------------------------------
# Protocol — OBSERVATION ONLY (do not name a mechanism; that's the loop's job)
# ---------------------------------------------------------------------------

def build_protocol():
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "An audio-visual LLM answers questions about a short video that "
            "require combining information from BOTH the audio track and the "
            "visual frames (e.g. counting instruments seen and/or heard, "
            "comparing which of two instruments is louder, judging whether an "
            "instrument visible in frame is the one currently making sound, "
            "localizing which visible object is the sound source). Some "
            "answers are wrong even though the relevant object or event is "
            "clearly present in at least one modality. Failure cases are "
            "questions the model answers incorrectly against the dataset's "
            "ground-truth answer; success cases are questions drawn from the "
            "same video pool and the same question templates that it answers "
            "correctly."
        ),
        task_domain="audio-visual question answering / cross-modal hallucination",
        success_criteria=(
            "the generated answer contains or matches the ground-truth answer "
            "for the question"
        ),
        failure_patterns=(
            "wrong answers concentrated on Audio-Visual and Audio-only "
            "questions (counting, comparative loudness, existential 'is X "
            "playing', sound-source localization) rather than Visual-only "
            "questions — a model that effectively answers as if it only sees "
            "the video and cannot hear it (or vice versa) will show exactly "
            "this pattern; a fix that degrades Visual-only accuracy to gain "
            "Audio accuracy is a different error, not an improvement"
        ),
        target_modalities=frozenset({"text", "video", "audio"}),
    )
