"""MMAU test-mini (Sakshi et al. 2024): four-way multiple choice over a short clip.

Port of examples/m1_m5/mmau_qwen2_audio/download_mmau.py (pyarrow over the
official parquet, embedded audio decoded through ffmpeg, >29.5 s clips skipped
with a recorded count). The prompt is composed here, once, so every model sees
the same instruction + options + "reply with only the option letter".
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from . import _audio
from .base import Task, _protocol, write_manifest

REPO = "gamma-lab-umd/MMAU-test-mini"
FILENAME = "test_mini.parquet"
LETTERS = ["A", "B", "C", "D"]


def _letter(choices: list[str], answer: str) -> str:
    """MMAU's answer is the full "(A) Man" string; match the leading letter."""
    answer = answer.strip()
    for choice in choices:
        if choice.strip() == answer:
            marker = choice.strip()[:2]
            if len(marker) == 2 and marker[0] == "(" and marker[1].isalpha():
                return marker[1].upper()
    if len(answer) >= 2 and answer[0] == "(" and answer[1].isalpha():
        return answer[1].upper()
    raise ValueError(f"could not extract an option letter from answer={answer!r} choices={choices!r}")


def task_prompt(instruction: str, choices: list[str]) -> str:
    options = "\n".join(choices)
    return (f"{instruction}\n\n{options}\n\n"
            "Listen to the audio and reply with only the option letter (A, B, C, or D).")


def download(out_dir: Path, limit: int = 256, seed: int = 20260814, scan_rows: int = 1000) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    _audio.require_ffmpeg()
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(hf_hub_download(REPO, FILENAME, repo_type="dataset"))
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    order = order[: min(scan_rows, table.num_rows)]
    rows, n_dur, n_bad = [], 0, 0
    for row_index in order:
        if len(rows) >= limit:
            break
        row = table.slice(row_index, 1).to_pylist()[0]
        other = json.loads(row["other_attributes"])
        row_id = str(other.get("id", row_index))
        choices = list(row["choices"])
        try:
            letter = _letter(choices, str(row["answer"]))
        except ValueError:
            n_bad += 1
            continue
        wav_path = audio_dir / f"{row_id}.wav"
        if wav_path.is_file():
            duration = _audio.duration_seconds(wav_path)
        else:
            duration = _audio.decode_to_wav(row["context"]["bytes"], wav_path)
        if duration > _audio.MAX_DURATION_SEC:
            n_dur += 1
            wav_path.unlink(missing_ok=True)
            continue
        rows.append({
            "id": row_id, "dataset": REPO, "subset": "test-mini", "source_index": row_index,
            "sample_seed": seed,
            "image": None, "audio": f"audio/{row_id}.wav",
            "prompt": task_prompt(row["instruction"], choices),
            "answers": [letter], "choices": LETTERS,
            "task": "multiple_choice_letter", "numeric_tolerance": 0.0,
            "metadata": {
                "options": choices, "duration_sec": round(duration, 3),
                "mmau_task": other.get("task", ""), "category": other.get("category", ""),
                "sub_category": other.get("sub-category", ""),
                "difficulty": other.get("difficulty", ""),
                "source_dataset": other.get("dataset", ""),
            },
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "skipped_duration_over_limit": n_dur,
            "skipped_unparseable_answer": n_bad, "scanned": len(order),
            "manifest": str(out_dir / "manifest.json")}


def protocol(model_label: str):
    return _protocol(
        description=(
            f"We evaluate an audio-language model ({model_label}) on MMAU test-mini: "
            "four-way multiple-choice questions about a short audio clip, spanning "
            "three domains — sound, music, and speech. The model must select the "
            "option that best matches what is actually audible in the clip. We want "
            "to know what distinguishes the questions it gets right from the ones it "
            "gets wrong."
        ),
        task_domain="audio question answering (MMAU)",
        success_criteria="The selected option letter must match the gold answer for the clip.",
        target_modalities=frozenset({"text", "audio"}),
        output_contract={"kind": "multiple_choice_letter", "choices": LETTERS},
    )


TASK = Task(
    name="mmau", modality="alm", kind="multiple_choice_letter", title="MMAU/test-mini",
    download=download, protocol=protocol,
    pinned_m1=(
        "answer_extraction_audit", "termination_audit", "selfcheck_consistency",
        "format_sensitivity", "self_consistency", "calibration", "logprob_entropy",
        "coverage_verification_gap",
    ),
    default_limit=256, default_seed=20260814, max_new_tokens=64,
    output_contract={"kind": "multiple_choice_letter", "choices": LETTERS},
    source="gamma-lab-umd/MMAU-test-mini",
)
