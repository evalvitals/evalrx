"""AudioCaps object-hallucination (Kuan et al. 2024): yes/no "is sound X present?".

Port of examples/m1_m4/audiocaps_hallucination_qwen2_audio/download_audiohallucination.py:
question rows from kuanhuggingface/AudioHallucination_AudioCaps-<sampling> joined on
youtube_id to the audio in OpenSound/AudioCaps, both read through pyarrow, clips
re-encoded to 16 kHz mono WAV via ffmpeg.
"""

from __future__ import annotations

import random
from pathlib import Path

from . import _audio
from .base import Task, _protocol, write_manifest

QUESTIONS_REPO_FMT = "kuanhuggingface/AudioHallucination_AudioCaps-{sampling}"
AUDIO_REPO = "OpenSound/AudioCaps"


def _find_parquets(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    candidates = [f for f in files if f.endswith(".parquet")]
    if not candidates:
        return ["default/test/0000.parquet"]
    test_files = [f for f in candidates if "test" in f.lower()]
    return sorted(test_files or candidates)


def _read_parquets(repo: str, filenames: list[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tables = [pq.read_table(hf_hub_download(repo, name, repo_type="dataset")) for name in filenames]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def download(out_dir: Path, limit: int = 256, seed: int = 20260814, scan_rows: int = 2000,
             sampling: str = "Random") -> dict:
    _audio.require_ffmpeg()
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    questions_repo = QUESTIONS_REPO_FMT.format(sampling=sampling)
    q_table = _read_parquets(questions_repo, _find_parquets(questions_repo))
    a_table = _read_parquets(AUDIO_REPO, _find_parquets(AUDIO_REPO))
    audio_by_ytid: dict[str, int] = {}
    for idx, ytid in enumerate(a_table.column("youtube_id").to_pylist()):
        audio_by_ytid.setdefault(ytid, idx)
    order = list(range(q_table.num_rows))
    random.Random(seed).shuffle(order)
    order = order[: min(scan_rows, q_table.num_rows)]
    rows, n_nomatch, n_dur = [], 0, 0
    seen: set[str] = set()
    for row_index in order:
        if len(rows) >= limit:
            break
        row = q_table.slice(row_index, 1).to_pylist()[0]
        ytid = str(row["audio_index"]).removeprefix("Y")
        if ytid not in audio_by_ytid:
            n_nomatch += 1
            continue
        wav_path = audio_dir / f"{ytid}.wav"
        if ytid not in seen:
            if wav_path.is_file():
                duration = _audio.duration_seconds(wav_path)
            else:
                a_row = a_table.slice(audio_by_ytid[ytid], 1).to_pylist()[0]
                field = a_row["audio"]
                duration = _audio.decode_to_wav(field["bytes"] if isinstance(field, dict) else field, wav_path)
            if duration > _audio.MAX_DURATION_SEC:
                n_dur += 1
                wav_path.unlink(missing_ok=True)
                continue
            seen.add(ytid)
        elif not wav_path.is_file():
            n_dur += 1
            continue
        rows.append({
            "id": str(row["entry_id"]), "dataset": questions_repo, "subset": sampling,
            "source_index": row_index, "sample_seed": seed,
            "image": None, "audio": f"audio/{ytid}.wav",
            "prompt": f"{str(row['prompt_text']).strip()} Answer with only the single word Yes or No.",
            "answers": [str(row["label"]).strip().capitalize()],
            "task": "yes_no", "numeric_tolerance": 0.0,
            "metadata": {
                "object": row.get("object", ""), "attribute": row.get("attribute", ""),
                "sampling": row.get("sampling", sampling), "youtube_id": ytid,
                "source_dataset": "AudioCaps",
            },
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "skipped_no_audio_match": n_nomatch,
            "skipped_duration_over_limit": n_dur, "scanned": len(order),
            "sampling": sampling, "manifest": str(out_dir / "manifest.json")}


def protocol(model_label: str):
    return _protocol(
        description=(
            f"We evaluate an audio-language model ({model_label}) on a discriminative "
            "object-hallucination benchmark: for a short AudioCaps clip, the model is "
            "asked a binary yes/no question about whether a specific sound is present "
            "('Is there a sound of X in the audio?'). Half the questions name a sound "
            "that IS present, half name one that is absent (random/popular/adversarial "
            "negative sampling). We want to know what distinguishes the questions it "
            "gets right from the ones it gets wrong."
        ),
        task_domain="audio object-hallucination detection (AudioCaps discriminative)",
        success_criteria=(
            "The model's Yes/No answer must match the gold label for whether that "
            "sound is actually present in the clip."
        ),
        failure_patterns=(
            "wrong answers concentrated on absent-sound (gold=No) questions, where the "
            "model answers Yes anyway -- a model that asserts sounds it has not actually "
            "heard, falling back on what the question text alone makes plausible rather "
            "than the audio evidence, will show exactly this asymmetric pattern; a fix "
            "that trades away present-sound (gold=Yes) accuracy to gain absent-sound "
            "accuracy is a different error, not an improvement"
        ),
        target_modalities=frozenset({"text", "audio"}),
    )


TASK = Task(
    name="audiocaps_hallu", modality="alm", kind="yes_no", title="AudioCaps-Hallucination/Random",
    download=download, protocol=protocol,
    pinned_m1=(
        "answer_extraction_audit", "termination_audit", "selfcheck_consistency",
        "self_consistency", "calibration", "logprob_entropy", "perturbation_battery",
    ),
    default_limit=300, default_seed=20260814, max_new_tokens=16,
    source="kuanhuggingface/AudioHallucination_AudioCaps-Random + OpenSound/AudioCaps",
)
