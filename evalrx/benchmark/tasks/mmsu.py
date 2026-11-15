"""MMSU (Wang et al. 2025): spoken-language understanding and reasoning.

MMSU contains 5,000 four-way multiple-choice audio questions spanning 47
fine-grained tasks. The official Hugging Face parquet embeds every clip and is
about 1.47 GB. This downloader uses Parquet column projection over HTTP range
reads to obtain the complete text/label census, takes one deterministic sample,
and downloads only the selected media files from the pinned dataset revision.
"""

from __future__ import annotations

import random
from pathlib import Path

from . import _audio
from .base import Task, _protocol, write_manifest

REPO = "ddwang2000/MMSU"
REVISION = "548e2283105825bf908a7db5c09c00dbcf42bd4c"
SPLIT = "train"  # MMSU publishes its 5,000-item evaluation set under this split name.
LETTERS = ["A", "B", "C", "D"]
PARQUET_FILES = [f"data/train-{part:05d}-of-00003.parquet" for part in range(3)]
METADATA_COLUMNS = [
    "id",
    "task_name",
    "question",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "answer_gt",
    "category",
    "sub-category",
    "sub-sub-category",
    "linguistics_sub_discipline",
]


def _choices(row: dict) -> list[str]:
    return [str(row.get(f"choice_{letter.lower()}") or "").strip() for letter in LETTERS]


def _letter(choices: list[str], answer: str) -> str:
    """Map MMSU's answer text back to its unique option letter."""
    answer = str(answer).strip()
    matches = [letter for letter, choice in zip(LETTERS, choices) if choice.strip() == answer]
    if len(matches) != 1:
        raise ValueError(
            f"answer must match exactly one option: answer={answer!r} choices={choices!r}"
        )
    return matches[0]


def task_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"({letter}) {choice}" for letter, choice in zip(LETTERS, choices))
    return (
        f"{question.strip()}\n\n{options}\n\n"
        "Listen to the audio and reply with only the option letter (A, B, C, or D)."
    )


def _metadata_census() -> list[tuple[int, dict]]:
    """Return all rows in source order without downloading embedded audio blobs."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    rows: list[dict] = []
    for filename in PARQUET_FILES:
        remote = f"datasets/{REPO}@{REVISION}/{filename}"
        with filesystem.open(remote, "rb") as handle:
            rows.extend(pq.read_table(handle, columns=METADATA_COLUMNS).to_pylist())
    census = list(enumerate(rows))
    if len(census) != 5000:
        raise RuntimeError(f"incomplete MMSU metadata census: received {len(census)} of 5000 rows")
    return census


def download(out_dir: Path, limit: int = 256, seed: int = 20260814) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    _audio.require_ffmpeg()
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    current_revision = api.dataset_info(REPO).sha
    if current_revision != REVISION:
        raise RuntimeError(
            f"MMSU changed upstream ({REVISION} -> {current_revision}); audit the new revision "
            "before updating the pinned downloader"
        )
    census = _metadata_census()
    media = {}
    for filename in api.list_repo_files(REPO, repo_type="dataset", revision=REVISION):
        path = Path(filename)
        if path.parent == Path("audio"):
            if path.stem in media:
                raise RuntimeError(f"multiple MMSU audio files share ID {path.stem!r}")
            media[path.stem] = filename
    order = list(range(len(census)))
    random.Random(seed).shuffle(order)
    requested = len(order) if limit <= 0 else min(limit, len(order))

    rows, n_dur, n_bad, n_noaudio, scanned = [], 0, 0, 0, 0
    for order_index in order:
        if len(rows) >= requested:
            break
        scanned += 1
        row_index, row = census[order_index]
        choices = _choices(row)
        try:
            letter = _letter(choices, row["answer_gt"])
        except ValueError:
            n_bad += 1
            continue

        row_id = str(row["id"])
        source_filename = media.get(row_id)
        if source_filename is None:
            n_noaudio += 1
            continue
        wav_path = audio_dir / f"{row_id}.wav"
        if wav_path.is_file():
            duration = _audio.duration_seconds(wav_path)
        else:
            source = Path(
                hf_hub_download(
                    REPO,
                    source_filename,
                    repo_type="dataset",
                    revision=REVISION,
                )
            )
            duration = _audio.decode_to_wav(source.read_bytes(), wav_path)
        if duration > _audio.MAX_DURATION_SEC:
            n_dur += 1
            wav_path.unlink(missing_ok=True)
            continue

        rows.append(
            {
                "id": row_id,
                "dataset": REPO,
                "subset": SPLIT,
                "source_index": row_index,
                "sample_seed": seed,
                "image": None,
                "audio": f"audio/{row_id}.wav",
                "prompt": task_prompt(row["question"], choices),
                "answers": [letter],
                "choices": LETTERS,
                "task": "multiple_choice_letter",
                "numeric_tolerance": 0.0,
                "metadata": {
                    "options": [f"({letter}) {choice}" for letter, choice in zip(LETTERS, choices)],
                    "duration_sec": round(duration, 3),
                    "mmsu_task": row["task_name"],
                    "category": row["category"],
                    "sub_category": row["sub-category"],
                    "sub_sub_category": row["sub-sub-category"],
                    "linguistics_sub_discipline": row["linguistics_sub_discipline"],
                    "dataset_revision": REVISION,
                },
            }
        )

    write_manifest(out_dir / "manifest.json", rows)
    return {
        "kept": len(rows),
        "requested": requested,
        "scanned": scanned,
        "source_rows": len(census),
        "skipped_duration_over_limit": n_dur,
        "skipped_unparseable_answer": n_bad,
        "skipped_audio_not_found": n_noaudio,
        "dataset_revision": REVISION,
        "manifest": str(out_dir / "manifest.json"),
    }


def protocol(model_label: str):
    return _protocol(
        description=(
            f"We evaluate an audio-language model ({model_label}) on MMSU: 5,000 "
            "multiple-choice spoken-language questions across 47 fine-grained tasks. "
            "The benchmark covers perception and reasoning over semantics, phonology, "
            "and paralinguistics. We want to know what distinguishes the questions it "
            "gets right from the ones it gets wrong."
        ),
        task_domain="spoken-language understanding and reasoning (MMSU)",
        success_criteria="The selected option letter must identify the gold answer for the clip.",
        target_modalities=frozenset({"text", "audio"}),
        output_contract={"kind": "multiple_choice_letter", "choices": LETTERS},
    )


TASK = Task(
    name="mmsu",
    modality="alm",
    kind="multiple_choice_letter",
    title="MMSU",
    download=download,
    protocol=protocol,
    pinned_m1=(
        "answer_extraction_audit",
        "termination_audit",
        "selfcheck_consistency",
        "format_sensitivity",
        "self_consistency",
        "calibration",
        "logprob_entropy",
        "coverage_verification_gap",
    ),
    default_limit=256,
    default_seed=20260814,
    max_new_tokens=64,
    output_contract={"kind": "multiple_choice_letter", "choices": LETTERS},
    source="ddwang2000/MMSU (used in the Audio Flamingo evaluation ecosystem)",
)
