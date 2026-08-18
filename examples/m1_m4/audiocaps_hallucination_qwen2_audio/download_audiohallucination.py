#!/usr/bin/env python3
"""Download the AudioCaps object-hallucination benchmark and freeze a local sample.

Kuan et al. 2024 (Interspeech 2024, arXiv:2406.08402, "Understanding Sounds,
Missing the Questions: The Challenge of Object Hallucination in Large
Audio-Language Models") -- the paper AAD (Hsu et al. 2025, arXiv:2506.07233)
itself evaluates against, on the exact model this example uses
(Qwen2-Audio-7B-Instruct). Discriminative yes/no questions ("Is there a sound
of X in the audio?") over AudioCaps clips, three negative-sampling
strategies (random / popular / adversarial); this script defaults to
``random`` (the most representative slice) but any strategy works.

The questions dataset (``kuanhuggingface/AudioHallucination_AudioCaps-*``)
carries no audio itself -- only an ``audio_index`` that is an AudioCaps
``youtube_id``. Audio bytes come from a second dataset
(``OpenSound/AudioCaps``, an HF mirror of AudioCaps that DOES embed decoded
audio) via a plain join on that id. Both are read directly through pyarrow
off the auto-converted parquet files (same reasoning as
examples/m1_m4/mmau_qwen2_audio/download_mmau.py: reading an ``Audio``
feature through ``datasets`` pulls in torchcodec, which needs a newer
ffmpeg build than most base images ship; pyarrow reads the identical parquet
rows without that dependency), each row's audio re-encoded via ``ffmpeg`` to
mono float32 WAV at 16 kHz -- the sample rate contract
``hf_local._resolve_audio`` assumes throughout the rest of this pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
DATA = HERE / "data"
QUESTIONS_REPO_FMT = "kuanhuggingface/AudioHallucination_AudioCaps-{sampling}"
AUDIO_REPO = "OpenSound/AudioCaps"
# Qwen2-Audio-7B-Instruct's WhisperFeatureExtractor.chunk_length is 30s
# (verified live against the processor in mmau_qwen2_audio's download
# script); AudioCaps clips are ~10s by construction so this rarely bites,
# kept as the same safety margin for consistency.
MAX_DURATION_SEC = 29.5


def _duration_seconds(wav_path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _find_parquets(repo: str) -> list[str]:
    """Resolve ALL auto-converted parquet shards for a dataset's default
    config/test split, without depending on the exact refs/convert/parquet
    branch layout (checked once against the live Hub instead of assumed).

    HF's auto-convert shards large splits (e.g. ``test-00000-of-00041.parquet``
    .. ``test-00040-of-00041.parquet``) -- returning only the alphabetically
    first shard silently reads ~1/N of the split (caught live: OpenSound/
    AudioCaps's 41-shard test split returned 108 of its ~4.4k rows this way).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    candidates = [f for f in files if f.endswith(".parquet")]
    if not candidates:
        # Fall back to the well-known auto-convert location.
        return ["default/test/0000.parquet"]
    # Prefer "test" split file(s) if present, else every parquet found.
    test_files = [f for f in candidates if "test" in f.lower()]
    return sorted(test_files or candidates)


def _read_parquets(repo: str, filenames: list[str]) -> "pq.Table":
    from huggingface_hub import hf_hub_download

    tables = [
        pq.read_table(hf_hub_download(repo, name, repo_type="dataset"))
        for name in filenames
    ]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def _decode_row_audio(raw_bytes: bytes, wav_path: Path) -> float:
    with tempfile.NamedTemporaryFile(suffix=".src", delete=False) as raw:
        raw.write(raw_bytes)
        raw_path = raw.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", raw_path,
                "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path),
            ],
            check=True, capture_output=True,
        )
    finally:
        Path(raw_path).unlink(missing_ok=True)
    return _duration_seconds(wav_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", default="Random", choices=["Random", "Popular", "Adversarial"])
    parser.add_argument("--limit", type=int, default=120, help="rows to keep after filtering")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--scan-rows", type=int, default=1000, help="question rows to scan before giving up")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg/ffprobe must be on PATH to decode AudioCaps' embedded audio")

    questions_repo = QUESTIONS_REPO_FMT.format(sampling=args.sampling)
    q_table = _read_parquets(questions_repo, _find_parquets(questions_repo))
    print(f"questions: {questions_repo} ({q_table.num_rows} rows)")

    a_table = _read_parquets(AUDIO_REPO, _find_parquets(AUDIO_REPO))
    print(f"audio source: {AUDIO_REPO} ({a_table.num_rows} rows)")

    # Build youtube_id -> row index once; AudioCaps' test split can have more
    # than one audiocap_id per youtube_id (multiple captions/timestamps for
    # the same clip) -- any one of them is the same underlying audio.
    audio_by_ytid: dict[str, int] = {}
    ytid_col = a_table.column("youtube_id").to_pylist()
    for idx, ytid in enumerate(ytid_col):
        audio_by_ytid.setdefault(ytid, idx)

    import random

    order = list(range(q_table.num_rows))
    random.Random(args.seed).shuffle(order)
    order = order[: min(args.scan_rows, q_table.num_rows)]

    audio_dir = DATA / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    n_skipped_no_audio_match = 0
    n_skipped_duration = 0
    seen_ytids: set[str] = set()

    for row_index in order:
        if len(records) >= args.limit:
            break
        row = q_table.slice(row_index, 1).to_pylist()[0]
        # audio_index carries AudioSet/AudioCaps' own "Y<11-char-id>" prefix
        # (checked live: 'Y7fmOlUlwoNg'); OpenSound/AudioCaps' youtube_id
        # column strips it ('7fmOlUlwoNg') -- same clip, different id
        # convention, not a mismatch in content.
        ytid = str(row["audio_index"]).removeprefix("Y")
        if ytid not in audio_by_ytid:
            n_skipped_no_audio_match += 1
            continue

        entry_id = str(row["entry_id"])
        wav_path = audio_dir / f"{ytid}.wav"
        if ytid not in seen_ytids:
            a_row = a_table.slice(audio_by_ytid[ytid], 1).to_pylist()[0]
            audio_field = a_row["audio"]
            raw_bytes = audio_field["bytes"] if isinstance(audio_field, dict) else audio_field
            duration = _decode_row_audio(raw_bytes, wav_path)
            if duration > MAX_DURATION_SEC:
                n_skipped_duration += 1
                wav_path.unlink(missing_ok=True)
                continue
            seen_ytids.add(ytid)
        elif not wav_path.is_file():
            # A dedup'd id that got skipped for duration earlier -- skip
            # every question that shares it too, don't re-decode.
            n_skipped_duration += 1
            continue

        records.append({
            "id": entry_id,
            "instruction": (
                f"{str(row['prompt_text']).strip()} "
                "Answer with only the single word Yes or No."
            ),
            "expected": str(row["label"]).strip(),
            "audio_path": str(wav_path.relative_to(DATA)),
            "task": "yes_no",
            "metadata": {
                "object": row.get("object", ""),
                "attribute": row.get("attribute", ""),
                "sampling": row.get("sampling", args.sampling),
                "youtube_id": ytid,
                "source_dataset": "AudioCaps",
            },
        })

    manifest_path = DATA / "audiohallucination.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "kept": len(records),
        "skipped_no_audio_match": n_skipped_no_audio_match,
        "skipped_duration_over_limit": n_skipped_duration,
        "scanned": len(order),
        "source_total_rows": q_table.num_rows,
        "sampling": args.sampling,
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    if len(records) < args.limit:
        print(
            f"warning: kept only {len(records)}/{args.limit} requested rows "
            f"after scanning {len(order)}/{q_table.num_rows}; raise --scan-rows or lower --limit"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
