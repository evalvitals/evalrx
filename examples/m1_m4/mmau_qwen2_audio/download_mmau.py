#!/usr/bin/env python3
"""Download MMAU test-mini (Sakshi et al. 2024) and freeze a local sample.

The paper's own benchmark for TCD (Table 1). Source:
``gamma-lab-umd/MMAU-test-mini`` on the Hub -- a single parquet with the
official 1000-row test-mini split, audio embedded as raw file bytes (not an
HF ``Audio`` feature -- reading it through ``datasets`` pulls in torchcodec,
which needs an ffmpeg build newer than most base images ship; pyarrow reads
the same parquet without that dependency, so this script never imports
``datasets`` at all).

Each row's audio bytes are decoded via ``ffmpeg`` (the same binary
``hf_local._resolve_audio`` already requires) straight to mono float32 WAV
at 16 kHz, matching the sample rate contract the whole audio pipeline
assumes. Clips longer than Qwen2-Audio-7B-Instruct's 30s encoder window are
SKIPPED here, with a recorded count -- letting one through would surface as
a mid-run ``_check_audio_duration`` crash instead of a clean, auditable
exclusion.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).parent
DATA = HERE / "data"
REPO = "gamma-lab-umd/MMAU-test-mini"
FILENAME = "test_mini.parquet"
# Qwen2-Audio-7B-Instruct's WhisperFeatureExtractor.chunk_length is 30s
# (verified live against the processor, see hf_local._check_audio_duration).
# A small safety margin below that avoids a clip that's a few frames over
# after resampling from silently tripping the guard.
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


def _letter(choices: list[str], answer: str) -> str:
    """MMAU's ``answer`` is the full ``"(A) Man"`` string; choices are the same
    format. Match on the leading letter rather than exact text, since only the
    letter is the task contract we ask the model for."""
    answer = answer.strip()
    for choice in choices:
        if choice.strip() == answer:
            marker = choice.strip()[:2]  # "(A"
            if len(marker) == 2 and marker[0] == "(" and marker[1].isalpha():
                return marker[1].upper()
    # Fall back to a leading "(X)" pattern directly on the answer string.
    stripped = answer.strip()
    if len(stripped) >= 2 and stripped[0] == "(" and stripped[1].isalpha():
        return stripped[1].upper()
    raise ValueError(f"could not extract an option letter from answer={answer!r} choices={choices!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000, help="rows to keep after filtering")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--scan-rows", type=int, default=1000,
        help="rows to scan before giving up (the official test-mini split has 1000 rows)",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg/ffprobe must be on PATH to decode MMAU's embedded audio")

    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(REPO, FILENAME, repo_type="dataset")
    table = pq.read_table(parquet_path)
    n_total = table.num_rows
    order = list(range(n_total))
    random.Random(args.seed).shuffle(order)
    order = order[: min(args.scan_rows, n_total)]

    audio_dir = DATA / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    n_skipped_duration = 0
    n_skipped_unparseable = 0

    for row_index in order:
        if len(records) >= args.limit:
            break
        row = table.slice(row_index, 1).to_pylist()[0]
        other = json.loads(row["other_attributes"])
        row_id = str(other.get("id", row_index))
        choices = list(row["choices"])
        try:
            answer_letter = _letter(choices, str(row["answer"]))
        except ValueError:
            n_skipped_unparseable += 1
            continue

        wav_path = audio_dir / f"{row_id}.wav"
        with tempfile.NamedTemporaryFile(suffix=".src", delete=False) as raw:
            raw.write(row["context"]["bytes"])
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

        duration = _duration_seconds(wav_path)
        if duration > MAX_DURATION_SEC:
            n_skipped_duration += 1
            wav_path.unlink(missing_ok=True)
            continue

        records.append(
            {
                "id": row_id,
                "instruction": row["instruction"],
                "choices": choices,
                "expected": answer_letter,
                "audio_path": str(wav_path.relative_to(DATA)),
                "duration_sec": round(duration, 3),
                "task": "multiple_choice",
                "metadata": {
                    "mmau_task": other.get("task", ""),           # sound/music/speech
                    "category": other.get("category", ""),
                    "sub_category": other.get("sub-category", ""),
                    "difficulty": other.get("difficulty", ""),
                    "source_dataset": other.get("dataset", ""),
                },
            }
        )

    manifest_path = DATA / "mmau_test_mini.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "kept": len(records),
        "skipped_duration_over_limit": n_skipped_duration,
        "skipped_unparseable_answer": n_skipped_unparseable,
        "scanned": len(order),
        "source_total_rows": n_total,
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    if len(records) < args.limit:
        print(
            f"warning: kept only {len(records)}/{args.limit} requested rows "
            f"after scanning {len(order)}/{n_total}; raise --scan-rows or lower --limit"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
