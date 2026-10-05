"""AF-Reasoning-Eval / AQA-MCQ (NVIDIA Audio Flamingo): four-way multiple choice
requiring reasoning over closely-related options, not just single-event recall.

The eval set itself (76 items) ships as JSON in the audio-flamingo repo
(``AF_Reasoning_Eval/AQA_MCQ.json``, branch ``soundCoT``) and only references
its audio by filename (``dataset_path: "Clotho-AQA/audio_files"``) -- it does
not embed or host the clips. ``gijs/clothoaqa`` mirrors Clotho-AQA's audio
files as an embedded HF ``Audio`` column keyed by that same ``file_name``, so
this downloader joins the two: NVIDIA's questions/choices/gold answer, Clotho's
audio bytes. Every question is verified present in the mirror's ``test`` split
before being kept (see the module docstring's own commit for the verification
script) -- 70 unique clips answer all 76 questions (some clips carry more than
one question).
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from . import _audio
from .base import Task, _protocol, write_manifest

EVAL_JSON_URL = (
    "https://raw.githubusercontent.com/NVIDIA/audio-flamingo/soundCoT/"
    "AF_Reasoning_Eval/AQA_MCQ.json"
)
AUDIO_REPO = "gijs/clothoaqa"
AUDIO_SPLIT_PREFIX = "data/test-"
AUDIO_SPLIT_SHARDS = 35
LETTERS = ["A", "B", "C", "D"]
_OUTPUT_RE = re.compile(r"^\(([A-D])\)\s*(.*)$")
_OPTION_RE = re.compile(r"\([A-D]\)\s*[^()]*?(?=\s*\([A-D]\)|$)")


def _parse_item(item: dict) -> tuple[str, list[str], str]:
    """``(question, ["(A) ...", ...], gold_letter)`` from one AQA_MCQ.json entry.

    The source prompt embeds NVIDIA's own CoT-tag instruction ("Output the
    answer with <SUMMARY>, <CAPTION>, ..."); this codebase's own house style
    (plain "reply with only the option letter", matching MMAU) replaces it in
    :func:`task_prompt` rather than carrying that scaffolding into every model.
    """
    m = _OUTPUT_RE.match(str(item["output"]).strip())
    if not m:
        raise ValueError(f"unparseable output {item['output']!r}")
    letter = m.group(1)
    q_part = str(item["prompt"]).split("Output the answer")[0].strip()
    question, _, opts_part = q_part.partition("\n(A)")
    opts_part = "(A)" + opts_part
    options = [o.strip().rstrip(".").strip() for o in _OPTION_RE.findall(opts_part)]
    if len(options) != 4:
        raise ValueError(f"expected 4 options, got {options!r} from prompt {item['prompt']!r}")
    return question.strip(), options, letter


def task_prompt(question: str, options: list[str]) -> str:
    return (f"{question}\n\n" + "\n".join(options) + "\n\n"
            "Listen to the audio and reply with only the option letter (A, B, C, or D).")


def download(out_dir: Path, limit: int = 76, seed: int = 20260814) -> dict:
    """*seed*/shuffling are accepted for interface parity but unused: the source
    is a fixed 76-item eval set, not a pool to sample from -- every item is
    kept up to *limit*."""
    from huggingface_hub import hf_hub_download

    _audio.require_ffmpeg()
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(EVAL_JSON_URL, timeout=30) as resp:
        payload = json.loads(resp.read())
    items = list(payload["data"].items())[: max(0, limit)]

    parsed: dict[str, dict] = {}
    n_bad = 0
    for item_id, item in items:
        try:
            question, options, letter = _parse_item(item)
        except ValueError:
            n_bad += 1
            continue
        parsed[str(item["name"])] = {
            "item_id": item_id, "question": question, "options": options, "letter": letter,
        }
    wanted = set(parsed)

    # Scan the Clotho-AQA mirror's test shards for the filenames the eval set
    # actually needs, decoding audio only for those -- the mirror is >1 GB
    # across 35 shards but the eval set draws only ~70 unique clips from it.
    import pyarrow.parquet as pq

    resolved: dict[str, str] = {}  # file_name -> relative wav path
    for shard in range(AUDIO_SPLIT_SHARDS):
        remaining = wanted - set(resolved)
        if not remaining:
            break
        shard_path = hf_hub_download(
            AUDIO_REPO, f"{AUDIO_SPLIT_PREFIX}{shard:05d}-of-{AUDIO_SPLIT_SHARDS:05d}.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(shard_path, columns=["file_name", "audio"])
        names = table.column("file_name").to_pylist()
        for row_index, name in enumerate(names):
            if name not in remaining:
                continue
            wav_path = audio_dir / f"{Path(name).stem}.wav"
            if not wav_path.is_file():
                audio_cell = table.column("audio")[row_index].as_py()
                _audio.decode_to_wav(audio_cell["bytes"], wav_path)
            resolved[name] = f"audio/{wav_path.name}"
            remaining.discard(name)

    rows, n_noaudio, n_dur = [], 0, 0
    for name, meta in parsed.items():
        rel_path = resolved.get(name)
        if rel_path is None:
            n_noaudio += 1
            continue
        duration = _audio.duration_seconds(audio_dir / Path(rel_path).name)
        if duration > _audio.MAX_DURATION_SEC:
            n_dur += 1
            (audio_dir / Path(rel_path).name).unlink(missing_ok=True)
            continue
        rows.append({
            "id": meta["item_id"], "dataset": "audio-flamingo/AF-Reasoning-Eval-AQA-MCQ",
            "source_index": int(meta["item_id"]), "sample_seed": seed,
            "image": None, "audio": rel_path,
            "prompt": task_prompt(meta["question"], meta["options"]),
            "answers": [meta["letter"]], "choices": LETTERS,
            "task": "multiple_choice_letter", "numeric_tolerance": 0.0,
            "metadata": {
                "options": meta["options"], "clotho_file_name": name,
                "duration_sec": round(duration, 3),
                "source_dataset": "Clotho-AQA (via gijs/clothoaqa mirror)",
            },
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {
        "kept": len(rows), "skipped_unparseable": n_bad, "skipped_audio_not_found": n_noaudio,
        "skipped_duration_over_limit": n_dur, "scanned": len(items),
        "manifest": str(out_dir / "manifest.json"),
    }


def protocol(model_label: str):
    return _protocol(
        description=(
            f"We evaluate an audio-language model ({model_label}) on AF-Reasoning-Eval's "
            "AQA-MCQ set (NVIDIA Audio Flamingo): four-way multiple-choice questions over "
            "a short clip, built to require discriminating among closely related choices "
            "rather than recalling a single salient event. We want to know what "
            "distinguishes the questions it gets right from the ones it gets wrong."
        ),
        task_domain="audio question answering, closely-related-choice discrimination (AF-Reasoning-Eval)",
        success_criteria="The selected option letter must match the gold answer for the clip.",
        target_modalities=frozenset({"text", "audio"}),
        output_contract={"kind": "multiple_choice_letter", "choices": LETTERS},
    )


TASK = Task(
    name="af_reasoning_mcq", modality="alm", kind="multiple_choice_letter",
    title="AF-Reasoning-Eval/AQA-MCQ",
    download=download, protocol=protocol,
    pinned_m1=(
        "answer_extraction_audit", "termination_audit", "selfcheck_consistency",
        "format_sensitivity", "self_consistency", "calibration", "logprob_entropy",
        "coverage_verification_gap",
    ),
    default_limit=76, default_seed=20260814, max_new_tokens=64,
    output_contract={"kind": "multiple_choice_letter", "choices": LETTERS},
    source="NVIDIA/audio-flamingo AF_Reasoning_Eval (AQA_MCQ) + gijs/clothoaqa (audio)",
)
