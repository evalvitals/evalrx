"""HotpotQA, the GEPA split (arXiv:2507.19457) — the 300-question test set.

GEPA's appendix fixes a 150/300/300 train/val/test split of HF
``hotpotqa/hotpot_qa`` (config ``fullwiki``, split ``train``, 90,447 rows):
slice the split IN ORDER into a test pool (first 40 %), a val pool (middle
40 %) and a train pool (last 20 %), then sample each pool with a fresh
``random.Random(); rng.seed(1)`` (``gepa-artifact`` ``benchmarks/benchmark.py``).
``download`` re-runs that recipe bit-for-bit — sampling indices instead of row
objects, which picks the identical positions — and freezes the TEST pool's 300
questions in sample order; these are the same items behind GEPA Table 1
(Qwen3-8B 42.33 EM).

The protocol here is NOT GEPA's: their program answers via two-hop ColBERTv2
retrieval over a hosted Wikipedia-2017 index that is not reproducible locally.
We hand the model HotpotQA's own 10 candidate paragraphs (2 gold + 8
distractors, the dataset's distractor setting, formatted exactly like
``examples/dataset_selection`` ``band_locate._hotpot_context``) so the task
stays multi-hop READING over the same items, and grade with ``short_answer_em`` —
the same SQuAD normalisation as GEPA's ``dspy.evaluate.answer_exact_match``. Published GEPA numbers are
therefore a distribution anchor for these items, not a per-item comparison.
Everything is drawn from the 2018 TRAIN split: a contamination risk worth
naming in any writeup.
"""

from __future__ import annotations

import random
from pathlib import Path

from .base import Task, _protocol, write_manifest
from .llm import PINNED_M1

HF_REPO = "hotpotqa/hotpot_qa"
EXPECTED_TRAIN_ROWS = 90_447
SIZES = {"train": 150, "val": 300, "test": 300}
GEPA_SEED = 1
_CONTEXT_CHAR_CAP = 14_000
INSTRUCTION = ("Solve the problem. Put the final answer on its own last line "
               "as 'Answer: <answer>'.")


def _load_train_rows() -> list[dict]:
    """``fullwiki/train`` in the hub's canonical row order (the split logic
    depends on nothing but this order)."""
    from datasets import load_dataset

    rows = load_dataset(HF_REPO, "fullwiki", split="train")
    if rows.num_rows != EXPECTED_TRAIN_ROWS:
        raise SystemExit(
            f"{HF_REPO} fullwiki/train has {rows.num_rows} rows, expected "
            f"{EXPECTED_TRAIN_ROWS}; the GEPA split is defined on that revision "
            "and cannot be reconstructed from a different one"
        )
    return list(rows)


def _pool_bounds(split: str, n: int) -> tuple[int, int]:
    """GEPA ``create_splits``: test = first 40 %, val = middle 40 %, train = last 20 %."""
    cuts = {"test": (0, int(0.4 * n)), "val": (int(0.4 * n), int(0.8 * n)),
            "train": (int(0.8 * n), n)}
    return cuts[split]


def _gepa_indices(pool_len: int, size: int, seed: int) -> list[int]:
    """GEPA ``trim_dataset`` on index space: ``rng.sample`` picks positions from
    ``(len, k, seed)`` alone, so sampling ``range(pool_len)`` yields exactly the
    rows that sampling the row list would, in the same order; a pool smaller
    than ``size`` is kept whole in pool order (``trim_dataset`` returns it as is)."""
    if size >= pool_len:
        return list(range(pool_len))
    return random.Random(seed).sample(range(pool_len), size)


def _context(row: dict) -> str:
    """The 10 candidate paragraphs (2 gold + 8 distractors). ``context`` is a
    dict of PARALLEL lists (``title`` / ``sentences``), so zipping is the only
    correct read; same formatting + cap as ``band_locate._hotpot_context``."""
    ctx = row.get("context")
    if not isinstance(ctx, dict):
        return ""
    paras = [f"{title}: {''.join(sents).strip()}"
             for title, sents in zip(ctx.get("title") or [], ctx.get("sentences") or [])
             if str(title).strip()]
    return "\n\n".join(paras)[:_CONTEXT_CHAR_CAP]


def download(out_dir: Path, limit: int = 300, seed: int = GEPA_SEED, split: str = "test") -> dict:
    """Freeze the GEPA *split* (test by default) in GEPA sample order.

    ``limit`` > 0 keeps the first ``limit`` rows of that order (a deterministic
    prefix); 0 keeps the whole split. ``seed`` = 1 is the published split —
    anything else is a same-recipe robustness variant, not GEPA's set.
    """
    if split not in SIZES:
        raise ValueError(f"unknown GEPA split {split!r}; expected one of {tuple(SIZES)}")
    out_dir = Path(out_dir)
    all_rows = _load_train_rows()
    start, end = _pool_bounds(split, len(all_rows))
    picked = _gepa_indices(end - start, SIZES[split], seed)
    if limit and limit > 0:
        picked = picked[:limit]
    rows = []
    for rank, pool_index in enumerate(picked):
        source_index = start + pool_index
        row = all_rows[source_index]
        context = _context(row)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not (context and question and answer):
            raise SystemExit(f"row {source_index} is not gradable (empty context/question/answer)")
        supporting = row.get("supporting_facts") or {}
        rows.append({
            "id": f"hotpotqa-gepa-{split}-{source_index:05d}",
            "dataset": HF_REPO, "subset": f"fullwiki/train (GEPA {split} pool)",
            "source_index": source_index, "sample_rank": rank, "sample_seed": seed,
            "image": None, "audio": None,
            "prompt": f"Context:\n{context}\n\nQuestion: {question}\n\n{INSTRUCTION}",
            "answers": [answer],
            "task": "short_answer_em", "numeric_tolerance": 0.0,
            "metadata": {"hotpot_id": str(row.get("id", "")), "type": str(row.get("type", "")),
                         "level": str(row.get("level", "")), "gepa_split": split,
                         "supporting_titles": sorted(set(supporting.get("title") or []))},
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "split": split, "train_rows": len(all_rows),
            "pool": [start, end], "manifest": str(out_dir / "manifest.json")}


def protocol(model_label: str):
    return _protocol(
        description=(
            f"A text-only LLM ({model_label}) answers the 300-question test set of the GEPA "
            "split of HotpotQA (fullwiki/train, seed-1 sample — the same items as GEPA's "
            "Table 1). Each prompt carries the dataset's own 10 candidate Wikipedia "
            "paragraphs (2 gold + 8 distractors); answering needs a two-hop combination "
            "of two of them (bridge questions) or a comparison of two entities "
            "(comparison questions, often with a yes/no answer). Failure cases are items "
            "answered incorrectly under normalised exact match; the batch also contains, "
            "as controls, items the model answered correctly."
        ),
        task_domain="multi-hop reading comprehension (HotpotQA, distractor setting)",
        success_criteria=(
            "The final answer must equal the gold answer after SQuAD-style normalisation "
            "(case, punctuation, articles); no partial credit, no LLM judge. Verbose "
            "answers that contain the gold span plus extra words score 0."
        ),
        failure_patterns=(
            "failed second hops (the bridge entity is found but its property is not); "
            "answers copied from a distractor paragraph about a similar entity; "
            "unsupported recall that ignores the provided context; comparison questions "
            "flipped to the wrong side; correct content lost to the exact-match surface "
            "(extra words, wrong granularity such as a full date for a year)"
        ),
        target_modalities=frozenset({"text"}),
        metadata={
            "gepa_reference": ("GEPA (arXiv:2507.19457) reports Qwen3-8B 42.33 EM on these "
                               "same 300 items under its two-hop ColBERTv2 retrieval "
                               "protocol; with the 10 in-prompt paragraphs used here that "
                               "number anchors the distribution, not per-item difficulty"),
            "contamination_note": "all items come from HotpotQA's public 2018 TRAIN split",
        },
    )


TASK = Task(
    name="hotpotqa_gepa", modality="llm", kind="short_answer_em", title="HotpotQA/GEPA-test",
    download=download, protocol=protocol,
    pinned_m1=PINNED_M1,
    default_limit=300, default_seed=GEPA_SEED, max_new_tokens=512,
    short_answer=False,
    source="hotpotqa/hotpot_qa fullwiki/train, GEPA split seed 1 (150/300/300), arXiv:2507.19457",
)
